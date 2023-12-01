

import gpflow
import tensorflow_probability as tfp
from candas.utils.gp_utils import get_ℓ_prior
from candas.learn import ParameterArray, ParameterSet, Standardizer #, Linear_offset
from candas.learn import ParameterArray as parray

from candas.learn.regression import Regressor

from candas.plotting.regression_plots import plot_uparray, contourf_uparray, unstandardize_axis_labels

from typing import Optional
import numpy as np
import tensorflow as tf
from candas.utils.misc import assert_in, assert_is_subset
from gpflow import covariances, kernels, likelihoods
from gpflow.base import Parameter, _cast_to_dtype
from gpflow.config import default_float, default_jitter
from gpflow.expectations import expectation
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction, Zero
from gpflow.probability_distributions import DiagonalGaussian
from gpflow.utilities import positive, to_default_float, ops
from gpflow.utilities.ops import pca_reduce
from gpflow.models.gpr import GPR
from gpflow.models.model import GPModel, MeanAndVariance
from gpflow.models.training_mixins import InputData, InternalDataTrainingLossMixin, OutputData
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper
from gpflow.covariances.dispatch import Kuf, Kuu
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
# gpflow.config.set_default_jitter(tf.convert_to_tensor(1e-6, dtype=default_float()))


class LVMOGP(GPModel, InternalDataTrainingLossMixin):
    def __init__(
        self,
        data: OutputData,
        X_data: tf.Tensor,
        X_data_fn: tf.Tensor,
        H_data_mean: tf.Tensor,
        H_data_var: tf.Tensor,
        kernel: Kernel,
        num_inducing_variables: Optional[int] = None,
        inducing_variable=None,
        H_prior_mean=None,
        H_prior_var=None,
        MAP=False,
    ):
        """
        Initialise  LVMOGP object. This method only works with a Gaussian likelihood.

        :param data: data matrix, size N (number of points) x D_out (output dimensions)
        :param X_data: observed inputs, size N (number of points) x D (input dimensions)
        :param H_data: initial latent positions, size N (number of points) x Q (latent dimensions).
        :param H_data_var: variance of latent positions ([N, Q]), for the initialisation of the latent space.
        :param kernel: kernel specification
        :param num_inducing_variables: number of inducing points, M
        :param inducing_variable: matrix of inducing points, size M (inducing points) x Q (latent dimensions). By default
            random permutation of X_data_mean.
        :param H_prior_mean: prior mean used in KL term of bound. By default 0. Same size as X_data_mean.
        :param H_prior_var: prior variance used in KL term of bound. By default 1.
        :param MAP: boolean if the elbo should just be the datafit term or if it should include the KL
        """
        num_data, num_latent_gps = X_data.shape
        num_fns, num_latent_dims = H_data_mean.shape
        super().__init__(kernel, likelihoods.Gaussian(), num_latent_gps=num_latent_gps)
        self.data = data_input_to_tensor(data)
        self.MAP = MAP

        self.X_data = Parameter(X_data, trainable=False)
        self.X_data_fn = Parameter(X_data_fn, trainable=False)
        self.H_data_mean = Parameter(H_data_mean)
        self.H_data_var = Parameter(H_data_var, transform=positive())

        self.num_fns = num_fns
        self.num_latent_dims = num_latent_dims
        self.num_data = num_data
        self.output_dim = self.data.shape[-1]


        assert X_data.shape[0] == self.data.shape[0], "X mean and Y must be same size."
        assert H_data_mean.shape[0] == H_data_var.shape[0], "H mean and var should be the same length"

        if (inducing_variable is None) == (num_inducing_variables is None):
            raise ValueError(
                "BayesianGPLVM needs exactly one of `inducing_variable` and `num_inducing_variables`"
            )

        if inducing_variable is None:
            # By default we initialize by subset of initial latent points
            # Note that tf.random.shuffle returns a copy, it does not shuffle in-
            X_mean_tilde, X_var_tilde = self.fill_Hs()
            Z = tf.random.shuffle(X_mean_tilde)[:num_inducing_variables]
            inducing_variable = InducingPoints(Z)

        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        assert X_data.shape[1] == self.num_latent_gps

        # deal with parameters for the prior mean variance of X
        if H_prior_mean is None:
            H_prior_mean = tf.zeros((self.num_fns, self.num_latent_dims), dtype=default_float())
        if H_prior_var is None:
            H_prior_var = tf.ones((self.num_fns, self.num_latent_dims))

        self.H_prior_mean = tf.convert_to_tensor(np.atleast_1d(H_prior_mean), dtype=default_float())
        self.H_prior_var = tf.convert_to_tensor(np.atleast_1d(H_prior_var), dtype=default_float())

        assert self.H_prior_mean.shape[0] == self.num_fns
        assert self.H_prior_mean.shape[1] == self.num_latent_dims
        assert self.H_prior_var.shape[0] == self.num_fns
        assert self.H_prior_var.shape[1] == self.num_latent_dims

    def fill_Hs(self, X_data=None, X_data_fn=None):
        """append latent Hs to Xs by function number, to give X_tilde"""

        if X_data is None:
            X_data = self.X_data
        if X_data_fn is None:
            X_data_fn = self.X_data_fn

        H_mean_vect =  tf.reshape(tf.gather(_cast_to_dtype(self.H_data_mean, dtype=default_float()),
                                      _cast_to_dtype(X_data_fn, dtype=tf.int64)),
                                   [X_data.shape[0], self.num_latent_dims])
        H_var_vect = tf.reshape(tf.gather(_cast_to_dtype(self.H_data_var, dtype=default_float()),
                                           _cast_to_dtype(X_data_fn, dtype=tf.int64)),
                                   [X_data.shape[0], self.num_latent_dims])

        return tf.concat([X_data, H_mean_vect], axis=1),  \
               tf.concat([tf.ones(X_data.shape, dtype=default_float())*1e-5, H_var_vect], axis=1)

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        Y_data = self.data
        mu, var = self.fill_Hs()
        pH = DiagonalGaussian(mu, var)

        num_inducing = self.inducing_variable.num_inducing
        psi0 = tf.reduce_sum(expectation(pH, self.kernel))
        psi1 = expectation(pH, (self.kernel, self.inducing_variable))
        psi2 = tf.reduce_sum(
            expectation(
                pH, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
            ),
            axis=0,
        )
        cov_uu = covariances.Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(cov_uu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        # tf.print(B)
        LB = tf.linalg.cholesky(B)
        log_det_B = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma

        # KL[q(x) || p(x)]
        dH_data_var = (
            self.H_data_var
            if self.H_data_var.shape.ndims == 2
            else tf.linalg.diag_part(self.H_data_var)
        )
        NQ = to_default_float(tf.size(self.H_data_mean))
        D = to_default_float(tf.shape(Y_data)[1])
        KL = -0.5 * tf.reduce_sum(tf.math.log(dH_data_var))
        KL += 0.5 * tf.reduce_sum(tf.math.log(self.H_prior_var))
        KL -= 0.5 * NQ
        KL += 0.5 * tf.reduce_sum(
            (tf.square(self.H_data_mean - self.H_prior_mean) + dH_data_var) / self.H_prior_var
        )

        # compute log marginal bound
        ND = to_default_float(tf.size(Y_data))
        bound = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(Y_data)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.linalg.diag_part(AAT)))

        if not self.MAP:
            bound -= KL
        # tf.print(bound)
        # tf.print(bound)
        # if tf.math.is_nan(bound):
        #     raise ValueError("elbo is NaN")

        return bound

    def predict_f(
            self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points.
        Note that this is very similar to the SGPR prediction, for which
        there are notes in the SGPR notebook.
        Note: This model does not allow full output covariances.
        :param Xnew: points at which to predict
        """
        if full_output_cov:
            raise NotImplementedError

        X_mean_tilde, X_var_tilde = self.fill_Hs()
        pH = DiagonalGaussian(X_mean_tilde, X_var_tilde)

        Xnew_mean, Xnew_var = self.fill_Hs(X_data=Xnew[:, :-1], X_data_fn=Xnew[:,-1])

        # Xnew_mean = Xnew[0]
        # Xnew_var = Xnew[1]
        pH_new = DiagonalGaussian(Xnew_mean, Xnew_var)
        psi1_new = expectation(pH_new, (self.kernel, self.inducing_variable))
        pX = pH
        Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing
        psi1 = expectation(pX, (self.kernel, self.inducing_variable))
        psi2 = tf.reduce_sum(
            expectation(
                pX, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
            ),
            axis=0,
        )
        jitter = default_jitter()
        sigma2 = self.likelihood.variance
        L = tf.linalg.cholesky(covariances.Kuu(self.inducing_variable, self.kernel, jitter=jitter))

        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True)
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma2
        tmp1 = tf.linalg.triangular_solve(L, tf.transpose(psi1_new), lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                    self.kernel(Xnew_mean)
                    + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                    - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            shape = tf.stack([1, 1, tf.shape(Y_data)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = (
                    self.kernel(Xnew_mean, full_cov=False)
                    + tf.reduce_sum(tf.square(tmp2), axis=0)
                    - tf.reduce_sum(tf.square(tmp1), axis=0)
            )
            shape = tf.stack([1, tf.shape(Y_data)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew_mean), var


    def predict_log_density(self, data: OutputData) -> tf.Tensor:
        raise NotImplementedError

    def predict_points(self, points, param=None, with_noise=True, lg10_Copies=None, **kwargs):
        """Make predictions at supplied points

        Parameters
        ----------
        points : ParameterArray
            1-D ParameterArray vector of coordinates for prediction, must have one layer per ``self.dims``
        param : str or list of str, optional
            Parameter for which to make predictions
        with_noise : bool, default True
            Whether to incorporate aleatoric uncertainty into prediction error
        lg10_Copies : float, optional
            Single value of lg10_Copies applied to all points
        **kwargs
            Additional keyword arguments passed to subclass-specific :meth:`predict` method

        Returns
        -------
        prediction : UncertainParameterArray
            Predictions as a `uparray`
        """

        points = np.atleast_1d(points)
        assert points.ndim == 1
        assert set(self.dims) - set(['Parameter']) == set(points.names), \
            'All model dimensions must be present in "points" parray.'

        if 'Parameter' in self.coregion_dims:
            # Multiple parameters are possible, determine which ones to predict
            if param is None:
                # predict all parameters in model
                param = self.coregion_levels['Parameter']
            elif isinstance(param, list):
                assert_is_subset('Parameters', param, self.coregion_levels['Parameter'])
            elif isinstance(param, str):
                param = [param]
                assert_is_subset('Parameters', param, self.coregion_levels['Parameter'])
            else:
                raise ValueError('"param" must be list, string, or None')

            # Get model coordinates for each parameter to be predicted
            param_coords = [self.coregion_coords['Parameter'][p] for p in param]

            # Convert input points to tall array and tile once for each parameter, adding the respective coordinate
            tall_points = parray.vstack([points.add_layers(Parameter=coord)[:, None] for coord in param_coords])
        else:
            # If 'Parameter' is not in coregion_dims, it must be in filter_dims, and only one is possible
            param = self.filter_dims['Parameter']
            # Convert input points to tall array

            tall_points = points[:, None]

        points_array = np.hstack([tall_points[dim].z.values() for dim in self.dims])

        idx_c = [self.dims.index(dim) for dim in self.coregion_dims]
        if hasattr(self, 'lvmogps'):
            test = [self.model.H_data_mean[int(point)] for point in points_array[:, 2]]
            test2 = points_array[:, -1 * len(self.coregion_dims)]
            points_array_mean = np.hstack([points_array[:, -1 * len(self.coregion_dims)].reshape(1, 1),
                                      np.array(
                                          [self.model.H_data_mean[int(point)] for point in points_array[:, 2]]).reshape(
                                          1, 2)])
            points_array_var = np.hstack([np.zeros(points_array.shape),
                                      np.array(
                                          [self.model.H_data_var[int(point)] for point in points_array[:, 2]]).reshape(
                                          1, 2)])
            points_array = [points_array_mean, points_array_var]

        # Prediction means and variance as a list of numpy vectors
        pred_mean, pred_variance = self.predict(points_array, with_noise=with_noise, **kwargs)
        self.predictions_X = points

        if 'τ' in param:
            # Extract copy number from `points` if present, otherwise use the value provided in the function call
            lg10_Copies = points.get('lg10_Copies', lg10_Copies)
            if lg10_Copies is None and 'lg10_Copies' in self.filter_dims.keys():
                lg10_Copies = self.filter_dims['lg10_Copies']
            elif lg10_Copies is None:
                raise ValueError('Cannot predict τ without lg10_Copies')

            # Get standardized copy number
            if type(lg10_Copies) in [int, float, list, np.ndarray]:
                lg10_Copies = self.parray(lg10_Copies=lg10_Copies)
            lg10_Copies = lg10_Copies.z.values()
        else:
            lg10_Copies = None

        # Store predictions in appropriate structured array format
        if len(param) == 1:
            # Predicting one parameter, return an UncertainParameterArray
            self.predictions = self.uparray(param[0], pred_mean, pred_variance, lg10_Copies=lg10_Copies, stdzd=True)
        # else:
        #     # Predicting multiple parameters, return an MVUncertainParameterArray
        #     # First split prediction into UncertainParameterArrays
        #     uparrays = []
        #     for i, name in enumerate(param):
        #         idx = (tall_points['Parameter'].values() == param_coords[i]).squeeze()
        #         μ = pred_mean[idx]
        #         σ2 = pred_variance[idx]
        #         uparrays.append(self.uparray(name, μ, σ2, lg10_Copies=lg10_Copies, stdzd=True))
        #
        #     # Store predictions as MVUncertainParameterArray
        #     self.predictions = self.mvuparray(*uparrays, cor=cor)

        return self.predictions

class GP_gpflow(Regressor):
    r"""Gaussian Process surface learning and prediction.

    See Also
    --------
    :class:`Regressor`

    Notes
    -----
    This is the same as the GP class, expect it is implemented using GPflow rather than pymc3. The GP_gpflow
     class is built from a dataframe in the form of a :class:`ParameterSet` object. This is stored as
    :attr:`data`. The model inputs are constructed by filtering this dataframe, extracting column values, and
    converting these to numerical input coordinates. The main entry point will be :meth:`fit`, which parses the
    dimensions of the model with :meth:`specify_model`, extracts numerical input coordinates with
    :meth:`get_shaped_data`, compiles the Pymc3 model with :meth:`build_model`, and finally learns the
    hyperparameters with :meth:`find_MAP`.

    Dimensions fall into several categories:

    * Filter dimensions, those with only one level, are used to subset the dataframe but are not included as explicit
      inputs to the model. These are not specified explicitly, but rather any spatial or coregion dimension with only one
      level is treated as a filter dimension.
    * Spatial dimensions are treated as explicit coordinates and given a Radial Basis Function kernel

      * Linear dimensions (which must be a subset of `spatial_dims`) have an additional linear kernel.

    * Coregion dimensions imply a distinct but correlated output for each level

      * If more than one parameter is specified, ``'Parameter'`` is treated as a coregion dim.

    A non-additive model has the form:

    .. math::

        y &\sim \text{Normal} \left( \mu, \sigma \right) \\
        mu &\sim \mathcal{GP} \left( K \right) \\
        K &= \left( K^\text{spatial}+K^\text{lin} \right) K^\text{coreg}_\text{params} \prod_{n} K^\text{coreg}_{n} \\
        K^\text{spatial} &= \text{RBF} \left( \ell_{i}, \eta \right) \\
        K^\text{lin} &= \text{LIN} \left( c_{j}, \tau \right) \\
        K^\text{coreg} &= \text{Coreg} \left( \boldsymbol{W}, \kappa \right) \\
        \sigma &\sim \text{Exponential} \left( 1 \right) \\

    Where :math:`i` denotes a spatial dimension, :math:`j` denotes a linear dimension, and :math:`n` denotes a
    coregion dimension (excluding ``'Parameter'``). :math:`K^\text{spatial}` and :math:`K^\text{lin}` each consist of a
    joint kernel encompassing all spatial and linear dimensions, respectively, whereas :math:`K^\text{coreg}_{n}` is
    a distinct kernel for a given coregion dimension.

    The additive model has the form:

    .. math::

        mu &\sim \mathcal{GP}\left( K^\text{global} \right) + \sum_{n} \mathcal{GP}\left( K_{n} \right) \\
        K^\text{global} &= \left( K^\text{spatial}+K^\text{lin} \right) K^\text{coreg}_\text{params} \\
        K_{n} &= \left( K^\text{spatial}_{n}+K^\text{lin}_{n} \right) K^\text{coreg}_\text{params} K^\text{coreg}_{n} \\

    Note that, in the additive model, :math:`K^\text{spatial}_{n}` and :math:`K^\text{lin}_{n}` still consist of
    only the spatial and linear dimensions, respectively, but have unique priors corresponding to each coregion
    dimension. However, there is only one :math:`K^\text{coreg}_\text{params}` kernel.

    Internally, GC content is always on [0, 1], though it may be plotted on [0,100].

    Parameters
    ----------
    parameter_set : ParameterSet
        Data for fitting.
    params : str or list of str, default "r"
        Name(s) of parameter(s) to learn.
    seed : int
        Random seed

    Examples
    --------
    A GP object is created from a :class:`ParameterSet` and can be fit immediately with the default dimension
    configuration (regressing `r` with RBF + linear kernels for `BP` and `GC`):

    >>> from candas.learn import ParameterSet, GP
    >>> ps = ParameterSet.load('my_ParameterSet.pkl')
    >>> gp = GP_gpflow(ps).fit()

    Note that the last line is equivalent to

    >>> gp = GP_gpflow(ps)
    >>> gp.specify_model()
    >>> gp.build_model()
    >>> gp.find_MAP()

    The model can be specified with various spatial, linear, and coregion dimensions.
    `GC` and `BP` are always included in both ``spatial_dims`` and ``linear_dims``.

    >>> gp.specify_model(spatial_dims='lg10_Copies', linear_dims='lg10_Copies', coregion_dims='PrimerPair')
    >>> GP_gpflow(ps).fit(spatial_dims='lg10_Copies', linear_dims='lg10_Copies', coregion_dims='PrimerPair')  # equivalent

    After the model is fit, define a grid of points at which to make predictions. The result is a
    :class:`ParameterArray`:

    >>> gp.spatial_grid()
    >>> gp.grid_points
    ('GC', 'BP'): [(0.075     ,  10.) (0.08358586,  10.) (0.09217172,  10.) ...
     (0.90782828, 800.) (0.91641414, 800.) (0.925     , 800.)]

    Make predictions, returning an :class:`UncertainParameterArray`

    >>> gp.predict_grid()
    >>> gp.predictions
    r['?', '?2']: [[(0.70728056, 0.16073197) (0.70728172, 0.16073197)
                    (0.70728502, 0.16073197) ... (0.70727954, 0.16073197)
                    (0.7072809 , 0.16073197) (0.70728058, 0.16073197)]
                   ...
                   [(0.70749247, 0.1607318 ) (0.70773573, 0.16073116)
                    (0.70806603, 0.16072949) ... (0.70728449, 0.16073197)
                    (0.70728194, 0.16073197) (0.7072807 , 0.16073197)]]

    The `uparray` makes it easy to calculate standard statistics in natural or transformed/standardized space while
    maintaining the original shape of the array:

    >>> gp.predictions.z.dist.ppf(0.025)
    array([[-3.1887916 , -3.18878491, -3.18876601, ..., -3.18879744,
            -3.18878966, -3.18879146],
           ...,
           [-3.1875742 , -3.18617286, -3.18426272, ..., -3.18876906,
            -3.18878366, -3.18879081]])

    Finally, plot the results:

    >>> import matplotlib.pyplot as plt
    >>>
    >>> plt.style.use(str(futura))
    >>> gp.plot_predictions()

    Plot a slice down the center of the prediction along each axis

    >>> x_pa, y_upa = gp.get_conditional_prediction(BP=88)
    >>>
    >>> ax = plot_uparray(x_pa, y_upa)
    >>> ax.set_xticklabels([int(float(txt.get_text())*100) for txt in ax.get_xticklabels()]);

    Plot a slice down the center of the prediction along each axis

    >>> x_pa, y_upa = gp.get_conditional_prediction(GC=0.5)
    >>>
    >>> ax = plot_uparray(x_pa, y_upa)
    >>> ax.set_xticklabels([int(float(txt.get_text())*100) for txt in ax.get_xticklabels()]);


    Attributes
    ----------
    data : ParameterSet
        Data for fitting.
    params : list of str
        Name(s) of parameter(s) to learn.
    seed : int
        Random seed
    spatial_dims : list of str
        Columns of dataframe used as spatial dimensions
    linear_dims : list of str
        Subset of spatial dimensions to apply an additional linear kernel.
    spatial_levels : dict
        Values considered within each spatial column as ``{dim: [level1, level2]}``
    spatial_coords : dict
        Numerical coordinates of each spatial level within each spatial dimension as ``{dim: {level: coord}}``
    coregion_dims : list of str
        Columns of dataframe used as coregion dimensions
    coregion_levels : dict
        Values considered within each coregion column as ``{dim: [level1, level2]}``
    coregion_coords : dict
        Numerical coordinates of each coregion level within each coregion dimension as ``{dim: {level: coord}}``
    additive : bool
        Whether to treat coregion dimensions as additive or joint
    filter_dims : dict
        Dictionary of column-value pairs used to filter dataset before fitting
    X : array
        A 2D tall array of input coordinates.
    y : array
        A 1D vector of observations
    model : pymc3.model.Model
        Compiled pymc3 model
    gp_dict : dict
        Dictionary of model GP objects. Contains at least 'total'.
    """

    def __init__(self, parameter_set: ParameterSet, params='r', seed=2021):
        super(GP_gpflow, self).__init__(parameter_set, params, seed)

        self.model = None
        self.gp_dict = None
        self.MAP = None
        self.linear_cov_type = None

    ################################################################################
    # Model building and fitting
    ################################################################################

    def fit(self, params=None, linear_dims=None, spatial_dims=None, spatial_levels=None, spatial_coords=None,
            coregion_dims=None, coregion_levels=None, additive=False, seed=None, heteroskedastic_inputs=False,
            heteroskedastic_outputs=True, sparse=False, n_u=100, coregion_rank=None, lengthscales_init='stats', **MAP_kwargs):
        """Fits a GP surface

        Parses inputs, compiles a gpflow model, then finds the MAP value for the hyperparameters. `{}_dims` arguments
        indicate the columns of the dataframe to be included in the model, with `{}_levels` indicating which values of
        those columns are to be included (``None`` implies all values).

        If ``additive==True``, the model is constructed as the sum of a global GP and a distinct GP for each coregion
        dimension. Each of these GPs, including the global GP, consists of an RBF+linear kernel multiplied by a
        coregion kernel for ``'Parameter'`` if necessary. Although the same spatial kernel structure is used for each
        GP in this model, unique priors are assigned to each distinct kernel. However, there is always only one
        coregion kernel for ``'Parameter'``. The kernel for each dimension-specific GP is further multiplied by a
        coregion kernel that provides an output for each level in that dimension.

        See Also
        --------
        :meth:`build_model`


        Parameters
        ----------
        params : str or list of str, default "r"
            Name(s) of parameter(s) to learn. If ``None``, :attr:`params` is used.
        linear_dims : str or list of str, optional
            Subset of spatial dimensions to apply an additional linear kernel. If ``None``, defaults to ``['BP','GC']``.
        spatial_dims : str or list of str, optional
            Columns of dataframe used as spatial dimensions.
        spatial_levels : str, list, or dict, optional
            Values considered within each spatial column as ``{dim: [level1, level2]}``.
        spatial_coords : list or dict, optional
            Numerical coordinates of each spatial level within each spatial dimension as ``{dim: {level: coord}}``.
        coregion_dims : str or list of str, optional
            Columns of dataframe used as coregion dimensions.
        coregion_levels : str, list, or dict, optional
            Values considered within each coregion column as ``{dim: [level1, level2]}``.
        additive : bool, default False
            Whether to treat coregion_dims as additive or joint (default).
        seed : int, optional.
            Random seed for model instantiation. If ``None``, :attr:`seed` is used.
        heteroskedastic_inputs: bool, default False
            Whether to allow heteroskedasticity along spatial dimensions (input-dependent noise)
        heteroskedastic_outputs: bool, default True
            Whether to allow heteroskedasticity between multiple Parameter outputs (output-dependent noise)


        Returns
        -------
        self : :class:`GP`
        """

        self.specify_model(params=params, linear_dims=linear_dims, spatial_dims=spatial_dims,
                           spatial_levels=spatial_levels, spatial_coords=spatial_coords,
                           coregion_dims=coregion_dims, coregion_levels=coregion_levels,
                           additive=additive, coregion_rank=coregion_rank)

        self.build_model(seed=seed,
                         heteroskedastic_inputs=heteroskedastic_inputs,
                         heteroskedastic_outputs=heteroskedastic_outputs,
                         sparse=sparse, n_u=n_u, lengthscales_init=lengthscales_init)

        self.train_model()

        return self

    # TODO: add full probabilistic model description to docstring
    def build_model(self, seed=None, heteroskedastic_inputs=False, heteroskedastic_outputs=True, sparse=False, n_u=100,
                    linear_cov_type='gpflow_linear', priors=False, lengthscales_init='random', W_init=None,
                    kappa_init=None):
        r"""Compile a gpflow model for the GP.

        Each dimension in :attr:`spatial_dims` is combined in an ExpQuad kernel with a principled
        :math:`\text{InverseGamma}` prior for each lengthscale (as `suggested by Michael Betancourt`_) and a
        :math:`\text{Gamma}\left(2, 1\right)` prior for variance.

        .. _suggested by Michael Betancourt: https://betanalpha.github.io/assets/case_studies/gp_part3/part3.html#4_adding_an_informative_prior_for_the_length_scale

        Parameters
        ----------
        seed : int, optional.
            Random seed. If ``None``, :attr:`seed` is used.
        heteroskedastic_inputs: bool, default False
            Whether to allow heteroskedasticity along spatial dimensions (input-dependent noise)
        heteroskedastic_outputs: bool, default True
            Whether to allow heteroskedasticity between multiple Parameter outputs (output-dependent noise)

        Returns
        -------
        self : :class:`GP`
        """

        if lengthscales_init not in ['random', 'stats']:
            AssertionError("lengthscales_init must be in ['random', 'stats']")

        if len(self.coregion_dims) > 0:
            if W_init is None:
                W_init = np.random.uniform(-2, 2, (len(self.coregion_levels['PrimerPairReporter']), self.coregion_rank))
            if kappa_init is None:
                kappa_init = np.random.uniform(1e-5, 2, len(self.coregion_levels['PrimerPairReporter']))

        self.linear_cov_type = linear_cov_type
        self.X, self.y = self.get_shaped_data(metric='mean')

        if (linear_cov_type is None) & (self.linear_dims is not None):
            ValueError("Must specify the type of linear kernel out of 'gpflow_linear', "
                       "'linear+constant' and 'linear_offset'")

        # Convert ParameterArray into plain numpy tall array
        if 'Parameter' in self.dims:
            ordered_params = {k: v for k, v in sorted(self.coords['Parameter'].items(), key=lambda item: item[1])}
            y = np.hstack([self.y.z[param+'_z'].values() for param in ordered_params.keys()])
            X = np.atleast_2d(self.X)
            X = parray.vstack([X.add_layers(Parameter=coord) for coord in ordered_params.values()])
            X = np.atleast_2d(np.column_stack([X[dim].z.values().squeeze() for dim in self.dims]))

        else:
            y = np.atleast_1d(self.y.z.drop('lg10_Copies_z').values().squeeze())
            X = np.atleast_2d(np.column_stack([self.X[dim].z.values().squeeze() for dim in self.dims]))

        # print(y.shape)
        # print(y)
        if len(y.shape) == 1:
            y = y.reshape(len(y), 1)

        seed = self.seed if seed is None else seed

        n_l = len(self.linear_dims)
        n_s = len(self.spatial_dims)
        n_c = len(self.coregion_dims)
        n_p = len(self.params)

        D_in = len(self.dims)
        assert X.shape[1] == D_in

        idx_l = [self.dims.index(dim) for dim in self.linear_dims] # linear (can ignore for now)
        idx_s = [self.dims.index(dim) for dim in self.spatial_dims] # spatial
        idx_c = [self.dims.index(dim) for dim in self.coregion_dims] # coregionalisation
        idx_p = self.dims.index('Parameter') if 'Parameter' in self.dims else None

        X_s = X[:, idx_s]

        ℓ_μ, ℓ_σ = [stat for stat in np.array([get_ℓ_prior(dim) for dim in X_s.T]).T] # this gets the values for the prior on X dims
        # these 3 functions define the different possible covariance matrices. One for X, one for linear (that can be ignored) and one for coreg

        if lengthscales_init == 'random':
            L_init = np.random.uniform(0.4, 2, n_s)
        else:
            L_init = ℓ_μ

        def spatial_cov(suffix):

            lengthscales = tf.convert_to_tensor(L_init, dtype=default_float(), name='X_lengthscales')

            k = gpflow.kernels.RBF(lengthscales=lengthscales, active_dims=idx_s)
            if priors:
                alphas = [mean ** 2 / ℓ_σ[i] + 2 for i, mean in enumerate(ℓ_μ)]
                betas = [mean * (mean ** 2 / ℓ_σ[i] + 1) for i, mean in enumerate(ℓ_μ)]
                k.lengthscales.prior = tfp.distributions.InverseGamma(to_default_float(alphas), to_default_float(betas))
                k.variance.prior = tfp.distributions.Gamma(to_default_float(2), to_default_float(1))
            return k

        def linear_cov(suffix):
            "Must specify the type of linear kernel out of 'gpflow_linear', "
            "'linear+constant' and 'linear_offset'"

            var = tf.transpose(tf.convert_to_tensor([1.0]* n_l, dtype=default_float()))

            if self.linear_cov_type == 'gpflow_linear':
                k_l = gpflow.kernels.Linear(variance=var, active_dims=idx_l)
                if priors:
                    k_l.variance.prior = tfp.distributions.HalfNormal(scale=to_default_float(10.0))

                return k_l

            if self.linear_cov_type == 'linear+constant':
                k_l = gpflow.kernels.Linear(variance=var, active_dims=idx_l)
                k_c = gpflow.kernels.Constant(variance=to_default_float(1.0), active_dims=idx_l)

                if priors:
                    k_l.variance.prior = tfp.distributions.HalfNormal(scale=to_default_float(10.0))
                    k_c.variance.prior = tfp.distributions.Normal(to_default_float(0.0), to_default_float(10.0))

                return k_l + k_c

            if self.linear_cov_type == 'linear_offset':
                c = tf.transpose(tf.convert_to_tensor([1.0] * n_l, dtype=default_float()))
                k_l = kernels.Linear_offset(variance=var, offset=c, active_dims=idx_l)
                if priors:
                    k_l.variance.prior = tfp.distributions.HalfNormal(scale=to_default_float(10.0))
                    k_l.offset.prior = tfp.distributions.Normal(to_default_float(0.0), to_default_float(10.0))

                return k_l

        def coreg_cov(suffix, D_out, idx):

            coreg_k = gpflow.kernels.Coregion(output_dim=D_out, rank=self.coregion_rank, active_dims=[idx]) #TODO: change rank to be H_dims
            coreg_k.W.assign(W_init)
            coreg_k.kappa.assign(kappa_init)

            if priors:
                coreg_k.W.prior = tfp.distributions.Normal(to_default_float(0), to_default_float(3))
                coreg_k.kappa.prior = tfp.distributions.Gamma(to_default_float(2), to_default_float(1))
            return coreg_k

        # Define a "global" spatial kernel regardless of additive structure
        cov = spatial_cov('total')
        if n_l > 0:
            cov += linear_cov('total')

        # Construct a coregion kernel for each coregion_dims
        if n_c > 0 and not self.additive: # I think I can probably ignore this additive parameter
            for dim, idx in zip(self.coregion_dims, idx_c):
                if dim == 'Parameter':
                    continue
                D_out = len(self.coregion_levels[dim])
                cov = cov * coreg_cov(dim, D_out, idx)

        # Coregion kernel for parameters, if necessary
        if 'Parameter' in self.coregion_dims:  # not sure what this is for
            D_out = len(self.coregion_levels['Parameter'])
            cov_param = coreg_cov('Parameter', D_out, idx_p)
            cov *= cov_param

        if sparse:
            Z = tf.random.shuffle(X)[:n_u]
            pm_gp = gpflow.models.SGPR(data=(tf.convert_to_tensor(X, dtype=default_float()),
                                             tf.convert_to_tensor(y, dtype=default_float())), kernel=cov, inducing_variable=Z)

        else:
            # pm_gp = gpflow.models.GPR(data=(tf.convert_to_tensor(X, dtype=default_float()),
            #                                 tf.convert_to_tensor(y, dtype=default_float())), kernel=cov)
            pm_gp = gpflow.models.GPR(data=(tf.convert_to_tensor(X, dtype=default_float()),
                                             tf.convert_to_tensor(y, dtype=default_float())), kernel=cov)

        if len(self.y.z.values()) > 2:
            var_assign = np.var(self.y.z.values())*0.1
            if var_assign < 1e-4:
                var_assign = 1e-4
            pm_gp.likelihood.variance.assign(var_assign)
        else:
            pm_gp.likelihood.variance.assign(0.1)

        print(pm_gp.likelihood.variance)

        if priors:
            pm_gp.likelihood.variance.prior = tfp.distributions.InverseGamma(to_default_float(2), to_default_float(1))
        gp_dict = {'total': pm_gp}

        self.gp_dict = gp_dict
        self.model = pm_gp
        return self

    def train_model(self, *args, **kwargs):
        """Finds maximum a posteriori value for hyperparameters in model using gpflow optimizer

        """
        assert self.model is not None


        maxiter = 2000
        res_LMC = gpflow.optimizers.Scipy().minimize(
            self.model.training_loss, self.model.trainable_variables, options=dict(maxiter=maxiter), method="L-BFGS-B")

    def predict(self, points_array, with_noise=True, additive_level='total', **kwargs):
        """Make predictions at supplied points using specified gp

        Parameters
        ----------
        param : str
        points : ParameterArray
            Tall ParameterArray vector of coordinates for prediction, must have one layer per ``self.dims``
        with_noise : bool, default True
            Whether to incorporate aleatoric uncertainty into prediction error

        Returns
        -------
        prediction : UncertainParameterArray
            Predictions as a `uparray`
        """

        # TODO: need to supply "given" dict for additive GP sublevel predictions
        if additive_level != 'total':
            raise NotImplementedError('Prediction for additive sublevels is not yet supported.')



        # Prediction means and variance as a numpy vector
        # predictions = self.gp_dict[additive_level].predict(points_array, point=self.MAP, diag=True,
        #                                                    pred_noise=with_noise, **kwargs)
        self.predictions = self.gp_dict[additive_level].predict_y(points_array)

        return self.predictions

class LVMOGP_GP(GP_gpflow):

    def __init__(self, lvmogp_latent_dims, parameter_set: ParameterSet):

        super().__init__(parameter_set)
        self.lvmogp_latent_dims = lvmogp_latent_dims

        self.additive = False
        self.predictions = None

    def fit(self, params=None, linear_dims=None, spatial_dims=None, spatial_levels=None, spatial_coords=None,
            coregion_dims=None, coregion_levels=None, additive=False, seed=None, heteroskedastic_inputs=False,
            heteroskedastic_outputs=True, sparse=False, n_u=100, **MAP_kwargs):

        self.specify_model(params=params, linear_dims=linear_dims, spatial_dims=spatial_dims,
                           spatial_levels=spatial_levels, spatial_coords=spatial_coords,
                           coregion_dims=coregion_dims, coregion_levels=coregion_levels,
                           additive=additive)
        self.build_model(seed=seed, n_u=n_u)
        self.train_model()

    def build_model(self, seed=None, heteroskedastic_inputs=False, heteroskedastic_outputs=True, sparse=False, n_u=100,
                    plot_BGPLVM=False, n_restarts=4, lengthscales_init='random', initialisation='mo_PCA', priors=False, MAP=False,
                    set_inducing_points=True, train_inducing=False):

        n_fun = len(self.coregion_levels['PrimerPairReporter'])

        if lengthscales_init not in ['random', 'stats']:
            AssertionError("lengthscales_init must be in ['random', 'stats']")

        if initialisation not in ['Gpy', 'mo_PCA', 'random']:
            AssertionError("initialisation needs to be in ['Gpy', 'mo_PCA', 'random']")

        self.X, self.y = self.get_shaped_data(metric='mean')

        if 'Parameter' in self.dims:
            ordered_params = {k: v for k, v in sorted(self.coords['Parameter'].items(), key=lambda item: item[1])}
            y = np.hstack([self.y.z[param+'_z'].values() for param in ordered_params.keys()])
            X = np.atleast_2d(self.X)
            X = parray.vstack([X.add_layers(Parameter=coord) for coord in ordered_params.values()])
            X = np.atleast_2d(np.column_stack([X[dim].z.values().squeeze() for dim in self.dims]))

        else:
            y = self.y.z.drop('lg10_Copies_z').values().squeeze()
            X = np.atleast_2d(np.column_stack([self.X[dim].z.values().squeeze() for dim in self.dims]))

        if len(y.shape) == 1:
            y = y.reshape(len(y), 1)

        idx_s = [self.dims.index(dim) for dim in self.spatial_dims]
        idx_c = [self.dims.index(dim) for dim in self.coregion_dims]
        X_s = X[:, idx_s]

        ℓ_μ, ℓ_σ = [stat for stat in
                    np.array([get_ℓ_prior(dim) for dim in X_s.T]).T]  # this gets the values for the prior on X dims

        n_s = len(self.spatial_dims)

        if lengthscales_init == 'random':
            L_init = np.random.uniform(0.01, 1, n_s)

        else:
            L_init = ℓ_μ

        if initialisation == 'Random':
            Ls = L_init
            L_h_init = np.random.uniform(0.01, 1, self.lvmogp_latent_dims)
            Ls = L_init.tolist() + L_h_init.tolist()
            lengthscales = tf.convert_to_tensor(Ls, dtype=default_float(), name='lengthscales')
            kern_variance = np.random.uniform(0, 1)
            H_mean = np.random.uniform(-1, 1, (n_fun, self.lvmogp_latent_dims))
            H_mean_init = tf.convert_to_tensor(H_mean, dtype=default_float())
            H_var_init = tf.ones((len(H_mean), self.lvmogp_latent_dims), dtype=default_float()) * 1e-6

        else:
            n_x_inducing = 20
            self.spatial_grid(resolution=n_x_inducing)

            inducing_point_xs_grids = self.grid_points.z.values()

            inducing_point_xs = np.vstack([np.hstack([grid.ravel().reshape(n_x_inducing*n_x_inducing, 1)
                                                      for grid in inducing_point_xs_grids])]*n_fun)
            inducing_point_fns = tf.convert_to_tensor(np.atleast_2d(np.hstack([[fun_no] * n_x_inducing * n_x_inducing for
                                                                 fun_no in
                                                                 self.coregion_coords['PrimerPairReporter'].values()])).T,
                                                      dtype=default_float())
            inducing_point_xs = tf.convert_to_tensor(inducing_point_xs, dtype=default_float())
            inducing_point_xs = tf.concat([inducing_point_xs, inducing_point_fns], axis=1)

            mo_indi = self.fit_mo_indi(X, y,  L_init, active_dims=n_s, inducing_points=inducing_point_xs)

            mo_indi_mu, mo_indi_sig2 = mo_indi.predict_f(inducing_point_xs)

            # plt.scatter(inducing_point_xs[:, 0], inducing_point_xs[:, 1])
            # plt.scatter(X[:, 0], X[:, 1], color='red')
            # plt.show()

            mo_indi_mean = mo_indi_mu.numpy().reshape(int(n_fun), int(len(inducing_point_xs.numpy()) / n_fun))

            H_mean_init, fracs = self.pca_reduce(tf.convert_to_tensor(mo_indi_mean, dtype=default_float()),
                                            self.lvmogp_latent_dims)

            H_var_init = tf.ones((len(H_mean_init), self.lvmogp_latent_dims), dtype=default_float()) * 1e-6
            fracs = 0.1 * fracs
            Ls_hs = tf.reduce_max(fracs) / fracs

            Ls = mo_indi.kernel.kernels[
                     0].lengthscales.numpy().ravel().tolist() + Ls_hs.numpy().ravel().tolist()
            lengthscales = tf.convert_to_tensor(Ls, dtype=default_float(), name='lengthscales')
            kern_variance = mo_indi.kernel.kernels[0].variance.numpy()

            if initialisation == 'Gpy':
                gplvm = self.fit_gplvm(mo_indi_mean, Ls_hs, H_mean_init, n_fun)

                Ls = mo_indi.kernel.kernels[
                         0].lengthscales.numpy().ravel().tolist() + gplvm.kernel.lengthscales.numpy().ravel().tolist()

                lengthscales = tf.convert_to_tensor(Ls, dtype=default_float(), name='lengthscales')

                kern_variance = np.mean([mo_indi.kernel.kernels[0].variance.numpy(), gplvm.kernel.variance.numpy()])

        if set_inducing_points:
            inducing_points, _ = self.fill_Hs_(inducing_point_xs[:, :2], inducing_point_fns, H_mean_init, H_var_init,
                                           self.lvmogp_latent_dims)
            n_u = None
        else:
            inducing_points = None

        lvmogp = self.init_lvmogp(X[:, :(len(self.dims)-1)], y, X[:, -1], lengthscales=lengthscales, kern_variance=kern_variance,
                                  H_mean=H_mean_init, H_var=H_var_init, lik_variance=np.var(y)*0.01,
                                  inducing_points=inducing_points, train_inducing=train_inducing, n_u=n_u, MAP=MAP)
        self.lvmogp = lvmogp
        return lvmogp

    def fill_Hs_(self, X_data, X_data_fn, H_data_mean, H_data_var, latent_dims):
        """append Hs to Xs by function number"""

        H_mean_vect = tf.reshape(tf.gather(_cast_to_dtype(H_data_mean, dtype=default_float()),
                                           _cast_to_dtype(X_data_fn, dtype=tf.int64)),
                                 [len(X_data), latent_dims])
        H_var_vect = tf.reshape(tf.gather(_cast_to_dtype(H_data_var, dtype=default_float()),
                                          _cast_to_dtype(X_data_fn, dtype=tf.int64)),
                                [len(X_data), latent_dims])

        return tf.concat([X_data, H_mean_vect], axis=1), \
               tf.concat([tf.zeros(X_data.shape, dtype=default_float()), H_var_vect], axis=1)

    def pca_reduce(self, X: tf.Tensor, latent_dim: tf.Tensor) -> tf.Tensor:
        """
        A helpful function for linearly reducing the dimensionality of the input
        points X to `latent_dim` dimensions.

        :param X: data array of size N (number of points) x D (dimensions)
        :param latent_dim: Number of latent dimensions Q < D
        :return: PCA projection array of size [N, Q].
        """
        if latent_dim > X.shape[1]:  # pragma: no cover
            raise ValueError("Cannot have more latent dimensions than observed")
        X_cov = tfp.stats.covariance(X)
        evals, evecs = tf.linalg.eigh(X_cov)
        W = evecs[:, -latent_dim:]
        fracs = evals / tf.reduce_sum(evals)
        return (X - tf.reduce_mean(X, axis=0, keepdims=True)) @ W, fracs[-latent_dim:]

    def fit_mo_indi(self, X, y,  L_init, active_dims, inducing_points):

        k = gpflow.kernels.RBF(lengthscales=L_init, active_dims=range(active_dims))
        coreg_k = gpflow.kernels.Coregion(output_dim=len(self.coregion_levels['PrimerPairReporter']),
                                          rank=self.lvmogp_latent_dims,
                                          active_dims=[active_dims])
        cov = k * coreg_k

        mo_indi = gpflow.models.GPR(data=(tf.convert_to_tensor(X, dtype=default_float()),
                                           tf.convert_to_tensor(y, dtype=default_float())), kernel=cov)
        # mo_indi.inducing_variable.Z.assign(inducing_variable)
        mo_indi.likelihood.variance.assign(np.var(y) * 0.01)

        mo_indi.kernel.kernels[1].W.assign(
            np.hstack([np.array([[1e-6] * len(self.coregion_levels['PrimerPairReporter'])]).T]*self.lvmogp_latent_dims))
        gpflow.set_trainable(mo_indi.kernel.kernels[1].W, False)
        mo_indi.kernel.kernels[1].kappa.assign(np.array([1] * len(self.coregion_levels['PrimerPairReporter'])))
        gpflow.set_trainable(mo_indi.kernel.kernels[1].kappa, False)

        maxiter = 2000

        mo_opt = gpflow.optimizers.Scipy().minimize(
            mo_indi.training_loss, mo_indi.trainable_variables, options=dict(maxiter=maxiter), method="L-BFGS-B",
        )
        return mo_indi

    def fit_gplvm(self, data, Ls, H_mean_init, n_fun):

        kernel_H = gpflow.kernels.RBF(lengthscales=Ls,
                                     active_dims=list(range(0, self.lvmogp_latent_dims)))

        # Initialise the variances randomly between 0 and 0.1
        H_var_init = tf.ones((len(H_mean_init.numpy()), self.lvmogp_latent_dims), dtype=default_float()) * np.random.uniform(
            0, 0.1)

        gplvm = gpflow.models.BayesianGPLVM(tf.convert_to_tensor(data, dtype=default_float()),
                                            X_data_mean=H_mean_init,
                                            X_data_var=H_var_init,
                                            kernel=kernel_H,
                                            num_inducing_variables=n_fun,
                                            )
        gplvm.likelihood.variance.assign(
            np.var(data) * 0.01)  # initalise the variance as 0.01 times the variance of the data

        H_init_gplvm = gplvm.X_data_mean.numpy()
        H_var_init_gplvm = gplvm.X_data_var.numpy()

        opt = gpflow.optimizers.Scipy()
        maxiter = 2000
        res = opt.minimize(
            gplvm.training_loss,
            method="BFGS",
            variables=gplvm.trainable_variables,
            options=dict(maxiter=maxiter),
        )

        return gplvm


    def init_lvmogp(self, data_X, data_y, fun_nos, lengthscales, kern_variance, H_mean, H_var, lik_variance,
                    inducing_points, train_inducing, n_u=100, MAP=True):
        """only one of n_u and inducing points should have a value, the other should be None"""

        kernel_lvmogp = gpflow.kernels.RBF(lengthscales=lengthscales, variance=kern_variance)

        lvmogp = LVMOGP(data=data_y,
                        X_data=data_X,
                        X_data_fn=fun_nos,
                        H_data_mean=H_mean,
                        H_data_var=H_var,
                        kernel=kernel_lvmogp,
                        num_inducing_variables=n_u,
                        inducing_variable=inducing_points,
                        H_prior_mean=None,
                        H_prior_var=None,
                        MAP=MAP)

        if not train_inducing:
            gpflow.utilities.set_trainable(lvmogp.inducing_variable.Z, False)

        lvmogp.likelihood.variance.assign(to_default_float(lik_variance))

        return lvmogp


    def train_model(self, *args, **kwargs):

        assert self.lvmogp is not None

        opt = gpflow.optimizers.Scipy()
        maxiter = 1000
        # print(lvmogp.elbo())
        res = opt.minimize(
            self.lvmogp.training_loss,
            method="BFGS",
            variables=self.lvmogp.trainable_variables,
            options=dict(maxiter=maxiter))

        self.model = self.lvmogp
        self.gp_dict = {'total': self.lvmogp}
