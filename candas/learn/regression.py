import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from scipy.interpolate import interpn

from scipy.stats import norm

from candas.utils.misc import assert_in, assert_is_subset
from candas.learn import ParameterArray, ParameterSet
from candas.learn import ParameterArray as parray
from candas.learn import UncertainParameterArray as uparray
from candas.learn import MVUncertainParameterArray as mvuparray

# from candas.plotting.regression_plots import plot_uparray, contourf_uparray, unstandardize_axis_labels


class Regressor:
    r"""Surface learning and prediction.

    A Regressor is built from a dataframe in the form of a :class:`ParameterSet` object. This is stored as
    :attr:`data`. The model inputs are constructed by filtering this dataframe, extracting column values, and
    converting these to numerical input coordinates. Each subclass defines at least `build_model`, `fit`, and `predict_points`
    methods in addition to subclass-specific methods.

    Dimensions fall into several categories:

    * Filter dimensions, those with only one level, are used to subset the dataframe but are not included as explicit
      inputs to the model. These are not specified explicitly, but rather any spatial or coregion dimension with only one
      level is treated as a filter dimension.
    * Spatial dimensions are treated as explicit coordinates and given a Radial Basis Function kernel

      * Linear dimensions (which must be a subset of `continuous_dims`) have an additional linear kernel.

    * Coregion dimensions imply a distinct but correlated output for each level

      * If more than one parameter is specified, ``'Parameter'`` is treated as a coregion dim.

    Notes
    -----
    Internally, GC content is always on [0, 1], though it may be plotted on [0,100].

    Parameters
    ----------
    parameter_set : ParameterSet
        Data for fitting.
    params : str or list of str, default "r"
        Name(s) of parameter(s) to learn.
    seed : int
        Random seed

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
    """

    def __init__(self, parameter_set: ParameterSet, params='r', seed=2021):
        if not isinstance(parameter_set, ParameterSet):
            raise TypeError('Learner instance must be initialized with a ParameterSet object')

        self.data = parameter_set
        self.stdzr = parameter_set.stdzr
        self.params = params if isinstance(params, list) else [params]
        self.seed = seed

        self.spatial_dims = []
        self.linear_dims = []
        self.spatial_levels = {}
        self.spatial_coords = {}
        self.coregion_dims = []
        self.coregion_levels = {}
        self.coregion_coords = {}
        self.additive = False

        self.X = None
        self.y = None

        self.grid_vectors = None
        self.grid_parray = None
        self.grid_points = None
        self.ticks = None

        self.predictions = None

    ################################################################################
    # Model building and fitting
    ################################################################################

    def fit(self, *args, **kwargs):
        """Defined by subclass

        See Also
        --------
        :meth:`GP.fit`
        :meth:`GLM.fit`
        """
        pass

    def build_model(self, *args, **kwargs):
        """Defined by subclass

        See Also
        --------
        :meth:`GP.build_model`
        :meth:`GLM.build_model`
        """
        pass

    ################################################################################
    # Properties and convenience methods
    ################################################################################

    def parray(self, **kwargs) -> parray:
        """Creates a parray with the current instance's stdzr attached"""
        return parray(stdzr=self.stdzr, **kwargs)

    def uparray(self, name: str, μ: np.ndarray, σ2: np.ndarray, **kwargs) -> uparray:
        """Creates a uparray with the current instance's stdzr attached"""
        return uparray(name, μ, σ2, stdzr=self.stdzr, **kwargs)

    def mvuparray(self, *uparrays, cor, **kwargs) -> mvuparray:
        """Creates a uparray with the current instance's stdzr attached"""
        return mvuparray(*uparrays, cor=cor, stdzr=self.stdzr, **kwargs)

    @property
    def dims(self) -> list:
        """List of all dimensions under consideration"""
        return self.spatial_dims + self.coregion_dims

    @property
    def levels(self) -> dict:
        """Dictionary of values considered within each dimension as ``{dim: [level1, level2]}``"""
        return {**self.spatial_levels, **self.coregion_levels}

    @property
    def coords(self) -> dict:
        """ Dictionary of numerical coordinates of each level within each dimension as ``{dim: {level: coord}}``"""
        return {**self.spatial_coords, **self.coregion_coords}

    ################################################################################
    # Preprocessing
    ################################################################################

    def specify_model(self, params=None, linear_dims=None, spatial_dims=None, spatial_levels=None, spatial_coords=None,
                      coregion_dims=None, coregion_levels=None, additive=False, coregion_rank=None):
        """Checks for consistency among dimensions and levels and formats appropriately.

        Parameters
        ----------
        params : str or list of str, default "r"
            Name(s) of parameter(s) to learn. If ``None``, :attr:`params` is used.
        linear_dims : str or list of str, optional
            Subset of spatial dimensions to apply an additional linear kernel. If ``None``, defaults to ``['BP','GC']``.
        spatial_dims : str or list of str, optional
            Columns of dataframe used as spatial dimensions
        spatial_levels : str, list, or dict, optional
            Values considered within each spatial column as ``{dim: [level1, level2]}``
        spatial_coords : list or dict, optional
            Numerical coordinates of each spatial level within each spatial dimension as ``{dim: {level: coord}}``
        coregion_dims : str or list of str, optional
            Columns of dataframe used as coregion dimensions
        coregion_levels : str, list, or dict, optional
            Values considered within each coregion column as ``{dim: [level1, level2]}``
        additive : bool, default False
            Whether to treat categorical_dims as additive or joint (default)

        Returns
        -------
        self : :class:`GP`
        """

        # Ensure parameter is valid and format as list
        params = params if params is not None else self.params
        assert_is_subset('Parameter', params, self.data.data.Parameter)
        self.params = params if isinstance(params, list) else [params]

        # Ensure dimensions are valid and format as list
        self.spatial_dims = self._parse_dimensions(spatial_dims)
        self.linear_dims = self._parse_dimensions(linear_dims)
        self.coregion_dims = self._parse_dimensions(coregion_dims)

        if (coregion_dims is not None) & (coregion_rank is None):
            self.coregion_rank = 2
        else:
            self.coregion_rank = coregion_rank

        if set(self.coregion_dims) & set(self.spatial_dims) != set():
            raise ValueError('Overlapping items in categorical_dims, continuous_dims, and/or additive_dims')

        # Ensure levels are valid and format as dict
        self.spatial_levels = self._parse_levels(self.spatial_dims, spatial_levels)
        self.coregion_levels = self._parse_levels(self.coregion_dims, coregion_levels)

        # Add 'Parameter' to the end of the coregion list
        self.coregion_dims += ['Parameter']
        self.coregion_levels['Parameter'] = self.params

        # Move dims with only one level to separate list
        self.filter_dims = {}
        for dim in self.dims:
            levels = self.levels[dim]
            if len(levels) == 1:
                self.filter_dims[dim] = levels
                # self.continuous_dims = [d for d in self.continuous_dims if d != dim]
                self.coregion_dims = [d for d in self.coregion_dims if d != dim]
                # self.continuous_levels = {d: l for d, l in self.continuous_levels.items() if d != dim}
                self.coregion_levels = {d: l for d, l in self.coregion_levels.items() if d != dim}

        # Ensure coordinates are valid and format as dict-of-dicts
        self.spatial_coords = self._parse_coordinates(self.spatial_dims, self.spatial_levels, spatial_coords)
        self.coregion_coords = self._parse_coordinates(self.coregion_dims, self.coregion_levels, None)

        # Add 'GC' and 'BP' to the beginning of the spatial list
        # if 'BP' not in self.continuous_dims:
        #     self.continuous_dims = ['BP'] + self.continuous_dims
        # if 'GC' not in self.continuous_dims:
        #     self.continuous_dims = ['GC'] + self.continuous_dims

        # self.continuous_levels = {**{dim: self.data.data[dim].unique() for dim in ['GC', 'BP']},
        #                        **self.continuous_levels}
        # self.continuous_coords = {**{dim: {level: level for level in self.continuous_levels[dim]} for dim in ['GC', 'BP']},
        #                        **self.continuous_coords}
        assert_is_subset('spatial dimensions', self.linear_dims, self.spatial_dims)
        self.additive = additive
        return self

    def _parse_dimensions(self,
                          dims: None or str or list) -> list:
        """Ensure dimensions are possible and formatted as list"""
        if dims is not None:
            assert 'Parameter' not in dims
            dims = dims if isinstance(dims, list) else [dims]
            assert_is_subset('columns', dims, self.data.data.columns)
        else:
            dims = []
        return dims

    def _parse_levels(self, dims: list, levels: None or str or list or dict) -> dict:
        """Perform consistency checks between dimensions and levels and format `levels` as dict"""
        if len(dims) != 0:
            if levels is None:
                # Use all levels of all dims
                levels = {dim: list(self.data.data[dim].unique()) for dim in dims}
            elif any(isinstance(levels, typ) for typ in [str, list]):
                # If only a single dim is supplied, convert levels to dictionary
                assert len(dims) == 1, 'Non-dict argument for `levels` only allowed if `len(dims)==1`'
                levels = levels if isinstance(levels, list) else [levels]
                levels = {dims[0]: levels}
            elif isinstance(levels, dict):
                assert (dim in dims for dim in levels.keys())
            else:
                raise TypeError('`levels` must be of type str, list, or dict')

            assert all(set(levels[dim]).issubset(set(self.data.data[dim])) for dim in dims)
        else:
            levels = {}
        return levels

    def _parse_coordinates(self, dims: list, levels: dict, coords: None or list or dict) -> dict:
        """Check for consistency between supplied dims/levels/coords or generate coords automatically"""
        if coords is not None:
            if isinstance(coords, dict):
                # Ensure all dim-level pairs in ``levels`` and ``coords`` match exactly
                level_tuples = [(dim, level) for dim, levels_list in levels.items() for level in levels_list]
                coord_tuples = [(dim, level) for dim, coord_dict in coords.items() for level in coord_dict.keys()]
                assert_is_subset('coordinates', coord_tuples, level_tuples)
                assert_is_subset('coordinates', level_tuples, coord_tuples)
            elif isinstance(coords, list):
                assert len(levels.keys()) == 1, \
                    'Non-dict argument for `continuous_coords` only allowed if `len(continuous_dims)==1`'
                dim = dims[0]
                assert len(coords) == len(levels[dim])
                coords = {dim: {level: coord for level, coord in zip(levels[dim], coords)}}
            else:
                raise TypeError('Coordinates must be of type list or dict')
            if not all(isinstance(coord, int) or isinstance(coord, float) for coord in coords.values()):
                raise TypeError('Coordinates must be numeric')
        elif dims is not None and levels is not None:
            coords = {dim: self._make_coordinates(dim, levels_list) for dim, levels_list in levels.items()}
        else:
            coords = {}
        return coords

    def _make_coordinates(self, dim: str, levels_list: list) -> dict:
        """Generate numerical coordinates for each level in each dim under consideration"""

        df = self.data.data
        col = df[df[dim].isin(levels_list)][dim]

        if col.dtype in [np.float32, np.float64, np.int32, np.int64]:
            coords = {level: level for level in levels_list}
        else:
            coords = {level: col.astype('category').cat.categories.to_list().index(level) for level in levels_list}

        return coords

    def get_filtered_data(self, standardized=False, metric='mean'):
        """The portion of the dataset under consideration

        A filter is built by comparing the values in the unstandardized dataframe with those in :attr:`filter_dims`,
        :attr:`categorical_levels`, and :attr:`continuous_levels`, then the filter is applied to the standardized or
        unstandardized dataframe as indicated by the `standardized` input argument.

        Parameters
        ----------
        standardized : bool, default True
            Whether to return a subset of the raw data or the centered and scaled data
        metric : str, default 'mean'
            Which summary statistic to return (must be a value in the `Metric` column)

        Returns
        -------
        data : pd.DataFrame
        """

        assert_in('Metric', metric, self.data.data['Metric'].unique())
        df = self.data.data

        allowed = df.isin(self.filter_dims)[self.filter_dims.keys()].all(axis=1)
        allowed &= df['Metric'] == metric
        for dim, levels in self.levels.items():
            allowed &= df[dim].isin(levels)

        return df[allowed] if not standardized else self.data.zdata[allowed]

    def get_shaped_data(self, metric='mean'):
        """Formats input data and observations as parrays

        Parameters
        ----------
        metric : str, default 'mean'
            Which summary statistic to return (must be a value in the `Metric` column)

        Returns
        -------
        X : parray
            A multilayered column vector of input coordinates.
        y : parray
            A multilayered (1D) vector of observations

        See Also
        --------
        :meth:`get_filtered_data`

        """

        df = self.get_filtered_data(standardized=False, metric=metric)

        # Ensure same number of observations for every parameter (only possible if something broke)
        assert len(set(sum(df.Parameter == param) for param in self.params)) == 1

        # Assuming all parameters observed at the same points
        # Extract the model dimensions from the dataframe for one of the parameters
        dims = set(self.dims) - set(['Parameter'])
        dim_values = {dim: df[df.Parameter == self.params[0]].replace(self.coords)[dim].values for dim in dims}
        X = self.parray(**dim_values, stdzd=False)[:, None]

        lg10_Copies = df[df.Parameter == 'τ'].lg10_Copies if 'τ' in self.params else None

        # List of parrays for each parameter
        params = {param: df[df.Parameter == param]['Value'].values for param in self.params}
        y = self.parray(**params, stdzd=False, lg10_Copies=lg10_Copies)

        return X, y

    ################################################################################
    # Prediction
    ################################################################################

    def predict(self, points_array, with_noise=True, **kwargs):
        """Defined by subclass.

        It is not recommended to call :meth:`predict` directly, since it requires a very specific formatting for inputs,
        specifically a tall array of standardized coordinates in the same order as :attr:`dims`. Rather, one of the
        convenience functions :meth:`predict_points` or :meth:`predict_grid` should be used, as these have a more
        intuitive input structure and format the data appropriately prior to calling :meth:`predict`.

        See Also
        --------
        :meth:`GP.predict`
        :meth:`GLM.predict`

        Returns
        -------
        prediction_mean, prediction_var : list of np.ndarray
            Mean and variance of predictions at each supplied points
        """
        pass

    def _check_has_prediction(self):
        """Does what it says on the tin"""
        if self.predictions is None:
            raise ValueError('No predictions found. Run self.predict_grid or related method first.')

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
            # If 'Parameter' is not in categorical_dims, it must be in filter_dims, and only one is possible
            param = self.filter_dims['Parameter']
            # Convert input points to tall array

            tall_points = points[:, None]

        # Combine standardized coordinates into an ordinary tall numpy array for prediction

        # if hasattr(self, 'lvmogps'):
        #     points_array = np.hstack([tall_points[dim].z.values() for dim in self.continuous_dims + self.linear_dims])
        #     points_array = np.hstack(points_array, [self.model.H_data_mean[tall_points[dim].z.values(), :] for dim in self.categorical_dims])
        #
        # else:
        points_array = np.hstack([tall_points[dim].z.values() for dim in self.dims])

        # if hasattr(self, 'lvmogps'):
        #     idx_c = [self.dims.index(dim) for dim in self.coregion_dims]
        #     H_mean_vect = tf.reshape(tf.gather(_cast_to_dtype(self.model.H_data_mean, dtype=default_float()),
        #                                        _cast_to_dtype(points_array[:, -1*len(self.coregion_dims)], dtype=tf.int64)),
        #                              [len(points_array), self.lvmogp_latent_dims])
        #     H_var_vect = tf.reshape(tf.gather(_cast_to_dtype(self.model.H_data_var, dtype=default_float()),
        #                                       _cast_to_dtype(points_array[:, -1*len(self.coregion_dims)], dtype=tf.int64)),
        #                             [len(points_array), self.lvmogp_latent_dims])
        # 
        #     points_array_mean = tf.concat([tf.convert_to_tensor(points_array[:, :len(self.spatial_dims)], default_float()), H_mean_vect], axis=1).numpy()
        #     points_array_var = tf.concat([tf.zeros(points_array[:, :len(self.spatial_dims)].shape, dtype=default_float()), H_var_vect], axis=1).numpy()
        # 
        #     #
        #     # points_array_mean = np.hstack([points_array[:, -1 * len(self.coregion_dims)],
        #     #                                np.array(
        #     #                                    [self.model.H_data_mean[int(point)] for point in
        #     #                                     points_array[:, 2]]).reshape(
        #     #                                    1, 2)])
        #     # points_array_var = np.hstack([np.zeros(points_array.shape),
        #     #                               np.array(
        #     #                                   [self.model.H_data_var[int(point)] for point in
        #     #                                    points_array[:, 2]]).reshape(
        #     #                                   1, 2)])
        #     points_array = [points_array_mean, points_array_var]

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
        else:
            # Predicting multiple parameters, return an MVUncertainParameterArray
            # First split prediction into UncertainParameterArrays
            uparrays = []
            for i, name in enumerate(param):
                idx = (tall_points['Parameter'].values() == param_coords[i]).squeeze()
                μ = pred_mean[idx]
                σ2 = pred_variance[idx]
                uparrays.append(self.uparray(name, μ, σ2, lg10_Copies=lg10_Copies, stdzd=True))

            # Calculate the correlation matrix from the hyperparameters of the coregion kernel
            W = self.MAP['W_Parameter'][param_coords, :]
            κ = self.MAP['κ_Parameter'][param_coords]
            B = W @ W.T + np.diag(κ)  # covariance matrix
            D = np.atleast_2d(np.sqrt(np.diag(B)))  # standard deviations
            cor = B / (D.T @ D)  # correlation matrix

            # Store predictions as MVUncertainParameterArray
            self.predictions = self.mvuparray(*uparrays, cor=cor)

        return self.predictions

    def spatial_grid(self, limits=None, at=None, resolution=100):
        """Prepare unobserved input coordinates for specified spatial dimensions.

        Parameters
        ----------
        limits : ParameterArray
            List of min/max values as a single parray with one layer for each of a subset of `continuous_dims`.
        at : ParameterArray
            A single parray of length 1 with one layer for each remaining `continuous_dims` by name.
        ticks : dict
        resolution : dict or int
            Number of points along each dimension, either as a dictionary or one value applied to all dimensions

        Returns
        -------

        """

        # Remove any previous predictions to avoid confusion
        self.predictions = None
        self.predictions_X = None

        ##
        ## Check inputs for consistency and completeness
        ##

        # Ensure "at" is supplied correctly
        if at is None:
            at = self.parray(none=[])
        elif not isinstance(at, ParameterArray):
            raise TypeError('"at" must be a ParameterArray')
        elif at.ndim != 0:
            raise ValueError('"at" must be single point, potentially with multiple layers')

        # Ensure a grid can be built
        at_dims = set(at.names)
        spatial_dims = set(self.spatial_dims)
        limit_dims = spatial_dims-at_dims

        # If there are no remaining dimensions
        if limit_dims == set():
            raise ValueError('At least one dimension must be non-degenerate to generate grid.')

        # If no limits are supplied
        if limits is None:
            # Fill limits with default values
            limits = self.parray(**{dim: [-2.5, +2.5] for dim in self.dims if dim in limit_dims}, stdzd=True)
        else:
            # Append default limits to `limits` for unspecified dimensions
            if not isinstance(limits, ParameterArray):
                raise TypeError('"limits" must be a ParameterArray')
            remaining_dims = limit_dims-set(limits.names)
            if remaining_dims:
                dflt_limits = self.parray(**{dim: [-2.5, +2.5] for dim in remaining_dims}, stdzd=True)
                limits = limits.add_layers(**dflt_limits.to_dict())

        # Ensure all dimensions are specified without conflicts
        limit_dims = set(limits.names)

        if limit_dims.intersection(at_dims):
            raise ValueError('Dimensions specified via "limits" and in "at" must not overlap.')
        elif not spatial_dims.issubset(at_dims.union(limit_dims)-set(['none'])):
            raise ValueError('Not all spatial dimensions are specified by "limits" or "at".')

        # Format "resolution" as dict if necessary
        if isinstance(resolution, int):
            resolution = {dim: resolution for dim in self.spatial_dims}
        elif not isinstance(resolution, dict):
            raise TypeError('"resolution" must be a dictionary or an integer')
        else:
            assert_is_subset('spatial dimensions', resolution.keys(), self.spatial_dims)


        ##
        ## Build grids
        ##

        # Store a dictionary with one 1-layer 1-D parray for the grid points along each dimension
        # Note they may be different sizes
        grid_vectors = {dim:
            self.parray(
                **{dim: np.linspace(*limits[dim].z.values(), resolution[dim])[:, None]},
                stdzd=True)
            for dim in limit_dims}

        # Create a single n-layer n-dimensional parray for all evaluation points
        grids = np.meshgrid(*[grid_vectors[dim] for dim in self.dims if dim in limit_dims])
        grid_parray = self.parray(**{array.names[0]: array.values() for array in grids})

        # Add values specified in "at" to all locations in grid_parray
        if at.names != ['none']:
            at_arrays = {dim: np.full(grid_parray.shape, value) for dim, value in at.to_dict().items()}
            grid_parray = grid_parray.add_layers(**at_arrays)

        # Store dimensions along which grid was formed, ensuring the same order as self.dims
        self.prediction_dims = [dim for dim in self.dims if dim in limit_dims]
        self.grid_vectors = grid_vectors
        self.grid_parray = grid_parray
        self.grid_points = self.grid_parray.ravel()

    def predict_grid(self, param=None, coregion_levels=None, with_noise=True, **kwargs):
        """Make predictions and reshape into grid.

        If the model has :attr:`categorical_dims`, a specific level for each dimension must be specified as key-value pairs
        in `categorical_levels`.

        Parameters
        ----------
        coregion_levels : dict, optional
            Level for each :attr:`categorical_dims` at which to make prediction
        with_noise : bool, default True
            Whether to incorporate aleatoric uncertainty into prediction error

        Returns
        -------
        prediction : UncertainParameterArray
            Predictions as a grid with len(:attr:`continuous_dims`) dimensions
        """

        if self.grid_points is None:
            raise ValueError('Grid must first be specified with `spatial_grid`')

        points = self.grid_points
        if self.coregion_dims:
            points = self.append_coregion_points(points, coregion_levels=coregion_levels)

        self.predict_points(points, param=param, with_noise=with_noise, **kwargs)
        self.predictions = self.predictions.reshape(self.grid_parray.shape)
        self.predictions_X = self.predictions_X.reshape(self.grid_parray.shape)

        return self.predictions

    ################################################################################
    # Proposals
    ################################################################################

    def propose(self, target, acquisition='EI'):
        """Bayesian Optimization with Expected Improvement acquisition function"""
        if self.predictions is None:
            raise ValueError('No predictions to make proposal from!')
        assert_in(acquisition, ['EI', 'PD'])
        param = self.predictions.name

        df = self.get_filtered_data(standardized=False).query('Parameter == @param')
        observed = self.parray(**{param: df['Values']}, stdzd=False)

        target = self.parray(**{param: target}, stdzd=False)
        best_yet = np.min(np.sqrt(np.mean(np.square(observed.z - target.z))))

        if acquisition == 'EI':
            self.proposal_surface = self.predictions.z.EI(target.z, best_yet.z)
        elif acquisition == 'PD':
            self.proposal_surface = self.predictions.z.nlpd(target.z)

        self.proposal_idx = np.argmax(self.proposal_surface)
        self.propsal = self.predictions_X.ravel()[self.proposal_idx]

        return self.proposal

    def append_coregion_points(self, spatial_parray, coregion_levels):
        """Appends coordinates for the supplied coregion dim-level pairs to tall array of spatial coordinates.

        Parameters
        ----------
        spatial_points : ParameterArray
            Tall :class:`ParameterArray` of coordinates, one layer per spatial dimension
        coregion_levels : dict
            Single level for each :attr:`categorical_dims` at which to make prediction

        Returns
        -------
        points : ParameterArray
            Tall `ParameterArray` of coordinates, one layer per spatial and coregion dimension
        """

        if coregion_levels is not None:
            if set(coregion_levels.keys()) != (set(self.coregion_dims) - set(['Parameter'])):
                raise AttributeError('Must specify level for every coregion dimension')

            points = spatial_parray.fill_with(**{dim: self.coregion_coords[dim][level]
                                                 for dim, level in coregion_levels.items()})
        else:
            points = spatial_parray
        return points

    ################################################################################
    # Evaluation
    ################################################################################

    def cross_validate(self, dims, n_train=None, pct_train=None, train_only=None, warm_start=None, seed=None,
                       **MAP_kws):
        """Fits model on random subset of data and evaluates accuracy of predictions on remaining observations.

        This method finds unique combinations of values in the columns specified by ``dims``, takes a random subset of
        these for training, and evaluates the predictions made for the remaining data.

        Parameters
        ----------
        dims : list of str
            Columns from which to take unique combinations as training and testing sets
        n_train : int, optional
            Number of training points to use. Exactly one of `n_train` and `pct_train` must be specified.
        pct_train : float, optional
            Percent of training points to use. Exactly one of `n_train` and `pct_train` must be specified.
        train_only : dict, optional
            Dimension and level names to be always included in the training set.
        warm_start : bool, default True
            Whether to include a minimum of one observation for each level in each `coregion_dim` in the training set.
        seed : int, optional
            Random seed
        **MAP_kws
            Additional

        Returns
        -------

        """
        if not (n_train is None) ^ (pct_train is None):
            raise ValueError('Exactly one of "n_train" and "pct_train" must be specified')

        train_only = {} if train_only is None else train_only
        seed = self.seed if seed is None else seed
        rg = np.random.default_rng(seed)

        df = self.get_filtered_data()
        n_train = n_train if n_train is not None else len(df.set_index(dims).index.unique()) // pct_train

        # Build up a list of dataframes that make up the training set
        train_list = []

        # Ensure levels in train_only are lists
        train_only = {dim: levels if isinstance(levels, list) else [levels] for dim, levels in train_only.items()}

        # Move items in `train_only` to training set
        for dim, levels in train_only.items():
            for level in levels:
                for_train = df[df[dim] == level]
                train_list.append(for_train)
                df = df[df[dim] != level]
                n_train -= len(for_train.set_index(dims).index.unique())
                if n_train <= 0:
                    raise ValueError('Adding `train_only` observations exceeded specified size of training set')

        if warm_start:
            # Add one random item from each coregion level to the training set
            for dim, levels in self.coregion_levels.items():
                for level in levels:
                    if level in train_only.get(dim, []):
                        continue
                    these = df[df[dim] == level].set_index(dims)
                    for_train = these.loc[rg.choice(these.index.unique(), 1, replace=False)]
                    train_list.append(for_train)
                    df = df[df[dim] != level]
                    n_train -= 1
                    if n_train <= 0:
                        raise ValueError('Adding `warm_start` observations exceeded specified size of training set')

        # Move a random subset of the remaining items to the training set
        df = df.set_index(dims)
        if n_train > len(df.index.unique()):
            raise ValueError('Specified size of training set exceeds number of unique combinations found in `dims`')
        train_idxs = rg.choice(df.index.unique(), n_train, replace=False)
        for_train = df.loc[train_idxs]
        train_list.append(for_train)
        train_df = pd.concat(train_list)
        test_df = df.drop(train_idxs)

        specifications = dict(params=self.params, linear_dims=self.linear_dims, spatial_dims=self.spatial_dims,
                              spatial_levels=self.spatial_levels, spatial_coords=self.spatial_coords,
                              coregion_dims=self.coregion_dims, coregion_levels=self.coregion_levels,
                              additive=self.additive)

        # Build and fit a new object of the current class (GP, GLM, etc) with the training set
        train_obj = self.__class__(ParameterSet(train_df), params=self.params, seed=seed)
        train_obj.specify_model(**specifications)
        train_obj.filter_dims = self.filter_dims
        train_obj.build_model()
        train_obj.find_MAP(**MAP_kws)

        # Get in-sample prediction metrics
        train_X, train_y = train_obj.get_shaped_data()
        train_predictions = train_obj.predict_points(train_X)
        train_nlpd = train_predictions.nlpd(train_y)
        train_rmse = np.sqrt(np.mean(np.square(train_y.z - train_predictions.z.μ)))

        if len(test_df.index.unique()) > 0:
            # If there's anything left for a testing set, build and fit a new object with the testing set
            test_obj = self.__class__(ParameterSet(test_df), params=self.params, seed=seed)
            test_obj.specify_model(**specifications)
            test_obj.filter_dims = self.filter_dims

            # Get out-of-sample prediction metrics
            test_X, test_y = test_obj.get_shaped_data()
            test_predictions = train_obj.predict_points(test_X)
            test_nlpd = train_predictions.z.nlpd(test_y.z)
            test_rmse = np.sqrt(np.mean(np.square(test_y.z - test_predictions.z.μ)))
        else:
            test_nlpd = np.nan
            test_rmse = np.nan

        metrics = {'train': {'NLPD': train_nlpd,
                             'RMSE': train_rmse},
                   'test': {'NLPD': test_nlpd,
                            'RMSE': test_rmse}}

        return metrics

    ################################################################################
    # Plotting
    ################################################################################

    def get_conditional_prediction(self, **dim_values):
        """The conditional prediction at the given values of the specified dimensions over the remaining dimension(s).

        Conditioning the prediction on specific values of `m` dimensions can be thought of as taking a "slice" along the
        remaining `n` dimensions.

        Performs `(m+n)`-dimensional interpolation over the entire prediction grid for each of the mean and variance
        separately, then returns the interpolation evaluated at the specified values for the provided dimensions and the
        original values for the remaining dimensions.

        Parameters
        ----------
        dim_values
            Keyword arguments specifying value for each dimension at which to return the conditional prediction of the
            remaining dimensions.

        Returns
        -------
        conditional_grid: ParameterArray
            `n`-dimensional grid with `n` parameters (layers) at which the conditional prediction is evaluated
        conditional_prediction: UncertainParameterArray
            `n`-dimensional grid of predictions conditional on the given values of the `m` specified dimensions
        """

        self._check_has_prediction()
        all_dims = self.prediction_dims

        # All points along every axis (parrays)
        # Note that these may not all be the same length
        all_margins = {dim: vec.squeeze() for dim, vec in self.grid_vectors.items() if dim in self.prediction_dims}

        # The dimensions to be "kept" are the ones not listed in kwargs
        keep = set(all_dims) - set(dim_values.keys())
        kept_margins = [all_margins[dim] for dim in self.prediction_dims if dim in keep]

        # parray grid of original points along all "kept" dimensions
        conditional_grid = self.parray(**{array.names[0]: array.values() for array in np.meshgrid(*kept_margins)})
        # Add specified value for each remaining dimension at all points, then unravel
        xi_parray = conditional_grid.add_layers(
            **{dim: np.full(conditional_grid.shape, value) for dim, value in dim_values.items()}
        ).ravel()

        # Stack standardized points into (ordinary) tall array, ensuring dimensions are in the right order for the model
        xi_pts = np.column_stack([xi_parray[dim].z.values() for dim in self.dims if dim in xi_parray.names])

        # Interpolate the mean and variance of the predictions
        # Swapping the first two axes is necessary because grids were generated using meshgrid's default "ij" indexing
        # but interpn expects "xy" indexing
        μ_arr = np.swapaxes(self.predictions.μ, 0, 1)
        μi = interpn([all_margins[dim].z.values() for dim in self.dims if dim in self.prediction_dims], μ_arr, xi_pts)
        σ2_arr = np.swapaxes(self.predictions.σ2, 0, 1)
        σ2i = interpn([all_margins[dim].z.values() for dim in self.dims if dim in self.prediction_dims], σ2_arr, xi_pts)

        conditional_prediction = self.uparray(self.predictions.name, μ=μi, σ2=σ2i).reshape(*conditional_grid.shape)

        return conditional_grid.squeeze(), conditional_prediction.squeeze()

    def plot_predictions(self, **kwargs):

        self._check_has_prediction()

        ndim = self.predictions.ndim

        if ndim==1:
            return self._plot_1d_predictions(**kwargs)
        elif ndim==2:
            return self._plot_2d_predictions(**kwargs)
        else:
            raise ValueError('Plotting predictions with more than two dimensions is not yet supported.')

    def _plot_1d_predictions(self, **kwargs):

        ax = plot_uparray(self.predictions_X, self.predictions, **kwargs)

        xticks = self.ticks[self.predictions_X.names[0]]
        ax.set_xticks(xticks.z.values())
        unstandardize_axis_labels(ax, 'x', xticks)

    def _plot_2d_predictions(self, **kwargs):

        ax = contourf_uparray(*self.predictions_X.to_list(), self.predictions, **kwargs)

        xticks, yticks = [self.ticks[dim] for dim in self.predictions_X.names]
        ax.set_xticks(xticks.z.values())
        unstandardize_axis_labels(ax, 'x', xticks)
        ax.set_xticklabels([f'{int(xt * 100)}' for xt in xticks.values()])
        ax.set_yticks(yticks.z.values())
        unstandardize_axis_labels(ax, 'y', yticks)
        ax.set_yticklabels([f'{int(yt)}' for yt in yticks.values()])

        return ax

