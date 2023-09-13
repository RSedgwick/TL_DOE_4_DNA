from __future__ import annotations  # Necessary for self-type annotations until Python >3.10

import copy
import pickle
import warnings
import numpy as np
import pandas as pd
import pymc3 as pm

from scipy.special import logit, expit
from scipy.stats import norm, lognorm, chi2, ncx2, rv_continuous, multivariate_normal
from uncertainties import unumpy as unp
from collections import namedtuple

from candas.utils import skip, assert_in, assert_is_subset, NotImplementedWrapper

__all__ = ['Standardizer', 'DistributionStandardizer', 'Parameter', 'ParameterSet',
           'LayeredArray', 'ParameterArray', 'UncertainArray', 'UncertainParameterArray', 'MVUncertainParameterArray']

class Standardizer(dict):
    r"""Container for dict of mean (μ) and standard deviation (σ) for every parameter.

    :class:`Standardizer` objects allow transformation and normalization of datasets. The main methods are :meth:`stdz`,
    which attempts to coerce the values of a given variable to a standard normal distribution (`z-scores`), and its
    complement :meth:`unstdz`. The steps are

    .. math::
        \mathbf{\text{data}} \rightarrow \text{transform} \rightarrow \text{mean-center} \rightarrow \text{scale}
        \rightarrow \mathbf{\text{zdata}}

    For example, reaction `rate` must clearly be strictly positive, so we use a `log` transformation so that it behaves
    as a normally-distributed random variable. We then mean-center and scale this transformed value to obtain `z-scores`
    indicating how similar a given estimate is to all the other estimates we've observed. `Standardizer` stores the
    transforms and population mean and standard deviation for every parameter, allowing us to convert back and forth
    between natural space (:math:`rate`), transformed space (:math:`\text{ln}\; rate`), and standardized space
    (:math:`\left( \text{ln}\; rate  - \mu_{\text{ln}\; rate} \right)/\sigma_{\text{ln}\; rate}`).


    Notes
    -----
    :class:`Standardizer` is just a `dictionary <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`_
    with some extra methods and defaults, so standard dictionary methods like :meth:`dict.update` still work.

    """

    # TODO: Standardizer: Make required_parameters and required_descriptors optional definition at init
    _required_parameters = ['τ', 'ρ', 'r', 'K', 'm', 'lg10_Copies', 'BP', 'GC']
    _required_descriptors = ['Parameter', 'lg10_Copies', 'BP', 'GC']

    # TODO: Standardizer: make `defaults` optional definition at init
    defaults = {
        'ρ': {'μ': -1.056, 'σ': 0.398},
        'τ': {'μ': 3.34, 'σ': 0.1501},
        'K': {'μ': -0.0368, 'σ': 0.351},
        'm': {'μ': -5.30, 'σ': 0.582},
        'offset': {'μ': 0.214, 'σ': 0.0725},
        'lg10_Copies': {'μ': 5, 'σ': 2},
        'BP': {'μ': 4.48, 'σ': 0.75},
        'GC': {'μ': -0.282, 'σ': 1},
        'r': {'μ': -0.307, 'σ': 0.158},
        'F0_lg': {'μ': -0.762, 'σ': 1.258},
        'bkg_F': {'μ': 0.200, 'σ': 0.0580},
        'bkg_Cycle': {'μ': 38.2, 'σ': 23.0}
    }

    # TODO: Standardizer: make `transforms` and `pymc_transforms` definable via string options
    # TODO: Standardizer: make transform suggestions based on provided data? e.g., all>0 -> log/exp
    transforms = {
        'r': [np.log, np.exp],
        'ρ': [logit, expit],
        'τ': [np.log, np.exp],
        'τ_': [np.log, np.exp],
        'K': [np.log, np.exp],
        'm': [np.log, np.exp],
        'offset': [skip, skip],
        'lg10_Copies': [skip, skip],
        'BP': [np.log, np.exp],
        'GC': [logit, expit]
    }

    pymc_transforms = {
        'r': [skip, skip],
        'ρ': [pm.math.logit, pm.math.invlogit],
        'τ': [pm.math.log, pm.math.exp],
        'τ_': [pm.math.log, pm.math.exp],
        'K': [pm.math.log, pm.math.exp],
        'm': [pm.math.log, pm.math.exp],
        'offset': [skip, skip],
        'Q': [skip, skip],
        'BP': [pm.math.log, pm.math.exp],
        'GC': [pm.math.logit, pm.math.invlogit]
    }

    def __init__(self, **kwargs):
        self.validate(kwargs)
        super().__init__(**kwargs)

    @classmethod
    def validate(cls, dct: dict):
        """Ensures provided dictionary has all required attributes"""
        assert_is_subset('Parameters', cls._required_parameters, dct.keys())

    @classmethod
    def default(cls):
        """Initializes Standardizer with default values"""
        return cls(**cls.defaults)

    def reset(self):
        """Revert to defaults"""
        self.update(**self.defaults)
        for k in self.keys():
            if k not in self.defaults.keys():
                del self[k]
        return self

    def save(self, filename: str):
        """Save to pickle file"""
        with open(filename, 'wb') as buff:
            pickle.dump(self, buff)

    @classmethod
    def load(cls, filename: str):
        """Load from pickle file"""
        with open(filename, 'rb') as buff:
            dct = pickle.load(buff)
        return cls(**dct)

    @classmethod
    def from_DataFrame(cls, df: pd.DataFrame):
        """Construct from DataFrame"""
        assert_in('"Parameter"', 'Parameter', df.columns)
        if 'Metric' in df.columns:
            if 'mean' in df['Metric'].unique():
                df = df[df['Metric'] == 'mean']
            else:
                raise ValueError('If DataFrame contains column "Metric", "means" must be present in that column')
        dct = (df
               .groupby('Parameter')
               .apply(cls.transform_series)
               .groupby('Parameter')
               .agg([np.mean, np.std])
               .rename(columns={"mean": "μ", "std": "σ"})
               .T
               .to_dict()
               )
        return cls(**{**cls.defaults, **dct})

    @classmethod
    def transform(cls, name: str, x: float, lg10_Copies=5., pymc3=False) -> float:
        """Apply appropriate forward transformation to parameter

        Parameters
        ----------
        x: float
            Value to be transformed
        name: str
            Name of parameter
        lg10_Copies: float, default 5.
            Corresponding log10 concentration; only necessary for τ
        pymc3: bool, optional
            Whether to use pymc3's transforms

        Returns
        -------
        float
        """
        _transforms = cls.transforms if not pymc3 else cls.pymc_transforms
        ftransform = _transforms.get(name, [skip, skip])[0]
        if name == 'τ':
            assert lg10_Copies is not None, 'Concentration must be supplied to transform τ'
            x = x + np.log2(10) * (lg10_Copies - 5)
        return ftransform(x)

    @classmethod
    def untransform(cls, name: str, x: float, lg10_Copies=5., pymc3=False) -> float:
        """Apply appropriate reverse transformation to parameter

        Parameters
        ----------
        x: float
            Value to be transformed
        name: str
            Name of parameter
        lg10_Copies: float, default 5.
            Corresponding log10 concentration; only necessary for τ
        pymc3: bool, optional
            Whether to use pymc3's transforms

        Returns
        -------
        float
        """
        _transforms = cls.transforms if not pymc3 else cls.pymc_transforms
        rtransform = _transforms.get(name, [skip, skip])[1]
        x_ = rtransform(x)
        if name == 'τ':
            assert lg10_Copies is not None, 'Concentration must be supplied to transform τ'
            x_ = x_ - np.log2(10) * (lg10_Copies - 5)
        return x_

    @classmethod
    def transform_series(cls, series: pd.Series, val_column='Value') -> float:
        """Apply appropriate transform to parameter in series

        Parameters
        ----------
        series: pd.Series
            Series containing value to be transformed. Series.name must be the name of the parameter and series must
             also contain a 'lg10_Copies' attribute.
        val_column: str, default 'Value'
            Name of series column containing value to be transformed.

        Returns
        -------
            float
        """
        assert series.name is not None
        return cls.transform(str(series.name), series[val_column], series.lg10_Copies)

    def stdz(self, name: str, x: float, lg10_Copies=5., pymc3=False) -> float:
        """Transforms, mean-centers, and scales parameter

        Parameters
        ----------
        x: float
        name: str
            Name of parameter
        lg10_Copies: float, default 5.
            Corresponding log10 concentration; only necessary for τ
        pymc3: bool, optional
            Whether to use pymc3's transforms

        Returns
        -------
        float
        """
        x_ = self.transform(name, x, lg10_Copies, pymc3)
        μ = self.get(name, {'μ': 0})['μ']
        σ = self.get(name, {'σ': 1})['σ']
        return (x_ - μ) / σ

    def stdz_series(self, series: pd.Series, val_column='Value') -> float:
        """Apply appropriate transform to parameter in series

        Parameters
        ----------
        series: pd.Series
            Series containing value to be transformed.
            Series.name must be the name of the parameter and series must also contain a 'lg10_Copies' attribute.
        val_column: str, default 'Value'
            Name of series column containing value to be transformed.

        Returns
        -------
            float
        """
        assert series.name is not None
        return self.stdz(str(series.name), series[val_column], series.lg10_Copies, pymc3=False)

    def unstdz(self, name: str, z: float, lg10_Copies=5., pymc3=False) -> float:
        """Un-scales, un-centers, and un-transforms parameter

        Parameters
        ----------
        z: float
        name: str
            Name of parameter
        lg10_Copies: float, default 5.
            Corresponding log10 concentration; only necessary for τ
        pymc3: bool, optional
            Whether to use pymc3's transforms

        Returns
        -------
        float
        """
        μ = self.get(name, {'μ': 0})['μ']
        σ = self.get(name, {'σ': 1})['σ']
        x_ = z * σ + μ
        return self.untransform(name, x_, lg10_Copies, pymc3)


class DistributionStandardizer(Standardizer):
    """Variation of :class:`Standardizer` that acts on mean-variance tuples"""

    # TODO: improve DistributionStandardizer support for logitnormal
    # TODO: DistributionStandardizer and UParray: store z values, display n values?
    _samples = 10000

    # TODO: unify DistributionStandardizer and Standardizer; behavior depends on whether or not σ2 is passed
    @property
    def stdzr(self):
        return Standardizer(**self)

    @classmethod
    def _LogitNormal(cls, μ, σ2):
        # from https://en.wikipedia.org/wiki/Logit-normal_distribution#Moments
        samples = cls._samples
        _norm = norm(loc=μ, scale=np.sqrt(σ2))
        quantiles = _norm.ppf(np.arange(1, samples)/samples)
        e_x = np.nanmean(logit(quantiles))
        e_x2 = np.nanmean(logit(quantiles)**2)
        var = e_x2-e_x**2
        return namedtuple('LogitNormal', 'mean var')(e_x, var)

    @property
    def mean_transforms(self):
        transforms = {skip: [lambda μ, σ2: μ,
                             lambda μ, σ2: μ],
                      # Note these are no longer strictly mean and variance. They are defined to be compatible with
                      # scipy.stats.lognormal definition
                      np.log: [lambda μ, σ2: np.log(μ),
                               lambda μ, σ2: np.exp(μ)],
                      logit: [lambda μ, σ2: self._LogitNormal(μ, σ2).mean,
                              lambda μ, σ2: np.mean(expit(norm(loc=μ, scale=np.sqrt(σ2)).rvs(self._samples)))]
                      }
        return transforms

    @property
    def var_transforms(self):
        transforms = {skip: [lambda μ, σ2: σ2,
                             lambda μ, σ2: σ2],
                      # Note these are no longer strictly mean and variance. They are defined to be compatible with
                      # scipy.stats.lognormal definition
                      np.log: [lambda μ, σ2: σ2,
                               lambda μ, σ2: σ2],
                      logit: [lambda μ, σ2: self._LogitNormal(μ, σ2).var,
                              lambda μ, σ2: np.var(expit(norm(loc=μ, scale=np.sqrt(σ2)).rvs(self._samples)))]
                      }
        return transforms

    def transform(self, name: str, mean: float, var: float, lg10_Copies=5.) -> tuple:
        f_transform = self.transforms.get(name, [skip, skip])[0]
        if f_transform is logit:
            warnings.warnings.warn('Transforming the moments of the LogitNormal distribution from (0,1) to (-inf, +inf) is unstable.')
        f_mean_transform = self.mean_transforms[f_transform][0]
        f_var_transform = self.var_transforms[f_transform][0]

        if name == 'τ':
            assert lg10_Copies is not None, 'Concentration must be supplied to transform τ'
            mean = mean + np.log2(10) * (lg10_Copies - 5)

        mean_ = f_mean_transform(mean, var)
        var_ = f_var_transform(mean, var)
        return mean_, var_

    def untransform(self, name: str, mean: float, var: float, lg10_Copies=5.) -> tuple:
        f_transform = self.transforms.get(name, [skip, skip])[0]
        r_mean_transform = self.mean_transforms[f_transform][1]
        r_var_transform = self.var_transforms[f_transform][1]

        mean_ = r_mean_transform(mean, var)
        var_ = r_var_transform(mean, var)

        if name == 'τ':
            assert lg10_Copies is not None, 'Concentration must be supplied to transform τ'
            mean_ = mean_ - np.log2(10) * (lg10_Copies - 5)

        return mean_, var_

    def stdz(self, name: str, mean: float, var: float, lg10_Copies=5.) -> float:
        """Transforms, mean-centers, and standardizes parameter

        Parameters
        ----------
        mean: float
            Standardized distribution mean
        var: float
            Standardized distribution variance
        name: str
            Name of parameter
        lg10_Copies: float, default 5.
            Corresponding log10 concentration; only necessary for τ

        Returns
        -------
        (mean_z, var_z) : tuple of float
        """
        mean_, var_ = self.transform(name, mean, var, lg10_Copies)
        μ = self[name]['μ']
        σ = self[name]['σ']
        mean_z = (mean_ - μ) / σ
        var_z = var_/σ**2
        return mean_z, var_z

    def unstdz(self, name: str, z_mean: float, z_var: float, lg10_Copies=5.) -> float:
        """Transforms, mean-centers, and standardizes parameter

        Parameters
        ----------
        z_mean: float
            Standardized distribution mean
        z_var: float
            Standardized distribution variance
        name: str
            Name of parameter
        lg10_Copies: float, default 5.
            Corresponding log10 concentration; only necessary for τ

        Returns
        -------
        tuple
        """
        μ = self[name]['μ']
        σ = self[name]['σ']
        mean_ =  z_mean * σ + μ
        var_ = z_var * σ**2
        mean, var = self.untransform(name, mean_, var_, lg10_Copies)
        return mean, var


class NamedFloat(float):
    def __new__(cls, name: str, val: float):
        parameter = float.__new__(cls, val)
        parameter.name = name
        return parameter

    def __repr__(self):
        return f'{self.name}: {super().__repr__()}'


class Parameter(NamedFloat):
    def __new__(cls, name: str, val: float, stdzr: Standardizer, lg10_Copies=None, stdzd=False):
        if stdzd:
            val = stdzr.unstdz(name, val, lg10_Copies=lg10_Copies)
        parameter = NamedFloat.__new__(cls, name, val)
        parameter.z = NamedFloat(name +'_z', stdzr.stdz(parameter.name, parameter, lg10_Copies))
        return parameter


class LayeredArray(np.ndarray):
    """An array with one or more named values at every index.

    Parameters
    ----------
    name : str
    array : array-like
    """
    def __new__(cls, **arrays):
        if arrays == {}:
            raise ValueError('Must supply at least one array')
        arrays = {name: np.asarray(array) for name, array in arrays.items() if array is not None}

        narray_dtype = np.dtype([(name, array.dtype) for name, array in arrays.items()])

        narray_prototype = np.empty(list(arrays.values())[0].shape, dtype=narray_dtype)
        for name, array in arrays.items():
            narray_prototype[name] = array

        larray = narray_prototype.view(cls)
        larray.names = list(narray_dtype.fields.keys())

        return larray

    def __array_finalize__(self, larray):
        if larray is None:
            return
        self.names = getattr(larray, 'names', None)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """https://numpy.org/doc/stable/user/basics.subclassing.html#array-ufunc-for-ufuncs"""
        args = []
        if len(set([larray.names[0] for larray in inputs if isinstance(larray, LayeredArray)]))>1:
            warnings.warnings.warn('Operating on arrays with different layer names, results may be unexpected.')
        for input_ in inputs:
            if isinstance(input_, LayeredArray):
                if len(input_.names) > 1:
                    raise ValueError('Cannot operate on array with multiple layer names')
                args.append(input_.astype(float).view(np.ndarray))
            else:
                args.append(input_)

        outputs = kwargs.get('out')
        if outputs:
            out_args = []
            for output in outputs:
                if isinstance(output, LayeredArray):
                    out_args.append(output.astype(float).view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)

        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        results = tuple(LayeredArray(**{self.names[0]: result})
                        if output is None else output
                        for result, output in zip(results, outputs))

        return results[0] if len(results) == 1 else results

    def __getitem__(self, item):
        default = super().__getitem__(item)
        if isinstance(item, str):
            return LayeredArray(**{item: default})
        elif isinstance(item, int) or (isinstance(item, tuple) and all(isinstance(val, int) for val in item)):
            return LayeredArray(**{name: value for name, value in zip(default.dtype.names, default)})
        return default

    def __repr__(self):
        return f'{tuple(self.names)}: {np.asarray(self)}'

    def get(self, name, default=None):
        """Return value given by `name` if it exists, otherwise return `default`"""
        if name in self.names:
            return self[name]
        elif default is None:
            return None
        else:
            return LayeredArray(**{name: default})

    def drop(self, name, missing_ok=True):
        if name in self.names:
            return LayeredArray(**{p: arr for p, arr in self.to_dict().items() if p != name})
        elif missing_ok:
            return self
        else:
            raise KeyError(f'Name {name} not found in array.')

    def values(self):
        """Values at each index stacked into regular ndarray"""
        stacked = np.stack([self[name].astype(float) for name in self.names])
        if len(self.names) > 1:
            return stacked
        else:
            return stacked[0]

    def to_list(self, order=None):
        order = self.names if order is None else order
        assert all(name in order for name in self.names)
        return [self[name] for name in order]

    def to_dict(self):
        """Values corresponding to each named level as a dictionary"""
        return {name: self[name].values() for name in self.names}

    def add_layers(self, **arrays):
        """Add additional layers at each index"""
        arrays_ = arrays.to_dict() if isinstance(arrays, LayeredArray) else arrays
        return LayeredArray(**self.to_dict(), **arrays_)


class ParameterArray(LayeredArray):
    """Array of parameter values, allowing simple transformation.

    :class:`ParameterArray` stores not only the value of the variable itself but also the corresponding lg10_Copies, if
    necessary, as well as a :class:`Standardizer` instance. This makes it simple to switch between the natural scale of
    the parameter and its transformed and standardized values through the :attr:`t` and :attr:`z` properties,
    respectively.

    This class can also be accessed through the alias :class:`parray`.

    Parameters
    ----------
    **arrays
        arrays to store with their names as keywords
    stdzr : Standardizer
        An instance  of :class:`Standardizer`
    stdzd : bool, default False
        Whether the supplied values are on standardized scale instead of the natural scale

    Examples
    --------

    A parray can created with a single parameter. In this case, `r` is treated as a `LogNormal` variable by the stdzr.

    >>> from candas.learn import ParameterArray as parray
    >>> stdzr = Standardizer.default()
    >>> rpa = parray(r=np.arange(5,10)/10, stdzr=stdzr)
    >>> rpa
    ('r',): [(0.5,) (0.6,) (0.7,) (0.8,) (0.9,)]
    >>> rpa.t
    ('r_t',): [(-0.69314718,) (-0.51082562,) (-0.35667494,) (-0.22314355,) (-0.10536052,)]
    >>> rpa.z
    ('r_z',): [(-2.4439695 ,) (-1.29003559,) (-0.31439838,) ( 0.53073702,) ( 1.27619927,)]

    Creating an array with `τ` values requires `lg10_Copies` to also be present. Note that `lg10_Copies` has no
    transformation associated with it in the stdzr, so its :attr:`t` values are the same as its natural-space values.

    >>> tpa = parray(τ=np.arange(25,30), stdzr=stdzr, lg10_Copies = np.full(5, 5))
    >>> tpa
    ('τ', 'lg10_Copies'): [(25, 5) (26, 5) (27, 5) (28, 5) (29, 5)]
    >>> tpa.t
    ('τ_t', 'lg10_Copies_t'): [(3.21887582, 5) (3.25809654, 5) (3.29583687, 5) (3.33220451, 5) (3.36729583, 5)]
    >>> tpa.z
    ('τ_z', 'lg10_Copies_z'): [(-0.80695653, 0.) (-0.54565931, 0.) (-0.29422474, 0.) (-0.05193531, 0.) ( 0.18185097, 0.)]

    If the parameter is completely absent from the stdzr, its natural, :attr:`t`, and :attr:`z` values are identical.

    >>> pa = parray(param=np.arange(5), stdzr=stdzr)
    >>> pa
    ('param',): [(0,) (1,) (2,) (3,) (4,)]
    >>> pa.t
    ('param_t',): [(0,) (1,) (2,) (3,) (4,)]
    >>> pa.z
    ('param_z',): [(0.,) (1.,) (2.,) (3.,) (4.,)]

    You can even do monstrous compositions like

    >>> np.min(np.sqrt(np.mean(np.square(rpa-rpa[0]-0.05)))).t
    ('r_t',): (-1.5791256,)

    If you `have` work with an ordinary numpy array, use :meth:`values`.

    >>> np.argmax(rpa.values())
    4

    Attributes
    ----------
    names : list of str
        Names of all stored parameters
    stdzr : Standardizer
        An instance  of :class:`Standardizer`
    """

    def __new__(cls, stdzr: Standardizer, stdzd=False, **arrays):
        if arrays == {}:
            raise ValueError('Must supply at least one array')

        if stdzd:
            if 'τ' in arrays.keys() and 'lg10_Copies' not in arrays.keys():
                raise KeyError('lg10_Copies must be provided with values of τ')
            lg10_Copies = stdzr.unstdz('lg10_Copies', arrays['lg10_Copies']) if 'τ' in arrays.keys() else None
            arrays = {name: stdzr.unstdz(name, np.array(array))
                      if name != 'τ' else stdzr.unstdz(name, array, lg10_Copies=lg10_Copies)
                      for name, array in arrays.items()}

        parray = LayeredArray.__new__(cls, **arrays)
        parray.stdzr = stdzr

        return parray

    def __array_finalize__(self, parray):
        if parray is None:
            return
        self.stdzr = getattr(parray, 'stdzr', None)
        self.names = getattr(parray, 'names', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if result is NotImplemented:
            return NotImplemented
        return ParameterArray(**result.to_dict(), stdzr=self.stdzr, stdzd=False)

    def __getitem__(self, item):
        default = super(LayeredArray, self).__getitem__(item)
        if isinstance(item, str):
            return ParameterArray(**{item: default}, stdzr=self.stdzr, stdzd=False)
        elif isinstance(item, int) or (isinstance(item, tuple) and all(isinstance(val, int) for val in item)):
            # lg10_Copies = self['lg10_Copies'][item] if 'lg10_Copies' in self.names else None
            arrays = {name: value for name, value in zip(default.dtype.names, default)}
            return ParameterArray(**arrays, stdzr=self.stdzr, stdzd=False)
        return default

    def get(self, name, default=None):
        """Return value given by `name` if it exists, otherwise return `default`"""
        lg10_Copies = self['lg10_Copies'] if 'lg10_Copies' in self.names and name == 'τ' else None
        if name in self.names:
            return self[name]
        elif default is None:
            return None
        else:
            return self.parray(**{name: default})

    def drop(self, name, missing_ok=True):
        if name in self.names:
            return self.parray(**{p: arr for p, arr in self.to_dict().items() if p != name})
        elif missing_ok:
            return self
        else:
            raise KeyError(f'Name {name} not found in array.')

    @property
    def z(self) -> LayeredArray:
        """Standardized values"""
        lg10_Copies = self.get('lg10_Copies', None)
        zdct = {name+'_z': self.stdzr.stdz(name, self[name].values(), lg10_Copies) for name in self.names}
        return LayeredArray(**zdct)

    @property
    def t(self) -> LayeredArray:
        """Transformed values"""
        lg10_Copies = self.get('lg10_Copies', None)
        tdct = {name+'_t': self.stdzr.transform(name, self[name].values(), lg10_Copies) for name in self.names}
        return LayeredArray(**tdct)

    def add_layers(self, stdzd=False, **arrays):
        """Add additional layers at each index"""
        narrays = super().add_layers(**arrays)
        if stdzd:
            for name in narrays.names:
                narrays[name] = self.stdzr.unstdz(name, narrays[name], lg10_Copies=narrays.get('lg10_Copies'))
        return self.parray(**narrays.to_dict(), stdzd=False)

    def fill_with(self, **params):
        assert all([isinstance(value, (float, int)) for value in params.values()])
        assert all([isinstance(key, str) for key in params.keys()])
        return self.add_layers(**{param: np.full(self.shape, value) for param, value in params.items()})

    def parray(self, *args, **kwargs):
        """Create a new ParameterArray using this instance's standardizer"""
        return ParameterArray(*args, **kwargs, stdzr=self.stdzr)

    @classmethod
    def stack(cls, parray_list, axis=0, **kwargs):
        all_names = [pa.names for pa in parray_list]
        if not all(names == all_names[0] for names in all_names):
            raise ValueError('Arrays do not have the same names!')
        new = np.stack(parray_list, axis=axis, **kwargs)
        stdzr = parray_list[0].stdzr
        return cls(**{dim: new[dim] for dim in new.dtype.names}, stdzr=stdzr)

    @classmethod
    def vstack(cls, parray_list, **kwargs):
        all_names = [pa.names for pa in parray_list]
        if not all(names == all_names[0] for names in all_names):
            raise ValueError('Arrays do not have the same names!')
        new = np.vstack(parray_list, **kwargs)
        stdzr = parray_list[0].stdzr
        return cls(**{dim: new[dim] for dim in new.dtype.names}, stdzr=stdzr)

    @classmethod
    def hstack(cls, parray_list, **kwargs):
        all_names = [pa.names for pa in parray_list]
        if not all(names == all_names[0] for names in all_names):
            raise ValueError('Arrays do not have the same names!')
        new = np.hstack(parray_list, **kwargs)
        stdzr = parray_list[0].stdzr
        return cls(**{dim: new[dim] for dim in new.dtype.names}, stdzr=stdzr)


class UncertainArray(np.ndarray):
    """Structured array containing mean and variance of a normal distribution at each point.

    The main purpose of this object is to correctly `propagate uncertainty`_ under transformations. Arithmetic
    operations between distributions or between distributions and scalars are handled appropriately via the
    `uncertainties`_ package.

    Additionally, a `scipy Normal distribution`_ object can be created at each point through the :attr:`dist` property,
    allowing access to that objects such as :meth:`rvs`, :meth:`ppf`, :meth:`pdf`, etc.

    This class can also be accessed through the alias :class:`uarray`.

    .. _`propagate uncertainty`: https://en.wikipedia.org/wiki/Propagation_of_uncertainty
    .. _`uncertainties`: https://pythonhosted.org/uncertainties/
    .. _`scipy Normal distribution`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html

    Notes
    -----
    The `name` argument is intended to be the general name of the value held, not unique to this instance. Combining two
    :class:`UncertainArray` objects with the same name results in a new object with that name; combining two objects
    with different names results in a new name that reflects this combination (so ``'A'+'B'`` becomes ``'(A+B)'``).

    Parameters
    ----------
    name : str
        Name of variable.
    μ : array
        Mean at each point
    σ2 : array
        Variance at each point
    **kwargs
        Names and values of additional arrays to store

    Examples
    --------
    >>> ua1 = UncertainArray('A', μ=1, σ2=0.1)
    >>> ua2 = uarray('A', μ=2, σ2=0.2)  #equivalent
    >>> ua2
    A['μ', 'σ2']: (2, 0.2)

    Addition of a scalar

    >>> ua1+1
    A['μ', 'σ2']: (2, 0.1)

    Addition and subtraction of two UncertainArrays:

    >>> ua2+ua1
    A['μ', 'σ2']: (3., 0.3)
    >>> ua2-ua1
    A['μ', 'σ2']: (1., 0.3)

    Note, however, that correlations are not properly accounted for (yet). Subtracting one UncertainArray from itself
    should give exactly zero with no uncertainty, but it doesn't:

    >>> ua2+ua2
    A['μ', 'σ2']: (4., 0.4)
    >>> ua2-ua2
    A['μ', 'σ2']: (0., 0.4)

    Mean of two `uarray` objects:

    >>> uarray.stack([ua1, ua2]).mean(axis=0)
    A['μ', 'σ2']: (1.5, 0.075)

    Mean within a single `uarray` object:

    >>> ua3 = uarray('B', np.arange(1,5)/10, np.arange(1,5)/100)
    >>> ua3
    B['μ', 'σ2']: [(0.1, 0.01) (0.2, 0.02) (0.3, 0.03) (0.4, 0.04)]
    >>> ua3.μ
    array([0.1, 0.2, 0.3, 0.4])
    >>> ua3.mean()
    B['μ', 'σ2']: (0.25, 0.00625)

    Adding two `uarrays` with differnt name creates an object with a new name

    >>> ua1+ua3.mean()
    (A+B)['μ', 'σ2']: (1.25, 0.10625)

    Accessing :attr:`dist` methods

    >>> ua3.dist.ppf(0.95)
    array([0.26448536, 0.43261743, 0.58489701, 0.72897073])
    >>> ua3.dist.rvs([3,*ua3.shape])
    array([[0.05361942, 0.14164882, 0.14924506, 0.03808633],
           [0.05804824, 0.09946732, 0.08727794, 0.28091272],
           [0.06291355, 0.47451576, 0.20756356, 0.2108717 ]])  # random

    Attributes
    ----------
    name : str
        Name of variable.
    μ : array
        Mean at each point
    σ2 : array
        Variance at each point
    fields : list of str
        Names of each level held in the array
    """

    def __new__(cls, name: str, μ: np.ndarray, σ2: np.ndarray, **kwargs):
        μ_ = np.asarray(μ)
        σ2_ = np.asarray(σ2)
        assert(μ_.shape == σ2_.shape)
        base_dtypes = [('μ', μ_.dtype), ('σ2', σ2_.dtype)]
        extra_dtypes = [(dim, np.asarray(arr).dtype) for dim, arr in kwargs.items() if arr is not None]
        uarray_dtype = np.dtype(base_dtypes+extra_dtypes)

        uarray_prototype = np.empty(μ_.shape, dtype=uarray_dtype)
        uarray_prototype['μ'] = μ_
        uarray_prototype['σ2'] = σ2_
        for dim, arr in kwargs.items():
            if arr is not None:
                uarray_prototype[dim] = np.asarray(arr)

        uarray = uarray_prototype.view(cls)
        uarray.name = name
        uarray.fields = list(uarray_dtype.fields.keys())

        return uarray

    def __array_finalize__(self, uarray):
        if uarray is None:
            return
        self.name = getattr(uarray, 'name', None)
        self.fields = getattr(uarray, 'fields', None)

    @property
    def μ(self) -> np.ndarray:
        # Nominal value (mean)
        return self['μ']

    @μ.setter
    def μ(self, val):
        self['μ'] = val

    @property
    def σ2(self) -> np.ndarray:
        # Variance
        return self['σ2']

    @σ2.setter
    def σ2(self, val):
        self['σ2'] = val

    @property
    def σ(self) -> np.ndarray:
        """Standard deviation"""
        return np.sqrt(self.σ2)

    @σ.setter
    def σ(self, val):
        self['σ2'] = val**2

    @property
    def _as_uncarray(self):
        return unp.uarray(self.μ, self.σ)

    @classmethod
    def _from_uncarray(cls, name, uncarray, **extra):
        return cls(name=name, μ=unp.nominal_values(uncarray), σ2=unp.std_devs(uncarray)**2, **extra)

    @property
    def dist(self) -> rv_continuous:
        """Array of :func:`scipy.stats.norm` objects"""
        return norm(loc=self.μ, scale=self.σ)

    @staticmethod
    def stack(uarray_list, axis=0) -> UncertainArray:
        new = np.stack(uarray_list, axis=axis)
        names = [ua.name for ua in uarray_list]
        if all(name == names[0] for name in names):
            name = uarray_list[0].name
        else:
            raise ValueError('Arrays do not have the same name!')
            # name = '('+', '.join(names)+')'
        return UncertainArray(name, **{dim: new[dim] for dim in new.dtype.names})

    def nlpd(self, target) -> float:
        """Negative log posterior density"""
        return -np.log(self.dist.pdf(target))

    def EI(self, target, best_yet, k=1) -> float:
        """Expected improvement

        Taken from https://github.com/akuhren/target_vector_estimation

        Parameters
        ----------
        target : float
        best_yet : float
        k : int

        Returns
        -------
        EI : float
        """

        nc = ((target - self.μ) ** 2) / self.σ2

        h1_nx = ncx2.cdf((best_yet / self.σ2), k, nc)
        h2_nx = ncx2.cdf((best_yet / self.σ2), (k+2), nc)
        h3_nx = ncx2.cdf((best_yet / self.σ2), (k+4), nc)

        t1 = best_yet * h1_nx
        t2 = self.σ2 * (k * h2_nx + nc * h3_nx)

        return t1 - t2

    def KLD(self, other):
        """Kullback–Leibler Divergence

        Parameters
        ----------
        other : UncertainArray

        Returns
        -------
        divergence : float
        """
        assert isinstance(other, UncertainArray)
        return np.log(other.σ / self.σ) + (self.σ2 + (self.μ - other.μ) ** 2) / (2 * other.σ2) - 1 / 2

    def BD(self, other):
        """Bhattacharyya Distance

        Parameters
        ----------
        other : UncertainArray

        Returns
        -------
        distance : float
        """
        assert isinstance(other, UncertainArray)
        return 1 / 4 * np.log(1 / 4 * (self.σ2 / other.σ2 + other.σ2 / self.σ2 + 2)) + 1 / 4 * ((self.μ - other.μ) ** 2 / (self.σ2 + other.σ2))


    def BC(self, other):
        """Bhattacharyya Coefficient

        Parameters
        ----------
        other : UncertainArray

        Returns
        -------
        coefficient : float
        """
        return np.exp(-self.BD(other))


    def HD(self, other):
        """Hellinger Distance

        Parameters
        ----------
        other : UncertainArray

        Returns
        -------
        distance : float
        """
        return np.sqrt(1 - self.BC(other))

    def __repr__(self):
        return f'{self.name}{self.fields}: {np.asarray(self)}'

    def __getitem__(self, item):
        default = super().__getitem__(item)
        if isinstance(item, int) or (isinstance(item, tuple) and all(isinstance(val, int) for val in item)):
            return UncertainArray(self.name, **{name: value for name, value in zip(default.dtype.names, default)})
        return default.view(np.ndarray)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, **kwargs) -> UncertainArray:
        """Summation with uncertainty propagation"""
        kwargs |= dict(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        new = self._as_uncarray.sum(**kwargs)
        extra = {dim: np.sum(self[dim]) for dim in self.fields if dim not in ['μ', 'σ2']}
        return self._from_uncarray(self.name, new, **extra)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, **kwargs) -> UncertainArray:
        """Mean with uncertainty propagation"""
        kwargs |= dict(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        new = self._as_uncarray.mean(**kwargs)
        extra = {dim: np.mean(self[dim]) for dim in self.fields if dim not in ['μ', 'σ2']}
        return self._from_uncarray(self.name, new, **extra)

    def __add__(self, other):
        new = self._as_uncarray
        if isinstance(other, UncertainArray):
            new += other._as_uncarray
            name = self.name if self.name == other.name else f'({self.name}+{other.name})'
        else:
            new += other
            name = self.name
        extra = {dim: np.mean(self[dim]) for dim in self.fields if dim not in ['μ', 'σ2']}
        return self._from_uncarray(name, new, **extra)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        new = self._as_uncarray
        if isinstance(other, UncertainArray):
            new -= other._as_uncarray
            name = self.name if self.name == other.name else f'({self.name}+{other.name})'
        else:
            new -= other
            name = self.name
        extra = {dim: np.mean(self[dim]) for dim in self.fields if dim not in ['μ', 'σ2']}
        return self._from_uncarray(name, new, **extra)

    def __rsub__(self, other):
        if isinstance(other, UncertainArray):
            new = other._as_uncarray
            name = self.name if self.name == other.name else f'({self.name}+{other.name})'
        else:
            new = copy.copy(other)
            name = self.name
        new -= self._as_uncarray
        extra = {dim: np.mean(self[dim]) for dim in self.fields if dim not in ['μ', 'σ2']}
        return self._from_uncarray(name, new, **extra)


class UncertainParameterArray(UncertainArray):
    r"""Structured array of parameter means and variances, allowing transformation with uncertainty handling.

    The primary role of this class is to compactly store the outputs of our regression and estimation models
    (:class:`candas.learn.Bayesian`, :class:`candas.learn.GP`, :class:`candas.learn.GLM`). We typically use these models
    to produce parameter predictions or estimates, but under some transformation. For example, reaction `rate` must
    clearly be strictly positive, so we fit a GP to the `log` of rate in order to more appropriately conform to the
    assumption of normality. For prediction and visualization, however, we often need to switch back and forth between
    natural space (:math:`rate`), transformed space (:math:`\text{ln}\; rate`), and standardized space
    (:math:`\left( \text{ln}\; rate  - \mu_{\text{ln}\; rate} \right)/\sigma_{\text{ln}\; rate}`), meanwhile calculating
    summary statistics such as means and percentiles. This class is intended to facilitate switching between those
    different contexts.

    :class:`UncertainParameterArray`, also accessible through the alias :class:`uparray`, combines the functionality of
    :class:`ParameterArray` and :class:`UncertainArray`. A `uparray` stores the mean and variance of the
    variable itself, the corresponding lg10_Copies (if necessary), as well as a :class:`DistributionStandardizer`
    instance. This  makes it simple to switch between the natural scale of the parameter and its transformed and
    standardized values through the :attr:`t` and :attr:`z` properties, respectively, with the accompanying variance
    transformed and scaled appropriately. This uncertainty is propagated under transformation, as with
    :class:`UncertainArray`, and a scipy distribution object can be created at each point through the :attr:`dist`
    property, allowing access to that objects such as :meth:`rvs`, :meth:`ppf`, :meth:`pdf`, etc.

    Notes
    -----
    The `name` argument is intended to be the general name of the value held, not unique to this instance. Combining two
    :class:`UncertainParameterArray` objects with the same name results in a new object with that name; combining two
    objects with different names results in a new name that reflects this combination (so ``'A'+'B'`` becomes
    ``'(A+B)'``).

    The behavior of this object depends on the transformation associated with it, as indicated by its `name` in its
    stored :class:`Standardizer` instance. If this transformation is :func:`np.log`, the parameter is treated as a
    `LogNormal` variable; otherwise it's treated as a `Normal` variable. This affects which distribution is returned by
    :attr:`dist` (`lognorm`_ vs `norm`_) and also the interpretation of :attr:`μ` and :attr:`σ2`.

     * For a `Normal` random
       variable, these are simply parameter's mean and variance in unstandardized space, :attr:`t.μ` and :attr:`t.σ2`
       are identical to :attr:`μ` and :attr:`σ2`, and :attr:`z.μ` and :attr:`z.σ2` are the parameter's mean and variance
       in standardized space.
     * For a `LogNormal` random variable ``Y``, however, :attr:`t.μ` and :attr:`t.σ2` are the mean and variance of a
       `Normal` variable ``X`` such that ``exp(X)=Y`` (:attr:`z.μ` and :attr:`z.σ2` are this mean and variance in
       standardized space). In this case, :attr:`μ` and :attr:`σ2` are the scale and shape descriptors of ``Y``, so
       ``self.μ = np.exp(self.t.μ)`` and ``self.σ2 = self.t.σ2``. Thus, :attr:`μ` and :attr:`σ2` are not strictly the
       mean and variance of the random variable in natural space, these can be obtained from the :attr:`dist`.

       * This behavior is most important, and potentially most confusing, when calculating the :meth:`mean`. Averaging
         is performed in `transformed` space, where the random variable exhibits a `Normal` distribution and the mean
         also exhibits a `Normal` distribution, allowing error propagation to be applied analytically. The :attr:`μ` and
         :attr:`σ2` returned are the descriptors of the `LogNormal` distribution that represents the reverse
         transformation of this new `Normal` distribution. Therefore, the result is more akin to marginalizing out the
         given dimensions in the underlying model than a true natural-space average.

    .. _norm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    .. _lognorm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html

    See Also
    --------
    `norm <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html>`_: \
    scipy `Normal` random variable

    `lognorm <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html>`_: \
    scipy `LogNormal` random variable

    :class:`ParameterArray`

    :class:`UncertainArray`

    :class:`DistributionStandardizer`

    Parameters
    ----------
    name : str
        Name of variable.
    μ : array
        Mean at each point
    σ2 : array
        Variance at each point
    stdzr : Standardizer
        An instance  of :class:`Standardizer`, converted internally to :class:`DistributionStandardizer`
    stdzd : bool, default False
        Whether the supplied values are on standardized scale instead of the natural scale

    Examples
    --------
    Create a `LogNormal` random variable, as indicated by its :class:`Standardizer`

    >>> from candas.learn import uparray, Standardizer
    >>> import numpy as np
    >>> stdzr = Standardizer.default()
    >>> upa = uparray('m', np.arange(1,5)/10, np.arange(1,5)/100, stdzr)
    >>> upa
    m['μ', 'σ2']: [(0.1, 0.01) (0.2, 0.02) (0.3, 0.03) (0.4, 0.04)]
    >>> stdzr.transforms['m']
    [<ufunc 'log'>, <ufunc 'exp'>]

    Mean and variance of the parameter in standardized space:

    >>> upa.z
    m_z['μ', 'σ2']: [(5.15019743, 0.02952256) (6.34117197, 0.05904512)
                     (7.03784742, 0.08856768) (7.53214651, 0.11809024)]

    Verify round-trip transformation:

    >>> upa.dstdzr.unstdz(upa.name, upa.z.μ, upa.z.σ2)
    (array([0.1, 0.2, 0.3, 0.4]), array([0.01, 0.02, 0.03, 0.04]))

    Create a `uparray` from already-standardized values and verify round-trip transformation:

    >>> uparray('m', np.arange(-2,3), np.arange(1,6)/10, stdzr, stdzd=True).z
    m_z['μ', 'σ2']: [(-2., 0.1) (-1., 0.2) ( 0., 0.3) ( 1., 0.4) ( 2., 0.5)]

    For `LogNormal` parameters, uparray follows the `scipy.stats` convention  of parameterizing a lognormal random
    variable in terms of it's natural-space mean and its log-space standard deviation. Thus, a LogNormal uparray defined
    as `m['μ', 'σ2']: (0.1, 0.01)` represents `exp(Normal(log(0.1), 0.01))`.

    Note that the mean is not simply the mean of each component, it is the parameters of the `LogNormal` distribution
    that corresponds to the mean of the underlying `Normal` distributions in `log` (transformed) space.

    >>> upa.μ.mean()
    0.25
    >>> upa.σ2.mean()
    0.025
    >>> upa.mean()
    m['μ', 'σ2']: (0.22133638, 0.00625)

    You can verify the mean and variance returned by averaging over the random variable explicitly.

    >>> upa.mean().dist.mean()
    2.2202914201059437e-01
    >>> np.exp(upa.t.mean().dist.rvs(10000, random_state=2021).mean())
    2.2133371283050837e-01
    >>> upa.mean().dist.var()
    3.0907071428047016e-04
    >>> np.log(upa.mean().dist.rvs(10000, random_state=2021)).var()
    6.304628046829242e-03

    Calculate percentiles

    >>> upa.dist.ppf(0.025)
    array([0.08220152, 0.1515835 , 0.21364308, 0.27028359])
    >>> upa.dist.ppf(0.975)
    array([0.12165225, 0.26388097, 0.42126336, 0.59197082])

    Draw samples

    >>> upa.dist.rvs([3, *upa.shape], random_state=2021)
    array([[0.11605116, 0.22006429, 0.27902589, 0.34041327],
           [0.10571616, 0.1810085 , 0.36491077, 0.45507622],
           [0.10106982, 0.21230397, 0.3065239 , 0.33827997]])

    You can compose the variable with numpy functions, though you may get a warning if the operation is poorly defined
    for the distribution (which is most transforms on `LogNormal` distributions). Transformations are applied in
    transformed space.

    >>> (upa+1+np.tile(upa, (3,1))[2,3]).mean().t.dist.ppf(0.5)
    UserWarning: Transform is poorly defined for <ufunc 'log'>; results may be unexpected.
    -1.8423623672812148


    Attributes
    ----------
    name : str
        Name of variable.
    μ : array
        Mean at each point
    σ2 : array
        Variance at each point
    fields : list of str
        Names of each level held in the array
    dstdzr : DistributionStandardizer
        An instance  of :class:`DistributionStandardizer` created from the supplied :class:`Standardizer` object
    """

    def __new__(cls, name: str, μ: np.ndarray, σ2: np.ndarray, stdzr: Standardizer, lg10_Copies=None, stdzd=False):
        μ_ = np.asarray(μ)
        σ2_ = np.asarray(σ2)
        assert(μ_.shape == σ2_.shape)
        dstdzr = DistributionStandardizer(**stdzr)

        if stdzd:
            if lg10_Copies is not None:
                lg10_Copies = stdzr.unstdz('lg10_Copies', lg10_Copies)
            μ_, σ2_ = dstdzr.unstdz(name, μ_, σ2_, lg10_Copies)

        if lg10_Copies is None:
            uparray_dtype = np.dtype([('μ', μ_.dtype), ('σ2', σ2_.dtype)])
        else:
            lg10_Copies_ = np.asarray(lg10_Copies)
            uparray_dtype = np.dtype([('μ', μ_.dtype), ('σ2', σ2_.dtype), ('lg10_Copies', lg10_Copies_.dtype)])

        uparray_prototype = np.empty(μ_.shape, dtype=uparray_dtype)
        uparray_prototype['μ'] = μ_
        uparray_prototype['σ2'] = σ2_
        if lg10_Copies is not None:
            uparray_prototype['lg10_Copies'] = lg10_Copies

        uparray = uparray_prototype.view(cls)
        uparray.name = name
        uparray.dstdzr = dstdzr
        uparray.fields = list(uparray_dtype.fields.keys())

        return uparray

    def __array_finalize__(self, uparray):
        if uparray is None:
            return
        self.name = getattr(uparray, 'name', None)
        self.fields = getattr(uparray, 'fields', None)
        self.dstdzr = getattr(uparray, 'dstdzr', None)

    @property
    def z(self) -> UncertainArray:
        """Standardized values"""
        lg10_Copies = self['lg10_Copies'] if 'lg10_Copies' in self.dtype.names else None

        zmean, zvar = self.dstdzr.stdz(self.name, self.μ, self.σ2, lg10_Copies)
        if lg10_Copies is not None:
            lg10_Copies = self.dstdzr.stdz('lg10_Copies', lg10_Copies, np.nan)[0]

        return UncertainArray(f'{self.name}_z', zmean, zvar, lg10_Copies=lg10_Copies)

    @property
    def t(self) -> UncertainArray:
        """Transformed values"""
        lg10_Copies = self['lg10_Copies'] if 'lg10_Copies' in self.dtype.names else None

        tmean, tvar = self.dstdzr.transform(self.name, self.μ, self.σ2, lg10_Copies)
        if lg10_Copies is not None:
            lg10_Copies = self.dstdzr.transform('lg10_Copies', lg10_Copies, 0)[0]

        return UncertainArray(f'{self.name}_t', tmean, tvar, lg10_Copies=lg10_Copies)

    @property
    def _ftransform(self):
        return self.dstdzr.transforms.get(self.name, [skip, skip])[0]

    @property
    def _as_uncarray(self):
        return unp.uarray(self.z.μ, self.z.σ)

    def _from_uncarray(self, name, uncarray):
        z = UncertainArray._from_uncarray(name, uncarray)
        return self._from_z(z)

    @property
    def dist(self) -> rv_continuous:
        """Array of :func:`scipy.stats.rv_continuous` objects.

        If the transformation associated with the array's parameter is log/exp, this is a `lognorm` distribution object
        with ``scale=self.μ`` and ``s=self.t.σ``. Otherwise it is a `norm` distribution with ``loc=self.μ`` and
        ``scale=self.σ``. See the scipy documentation on `LogNormal`_ and `Normal`_ random variables for more
        explanation and a list of methods.

        .. _Normal: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
        .. _LogNormal: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
        """
        dists = {
            skip: super().dist,
            np.log: lognorm(scale=self.μ, s=self.t.σ)
        }
        return dists[self._ftransform]

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, **kwargs):
        self._warn_if_poorly_defined()
        kwargs |= dict(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        z = self.z.sum(**kwargs)
        return self._from_z(z)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, **kwargs):
        """The natural-space distribution parameters which represent the mean of the transformed-space distributions"""
        kwargs |= dict(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        z = self.z.mean(**kwargs)
        return self._from_z(z)

    def _from_z(self, z):
        name = z.name.replace('_z', '')
        fields = {dim: z[dim] for dim in z.fields}
        return UncertainParameterArray(name, **fields, stdzr=self.dstdzr, stdzd=True)

    def _from_t(self, t):
        name = t.name.replace('_t', '')
        t.μ, t.σ2 = self.dstdzr.untransform(name, t.μ, t.σ2)
        return UncertainParameterArray(name, **{dim: t[dim] for dim in t.fields}, stdzr=self.dstdzr,
                                       stdzd=False)

    def _similar_lg10_Copies(self, other):
        if not isinstance(other, UncertainParameterArray):
            return False

        have_copies = 'lg10_Copies' in self.fields
        has_copies = 'lg10_Copies' in other.fields

        if have_copies ^ has_copies:
            return False
        elif ~have_copies and ~has_copies:
            return True
        else:
            return self['lg10_Copies'] == other['lg10_Copies']

    def _warn_if_dissimilar(self, other):
        if isinstance(other, UncertainParameterArray):
            if not self._similar_lg10_Copies(other):
                warnings.warn('uparrays have dissimilar lg10_Copies')
            if not self.dstdzr == other.dstdzr:
                warnings.warn('uparrays have dissimilar DistributionStandardizers')

    def _warn_if_poorly_defined(self):
        if self._ftransform is not skip:
            warnings.warn(f'Transform is poorly defined for {self._ftransform}; results may be unexpected.')

    # def __repr__(self):
    #     return f'{self.name}{self.fields}: {np.asarray(self)}'

    def __getitem__(self, item):
        default = super(UncertainArray, self).__getitem__(item)
        if isinstance(item, int) or (isinstance(item, tuple) and all(isinstance(val, int) for val in item)):
            return UncertainParameterArray(self.name, stdzr=self.dstdzr, stdzd=False,
                                           **{name: value for name, value in zip(default.dtype.names, default)})
        return default.view(np.ndarray)

    def __add__(self, other):
        self._warn_if_dissimilar(other)
        self._warn_if_poorly_defined()
        if isinstance(other, UncertainParameterArray):
            new = self._from_t(self.t.__add__(other.t))
            new.dstdzr = DistributionStandardizer(**(self.dstdzr | other.dstdzr))
        else:
            new = super().__add__(other)
        return new

    def __sub__(self, other):
        self._warn_if_dissimilar(other)
        self._warn_if_poorly_defined()
        if isinstance(other, UncertainParameterArray):
            new = self._from_t(self.t.__sub__(other.t))
            new.dstdzr = DistributionStandardizer(**(self.dstdzr | other.dstdzr))
        else:
            new = super().__sub__(other)
        return new

    def __rsub__(self, other):
        self._warn_if_dissimilar(other)
        self._warn_if_poorly_defined()
        if isinstance(other, UncertainParameterArray):
            new = self._from_t(self.t.__rsub__(other.t))
            new.dstdzr = DistributionStandardizer(**(other.dstdzr | self.dstdzr))
        else:
            new = super().__rsub__(other)
        return new


class MVUncertainParameterArray(np.ndarray):
    r"""Structured array of multiple parameter means and variances along with correlations.

    See Also
    --------
    `multivariate_normal <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html>`_: \
    scipy `Multivariate Normal` random variable

    :class:`ParameterArray`

    :class:`UncertainParameterArray`

    Parameters
    ----------
    \*uparrays: UncertainParameterArray
        UParrays of identical shape containing the marginal mean and variance of each parameter
    stdzr : Standardizer
        An instance  of :class:`Standardizer`, converted internally to :class:`DistributionStandardizer`
    stdzd : bool, default False
        Whether the supplied values are on standardized scale instead of the natural scale

    Examples
    --------
    Create an MVUParray from two UParrays with negative correlation

    >>> import numpy as np
    >>> from candas.learn import uparray, mvuparray, Standardizer
    >>>
    >>> stdzr = Standardizer.default()
    >>> m_upa = uparray('m', np.arange(1,5)/10, np.arange(1,5)/100, stdzr)
    >>> r_upa = uparray('r', np.arange(1,5)/10+0.5, np.arange(1,5)/100*2, stdzr)
    >>> cor = np.array([[1, -0.6], [-0.6, 1]])
    >>> cor
    array([[ 1. , -0.6],
           [-0.6,  1. ]])
    >>> mvup = mvuparray(m_upa, r_upa, cor=cor)

    The MVUParray is displayed as an array of tuples ((μ_a, μ_b), (σ2_a, σ2_b))

    >>> mvup
    ('m', 'r')['μ', 'σ2']: [((0.1, 0.6), (0.01, 0.02)) ((0.2, 0.7), (0.02, 0.04))
                            ((0.3, 0.8), (0.03, 0.06)) ((0.4, 0.9), (0.04, 0.08))]

    The marginal means or variances can be extracted as ParameterArrays:

    >>> mvup.μ
    ('m', 'r'): [(0.1, 0.6) (0.2, 0.7) (0.3, 0.8) (0.4, 0.9)]

    The component UncertainParameterArrays can be extracted with the :meth:`get` method:

    >>> mvup.get('r')
    r['μ', 'σ2']: [(0.6, 0.02) (0.7, 0.04) (0.8, 0.06) (0.9, 0.08)]

    Slicing and indexing works as normal:

    >>> mvup[::2]
    ('m', 'r'): [((0.1, 0.6), (0.01, 0.02)) ((0.3, 0.8), (0.03, 0.06))]

    Transformed and standardized distributions can be obtained as with UncertainParameterArray

    >>> mvup.t
    ('m_t', 'r_t')['μ', 'σ2']: [((-2.30258509, -0.51082562), (0.01, 0.02))
                                ((-1.60943791, -0.35667494), (0.02, 0.04))
                                ((-1.2039728 , -0.22314355), (0.03, 0.06))
                                ((-0.91629073, -0.10536052), (0.04, 0.08))]
    >>> mvup.z
    ('m_z', 'r_z')['μ', 'σ2']: [((5.15019743, -1.29003559), (0.02952256, 0.80115366))
                                ((6.34117197, -0.31439838), (0.05904512, 1.60230732))
                                ((7.03784742,  0.53073702), (0.08856768, 2.40346098))
                                ((7.53214651,  1.27619927), (0.11809024, 3.20461465))]

    For 0-d MVUParrays (or individual elements of larger arrays), the :meth:`dist` exposes the scipy
    `multivariate_normal` object. Because this distribution is defined in standardized space (by default, `stdzd=True`)
    or transformed spaced (`stdzd=False`), ParameterArrays may be the most convenient way to pass arguments to the
    distribution methods:

    >>> pa = mvup.parray(m=0.09, r=0.61)
    >>> mvup[0].dist().cdf(pa.z.values())
    0.023900979112885523
    >>> mvup[0].dist(stdzd=False).cdf(pa.t.values())
    0.023900979112885523

    Perhaps most importantly, the :meth:`rvs` allows drawing correlated samples from the joint distribution, returned as
    a ParameterArray:

    >>> mvup[0].rvs(10, random_state=2021)
    ('m', 'r'): [(0.0962634 , 0.74183363) (0.09627651, 0.56437764)
                 (0.09140986, 0.64790721) (0.09816149, 0.70518567)
                 (0.10271404, 0.60974628) (0.09288982, 0.60933939)
                 (0.0983131 , 0.63588871) (0.12262933, 0.45941758)
                 (0.1070759 , 0.4918009 ) (0.11118635, 0.49708401)]
    >>> mvup[0].rvs(1, random_state=2021).z
    ('m_z', 'r_z'): (5.08476443, 0.05297293)

   Allowing us to easily visualize the joint distribution:

    >>> import pandas as pd
    >>> import seaborn as sns
    >>> sns.jointplot(x='m', y='r', data=pd.DataFrame(mvup[0].rvs(1000).to_dict()), kind='kde')
    """

    def __new__(cls, *uparrays, cor, stdzr=None):

        shape = uparrays[0].shape
        assert all([upa.shape == shape for upa in uparrays])
        assert cor.shape[0] == len(uparrays)
        stdzr = uparrays[0].dstdzr.stdzr if stdzr is None else stdzr
        dstdzr = DistributionStandardizer(**stdzr)

        μ_ = ParameterArray(**{upa.name: upa.μ for upa in uparrays}, stdzr=stdzr)
        σ2_ = ParameterArray(**{upa.name: upa.σ2 for upa in uparrays}, stdzr=stdzr)

        lg10_Copies = [upa['lg10_Copies'] for upa in uparrays if 'lg10_Copies' in upa.fields]
        lg10_Copies = lg10_Copies[0] if lg10_Copies != [] else None

        if lg10_Copies is None:
            mvuparray_dtype = np.dtype([('μ', μ_.dtype), ('σ2', σ2_.dtype)])
        else:
            lg10_Copies_ = np.asarray(lg10_Copies)
            mvuparray_dtype = np.dtype([('μ', μ_.dtype), ('σ2', σ2_.dtype), ('lg10_Copies', lg10_Copies_.dtype)])

        mvuparray_prototype = np.empty(shape, dtype=mvuparray_dtype)
        mvuparray_prototype['μ'] = μ_
        mvuparray_prototype['σ2'] = σ2_
        if lg10_Copies is not None:
            mvuparray_prototype['lg10_Copies'] = lg10_Copies

        mvuparray = mvuparray_prototype.view(cls)
        mvuparray.names = [upa.name for upa in uparrays]
        mvuparray.stdzr = stdzr
        mvuparray.dstdzr = dstdzr
        mvuparray.fields = list(mvuparray_dtype.fields.keys())
        mvuparray.cor = cor

        return mvuparray

    def __array_finalize__(self, mvup):
        if mvup is None:
            return
        self.names = getattr(mvup, 'names', None)
        self.fields = getattr(mvup, 'fields', None)
        self.stdzr = getattr(mvup, 'stdzr', None)
        self.dstdzr = getattr(mvup, 'dstdzr', None)
        self.cor = getattr(mvup, 'cor', None)

    def __repr__(self):
        return f'{tuple(self.names)}{self.fields}: {np.asarray(self)}'

    def __getitem__(self, item):
        default = super().__getitem__(item)
        if isinstance(item, int) or (isinstance(item, tuple) and all(isinstance(val, int) for val in item)):
            return self.mvuparray(*[self.get(name)[item] for name in self.names])
        elif item == 'lg10_Copies':
            return default.view(np.ndarray).astype(float)
        else:
            return default.view(ParameterArray)

    def get(self, name, default=None) -> UncertainParameterArray | MVUncertainParameterArray:
        """Return one component parameter as an UncertainParameterArray or a subset as an MVUncertainParameterArray"""
        if isinstance(name, str):
            if name in self.names:
                lg10_Copies = self['lg10_Copies'] if 'lg10_Copies' in self.fields and name == 'τ' else None
                return self.uparray(name, self['μ'][name].values(), self['σ2'][name].values(), lg10_Copies=lg10_Copies)
            else:
                return default
        elif isinstance(name, list):
            idxs = [self.names.index(n) for n in name]
            return self.mvuparray([self.get(n) for n in name], cor=self.cor[idxs, :][:, idxs])

    @property
    def μ(self) -> ParameterArray:
        """Means"""
        return self['μ']

    @μ.setter
    def μ(self, val):
        self['μ'] = val

    @property
    def σ2(self) -> ParameterArray:
        """Variances"""
        return self['σ2']

    @σ2.setter
    def σ2(self, val):
        self['σ2'] = val

    @property
    def σ(self) -> ParameterArray:
        """Standard deviations"""
        return self.parray(**{k: np.sqrt(v) for k, v in self['σ2'].to_dict().items()})

    @property
    def z(self) -> MVUncertainParameterArray:
        """Standardized values"""
        return self.mvuparray(*[self.get(name).z for name in self.names])

    @property
    def t(self) -> MVUncertainParameterArray:
        """Transformed values"""
        return self.mvuparray(*[self.get(name).t for name in self.names])

    def parray(self, *args, **kwargs) -> ParameterArray:
        """Create a ParameterArray using this instance's Standardizer"""
        kwargs.setdefault('stdzr', self.dstdzr.stdzr)
        return ParameterArray(*args, **kwargs)

    def uparray(self, *args, **kwargs) -> UncertainParameterArray:
        """Create an UncertainParameterArray using this instance's Standardizer"""
        kwargs.setdefault('stdzr', self.dstdzr.stdzr)
        return UncertainParameterArray(*args, **kwargs)

    def mvuparray(self, *args, **kwargs) -> MVUncertainParameterArray:
        """Create an MVUncertainParameterArray using this instance's Standardizer"""
        kwargs.setdefault('stdzr', self.dstdzr.stdzr)
        kwargs.setdefault('cor', self.cor)
        return MVUncertainParameterArray(*args, **kwargs)

    def cov(self, stdzd=True):
        """Covariance matrix (only supported for 0-D MVUParrays)"""
        if self.ndim != 0:
            raise NotImplementedError('Multidimensional multivariate covariance calculations are not yet supported.')

        σ = self.z.σ.values() if stdzd else self.t.σ.values()

        return np.diag(σ) @ self.cor @ np.diag(σ)

    def dist(self, stdzd=True) -> rv_continuous:
        """Scipy :func:`multivariate_normal` object (only supported for 0-D MVUParrays)"""
        if self.ndim != 0:
            raise NotImplementedError('Multidimensional multivariate distributions are not yet supported.')
        if stdzd:
            μ = np.array([self.get(p).z.μ for p in self.names])
        else:
            μ = np.array([self.get(p).t.μ for p in self.names])
        cov = self.cov(stdzd=stdzd)
        return multivariate_normal(μ, cov)

    def rvs(self, *args, **kwargs) -> ParameterArray:
        """Generate random samples from the multivariate distribution."""
        # multivariate_normal throws a RuntimeWarning: covariance is not positive semidefinite if rvs() is called more
        # than once without calling dist().rvs() in between. Covariance isn't being altered as a side-effect of this
        # call, so this issue really makes no sense.....
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                samples = self.dist(stdzd=True).rvs(*args, **kwargs)
            except:
                self.dist().rvs()
                samples = self.dist(stdzd=True).rvs(*args, **kwargs)
            lg10_Copies = self.z['lg10_Copies'] if 'lg10_Copies' in self.fields else None
        return self.parray(**{p: samples[..., i] for i, p in enumerate(self.names)}, lg10_Copies=lg10_Copies, stdzd=True)

    def mahalanobis(self, parray: ParameterArray) -> float:
        """Calculates the Mahalanobis distance between the MVUParray distribution and a (ParameterArray) point."""
        cov_inv = np.linalg.inv(self.cov(stdzd=True))
        points = np.stack([parray.z.get(p+'_z').values() for p in self.names])
        μ = np.stack([self.z.μ.get(p+'_z').values() for p in self.names])
        diff = points-μ
        return np.sqrt(diff.T @ cov_inv @ diff)

    def outlier_pval(self, parray: ParameterArray) -> float:
        """Calculates the p-value that a given (ParameterArray) point is an outlier from the MVUParray distribution."""
        MD = self.mahalanobis(parray)
        n_params = len(self.names)
        pval = 1-chi2.cdf(MD**2, df=n_params)
        return pval


# TODO: Add ParameterSet save and load methods
class ParameterSet:
    """Container for parameter estimates that enforces data integrity.

    :class:`ParameterSet` is a container for a tidy dataframe (:attr:`data`) and a :class:`Standardizer`, allowing
    simple access to standardized data (:attr:`zdata`) and wide-form views of the data (:attr:`wide`/:attr:`zwide`).
    Ensures data integrity by enforcing a set of :attr:`required_columns` and a set of :attr:`required_parameters`.

    Notes
    -----
    :class:`ParameterSet` objects are created by the :meth:`VIResult.summarize` and :meth:`MCMCResult.summarize`
    methods, and forms the basis of the :class:`GP` class.

    Parameters
    ----------
    data: pd.DataFrame

    Attributes
    ----------

    """

    # TODO: ParameterSet: Make required_parameters and required_descriptors optional definition at init
    # TODO: Allow specification of stdz-able columns at init.
    required_columns = ['Target', 'BP', 'GC', 'lg10_Copies', 'Parameter', 'Metric', 'Value']
    required_parameters = ['τ', 'F0_lg', 'ρ', 'r', 'K', 'm']

    def __init__(self, data: pd.DataFrame):
        self.validate(data)
        self._data = data
        self._stdzr = Standardizer.from_DataFrame(data)

    # def _repr_html_(self):
    #     return self.data._repr_html_()

    @property
    def stdzr(self):
        """Standardizer: Container for dict of mean (μ) and standard deviation (σ) for every parameter."""
        return self._stdzr

    @stdzr.setter
    def stdzr(self, new_stdzr: Standardizer):
        assert isinstance(new_stdzr, Standardizer)
        self._stdzr = new_stdzr

    @property
    def data(self) -> pd.DataFrame:
        """pandas.DataFrame: Underlying dataframe"""
        return self._data

    @data.setter
    def data(self, df: pd.DataFrame):
        """Ensure new dataframe passes integrity checks"""
        self.validate(df)
        self._data = df
        self._stdzr = Standardizer.from_DataFrame(df)

    @property
    def wide(self) -> pd.DataFrame:
        """Wide-form copy of data"""
        idx_columns = [col for col in self.data.columns if col not in ['Parameter', 'Value']]
        return (self.data
                .pivot(index=idx_columns, columns='Parameter', values='Value')
                .reset_index()
                .rename_axis(columns=None)
                )

    @property
    def zdata(self) -> pd.DataFrame:
        """Long-form copy of standardized data"""
        return self.standardized

    @property
    def zwide(self) -> pd.DataFrame:
        """Wide-form copy of standardized data"""
        idx_columns = [col for col in self.zdata.columns if col not in ['Parameter', 'Value']]
        return (self.zdata
                .pivot(index=idx_columns, columns='Parameter', values='Value')
                .reset_index()
                .rename_axis(columns=None)
                )

    @property
    def standardized(self):
        """A copy of the instance's dataframe  with key parameters transformed and standardized.

        In addition to values in the ``Value`` column corresponding to the keys in :attr:`stdzr`, columns
        ``'BP'``, ``'GC'``, and ``'lg10_Copies'`` are also manipulated.
        """
        df_ = self.data.copy()
        df_['Value'] = (df_
            .groupby('Parameter')
            .apply(self.stdzr.stdz_series)
            .reset_index()
            .set_index('level_1')
            .sort_index()[0])
        for col in ['BP', 'GC', 'lg10_Copies']:
            df_[col] = (df_[col].map(lambda x: self.stdzr.stdz(col, x)))
        return df_

    @classmethod
    def validate(cls, df: pd.DataFrame):
        """Ensures provided DataFrame has all required attributes"""
        assert isinstance(df, pd.DataFrame)
        assert_is_subset('Columns', cls.required_columns, df.columns)
        assert_is_subset('Parameters', cls.required_parameters, df.Parameter.unique())
        assert 'mean' in df['Metric'].unique(), '"Metric" column must contain value "mean"'

    @property
    def valid(self):
        """Integrity check"""
        return self.validate(self.data) is None

    @classmethod
    def read_pickle(cls, *args, **kwargs):
        """Imports from pickle file

        Returns
        -------
        ParameterSet
        """
        df = pd.read_pickle(*args, **kwargs)
        return cls(df)

    @classmethod
    def read_csv(cls, *args, **kwargs):
        """Imports from comma delimited file

        Returns
        -------
        ParameterSet
        """
        df = pd.read_csv(*args, **kwargs)
        return cls(df)

    @classmethod
    def read_table(cls, *args, **kwargs):
        """Imports from generic delimited file

        Returns
        -------
        ParameterSet
        """
        df = pd.read_table(*args, **kwargs)
        return cls(df)

    def neaten(self):
        """Rearranges columns in a sensible order"""
        other_columns = [col for col in self.data.columns if col not in self.required_columns]
        self.data = self.data[other_columns + self.required_columns]

    @classmethod
    def from_wide(cls, wide, params=None):
        """Reshapes wide-form data to long-form, then instantiates class"""
        params = cls.required_parameters if params is None else params  # Might be broken
        meta = [col for col in wide.columns if col not in params]
        tidy = wide.melt(id_vars=meta, value_vars=params, var_name='Parameter', value_name='Value')
        return cls(tidy)

    def save(self, filename: str):
        """Pickles data in wide-form to save space"""
        self.wide.to_pickle(filename)

    @classmethod
    def load(cls, filename: str, params=None):
        """Un-pickles wide-form data, then reshapes as long-form"""
        wide = pd.read_pickle(filename)
        return cls.from_wide(wide, params)
