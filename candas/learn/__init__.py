from .aggregation import Standardizer, DistributionStandardizer, Parameter, ParameterSet
from .aggregation import LayeredArray, ParameterArray, UncertainArray, UncertainParameterArray, MVUncertainParameterArray
from .aggregation import ParameterArray as parray
from .aggregation import UncertainParameterArray as uparray
from .regression_gpflow import GP_gpflow, LVMOGP_GP

# Aliases
parray = ParameterArray
uarray = UncertainArray
uparray = UncertainParameterArray
mvuparray = MVUncertainParameterArray