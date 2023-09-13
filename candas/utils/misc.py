"""Internal utility functions, not intended for public consumption"""
from functools import wraps
import numpy as np

from typing import Iterable


def skip(x):
    """Returns input unmodified"""
    return x


def NotImplementedWrapper(func):
    """Blocks execution and raises an error"""
    @wraps(func)
    def block(*args, **kwargs):
        """Raises error"""
        raise NotImplementedError
    return block


def assert_in(name: str, arg, lst: Iterable):
    """Raises error if value not in list"""
    if arg not in lst:
        raise ValueError(f'{name} must be one of {lst}')


def assert_is_subset(name: str, subset: Iterable, superset: Iterable):
    """Raises error if any required value not in list"""
    l_set = set(superset)
    r_set = set(subset)
    if not l_set.issuperset(r_set):
        missing = list(r_set.difference(l_set))
        msg = f'{list_is_are(missing)} missing from {name}'
        raise ValueError(msg)


def assert_one(names: str, lst: Iterable):
    """Raises error if none or multiple of values in `lst` are not None"""
    if sum(el is not None for el in lst) != 1:
        raise ValueError(f'Exactly one of {names} must be supplied')


def list_is_are(lst: list) -> str or None:
    """String formats list to be grammatically correct"""
    if len(lst) == 0:
        return None
    elif len(lst) == 1:
        msg = f'{list_and(lst)} is'
    else:
        msg = f'{list_and(lst)} are'
    return msg


def list_and(lst: list) -> str:
    if len(lst) == 1:
        msg = f'{lst[0]}'
    else:
        msg = f'{", ".join(lst[:-1])} and {lst[-1]}'
    return msg


class Trigger(object):
    """Descriptor to create a property that calls an instance method when its value is changed.
    Note, must be specified as a class variable.

    Parameters
    ----------
    method: str
        The name of the instance method to be called
    default: optional
        Default value of variable
    """

    def __init__(self, method, default=None):
        self.default = default
        self.method = method
        self.data = dict()

    def __get__(self, instance, owner):
        return self.data.get(instance, self.default)

    def __set__(self, instance, value):
        self.data[instance] = value
        getattr(instance, self.method)()


class InstanceCopy:
    """A class that creates instances by copying all attributes of a parent instance"""

    def __init__(self, parent):
        assert isinstance(parent, self.__class__.__bases__[-1])

        for attr in parent.__dict__:
            setattr(self, attr, getattr(parent, attr))


def calc_tau(rate, init, norm):
    return -np.log(init / norm) / (rate * np.log(2))


def calc_rho(rate, init, norm):
    return -np.log(rate * np.log(2)) / np.log(calc_tau(rate, init, norm))


def calc_rate(τ, ρ):
    return τ ** (-ρ) / np.log(2)


def calc_init(τ, ρ, norm):
    return norm * np.exp(-τ ** (1 - ρ))