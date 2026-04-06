"""
Utilities
=========

General utilities used by Parkipy and
parikpy's users.

Helpers
-------
.. autosummary::
    :toctree: generated/

    get_dtype
    get_array_module
    get_execution_space

"""

import numpy as np
import pykokkos as pk
from typing import Union, Literal
import types
import math


def round_up(n: int, factor: int) -> int:
    """
    Round the number `n` to the nearest multiple
    of `factor`. That is, return `m>=n` such that
    `m % factor == 0`.

    Parameters
    ----------
    n: int
        number to round
    factor: int
        round `n` up to the nearest multiple of `factor`

    Returns
    -------
    int
    """
    return factor * math.ceil(n / factor)


def get_dtype(dtype: Union[Literal["fp32", "fp64"], np.dtype]) -> np.dtype:
    """
    Return a :class:`numpy.dtype` given a precision
    specification.

    Parameters
    ----------
    dtype: {"fp32", "fp64"} | numpy.dtype
        Data type specification.

    Returns
    -------
    :class:`numpy.dtype`
    """
    if dtype == "fp32" or dtype == np.float32:
        return np.float32
    elif dtype == "fp64" or dtype == np.float64:
        return np.float64
    else:
        raise ValueError(
            "only `np.float32` and np.float64` dtypes supported," " got {dtype}."
        )


def get_execution_space(
    execution_space: Union[
        None, Literal["Cuda", "HIP", "OpenMP"], pk.ExecutionSpace
    ] = None,
) -> pk.ExecutionSpace:
    """
    Return a Kokkos :class:`pk.ExecutionSpace`
    object given am execution space specification.

    Parameters
    ----------
    execution_space: None | {"GPU", "Cuda", "HIP", "OpenMP"} | pk.ExecutionSpace
        Kokkos execution space specification. If ``None``, return the Kokkos
        default execution space. The default is ``None``. If "GPU",
        choose one of "Cuda" or "HIP", if available.

    Returns
    -------
        :class:`pykokkos.ExecutionSpace`
    """
    valid_spaces = ("GPU", "Cuda", "HIP", "OpenMP", "CPU")
    if execution_space == None:
        return pk.get_default_space()
    else:
        if isinstance(execution_space, str):
            match execution_space.upper():
                case "GPU":
                    execution_space = pk.kokkos_manager.get_gpu_framework()
                case "CUDA":
                    execution_space = "Cuda"
                case "HIP":
                    execution_space = "HIP"
                case "OPENMP" | "CPU":
                    execution_space = "OpenMP"
                case "HOST":
                    execution_space = pk.get_default_space()
                    if not pk.interface.is_host_execution_space(execution_space):
                        raise ValueError(
                            f"PyKokkos default execution space {execution_space} is not"
                            " a host execution space, please specify one of"
                            " 'OpenMP', 'Threads', 'Serial' host execution spaces."
                        )
                case _:
                    raise ValueError(
                        f"execution space must be one of {valid_spaces}, got '{execution_space}'."
                    )
            return pk.ExecutionSpace(execution_space)
        elif isinstance(execution_space, pk.ExecutionSpace):
            return execution_space
        else:
            raise TypeError(f"""
                    `execution_space` expected to be of type
                    `str` or `pk.ExecutionSpace`, but is 
                    of type {type(execution_space)}
                    """)


def get_array_module(
    execution_space: Union[
        None, Literal["Cuda", "HIP", "OpenMP"], pk.ExecutionSpace
    ] = None,
) -> types.ModuleType:
    """
    Return the Numpy or Cupy array module used
    for a specific execution space.

    Parameters
    ----------
    execution_space: None | {"Cuda", "HIP", "OpenMP"} | pk.ExecutionSpace
        Kokkos execution space specification. If ``None``, return the Kokkos
        default execution space. The default is ``None``.

    Returns
    -------
    :class:`typing.ModuleType`


    Examples
    --------
    >>> import parkipy
    >>> am = parkipy.utils.get_array_module("Cuda")
    >>> am
    ... <module 'cupy' from '/opt/homebrew/lib/python3.13/site-packages/cupy/__init__.py'>
    >>> array = am.random.rand(3, 100000)

    """
    execution_space = get_execution_space(execution_space)
    if not pk.is_host_execution_space(execution_space):
        import cupy

        return cupy
    else:
        import numpy

        return numpy
