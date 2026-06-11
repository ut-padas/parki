"""
Distributed APIs
================

.. currentmodule:: parkipy.distributed

ParkiPy modules for high-performance distributed computing.
Distributed APIs are called from the host code and are executed on
multi-node systems.

Overview
--------
The distributed APIs behave similarly to their single-device counterparts,
with the exception that input arrays are initialised locally on each device
and programs are launched with ``mpiexec`` or an equivalent MPI launcher to
start multiple processes. Each rank owns a contiguous slab of the particle
domain in the first periodic direction.

.. warning::
   The distributed package currently only supports the ``'Cuda'`` execution
   space for device calls.

Contents
--------

- :func:`gather_points` — collect distributed particle arrays onto rank 0
- :ref:`parkipy.distributed.ewald <routines.distributed.ewald>` — distributed Ewald summation routines

"""

try:
    import mpi4py
except ImportError as e:
    raise ImportError("parkipy's distributed module requires mpi4py") from e


from ._utils import gather_points

__all__ = ["gather_points"]


def __getattr__(attr):
    match attr:
        case "ewald":
            import parkipy.distributed.ewald as ewald

            return ewald
        case _:
            raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")
