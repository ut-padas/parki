"""
Distributed APIs
================

.. currentmodule:: parkipy.distributed

ParkiPy modules for high-performance distributed computing.
Distributed APIs are called from the host code and are executed on
multi-node systems.

Overview
---------
The distributed APIs behave similarly to the single node counterparts,
with the exception that input arrays are initialized locally on each device
and programs are called with ``mpirun`` or similar specifications
to launch multiple processes.

========
Contents
========

.. toctree::
   :maxdepth: 2

   Spectral Ewald Summation <routines.distributed.ewald.rst>

.. warning:: Currently, the distributed package only supports the `'Cuda'` execution space for device calls.

"""

try:
    import mpi4py
except ImportError as e:
    raise ImportError("parkipy's distributed module requires mpi4py") from e


from ._utils import gather_points


def __getattr__(attr):
    match attr:
        case "ewald":
            import parkipy.distributed.ewald as ewald

            return ewald
        case _:
            raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")
