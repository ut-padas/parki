"""
Distributed Spectral Ewald Summation
====================================

.. currentmodule:: parkipy.distributed.ewald

Overview
--------
The distributed Ewald summation module :mod:`parkipy.distributed.ewald` uses a
Kokkos backend and MPI to provide fast, multi-node summation APIs for the Stokes
single and combined double-layer kernels [1]_.

Automatic parameter selection and distributed configurations are facilitated via the
:class:`DistributedEwaldOptions` class. This allows the user to easily configure
the execution space, Ewald tolerance, and particle scattering across MPI ranks.

.. warning::
   The distributed Ewald module currently has the following limitations:

   - Only the ``'Cuda'`` execution space is supported.
   - Only one periodic direction (``periodicity=1``) is supported.
   - The periodic box length must be divisible by the number of MPI ranks.

Distributed Ewald kernels are designed to be evaluated across ranks. For example,
to compute the Stokes single-layer potential across distributed GPUs, you can call:

>>> from mpi4py import MPI
>>> import parkipy
>>> mpi_comm = MPI.COMM_WORLD
>>> size = mpi_comm.Get_size()
>>> am = parkipy.utils.get_array_module(parkipy.utils.get_execution_space("Cuda"))
>>> # Define Box and Particles for this Rank
>>> box = [size, 1, 1]
>>> ns = nt = 10000
>>> src = am.random.rand(3, ns) * am.array(box).reshape(3, 1)
>>> trg = am.random.rand(3, nt) * am.array(box).reshape(3, 1)
>>> dens = am.random.randn(3, ns)
>>> # Define Distributed Ewald Options
>>> options = parkipy.distributed.ewald.DistributedEwaldOptions(
...     box=box, periodicity=1, tolerance=1e-4, execution_space="Cuda", cell_size=224
... )
>>> pot, trg_out = parkipy.distributed.ewald.stokes_sl(trg, src, dens, options)

Detailed documentation of the distributed Ewald kernels and options are given below.

Distributed Ewald Kernel Gallery
--------------------------------

.. autosummary::
   :toctree: generated/

   stokes_sl
   stokes_comb

Distributed Ewald Kernel Support
--------------------------------
.. autosummary::
    :toctree: generated/
    :template: dataclass.rst

    DistributedEwaldOptions

References
----------

.. [1] Bagge, J., & Tornberg, A.-K. (2023).
        Fast Ewald summation for Stokes flow with arbitrary periodicity.
        Journal of Computational Physics, 493, 112473.
        https://doi.org/10.1016/j.jcp.2023.112473
"""

from ._kernels import (
    DistributedEwaldOptions,
    stokes_sl,
    stokes_comb,
)

__all__ = ["stokes_sl", "stokes_comb", "DistributedEwaldOptions"]
