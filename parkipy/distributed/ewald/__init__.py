"""
Distributed Spectral Ewald Summation
====================================

.. currentmodule:: parkipy.distributed.ewald

Overview
--------
The distributed Ewald summation module :class:`parkipy.distributed.ewald` uses a Kokkos
backend to prodive distributed APIs for the :class:`parkipy.ewald` kernels.

.. warning::
    The Stokes combined single and double-layer potential is the **only currently supported distributed Ewald API**. Furthermore, the API is only supports a single perodic direction, assumes the periodic length of the computational box is divisible by the number of procssors. 

"""

from ._kernels import (
    stokes_comb,
    DistributedEwaldOptions,
)
