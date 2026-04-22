"""
Spectral Ewald Summation
========================

.. currentmodule:: parkipy.ewald

Overview
--------
The Ewald summation module :class:`parkipy.ewald` uses a Kokkos backend
to provide fast summation APIs for the combined
Stokes single and double-layer [1]_ and Laplace [2]_ kernels. The Ewald
method is extended to free directions with ideas from [3]_.

Automatic parameter selection is facilitated via the :class:`EwaldOptions`
class, where the user selects computational parameters and has the ability
to select different Ewald methods and threading configurations.

Ewald kernel are easy to call, e.g., if you are looking to compute the combined Stokes
single and double-layer potential in a fully periodic box on a GPU, you can call:

>>> import parkipy
>>> import cupy as cp
>>> nt = 43892
>>> ns = 876942
>>> # Particle Initialization
>>> trg = cp.random.rand(3, nt)
>>> src = cp.random.rand(3, ns)
>>> dsl = cp.random.rand(3, ns)
>>> ddl = cp.random.rand(3, ns)
>>> dns = cp.vstack((dsl, ddl))
>>> nrm = cp.random.rand(3, ns)
>>> # Define Ewald Options
>>> options = parkipy.ewald.EwaldOptions(box=[1,1,1], periodicity=3, tolerance=1e-5, execution_space="Cuda")
>>> potential = parkipy.ewald.stokes_comb(trg, src, dns, nrm, options)

Of course, detailed documentation of the Ewald kernels and options are given below.

Ewald Kernel Gallery
____________________

.. autosummary::
   :toctree: generated/

   stokes_sl
   stokes_comb
   laplace


Ewald Kernel Support
--------------------
.. autosummary::
    :toctree: generated/
    :template: dataclass.rst

    EwaldOptions
    SEParams
    PerfModel

References
__________

.. [1] Bagge, J., & Tornberg, A.-K. (2023).
        Fast Ewald summation for Stokes flow with arbitrary periodicity.
        Journal of Computational Physics, 493, 112473.
        https://doi.org/10.1016/j.jcp.2023.112473
.. [2] Shamshirgar, D. S., Bagge, J., & Tornberg, A.-K. (2021).
        Fast Ewald summation for electrostatic potentials with arbitrary periodicity.
        The Journal of Chemical Physics, 154(16), 164109.
        https://doi.org/10.1063/5.0044895
.. [3] Vico, F., Greengard, L., & Ferrando, M. (2016).
    Fast convolution with free-space Green’s functions.
    Journal of Computational Physics, 323, 191–203.
    https://doi.org/10.1016/j.jcp.2016.07.028
"""

from ._kernels import (
    EwaldOptions,
    stokes_sl,
    stokes_comb,
    laplace,
)

from ._params import SEParams
from ._perf import PerfModel

__all__ = ["stokes_comb", "laplace"]
