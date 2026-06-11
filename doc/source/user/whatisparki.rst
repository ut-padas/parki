.. _whatisparki:

****************
What is ParkiPy?
****************

The **(Par)ticle (K)ernel (I)nteractions** library for **(Py)thon** provides
performance-portable, parallel APIs on both CPUs and GPUs. 
ParkiPy leverages Kokkos [1]_ through its Python framework PyKokkos [2]_
to expose high-performance kernels behind a clean, NumPy-style interface.

ParkiPy is designed for simulations that require fast evaluation of pairwise
particle interactions---particularly problems in computational fluid dynamics
and potential theory that involve large numbers of particles under periodic
boundary conditions.


Key Features
------------

- **Ewald summation** for the Stokes and Laplace kernels in 1- and
  3-periodic domains via the :mod:`parkipy.ewald` module.
- **Distributed Ewald summation** in slab geometries across multiple GPUs
  via the :mod:`parkipy.distributed.ewald` module.
- **Cell list construction** for efficient local particle-interaction lookups
  via the :class:`parkipy.CellList` class.
- **Seamless NumPy/CuPy support**: pass :mod:`numpy` arrays for CPU execution
  or :mod:`cupy` arrays for GPU execution — the API is identical.
- **Just-in-time compiled kernels**: PyKokkos kernels are compiled on first
  use and cached automatically, so subsequent calls incur no recompilation
  overhead.


Supported Kernels
-----------------

The table below summarises which kernel and periodicity combinations are
currently available.

.. list-table::
   :header-rows: 1
   :widths: 20 30 35 15

   * - Periodicity
     - Stokes single layer
     - Stokes single + double layer
     - Laplace
   * - 0-periodic
     - ✗
     - ✗
     - ✗
   * - 1-periodic
     - ✓
     - ✓
     - ✗
   * - 2-periodic
     - ✗
     - ✗
     - ✗
   * - 3-periodic
     - ✓
     - ✓
     - ✓


How It Works
------------

ParkiPy splits the evaluation of a kernel sum into two parts following the
classical Ewald decomposition:

* **Near-field (P2P)**: direct particle-to-particle interactions within a
  cutoff radius, evaluated using a cell list for O(N) neighbour finding.
* **Far-field (G2P/P2G)**: long-range interactions handled in Fourier space
  via non-uniform FFTs on a regular grid.

This decomposition enables both accuracy control (via the ``tolerance``
parameter) and performance scaling to large particle counts on modern GPU
hardware.


Array Framework Compatibility
------------------------------

ParkiPy works with both CPU and GPU array frameworks through the same API:

.. code-block:: python

   import numpy as np      # CPU
   import cupy  as cp      # GPU

   import parkipy

   # The ewald.stokes_sl call accepts arrays from either framework.
   u_cpu = parkipy.ewald.stokes_sl(x_np, y_np, f_np, options)
   u_gpu = parkipy.ewald.stokes_sl(x_cp, y_cp, f_cp, options)

The execution space (``"OpenMP"`` or ``"CUDA"``) is specified once in the
:class:`~parkipy.ewald.EwaldOptions` object and controls which Kokkos backend
is used.


References
----------

.. [1] C. R. Trott et al., "Kokkos 3: Programming Model Extensions for the
       Exascale Era," in *IEEE Transactions on Parallel and Distributed
       Systems*, vol. 33, no. 4, pp. 805–817, 1 April 2022,
       https://doi.org/10.1109/TPDS.2021.3097283.

.. [2] Nader Al Awar, Steven Zhu, George Biros, and Milos Gligoric, 2021,
       "A performance portability framework for Python," in *Proceedings of
       the 35th ACM International Conference on Supercomputing (ICS '21)*,
       Association for Computing Machinery, New York, NY, USA, pp. 467–478,
       https://doi.org/10.1145/3447818.3460376.
