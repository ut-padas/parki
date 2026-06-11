.. _examples:
 
********
Examples
********
 
This page walks through the most common ParkiPy use cases. Full, runnable
scripts for each example can be found in the ``examples/`` directory of the
repository.
 
 
Stokes Single-Layer Potential (3-periodic)
------------------------------------------
 
The Stokes single-layer potential sums the Stokeslet kernel over a set of
source particles :math:`\{y_j\}` with associated force densities
:math:`\{f(y_j)\}` to obtain the velocity field at target locations
:math:`\{x_i\}`:
 
.. math::
 
   u(x_i) = \sum_{j=1}^{N_s}
   \left(
     \frac{I}{\|x_i - y_j\|}
     + \frac{(x_i - y_j) \otimes (x_i - y_j)}{\|x_i - y_j\|^3}
   \right) f(y_j)
 
With fully periodic boundary conditions this is straightforward to evaluate
using :func:`parkipy.ewald.stokes_sl`.
 
.. code-block:: python
 
   import cupy as cp       # swap for numpy for CPU execution
   import parkipy
 
   rng = cp.random.default_rng(123)
 
   # Source particles (y) and their force densities (f)
   y = rng.random(size=(3, 773))
   f = rng.random(size=(3, 773))
 
   # Target locations (x) where the velocity is evaluated
   x = rng.random(size=(3, 312))
 
   # Configure the Ewald summation
   options = parkipy.ewald.EwaldOptions(
       periodicity=3,       # fully 3-periodic (triply periodic)
       box=[1, 1, 1],       # unit cube domain
       tolerance=1e-8,      # relative accuracy target
       cell_size=23,        # grid cells per dimension for the far-field solve
       execution_space="CUDA",  # use "OpenMP" for CPU
   )
 
   # Evaluate the potential — u has shape (3, 312)
   u = parkipy.ewald.stokes_sl(x, y, f, options)
 
Key parameters in :class:`~parkipy.ewald.EwaldOptions`:
 
.. list-table::
   :header-rows: 1
   :widths: 20 80
 
   * - Parameter
     - Description
   * - ``periodicity``
     - Number of periodic dimensions. Currently ``1`` and ``3`` are supported
       for the Stokes kernel.
   * - ``box``
     - Side lengths of the periodic box as a list ``[Lx, Ly, Lz]``.
   * - ``tolerance``
     - Desired relative accuracy. Smaller values increase both the near-field
       cutoff radius and the far-field grid resolution.
   * - ``cell_size``
     - Number of grid cells per dimension for the far-field (Fourier-space)
       component of the Ewald sum.
   * - ``execution_space``
     - ``"Cuda"`` for NVIDIA GPUs, ``"HIP"`` for AMD GPUs, or ``"OpenMP"``
       for multi-threaded CPU execution.
   * - ``return_walltime``
     - If ``True``, the call returns a ``(potential, timing_dict)`` tuple
       instead of just the potential array. Useful for profiling.
 
 
Stokes Single + Double Layer Potential (3-periodic)
----------------------------------------------------
 
When both single- and double-layer contributions are needed (e.g. for a
completed double-layer representation in boundary integral methods), use
:func:`parkipy.ewald.stokes_comb`:
 
.. code-block:: python
 
   import cupy as cp
   import parkipy
 
   rng = cp.random.default_rng(42)
 
   y  = rng.random(size=(3, 500))   # source positions
   n  = rng.random(size=(3, 500))   # surface normals at sources
   f  = rng.random(size=(3, 500))   # single-layer densities
   g  = rng.random(size=(3, 500))   # double-layer densities
   x  = rng.random(size=(3, 200))   # target positions
 
   options = parkipy.ewald.EwaldOptions(
       periodicity=3,
       box=[1, 1, 1],
       tolerance=1e-6,
       cell_size=20,
       execution_space="CUDA",
   )
 
   u = parkipy.ewald.stokes_comb(x, y, n, f, g, options)
 
 
Stokes Combined Layer Potential (1-periodic)
---------------------------------------------
 
For problems periodic in only one direction (e.g. a periodic channel or tube),
set ``periodicity=1``. The combined single + double layer potential is
evaluated with :func:`parkipy.ewald.stokes_comb`, which takes a stacked
density array of shape ``(6, N_s)`` — the first three rows are the
single-layer density and the last three are the double-layer density:
 
.. code-block:: python
 
   import parkipy
   import numpy as np   # or cupy for GPU
 
   device = "Cuda"
   am = parkipy.utils.get_array_module(parkipy.utils.get_execution_space(device))
 
   box  = [1, 1, 1]
   tol  = 1e-6
   ns   = 600
   nt   = 400
   cell_size = 16
 
   src     = am.random.rand(3, ns) * am.array(box).reshape(3, 1)
   trg     = am.random.rand(3, nt) * am.array(box).reshape(3, 1)
   dens_sl = am.random.randn(3, ns)   # single-layer density
   dens_dl = am.random.randn(3, ns)   # double-layer density
   norms   = am.random.randn(3, ns)   # surface normals at sources
   dens    = am.vstack((dens_sl, dens_dl))   # shape (6, ns)
 
   options = parkipy.ewald.EwaldOptions(
       periodicity=1,
       box=box,
       tolerance=tol,
       execution_space=device,
       cell_size=cell_size,
   )
 
   u = parkipy.ewald.stokes_comb(trg, src, dens, norms, options)
 
 
Distributed Combined Stokes Potential (Multi-GPU, 1-periodic)
--------------------------------------------------------------
 
For very large particle counts, :mod:`parkipy.distributed` distributes the
computation across multiple GPUs using MPI. Each MPI rank allocates and owns
its own local slice of the source and target particles. The example below
evaluates the combined (single + double layer) Stokes potential with one
periodic direction across ``n`` MPI ranks, where the periodic box length in
the first dimension is scaled with the number of ranks:
 
.. code-block:: python
 
   from mpi4py import MPI
   import parkipy
   import numpy as np
 
   # Initialise MPI
   mpi_comm = MPI.COMM_WORLD
   size = mpi_comm.Get_size()
   rank = mpi_comm.Get_rank()
 
   # Use parkipy helpers to select the array module (numpy or cupy)
   # based on the chosen execution space
   device = "Cuda"   # or "OpenMP" for CPU
   execution_space = parkipy.utils.get_execution_space(device)
   am = parkipy.utils.get_array_module(execution_space)
 
   # Box scaled so each rank owns a unit-length slab in the periodic direction
   box = [size, 1, 1]
   tol = 1e-4
   cell_size = 224
 
   # Each rank generates its own local particles
   nt = int(4e6)
   ns = int(4e6)
   rc = np.ceil(ns / cell_size) ** (-1 / 3)   # near-field cutoff radius
 
   trg      = am.random.rand(3, nt) * am.array(box).reshape(3, 1)
   src      = am.random.rand(3, ns) * am.array(box).reshape(3, 1)
   dens_sl  = am.random.randn(3, ns)   # single-layer density
   dens_dl  = am.random.randn(3, ns)   # double-layer density
   norms    = am.random.randn(3, ns)   # surface normals at sources
   dens     = am.vstack((dens_sl, dens_dl))   # stacked density input
 
   # Evaluate the distributed combined Stokes potential
   pot, trg, timing = parkipy.distributed.ewald.stokes_comb(
       trg,
       src,
       dens,
       norms,
       1,         # periodicity
       box,
       tol,
       device,
       rc=rc,
       time=True,
   )
 
After the call, ``pot`` is a ``(3, nt_local)`` array on each rank containing
the local velocity potential. Timing information is returned in the ``timing``
dict when ``time=True``.
 
Once all ranks have finished, use :func:`parkipy.distributed.gather_points`
to collect results onto rank 0 for validation or output:
 
.. code-block:: python
 
   import numpy as np
 
   nt_local = trg.shape[1]
   nt_global = mpi_comm.allreduce(nt_local, op=MPI.SUM)
 
   pot_gathered = np.empty((3, nt_global))
   pot_gathered[:, :nt_local] = pot.get()
 
   parkipy.distributed.gather_points(mpi_comm, "openmp", pot_gathered.T, nt_local)
 
Run with ``mpiexec`` (or your scheduler's equivalent):
 
.. code-block:: bash
 
   mpiexec -n 4 python examples/distributed/ewald/stokes1p_mp.py --device Cuda
 
 
Cell List
---------
 
The :class:`parkipy.CellList` class provides an O(N) neighbour-finding data
structure for local (short-range) interactions. It is used internally by the
Ewald near-field routines but is also available as a standalone API.
 
As a motivating example, consider computing the short-range potential
:math:`u(x_i) = \sum_{j:\,\|x_i - y_j\| < r_c} 1 / \|x_i - y_j\|`. A
naïve double loop over all pairs is O(N²). Using a cell list reduces this to
O(N) by restricting the inner loop to the 27 neighbouring cells around each
source particle:
 
.. code-block:: python
 
   import parkipy
 
   device = "Cuda"   # or "OpenMP"
   am = parkipy.utils.get_array_module(parkipy.utils.get_execution_space(device))
 
   box    = [1, 1, 1]
   cutoff = 0.1
 
   Nx = 312
   Ny = 773
   x = am.random.rand(3, Nx) * am.array(box).reshape(3, 1)
   y = am.random.rand(3, Ny) * am.array(box).reshape(3, 1)
   u = am.zeros(x.shape[1])
 
   # Build separate cell lists for targets (x) and sources (y)
   x_list = parkipy.CellList(x, cutoff, box, execution_space=device)
   y_list = parkipy.CellList(y, cutoff, box, execution_space=device)
 
   # Loop over non-empty x-cells
   for cell_ne in range(x_list.num_nonempty_cells):
       off_x = cell_ne * x_list.cell_size
       cell  = x_list.nonempty_cells[cell_ne]
 
       # Loop over x-particles in this cell
       for ii in range(off_x, off_x + x_list.cell_size):
           i = x_list.particle_index[ii]
           if i == -1:
               continue   # padded slot — skip
 
           # Loop over the 27 neighbouring y-cells
           for k in range(27):
               neighbor = y_list.nonempty_neighbors[cell, k].get()
               if neighbor == -1:
                   continue   # no neighbour in this direction
 
               off_y = neighbor * y_list.cell_size
 
               # Loop over y-particles in the neighbour cell
               for jj in range(off_y, off_y + y_list.cell_size):
                   j = y_list.particle_index[jj]
                   if j == -1:
                       continue
 
                   dist_sq = am.sum((x[:, i] - y[:, j]) ** 2)
                   if dist_sq == 0:
                       continue
                   dist = am.sqrt(dist_sq)
                   if dist < cutoff:
                       u[i] += 1 / dist
 
Key attributes of :class:`~parkipy.CellList`:
 
.. list-table::
   :header-rows: 1
   :widths: 30 70
 
   * - Attribute
     - Description
   * - ``num_nonempty_cells``
     - Number of cells that contain at least one particle.
   * - ``nonempty_cells``
     - 1-D index array mapping the non-empty cell enumeration to the global
       cell index.
   * - ``cell_size``
     - Maximum number of particles per cell (padded with ``-1`` sentinels).
   * - ``particle_index``
     - Flat array of particle indices. Slice ``[off : off + cell_size]`` to
       get the particles in a given cell. Entries of ``-1`` are padding.
   * - ``nonempty_neighbors``
     - ``(num_cells, 27)`` array of neighbour cell indices. An entry of
       ``-1`` means no non-empty neighbour exists in that direction.

More Examples
-------------

Fully runnable scripts covering additional configurations and performance
benchmarks are provided in the ``examples/`` directory of the repository:

.. code-block:: text

        examples/
        ├── batched_celllist.py
        ├── celllist.py
        ├── distributed
        │   └── ewald
        │       ├── stokes1p_nosort.py
        │       └── stokes1p.py
        ├── ewald
        │   ├── laplace3p.py
        │   ├── stokes1p.py
        │   └── stokes3p.py
        └── periodic_celllist.py
