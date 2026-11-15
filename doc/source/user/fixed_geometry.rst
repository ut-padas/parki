.. _fixed-geometry:

Repeated Ewald calls with fixed geometry
========================================

In a GMRES solve, source positions, target positions, and source normals often
stay fixed while the density changes at every matrix-vector product. Prepare an
:class:`~parkipy.ewald.EwaldPlan` once and reuse it for those evaluations.
The plan retains Ewald parameters, cell lists, sorted geometry and normals,
grid and potential buffers, launch policies, and PyKokkos views. Each application
copies the new density into its existing source orderings and evaluates the
potential without rebuilding that geometry.

Plans support the Stokes single-layer and combined single/double-layer kernels
on a single CPU or GPU, with float32 or float64 arrays and periodicities 0–3.
They use the execution methods supported by the corresponding one-shot kernel.
Laplace and distributed plans are not currently provided. See
:ref:`laplace-plan-limitations` for the Laplace limitation.

.. _laplace-plan-limitations:

Why a Laplace plan is not supported yet
---------------------------------------

The existing :func:`~parkipy.ewald.laplace` routine selects Ewald parameters
using the charge-dependent quantity

.. math::

   Q = \sum_{j=1}^{N_s} q_j^2.

This quantity enters the error estimates used to select the splitting parameter,
Fourier grid resolution, window support, and upsampling. Fixed particle positions
therefore do not imply fixed parameters under the current Laplace API: changing
the charges can change the selected parameters. In contrast, the current Stokes
selectors default to a fixed ``source_quantity=1``.

Freezing parameters selected for the first charge vector would require an
accuracy policy for subsequent vectors, especially when their charge norm grows.
Recomputing parameters on each application would give up the cached setup and
could change the approximate operator during GMRES. Initializing the plan with
zero charges also cannot establish suitable parameters: the current parameter
selector requires positive quantities derived from ``Q``.

Prepare once, apply repeatedly
------------------------------

This complete CPU example evaluates a combined Stokes potential twice. For GPU
execution, replace NumPy with CuPy and select ``execution_space="GPU"``.

.. code-block:: python

   import numpy as np
   from parkipy import ewald

   rng = np.random.default_rng(17)
   ns, nt = 64, 32
   src = rng.uniform(0.1, 0.9, size=(3, ns))
   trg = rng.uniform(0.1, 0.9, size=(3, nt))
   normals = rng.normal(size=(3, ns))
   normals /= np.linalg.norm(normals, axis=0)
   density = rng.normal(size=(6, ns))
   options = ewald.EwaldOptions(
       box=[1.0, 1.0, 1.0],
       periodicity=1,
       tolerance=1e-5,
       execution_space="Serial",
       cell_size=16,
   )

   # Geometry preparation happens outside the iteration.
   plan = ewald.prepare_stokes_comb(trg, src, normals, options)
   first = plan.apply(density)
   second = plan.apply(2 * density)
   np.testing.assert_allclose(second, 2 * first, rtol=1e-10, atol=1e-10)

   # The prepared and one-shot operators evaluate the same potential.
   reference = ewald.stokes_comb(trg, src, density, normals, options)
   np.testing.assert_allclose(first, reference, rtol=1e-10, atol=1e-10)

For the single-layer operator, omit normals and provide three density rows:

.. code-block:: python

   sl_plan = ewald.prepare_stokes_sl(trg, src, options)
   single_layer = sl_plan.apply(density[:3])

Array shapes and output ownership
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Array
     - Shape
   * - Target positions
     - ``(3, Nt)``
   * - Source positions and source normals
     - ``(3, Ns)``
   * - Single-layer density
     - ``(3, Ns)``
   * - Combined density
     - ``(6, Ns)``: single-layer rows first, double-layer rows last
   * - Returned potential or supplied ``out``
     - ``(3, Nt)``

Source and target counts must be positive. Geometry, normals, density, and
output must use the same dtype and array backend; GPU arrays must also reside
on the same device. Noncontiguous density and output arrays are accepted.

By default, each call returns an independently owned array. Later applications
do not overwrite earlier results. To reuse output storage, pass ``out``:

.. code-block:: python

   potential = np.empty((3, nt), dtype=src.dtype)
   plan.apply(density, out=potential)
   plan.apply(2 * density, out=potential)  # overwrites potential

``plan(density)`` is equivalent to ``plan.apply(density)``. Both forms return only
the potential, even when ``options.return_walltime`` or ``options.return_params``
is true. Selected Ewald parameters are available as ``plan.params``.

Use inside GMRES
----------------

Create the plan before constructing the matrix-vector callback. For a combined
operator, fill the two density blocks with the layer densities required by your
formulation. Reuse a density buffer to avoid allocating a stacked array on every
iteration.

The following callback continues the CPU example with a manufactured operator
``I + alpha*K``. It uses the source points as targets so the linear system is
square:

.. code-block:: python

   from scipy.sparse.linalg import LinearOperator, gmres

   # Use the same points as sources and targets for this square operator.
   plan = ewald.prepare_stokes_comb(src, src, normals, options)
   work_density = np.empty((6, ns), dtype=src.dtype)
   work_potential = np.empty((3, ns), dtype=src.dtype)
   alpha = 1e-3 / ns

   def matvec(vector):
       values = vector.reshape(3, ns)
       work_density[:3] = values
       work_density[3:] = 0.1 * values
       plan.apply(work_density, out=work_potential)
       # GMRES retains vectors: return independent storage, not the scratch array.
       return vector + alpha * work_potential.reshape(-1)

   operator = LinearOperator((3 * ns, 3 * ns), matvec=matvec, dtype=src.dtype)
   exact = rng.normal(size=3 * ns)
   rhs = operator @ exact
   solution, info = gmres(operator, rhs, rtol=1e-8, atol=0.0)
   assert info == 0

For GPU execution, use CuPy arrays and
``cupyx.scipy.sparse.linalg.LinearOperator`` and ``gmres``. A runnable CPU/GPU
example checks the prepared operator against the one-shot API and reports the
GMRES residual:

.. code-block:: bash

   python examples/ewald/fixed_geometry_gmres.py --device Serial -N 64
   python examples/ewald/fixed_geometry_gmres.py --device GPU -N 512

When to rebuild a plan
----------------------

Plans own copies of positions, normals, and options. Mutating the original
arrays or options does not change an existing plan. Construct a new plan when
positions, normals, source or target counts, dtype, device, box, periodicity,
tolerance, cutoff, or execution settings change. Density values may change on
every application without rebuilding the plan. There is no in-place geometry
update or automatic invalidation.

A plan has mutable workspace. Apply it sequentially; use separate plans for
concurrent solves. ``out=`` reuses the final output array, but does not make the
whole evaluation allocation-free: FFT routines and some array operations can
still allocate temporaries. The plan also preserves PyKokkos's synchronization
behavior; it does not capture a CUDA graph or remove kernel fences.

See :func:`~parkipy.ewald.prepare_stokes_comb`,
:func:`~parkipy.ewald.prepare_stokes_sl`, and
:meth:`~parkipy.ewald.EwaldPlan.apply` for the API reference.
