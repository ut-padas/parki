"""Reusable, single-device Stokes Ewald operators for fixed geometry."""

from copy import deepcopy
import numpy as np

from parkipy.utils import get_array_module
from ._prepare import DevicePre
from ._ewald import (
    p2p,
    p2g,
    fft,
    cnv,
    ifft,
    g2p,
    adjust_potential,
    _cnv_global,
    _cnv_GLB,
)


class EwaldPlan:
    """A prepared Stokes operator with owned geometry and reusable workspace.

    Construct with :func:`prepare_stokes_comb` or :func:`prepare_stokes_sl`.
    Positions, normals, and options are copied at construction. Create a new plan
    when any of them change. Density values may change on every :meth:`apply`.

    Plans retain cell lists, sorted normals, grid buffers, launch policies, and
    PyKokkos views. They are stateful and must not be applied concurrently.
    ``apply`` returns only the potential, irrespective of the options' reporting
    flags; the selected Ewald parameters are available as ``plan.params``.
    FFT routines may still allocate temporary arrays. PyKokkos synchronization
    behavior is unchanged; this is not a CUDA graph or asynchronous launch API.
    """

    def __init__(self, trg, src, normals, options, kernel):
        if kernel not in ("stokes_comb", "stokes_sl"):
            raise ValueError("fixed-geometry plans support Stokes kernels only")
        self._options = deepcopy(options)
        self._am = am = get_array_module(options.execution_space)
        self._kernel = kernel
        self._dim_in = 6 if kernel == "stokes_comb" else 3
        for name, value in (("trg", trg), ("src", src)):
            self._validate_array(name, value)
            if value.ndim != 2 or value.shape[0] != 3 or value.shape[1] == 0:
                raise ValueError(f"{name} must have shape (3, N) with N > 0")
            if value.dtype not in (np.dtype("float32"), np.dtype("float64")):
                raise TypeError(f"{name} must have dtype float32 or float64")
        if trg.dtype != src.dtype:
            raise TypeError("trg and src must have the same dtype")
        self._device_id = getattr(getattr(src, "device", None), "id", None)
        if getattr(getattr(trg, "device", None), "id", None) != self._device_id:
            raise ValueError("trg and src must be on the same device")
        if normals is not None:
            self._validate_array("normals", normals, shape=src.shape, dtype=src.dtype)
        self._density_shape = (self._dim_in, src.shape[1])
        self._output_shape = (3, trg.shape[1])
        self._dtype = src.dtype
        density = am.zeros(self._density_shape, dtype=src.dtype)
        pre, self.params = DevicePre.from_particles(
            targets=trg.copy(),
            sources=src.copy(),
            forces=density,
            normals=None if normals is None else normals.copy(),
            kernel=kernel,
            box=np.asarray(self._options.box).copy(),
            tolerance=options.tolerance,
            periodicity=options.periodicity,
            cell_size=options.cell_size,
            rc=options.rc,
            execution_space=options.execution_space,
            fft_type=options.fft_type,
        )
        self._pre = pre
        self._near = p2p(
            pre,
            method=options.p2p_method,
            threads_x=options.p2p_threads_x,
            threads_y=options.p2p_threads_y,
            eps=options.puncture_radius,
            _prepare_only=True,
        )
        self._spread = p2g(
            pre,
            method=options.p2g_method,
            threads=options.p2g_threads,
            _prepare_only=True,
        )
        self._gather = g2p(
            pre,
            method=options.g2p_method,
            threads=options.g2p_threads,
            _prepare_only=True,
        )
        # Each source ordering has its own persistent density buffer. Padding
        # remains zero; normals and geometry never need refreshing.
        self._density_lists = []
        for stage in (self._near, self._spread):
            cells = stage.source_list
            if cells is not None:
                locations = am.nonzero(cells.particle_index >= 0)[0]
                indices = cells.particle_index[locations]
                values = cells.force_list
                if isinstance(values, tuple):
                    values = values[0]
                self._density_lists.append((values, locations, indices))
        pre._cnv_launches = _cnv_global(pre, options.cnv_threads, _prepare_only=True)
        # The 1P zero-plane input is overwritten every call, but its storage and
        # the views used by the compiled kernels stay fixed.
        index = (slice(None),) + (0,) * (options.periodicity or 3)
        pre._zero_source = am.empty_like(pre.data.Hg[index])
        pre._zero_launches = {}
        if options.periodicity == 1:
            _cnv_GLB(pre, options.cnv_threads, _prepare_only=True)
        self._p2g_transposed = options.p2g_method.upper() in ("BASE", "GRID")
        self._g2p_transposed = options.g2p_method.upper() == "BASE"

    def _validate_array(self, name, value, shape=None, dtype=None):
        if not isinstance(value, self._am.ndarray):
            raise TypeError(f"{name} must be a {self._am.__name__} array")
        if shape is not None and value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if dtype is not None and value.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
        device = getattr(getattr(value, "device", None), "id", None)
        if hasattr(self, "_device_id") and device != self._device_id:
            raise ValueError(f"{name} must be on the plan's device")

    def apply(self, density, out=None):
        """Apply to density of shape ``(6, Ns)`` (combined) or ``(3, Ns)``.

        Density and optional output must use the plan's array backend, device,
        and dtype. Noncontiguous arrays are accepted. With ``out=None`` a new,
        independently owned ``(3, Nt)`` array is returned. Otherwise write into
        and return ``out``. Input/output aliasing is supported.
        """
        self._validate_array("density", density, self._density_shape, self._dtype)
        if out is not None:
            self._validate_array("out", out, self._output_shape, self._dtype)
            if not getattr(out.flags, "writeable", True):
                raise ValueError("out must be writable")
        pre = self._pre
        pre.data.forces[...] = density
        for values, locations, indices in self._density_lists:
            values[:, locations] = pre.data.forces[:, indices]
        # Some P2P variants accumulate, and spreading adds into the grid.
        pre.near_potential.fill(0)
        pre.far_potential.fill(0)
        self._spread.grid.fill(0)
        self._near.launch()
        self._spread.launch()
        if self._p2g_transposed:
            pre.data.H[...] = self._spread.grid.transpose(3, 0, 1, 2)
        fft(pre, self._options)
        cnv(pre, threads=self._options.cnv_threads)
        ifft(pre, self._options)
        if self._g2p_transposed:
            self._gather.grid[...] = pre.data.H.transpose(1, 2, 3, 0)
        self._gather.launch()
        adjust_potential(pre)
        if out is None:
            out = self._am.empty(self._output_shape, dtype=self._dtype)
        self._am.add(pre.near_potential, pre.far_potential, out=out)
        return out

    __call__ = apply


def prepare_stokes_comb(trg, src, normals, options):
    """Prepare a combined Stokes operator for repeated fixed-geometry solves.

    ``trg``, ``src``, and ``normals`` have shape ``(3, N)`` and must use the
    array backend selected by ``options.execution_space``. Apply the returned
    :class:`EwaldPlan` to densities of shape ``(6, Ns)``. The first three rows
    are the single-layer density and the last three the double-layer density.
    """
    if normals is None:
        raise TypeError("normals are required for the combined Stokes operator")
    return EwaldPlan(trg, src, normals, options, "stokes_comb")


def prepare_stokes_sl(trg, src, options):
    """Prepare a Stokes single-layer operator; apply densities of shape ``(3, Ns)``."""
    return EwaldPlan(trg, src, None, options, "stokes_sl")
