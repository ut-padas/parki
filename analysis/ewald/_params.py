"""
Spectral Ewald parameter selection.
"""

from dataclasses import dataclass, field
from collections.abc import Callable
from fractions import Fraction
import numpy as np
import scipy as sp
import sys


def check_positive(val, s):
    if np.imag(val) / np.real(val) < 1e-16:
        val = np.real(val)
    if not np.isreal(val) or val <= 0:
        raise ValueError(f"Parameter failure: {s} did not become positive (={val})")
    return val


def rc_from_cell_size(box, num_sources, cell_size):
    """
    Compute ``rc`` given ``box``, ``num_sources`` and ``cell_size``. See e.g.
    ``se_params_stokeslet()`` for an explanation of these parameters. This
    function will raise ``TypeError`` if any input argument is ``None``.
    """
    if box is None or num_sources is None or cell_size is None:
        raise TypeError("Cannot compute rc due to missing information")
    num_cells = num_sources / cell_size
    rc = (np.prod(box) / num_cells) ** (1 / 3)
    return rc


def se_params_laplace(
    box_dict,
    tolerance=1e-16,
    periodicity=3,
    num_sources=None,
    cell_size=1024,
    rc=None,
    window_P=None,
    source_quantity=None,
    f=None,
    distributed=False,
):
    """
    Select Spectral Ewald parameters for the Laplace kernel.

    Parameters
    ----------
    box_dict : dict, optional
        Dictionary containing box and slab (if distributed) information.
        `box_dict['box']` is the size of the primary cell [L1, L2, L3],
        containing all sources and targets. Default: ``[1, 1, 1]``.
    tolerance : float, optional
        Tolerance for automatic parameter selection. Default: ``1e-16``. Note:
        the error estimates for computing the default values may not be good
        for a noncubic box.
    periodicity : {0, 1, 2, 3}, optional
        The number of periodic spatial directions. Default: 1.
    num_sources : int, optional
        The number of source points in the problem. This is only required if
        ``rc`` is to be computed from ``cell_size``; see below. If not given,
        ``rc`` becomes a required parameter.
    cell_size : float, optional
        The expected number of source points per cube of side length ``rc``. If
        ``num_sources`` is given, this value is used to compute ``rc``. Default:
        1024.
    rc : float, optional
        The cutoff radius of near-field interactions. If ``num_sources`` is
        given, ``rc`` is optional and will be computed if not given. Otherwise,
        ``rc`` becomes required. If a value for ``rc`` is given, both
        ``num_sources`` and ``cell_size`` are ignored.
    window_P : int, optional
        Window function support size, measured in number of grid subintervals.
        Default: computed from error estimates.
    source_quantity: float, optional
        A quantity that is computed from the sources and appears in the
        truncation error estimates. For the Leplace kernel,
        `source_quantity = np.sum(f**2)`. If None, the charges `f` must be
        provided.
    distributed : bool, optional
        Flag to identify if Ewald execution is distributed. Default: False
    Returns
    -------
    params : SEParams
        Data class containing all computed parameters.
    """
    box = box_dict["box"]
    if distributed:
        slab_box = box_dict["slab box"]
    else:
        slab_box = box_dict["box"]
    if rc is None:
        rc = rc_from_cell_size(slab_box, num_sources, cell_size)
    if source_quantity is None:
        if f is None:
            raise ValueError("One of `source_quantity` or `f` must be given.")
        source_quantity = float(np.sum(f**2))
    if f is not None and source_quantity is not None:
        assert source_quantity == np.sum(
            f**2
        ), "Source quantity expected to be `np.sum(f**2)."
    xi, grid_res, window_P, A_fun = _se_params_laplace_core(
        box, tolerance, rc, window_P, source_quantity
    )
    params = _check_options(
        box_dict,
        tolerance,
        periodicity,
        rc,
        xi,
        grid_res,
        window_P,
        source_quantity,
        A_fun,
        distributed,
    )
    return params


def se_params_stokeslet(
    box_dict,
    tolerance=1e-16,
    periodicity=1,
    num_sources=None,
    cell_size=1024,
    rc=None,
    window_P=None,
    source_quantity=1,
    distributed=False,
):
    """
    Select Spectral Ewald parameters for the stokeslet.

    Parameters
    ----------
    box_dict : dict, optional
        Dictionary containing box and slab (if distributed) information.
        `box_dict['box']` is the size of the primary cell [L1, L2, L3],
        containing all sources and targets. Default: ``[1, 1, 1]``.
    tolerance : float, optional
        Tolerance for automatic parameter selection. Default: ``1e-16``. Note:
        the error estimates for computing the default values may not be good
        for a noncubic box.
    periodicity : {0, 1, 2, 3}, optional
        The number of periodic spatial directions. Default: 1.
    num_sources : int, optional
        The number of source points in the problem. This is only required if
        ``rc`` is to be computed from ``cell_size``; see below. If not given,
        ``rc`` becomes a required parameter.
    cell_size : float, optional
        The expected number of source points per cube of side length ``rc``. If
        ``num_sources`` is given, this value is used to compute ``rc``. Default:
        1024.
    rc : float, optional
        The cutoff radius of near-field interactions. If ``num_sources`` is
        given, ``rc`` is optional and will be computed if not given. Otherwise,
        ``rc`` becomes required. If a value for ``rc`` is given, both
        ``num_sources`` and ``cell_size`` are ignored.
    window_P : int, optional
        Window function support size, measured in number of grid subintervals.
        Default: computed from error estimates.
    source_quantity: float, optional
        A quantity that is computed from the sources and appears in the
        truncation error estimates. Setting this to 1 and interpreting
        ``tolerance`` as a relative tolerance seems to work rather well, so
        there should be little reason to change this parameter. Default: 1.
    distributed : bool, optional
        Flag to identify if Ewald execution is distributed. Default: False

    Returns
    -------
    params : SEParams
        Data class containing all computed parameters.
    """
    box = box_dict["box"]
    if distributed:
        slab_box = box_dict["slab box"]
    else:
        slab_box = box_dict["box"]
    if rc is None:
        rc = rc_from_cell_size(slab_box, num_sources, cell_size)
    xi, grid_res, window_P, A_fun = _se_params_stokeslet_core(
        box, tolerance, rc, window_P, source_quantity
    )
    params = _check_options(
        box_dict,
        tolerance,
        periodicity,
        rc,
        xi,
        grid_res,
        window_P,
        source_quantity,
        A_fun,
        distributed,
    )
    return params


def se_params_stresslet(
    box_dict,
    tolerance=1e-16,
    periodicity=1,
    num_sources=None,
    cell_size=1024,
    rc=None,
    window_P=None,
    source_quantity=1,
    distributed=False,
):
    """
    Select Spectral Ewald parameters for the stresslet.

    Parameters
    ----------
    box_dict : dict, optional
        Dictionary containing box and slab (if distributed) information.
        `box_dict['box']` is the size of the primary cell [L1, L2, L3],
        containing all sources and targets. Default: ``[1, 1, 1]``.
    tolerance : float, optional
        Tolerance for automatic parameter selection. Default: ``1e-16``. Note:
        the error estimates for computing the default values may not be good
        for a noncubic box.
    periodicity : {0, 1, 2, 3}, optional
        The number of periodic spatial directions. Default: 1.
    num_sources : int, optional
        The number of source points in the problem. This is only required if
        ``rc`` is to be computed from ``cell_size``; see below. If not given,
        ``rc`` becomes a required parameter.
    cell_size : float, optional
        The expected number of source points per cube of side length ``rc``. If
        ``num_sources`` is given, this value is used to compute ``rc``. Default:
        1024.
    rc : float, optional
        The cutoff radius of near-field interactions. If ``num_sources`` is
        given, ``rc`` is optional and will be computed if not given. Otherwise,
        ``rc`` becomes required. If a value for ``rc`` is given, both
        ``num_sources`` and ``cell_size`` are ignored.
    window_P : int, optional
        Window function support size, measured in number of grid subintervals.
        Default: computed from error estimates.
    source_quantity: float, optional
        A quantity that is computed from the sources and appears in the
        truncation error estimates. Setting this to 1 and interpreting
        ``tolerance`` as a relative tolerance seems to work rather well, so
        there should be little reason to change this parameter. Default: 1.
    distributed : bool, optional
        Flag to identify if Ewald execution is distributed. Default: False

    Returns
    -------
    params : SEParams
        Data class containing all computed parameters.
    """
    box = box_dict["box"]
    if distributed:
        slab_box = box_dict["slab box"]
    else:
        slab_box = box_dict["box"]
    if rc is None:
        rc = rc_from_cell_size(slab_box, num_sources, cell_size)
    xi, grid_res, window_P, A_fun = _se_params_stresslet_core(
        box, tolerance, rc, window_P, source_quantity, distributed
    )
    params = _check_options(
        box_dict,
        tolerance,
        periodicity,
        rc,
        xi,
        grid_res,
        window_P,
        source_quantity,
        A_fun,
        distributed,
    )
    return params


def se_params_stokes_comb(
    box_dict,
    tolerance=1e-16,
    periodicity=1,
    num_sources=None,
    cell_size=1024,
    rc=None,
    window_P=None,
    source_quantity=1,
    distributed=False,
):
    """
    Select Spectral Ewald parameters for both the stokeslet and stresslet. This
    picks the maximum ``xi``, ``grid_res`` and ``window_P`` required by the two
    kernels.

    Parameters
    ----------
    box_dict : dict, optional
        Dictionary containing box and slab (if distributed) information.
        `box_dict['box']` is the size of the primary cell [L1, L2, L3],
        containing all sources and targets. Default: ``[1, 1, 1]``.
    tolerance : float, optional
        Tolerance for automatic parameter selection. Default: ``1e-16``. Note:
        the error estimates for computing the default values may not be good
        for a noncubic box.
    periodicity : {0, 1, 2, 3}, optional
        The number of periodic spatial directions. Default: 1.
    num_sources : int, optional
        The number of source points in the problem. This is only required if
        ``rc`` is to be computed from ``cell_size``; see below. If not given,
        ``rc`` becomes a required parameter.
    cell_size : float, optional
        The expected number of source points per cube of side length ``rc``. If
        ``num_sources`` is given, this value is used to compute ``rc``. Default:
        1024.
    rc : float, optional
        The cutoff radius of near-field interactions. If ``num_sources`` is
        given, ``rc`` is optional and will be computed if not given. Otherwise,
        ``rc`` becomes required. If a value for ``rc`` is given, both
        ``num_sources`` and ``cell_size`` are ignored.
    window_P : int, optional
        Window function support size, measured in number of grid subintervals.
        Default: computed from error estimates.
    source_quantity: float, optional
        A quantity that is computed from the sources and appears in the
        truncation error estimates. Setting this to 1 and interpreting
        ``tolerance`` as a relative tolerance seems to work rather well, so
        there should be little reason to change this parameter. Default: 1.
    distributed : bool, optional
        Flag to identify if Ewald execution is distributed. Default: False

    Returns
    -------
    params : SEParams
        Data class containing all computed parameters.
    """
    box = box_dict["box"]
    if distributed:
        slab_box = box_dict["slab box"]
    else:
        slab_box = box_dict["box"]
    if rc is None:
        rc = rc_from_cell_size(slab_box, num_sources, cell_size)
    xi_1, grid_res_1, window_P_1, A_fun_1 = _se_params_stokeslet_core(
        box, tolerance, rc, window_P, source_quantity
    )
    xi_2, grid_res_2, window_P_2, A_fun_2 = _se_params_stresslet_core(
        box, tolerance, rc, window_P, source_quantity
    )
    xi = np.nanmax([xi_1, xi_2])
    grid_res = np.nanmax([grid_res_1, grid_res_2])
    window_P = np.nanmax([window_P_1, window_P_2])
    A_fun = A_fun_1
    params = _check_options(
        box_dict,
        tolerance,
        periodicity,
        rc,
        xi,
        grid_res,
        window_P,
        source_quantity,
        A_fun,
        distributed,
    )
    return params


def _se_params_laplace_core(box, tolerance, rc, window_P, source_quantity):
    """
    Core function for selecting Laplace parameters
    """
    # Compute xi from rc
    vol = np.prod(box)
    factor_RS = source_quantity / (vol * tolerance**2)
    xi = (1.0 / rc) * np.sqrt(sp.special.lambertw(np.sqrt(rc * factor_RS)))
    xi = check_positive(xi, "xi")
    # Compute grid_res from xi
    vol_min = np.min(box) ** 3
    factor_FS = source_quantity / (vol_min * tolerance**2)
    grid_res = (np.sqrt(3) * xi / np.pi) * np.sqrt(
        sp.special.lambertw((4.0 / 3.0) * (factor_FS / (np.pi * xi)) ** (2 / 3))
    )
    grid_res = check_positive(grid_res, "grid_res")
    # Compute A fun
    co = [12.18, 1.815e-2, 1.080e-4]
    CC = 0.92
    xiLfun = lambda t: np.exp(-co[0] / (t * t)) * (1 + co[1] * t + co[2] * t * t)
    A_fun = lambda Q, xi, L: CC * np.sqrt(Q) * xiLfun(xi * L) / L
    # Compute window_P if not given
    if window_P is None:
        L = np.min(box)  # TODO: temporary assumption, check this for non-cubic boxes!
        A = A_fun(source_quantity, xi, L)
        shape_factor = 2.5  # default value for Kaiser windows
        extra_factor = 10  # from the estimates
        window_P = -np.log(tolerance / A / extra_factor) / shape_factor
        window_P = check_positive(window_P, "window_P")
    return xi, grid_res, window_P, A_fun


def _se_params_stokeslet_core(box, tolerance, rc, window_P, source_quantity):
    """
    Core function for selecting stokeslet parameters.
    """
    # Compute xi from rc
    vol = np.prod(box)
    factor_RS = source_quantity / (vol * tolerance**2)
    xi = np.sqrt(np.log(np.sqrt(4 * rc * factor_RS))) / rc
    xi = check_positive(xi, "xi")
    # Compute grid_res from xi
    Lmin = np.min(box)
    factor_FS = 4 * np.sqrt(source_quantity / 3) / (np.pi * Lmin * tolerance)
    grid_res = 2 * xi / np.pi * np.sqrt(np.log(factor_FS))
    grid_res = check_positive(grid_res, "grid_res")
    # Compute window_P only if not given
    co = [5.205, 1.323e-2, 2.469e-4]
    CC = 1.76
    xiLfun = lambda t: np.exp(-co[0] / (t * t)) * (1 + co[1] * t + co[2] * t * t)
    A_fun = lambda Q, xi, L: CC * np.sqrt(Q) * xiLfun(xi * L) / L
    if window_P is None:
        L = np.min(box)  # TODO: temporary assumption, check this for non-cubic boxes!
        A = A_fun(source_quantity, xi, L)
        shape_factor = 2.5  # default value for Kaiser windows
        extra_factor = 10  # from the estimates
        window_P = -np.log(tolerance / A / extra_factor) / shape_factor
        window_P = check_positive(window_P, "window_P")
    return xi, grid_res, window_P, A_fun


def _se_params_stresslet_core(box, tolerance, rc, window_P, source_quantity):
    """
    Core function for selecting stresslet parameters.
    """
    if isinstance(box, list):
        box = np.asarray(box)
    # Compute xi from rc
    vol = np.prod(box)
    factor = 3 * tolerance**2 / (112 * source_quantity)
    factor_RS = 3 * factor * vol
    xi = np.sqrt(-sp.special.lambertw(-np.sqrt(factor_RS * rc), -1)) / rc
    xi = check_positive(xi, "xi")
    # Compute grid_res from xi
    Ldiam2 = np.sum(box**2)  # diameter (diagonal) of box, squared
    factor_FS = np.pi**2 * factor * Ldiam2
    grid_res = (
        np.sqrt(2) * xi / np.pi * np.sqrt(-sp.special.lambertw(-factor_FS / xi**2, -1))
    )
    grid_res = check_positive(grid_res, "grid_res")
    # Compute window_P only if not given
    CC = 7.17
    A_fun = lambda Q, xi, L: CC * np.sqrt(Q) * np.sqrt(xi * L) / (L * L)
    if window_P is None:
        L = np.min(box)  # TODO: temporary assumption, check this for non-cubic boxes!
        A = A_fun(source_quantity, xi, L)
        shape_factor = 2.5  # default value for Kaiser windows
        extra_factor = 10  # from the estimates
        window_P = -np.log(tolerance / A / extra_factor) / shape_factor
        window_P = check_positive(window_P, "window_P")
    return xi, grid_res, window_P, A_fun


def _check_options(
    box_dict,
    tolerance,
    periodicity,
    rc,
    xi,
    grid_res,
    window_P,
    source_quantity,
    A_fun,
    distributed,
):
    """
    Check and complete options.
    """
    kwargs = {}
    if distributed:
        if periodicity != 1:
            raise NotImplementedError(
                "distributed Ewald only implemented for periodicity `1`,"
                f" got periodicity `{periodicity}`."
            )
    # Check that rc is not too large
    if distributed:
        box = box_dict["slab box"]
    else:
        box = box_dict["box"]
    if rc > np.min(box):
        raise ValueError(f"rc (={rc}) cannot be larger than min(box) (={np.min(box)})")
    # Check window_P and ensure it is even
    if window_P <= 0:
        raise ValueError(f"window_P (={window_P}) must be positive")
    window_P = int(2 * np.ceil(window_P / 2))
    # Create parameter dataclass
    params = SEParams(
        box_dict,
        tolerance,
        periodicity,
        rc,
        window_P,
        source_quantity,
        xi,
        grid_res,
        A_fun,
        distributed,
        **kwargs,
    )
    return params


@dataclass
class SEParams:
    r"""
    Data class with Spectral Ewald parameters, returned by the functions
    `se_params_stokeslet()`, `se_params_stresslet()` and `se_params_both()`.

    Input parameters
    ----------------
    box_dict, tolerance, periodicity, rc, window_P, source_quantity
        See the functions above for the meaning of these.
    xi : float
        Ewald decomposition parameter.
    grid_res : float
        Number of grid subintervals per unit length, i.e., 1/h, where h is the
        grid step size.
    A_fun : callable
        Function used in the error estimates.
    base_factor : int, optional
        All grid sizes will be rounded up to be multiples of ``base_factor``.
        Default: 4. The default value may speed up FFTs at the price of a
        slightly larger grid, and probably does not need to be changed.
    window_shape_factor : float, optional
        Shape parameter for window divided by ``window_P``. Default: 2.5. Cannot
        be varied for the ``'kaiser_poly'`` window, and probably does not need
        to be changed at all.
    stokeslet_k0_constant : float, optional
        Choice of gauge constant for the stokeslet kernel; :math:`\ell_B = \exp(C)`
        where :math:`C` is the ``stokeslet_k0_constant``. This adds a constant to
        the output potential. Default: 0.
    distributed : bool
        Flag to identify if Ewald execution is distributed. Default: False

    Derived parameters
    ------------------
    h : float
        The grid step size.
    grid_shape_ext : array_like
        Size of the extended grid [Mex1, Mex2, Mex3].
    box_ext : array_like
        Size of the extended primary cell [Lex1, Lex2, Lex3].
    off_1 : float
        Extended primary cell offset in the y direction.
    off_2 : float
        Extended primary cell offset in the z direction.
    greens_truncation_R : float
        This is the diagonal of the extended box in the free directions.
    actual_upsampling : float
        Actual upsampling factor used for the Fourier space grid
        (when using global upsampling).
    actual_upsampling_zero : float
        Actual upsampling factor used for the zero modes of the Fourier space
        grid (when using adaptive FFT).
    actual_upsampling_local : float
        Actual upsampling factor used for the local (near-zero) modes of the
        Fourier space grid (when using adaptive FFT).
    grid_shape_ups : array_like
        Size of the upsampled grid [Mup1, Mup2, Mup3], when using global
        upsampling.
    local_modes : array_like
        List of local (near-zero) modes to apply the local upsampling factor
        to (when using adaptive FFT).
    window_shape : float
        Computed by multiplying ``window_shape_factor`` by ``window_P``.
    kaiser_scaling : float
        Computed from ``window_shape``.
    """

    box_dict: dict
    tolerance: float
    periodicity: int
    rc: float
    window_P: int
    source_quantity: float
    xi: float
    grid_res: float
    A_fun: Callable[[float, float, float], float]
    distributed: bool
    base_factor: int = 4
    window_shape_factor: float = 2.5
    stokeslet_k0_constant: float = 0
    ghost_dist: float = 0

    box: list[float] = field(init=False)
    h: float = field(init=False)
    grid_shape_ext: list[int] = field(init=False)
    box_ext: list[float] = field(init=False)
    off_1: float = field(init=False)
    off_2: float = field(init=False)
    greens_truncation_R: float = field(init=False)
    actual_upsampling: float = field(init=False)
    actual_upsampling_zero: float = field(init=False)
    actual_upsampling_local: float = field(init=False)
    grid_shape_ups: list[int] = field(init=False)
    local_modes: list[int] = None
    window_shape: float = field(init=False)
    kaiser_scaling: float = field(init=False)

    def __post_init__(self):
        self.glb_box = [self.box_dict["box"][i] for i in range(3)]
        if self.distributed:
            self.box = self.box_dict["slab box"]
        else:
            self.box = self.box_dict["box"]
        # round base factor up to be a multiple of window P
        #   so cell list may partition the grid.
        #   NB: this is necessary for p2g-grid and p2g CellList creation
        self.base_factor = np.ceil(self.base_factor / self.window_P) * self.window_P
        # Adjust grid_res in periodic directions (with base_factor)
        grid_shape, grid_res = se_compute_grid_size(
            self.box, self.grid_res, self.base_factor, self.periodicity
        )
        glb_grid_shape, glb_grid_res = se_compute_grid_size(
            self.box_dict["box"], self.grid_res, self.base_factor, self.periodicity
        )
        # round grid res to be a multiple of self.window_P
        #   this ensures far-field cells form a full partition.
        if grid_res / self.grid_res > 2:
            print(
                "Warning: grid_res increased by more than a factor 2"
                f" (from {self.grid_res} to {grid_res})",
                file=sys.stderr,
            )
        self.grid_res = grid_res
        self.h = 1 / self.grid_res
        # ``grid_shape`` is the grid shape before extension; it is currently not
        # stored in the SEParams class since it is not needed. Also, it may not
        # consist of integers in the free directions.

        # compute self.grid_shape_ext, self.box_ext, self.off_1, self.off_2
        #   self.greens_truncation_R, self.actual_upsampling,
        #   self.actual_upsampling_zero, self.actual_upsampling_local
        #   self.grid_shape_ups, self.local_modes
        match self.periodicity:
            case 1:
                self._prepare_1p(grid_shape, glb_grid_shape)
            case 3:
                self._prepare_3p(grid_shape, glb_grid_shape)
            case _:
                raise ValueError(
                    "SEPre only suppored for periodicity 1 or 3,"
                    f" got {self.periodicity}."
                )

        # Check that h is the same in all directions
        thres = 4 * np.spacing(self.h)
        diff_0 = np.abs(self.h - self.box_ext[0] / self.grid_shape_ext[0])
        diff_1 = np.abs(self.h - self.box_ext[1] / self.grid_shape_ext[1])
        diff_2 = np.abs(self.h - self.box_ext[2] / self.grid_shape_ext[2])
        if diff_0 > thres or diff_1 > thres or diff_2 > thres:
            raise ValueError(
                f"Step size mismatch: [{diff_0}, {diff_1}, {diff_2}]" f" > {thres}"
            )

        # Window shape and scaling parameters
        self.window_shape = self.window_shape_factor * self.window_P
        self.kaiser_scaling = 1 / np.i0(self.window_shape)

    def _prepare_3p(self, grid_shape, glb_grid_shape):
        # grid_shape_ext = grid_shape, no need for upsampling
        self.grid_shape_ext = np.array([int(grid_shape[i]) for i in range(3)])
        self.box_ext = self.box
        self.glb_box_ext = np.array([self.box_dict["box"][0], *self.box_ext[1:]])
        self.glb_grid_shape_ext = np.array(
            [int(glb_grid_shape[0]), *self.grid_shape_ext[1:]]
        )
        self.off_1 = self.off_2 = 0
        self.actual_upsampling = np.array([1, 1])
        self.greens_truncation_R = 0

    def _prepare_1p(self, grid_shape, glb_grid_shape):
        # Compute extended grid shape (extend in free directions)
        assert self.periodicity == 1, "Periodicity must be 1"
        grid_ext_1 = grid_shape[1] + self.window_P + 1.4 * self.window_P
        grid_ext_2 = grid_shape[2] + self.window_P + 1.4 * self.window_P
        # TODO/FIXME: Maybe the grid increase should be adjusted by multiplying
        # it by new_grid_res/old_grid_res?
        grid_ext_1 = self.base_factor * np.ceil(
            grid_ext_1 / self.base_factor
        )  # round up
        grid_ext_2 = self.base_factor * np.ceil(
            grid_ext_2 / self.base_factor
        )  # round up
        self.grid_shape_ext = np.array(
            [int(grid_shape[0]), int(grid_ext_1), int(grid_ext_2)]
        )
        self.glb_grid_shape_ext = np.array(
            [int(glb_grid_shape[0]), *self.grid_shape_ext[1:]]
        )
        # Compute extended box shape
        self.box_ext = np.array(
            [
                self.box[0],
                self.h * self.grid_shape_ext[1],
                self.h * self.grid_shape_ext[2],
            ]
        )
        self.glb_box_ext = np.array([self.box_dict["box"][0], *self.box_ext[1:]])

        # Compute offsets in free directions
        self.off_1 = -(self.box_ext[1] - self.box[1]) / 2
        self.off_2 = -(self.box_ext[2] - self.box[2]) / 2

        # Computations for upsampled Fourier grid
        # Compute upsampling factor for the zero mode
        self.greens_truncation_R = np.linalg.norm(self.box_ext[1:], ord=2)
        upsampling_zero = 1 + self.greens_truncation_R / np.min(self.box_ext[1:])
        upsampling_zero = np.ceil(upsampling_zero * 10) / 10  # round up slightly
        # Compute upsampling factor for the local (near-zero) modes
        LextOverL = np.min(self.box_ext[1:] / self.box[1:])
        A = self.A_fun(self.source_quantity, self.xi, self.box_dict["box"][0])
        upsampling_local = (
            1 - 1 / (2 * np.pi) * np.log(2 * self.tolerance / A)
        ) / LextOverL
        # TODO/FIXME: This value may NOT work for a noncubic periodic box!
        local_modes_size = (
            -1 / (2 * np.pi) * np.log(2 * self.tolerance / A) / (LextOverL - 1) - 1
        )
        # TODO/FIXME: This value may NOT work for a noncubic periodic box!
        local_modes_size = int(np.ceil(np.maximum(local_modes_size, 0)))  # round up
        # Take maximum for global upsampling factor
        upsampling = np.maximum(upsampling_zero, upsampling_local)
        # Upsampling factors cannot be below 1
        upsampling_zero = np.maximum(upsampling_zero, 1)
        upsampling_local = np.maximum(upsampling_local, 1)
        upsampling = np.maximum(upsampling, 1)
        # Upsampled grids must be multiples of base_factor
        s0 = upsampling_zero
        sl = upsampling_local
        sg = upsampling
        grid_ups_s0 = self.base_factor * np.ceil(
            self.grid_shape_ext[1:] * s0 / self.base_factor
        )
        grid_ups_sl = self.base_factor * np.ceil(
            self.grid_shape_ext[1:] * sl / self.base_factor
        )
        grid_ups_sg = self.base_factor * np.ceil(
            self.grid_shape_ext[1:] * sg / self.base_factor
        )
        self.actual_upsampling_zero = grid_ups_s0 / self.grid_shape_ext[1:]
        self.actual_upsampling_local = grid_ups_sl / self.grid_shape_ext[1:]
        self.actual_upsampling = grid_ups_sg / self.grid_shape_ext[1:]
        self.grid_shape_ups = np.array(
            [self.grid_shape_ext[0], int(grid_ups_sg[0]), int(grid_ups_sg[1])]
        )

        # Compute local modes for local upsampling
        """
        n = local_modes_size
        Mx = self.grid_shape_ext[0]
        # The local pad consists of k = -n:n, and the grid is -M/2:(M/2-1)
        # [if M is even] or -(M-1)/2:(M-1)/2 [if M is odd]. Thus, n can
        # at most be M/2-1 [if M is even] or (M-1)/2 [if M is odd].
        # These are both captured by floor((M-1)/2).
        nx = np.minimum(np.floor((Mx - 1) / 2), n)
        # The zeroth index is the zero mode, which should not be included.
        self.local_modes = np.concatenate(
            (np.arange(1, nx + 1), np.arange(Mx - nx, Mx))
        )
        """

        return


def se_compute_grid_size(box, grid_res, base_factor, periodicity):
    """
    Compute grid given the box shape and desired ``grid_res``.

    Parameters
    ----------
    box : array_like
        Size of the primary cell [L1, L2, L3], containing all sources and
        targets.
    grid_res : float
        Desired number of grid subintervals per unit length, i.e., 1/h,
        where h is the grid step size.
    base_factor : int
        All grid sizes will be rounded up to be multiples of ``base_factor``.
    periodicity : {0, 1, 2, 3}
        The number of periodic spatial directions.

    Returns
    -------
    grid_shape : array_like
        Number of grid subintervals [M1, M2, M3].
    grid_res : float
        Actual number of grid subintervals per unit length.

    This code will adjust (increase) ``grid_res`` such that grid is divisible by
    ``base_factor`` in all periodic directions and ``grid_shape/box`` is the
    same in all periodic directions. The new value of ``grid_res`` is returned.
    """
    if periodicity == 0:
        # Nothing to do in 0P case
        grid_shape = grid_res * box
    elif periodicity == 1:
        # Determine number of "ticks" needed
        n = np.ceil(grid_res * box[0] / base_factor)
        # Compute number of subintervals
        M1 = n * base_factor
        # Compute new grid_res
        grid_res = M1 / box[0]
        # Collect grid in all directions
        grid_shape = np.array([M1, grid_res * box[1], grid_res * box[2]])
    elif periodicity == 2:
        # Simplify the fraction L1/L2 = a1/a2
        frac = Fraction(box[0] / box[1])
        a1 = frac.numerator
        a2 = frac.denominator
        # Determine number of "ticks" needed
        n = np.ceil(grid_res * box[0] / (a1 * base_factor))
        # Compute number of subintervals
        M1 = n * a1 * base_factor
        M2 = n * a2 * base_factor
        # Compute new grid_res
        grid_res = M1 / box[0]
        # Collect grid in all directions
        grid_shape = np.array([M1, M2, grid_res * box[2]])
    elif periodicity == 3:
        # First simplify the fraction L1/L2 = c1/c2
        frac = Fraction(box[0] / box[1])
        c1 = frac.numerator
        c2 = frac.denominator
        q = Fraction(box[0] / c1)
        # Then simplify the fraction q/L3 = b/a3
        frac = Fraction(q / box[2])
        b = frac.numerator
        a3 = frac.denominator
        a1 = c1 * b
        a2 = c2 * b
        # Determine number of "ticks" needed
        n = np.ceil(grid_res * box[0] / (a1 * base_factor))
        # Compute number of subintervals
        M1 = n * a1 * base_factor
        M2 = n * a2 * base_factor
        M3 = n * a3 * base_factor
        # Compute new grid_res
        grid_res = M1 / box[0]
        # Collect grid in all directions
        grid_shape = np.array([M1, M2, M3], dtype=np.float64)
    return grid_shape, grid_res
