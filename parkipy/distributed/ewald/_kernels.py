"""
Module for PDE kernels and their
evaluation using ewald summation.

Contains the `pyse.ewald.EwaldKernel` class.
"""

import warnings
import numpy as np
import pykokkos as pk
from dataclasses import dataclass, field

from parkipy.utils import get_array_module, get_execution_space
from parkipy.ewald import EwaldOptions
from parkipy.ewald._ewald import p2p, p2g, fft, cnv, ifft, g2p
from parkipy.ewald._prepare import DevicePre
from ._fft_utils import FFTMPBuffers

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None


class EwaldKernel:
    """
    Represents the Ewald split of a PDE kernel, i.e.,
    a fundamental solution (Green's function).

    Calling an instance of this class will evaluate the kernel
    using the spectral Ewald method.
    """

    def __init__(
        self, name, dim_in, dim_out, kernel, takes_normals=False, description=None
    ):
        """
        Create a new kernel.

        Input:
            - `name`: Name of the kernel (string). This will be used to build
              the `self.key` property and should therefore be unique.
            - `dim_in`: Input dimension, the number of dimensions of the layer
              density (or a point source).
            - `dim_out`: Output dimension, the number of dimensions of the layer
              potential (or potential from a point source).
            - `kernel`: The name of the kernel to be evaluated.
            - `takes_normals`: (bool) The kernel takes the normal vector as an
              argument (default: `False`).
            - `description`: Longer string describing the kernel (optional).
        """
        self._name = name
        self._dim_in = dim_in
        self._dim_out = dim_out
        self._kernel = kernel
        self._takes_normals = takes_normals
        self._description = description

        return

    @property
    def name(self):
        """
        Name of the kernel. Read-only.
        """
        return self._name

    @property
    def dim_in(self):
        """
        Input dimension. Read-only.

        See also `get_shape_in()`.
        """
        return self._dim_in

    @property
    def dim_out(self):
        """
        Output dimension. Read-only.

        See also `get_shape_out()`.
        """
        return self._dim_out

    @property
    def kernel(self):
        """
        Kernel name for evaluation. Read-only.
        """
        return self._kernel

    @property
    def takes_normals(self):
        """
        (bool) True if the kernel takes the normal vector as an argument.
        Read-only.
        """
        return self._takes_normals

    @property
    def description(self):
        """
        Longer string describing the kernel. Read-only.
        """
        return self._description

    @property
    def key(self):
        """
        Return a string suitable for use as a dict key, representing this
        kernel. Read-only.

        The key is formed as `'sph_kernel_' + self.name`.
        """
        return "sph_kernel_" + self.name

    def __call__(
        self,
        *args,
        options,
        cell_size=None,
        rc=None,
        p2p_method="GM-1D",
        p2g_method="HYBRID",
        g2p_method="TARGET",
        buffer_size=None,
        time=False,
        fft_type="R2C",
    ):
        """
        Evaluate the kernel using direct evaluation.

        Input:
            - `x_out`: Points to evaluate the potential at, shape `(3, N_out)`.
              May also be `(3,)` for a single point.
            - `x_in`: Source points, shape `(3, N_in)`.
            - `q_in`: Density values, shape `self.get_shape_in(N_in)`.
            - `n_in`: Only if `self.takes_normals==True`: Normal vectors, shape
              `(3, N_in)`.
            - `periodicity`:
                The number of periodic spatial directions.
            - `box`: Side lengths of computational domain.
               That is, `0 < min(x[i])`, `max(x[i]) < box[i]` for `i=0,1,2`
               and `x=x_in,x_out`.
            - `tol`: Spectral Ewald tolerance.
            - `execution_space`: PyKokkos execution space See
              https://kokkos.org/kokkos-core-wiki/API/core/execution_spaces.html
              for available execution spaces. Input is either a string
              (e.g., 'OpenMP') or a PyKokkos object (e.g., pk.OpenMP).
              Defaults to `pk.get_default_space()`
            - `cell_size`: Real space cell size. Defaults to `None`.
            - `p2p_method`: Must be one of `'GM-1D'`, `'GM-2D'`, `'SM-1D'`,
              or `'SM-2D'`. Defaults to `'GM-1D'`.
            - `p2g_method`: Must be one of `'BASE'`, `'SOURCE'`, `'GRID'`, `'HYBRID'`.
              Defaults to `'HYBRID'`.
            - `g2p_method`: Must be one of `'BASE'`, `'TARGET'`. Defaults to `'TARGET'`.
            - `mpi_comm`: MPI_COMM_WORLD for distributed execution.
            - `buffer_size`: Size of MPI buffers. Only relevant if `mpi_comm` is not `None`.
               Defaults to `2*MPI.Allreduce(N_in, MPI.MAX)`.
            - `time`: Flag to return `walltime` dictionary.

        Returns potential values at `x_out` as an array of shape
        `self.get_shape_out(N_out)`.
        """
        # parse arguments
        if len(args) != 7 + self.takes_normals:
            raise TypeError(
                f"Wrong number of positional arguments, "
                f"got {len(args)}, "
                f"expected {7 + self.takes_normals}"
            )
        x_out = args[0]
        if len(x_out.shape) == 1 and np.size(x_out) == 3:
            x_out = x_out.reshape((3, 1))
        N_out = x_out.shape[1]
        if x_out.shape != (3, N_out):
            raise ValueError(f"x_out must be of shape (3, N_out), found {x_out.shape}")
        x_in = args[1]
        N_in = x_in.shape[1]
        if x_in.shape != (3, N_in):
            raise ValueError(f"x_in must be of shape (3, N_in), found {x_in.shape}")
        q_in = args[2]
        if q_in.shape != self.get_shape_in(N_in):
            raise ValueError(
                f"q_in must be of shape {self.get_shape_in(N_in)}, found {q_in.shape}"
            )
        if q_in.dtype != x_in.dtype:
            raise TypeError(
                "q_in must have the same dtype as x_in,"
                f" got {q_in.dtype}, expected {x_in.dtype}."
            )
        n_in = None
        if self.takes_normals:
            n_in = args[3]
            if n_in.shape != (3, N_in):
                raise ValueError(f"n_in must be of shape (3, N_in), found {n_in.shape}")
        periodicity = args[3 + self.takes_normals]
        box = args[4 + self.takes_normals]
        if isinstance(box, list):
            box = np.asarray(box)
        tol = args[5 + self.takes_normals]
        execution_space = get_execution_space(args[6 + self.takes_normals])
        am = get_array_module(execution_space)
        dtype = x_in.dtype

        ### Spectral Ewald ###
        device_pre, params = DevicePre.from_particles(
            targets=x_out,
            sources=x_in,
            forces=q_in,
            normals=n_in,
            kernel=self.kernel,
            box=box,
            tolerance=tol,
            cell_size=cell_size,
            rc=rc,
            periodicity=periodicity,
            execution_space=execution_space,
            fft_type=fft_type,
            distributed=True,
            scatter=options.scatter,
        )
        # algorithm
        walltime = {}
        walltime["p2p"] = p2p(
            device_pre,
            method=p2p_method,
        )
        walltime["p2g"] = p2g(
            device_pre,
            method=p2g_method,
        )
        # TODO: allow for plan to be passed in if needed
        #   This will allow for speedups for repeated calls
        fft_buffers = FFTMPBuffers(
            fft_shape=device_pre.data.fft_shape,
            ifft_shape=device_pre.data.Hg.shape[1:-1],
            fft_type=device_pre.data.fft_type,
        )
        walltime["fft"] = fft(device_pre, None, fft_buffers)
        walltime["cnv"] = cnv(device_pre)
        walltime["ifft"] = ifft(device_pre, None, fft_buffers)
        del fft_buffers  # free symmetic heap if execution is distributed
        device_pre.data.communicate_ghost_grid_cells()
        walltime["g2p"] = g2p(
            device_pre,
            method=g2p_method,
        )
        walltime.update(
            device_pre.data.walltime
        )  # update walltimes with device_data comms
        val = device_pre.near_potential + device_pre.far_potential
        val = val.squeeze()
        shape = self.get_shape_out(device_pre.data.targets.shape[-1])
        if val.shape != shape:
            raise ValueError(
                f"Kernel function returned wrong shape, found {val.shape}, expected {shape}."
            )

        x_off = device_pre.data.box_dict["left"] - (device_pre.data.opt.ghost_dist)
        trg = device_pre.data.targets
        trg[0] += x_off
        if time:
            return val, trg, walltime
        else:
            return val, trg

    def get_shape_in(self, N_in):
        """
        Return the input shape expected by this kernel for `N_in` source points.

        This will be `(N_in,)` if `self.dim_in==1`, and `(self.dim_in, N_in)`
        otherwise.
        """
        if self.dim_in == 1:
            return (N_in,)
        return (self.dim_in, N_in)

    def get_shape_out(self, N_out):
        """
        Return the output shape expected by this kernel for `N_out` source points.

        This will be `(N_out,)` if `self.dim_out==1`, and `(self.dim_out,N_out)`
        otherwise.
        """
        if self.dim_out == 1:
            return (N_out,)
        return (self.dim_out, N_out)


@dataclass
class DistributedEwaldOptions(EwaldOptions):
    """
    Data class providing options for
    the distributed Ewald kernels.
    Subclass of :class:`EwaldOptions`.

    Parameters
    ----------
    scatter: bool
        Scatter targets and sources across ranks.
        This is an all-to-all operation and assumes
        no structure in the particle distribution.
        If particles are already slab distributed across
        ranks, scatter can be set to false. If false,
        target points are guaranteed to live *in the same order*
        on their original rank. Defaults to `True`.
    """

    scatter: bool = True


## Implementations of specific kernels ##


def stokes_sl(targets, sources, densities, options):
    r"""
    Compute the distributed Stokes single-layer potential.

    .. math::

        \boldsymbol{u}(\boldsymbol{x}_i) = \sum_{j=1}^{N} \sum_{p \in P}
        \left(
            \frac{\boldsymbol{q}_j}{\lVert \boldsymbol{r}_{ij} \rVert}
            +
            \frac{\boldsymbol{r}_{ij} (\boldsymbol{r}_{ij} \cdot \boldsymbol{q}_j)}
                 {\lVert \boldsymbol{r}_{ij} \rVert^3}
        \right),

    where :math:`\boldsymbol{r}_{ij} = \boldsymbol{x}_i - \boldsymbol{y}_j`
    and :math:`P` is the specified periodicity.

    The computation is distributed across MPI ranks. Each rank provides its
    local slice of targets, sources, and densities. If
    ``options.scatter == True`` (the default), particles are redistributed
    across ranks via an all-to-all before the Ewald sum; set it to ``False``
    if the particles are already slab-distributed.

    .. warning::
        Only ``periodicity=1`` is currently supported.

    Parameters
    ----------
    targets : ndarray
        Local target positions :math:`\boldsymbol{x}_i`, shape ``(3, nt_local)``
        with default array ordering (``'C'`` for :mod:`cupy`).

    sources : ndarray
        Local source positions :math:`\boldsymbol{y}_j`, shape ``(3, ns_local)``
        with default array ordering.

    densities : ndarray
        Single-layer density :math:`\boldsymbol{q}_j` at each local source
        point, shape ``(3, ns_local)`` with default array ordering.

    options : DistributedEwaldOptions
        Dataclass specifying Ewald and distributed execution parameters.
        See :class:`DistributedEwaldOptions`.

    Returns
    -------
    potential : ndarray
        Local Stokes single-layer potential at the (redistributed) target
        points, shape ``(3, nt_local_out)``.

    targets : ndarray
        Redistributed target positions after the slab scatter, shape
        ``(3, nt_local_out)``. Coordinates are in the global frame.

    walltime : dict, optional
        Per-stage wall-clock times (``'p2p'``, ``'p2g'``, ``'fft'``,
        ``'cnv'``, ``'ifft'``, ``'g2p'``, and MPI communication stages).
        Only returned if ``options.return_walltime == True``.

    Raises
    ------
    NotImplementedError
        If ``options.periodicity`` is not ``1``.

    Notes
    -----
    Parameter selection based off [1]_.

    References
    ----------
    .. [1] Bagge, J., & Tornberg, A.-K. (2023).
        Fast Ewald summation for Stokes flow with arbitrary periodicity.
        Journal of Computational Physics, 493, 112473.
        https://doi.org/10.1016/j.jcp.2023.112473

    Examples
    --------
    >>> from mpi4py import MPI
    >>> import parkipy
    >>> import numpy as np
    >>> mpi_comm = MPI.COMM_WORLD
    >>> size = mpi_comm.Get_size()
    >>> am = parkipy.utils.get_array_module(
    ...     parkipy.utils.get_execution_space("Cuda")
    ... )
    >>> box = [size, 1, 1]
    >>> ns = nt = 10000
    >>> src  = am.random.rand(3, ns) * am.array(box).reshape(3, 1)
    >>> trg  = am.random.rand(3, nt) * am.array(box).reshape(3, 1)
    >>> dens = am.random.randn(3, ns)
    >>> options = parkipy.distributed.ewald.DistributedEwaldOptions(
    ...     box=box, periodicity=1, tolerance=1e-4,
    ...     execution_space="Cuda", cell_size=224,
    ... )
    >>> pot, trg_out = parkipy.distributed.ewald.stokes_sl(trg, src, dens, options)
    """
    args = [
        targets,
        sources,
        densities,
        options.periodicity,
        options.box,
        options.tolerance,
        options.execution_space,
    ]
    exclude = {
        "box",
        "tolerance",
        "periodicity",
        "execution_space",
        "torch_fft",
        "p2p_threads_x",
        "p2p_threads_y",
        "p2g_threads",
        "g2p_threads",
        "return_walltime",
        "return_params",
        "scatter",
    }
    kwargs = {k: v for k, v in options.__dict__.items() if k not in exclude}
    pot = EwaldKernel(
        name="stokes_comb",
        dim_in=3,
        dim_out=3,
        kernel="stokes_sl",
        takes_normals=False,
        description="Stokes single layer potential.",
    )(*args, options=options, **kwargs)
    return pot


def stokes_comb(*args, **kwargs):
    r"""
    Compute the distributed combined Stokes single and double-layer potential.

    .. math::

        \boldsymbol{u}(\boldsymbol{x}_i) = \sum_{j=1}^{N} \sum_{p \in P}
        \left(
            \frac{\boldsymbol{q}_j}{\lVert \boldsymbol{r}_{ij} \rVert}
            + \frac{\boldsymbol{r}_{ij} (\boldsymbol{r}_{ij} \cdot \boldsymbol{q}_j)}
                   {\lVert \boldsymbol{r}_{ij} \rVert^3}
            + \epsilon_{jlm}
              \frac{(\boldsymbol{r}_{ij})_m}{\lVert \boldsymbol{r}_{ij} \rVert^3}
              (\boldsymbol{q}_l)_{\text{DL}} (\boldsymbol{n}_j)_m
        \right),

    where :math:`\boldsymbol{r}_{ij} = \boldsymbol{x}_i - \boldsymbol{y}_j`
    and :math:`P` is the specified periodicity.

    .. note::
        Unlike :func:`stokes_sl` and the single-device :func:`parkipy.ewald.stokes_comb`,
        this function uses a **positional argument** calling convention — it does not
        accept a :class:`DistributedEwaldOptions` object. Parameters are passed
        directly as positional and keyword arguments to the underlying
        :class:`EwaldKernel`.

    .. warning::
        Only ``periodicity=1`` is currently supported.

    Parameters
    ----------
    trg : ndarray
        Local target positions :math:`\boldsymbol{x}_i`, shape ``(3, nt_local)``
        with default array ordering (``'C'`` for :mod:`cupy`).

    src : ndarray
        Local source positions :math:`\boldsymbol{y}_j`, shape ``(3, ns_local)``
        with default array ordering.

    dens : ndarray
        Stacked density array of shape ``(6, ns_local)``, where ``dens[:3]``
        is the single-layer density :math:`\boldsymbol{q}_j` and ``dens[3:]``
        is the double-layer density.

    norms : ndarray
        Surface normal vectors at each source point, shape ``(3, ns_local)``
        with default array ordering.

    periodicity : int
        Number of periodic spatial directions. Must be ``1``.

    box : list[float]
        Global computational box side lengths ``[Lx, Ly, Lz]``. The periodic
        length ``Lx`` must be divisible by the number of MPI ranks.

    tol : float
        Tolerance for Ewald parameter selection.

    execution_space : str
        Kokkos execution space string, e.g. ``'Cuda'``.

    rc : float, optional
        Near-field cutoff radius. Either ``rc`` or ``cell_size`` must be
        provided.

    cell_size : int, optional
        Near-field cell size. Used to compute ``rc`` if ``rc`` is not given.

    time : bool, optional
        If ``True``, return per-stage wall-clock times as a third output.
        Default is ``False``.

    Returns
    -------
    potential : ndarray
        Local combined Stokes potential at the (redistributed) target points,
        shape ``(3, nt_local_out)``.

    targets : ndarray
        Redistributed target positions after the slab scatter, shape
        ``(3, nt_local_out)``. Coordinates are in the global frame.

    walltime : dict, optional
        Per-stage wall-clock times (``'p2p'``, ``'p2g'``, ``'fft'``,
        ``'cnv'``, ``'ifft'``, ``'g2p'``, and MPI communication stages).
        Only returned if ``time=True``.

    Raises
    ------
    NotImplementedError
        If ``periodicity`` is not ``1``.

    Notes
    -----
    Parameter selection based off [1]_.

    References
    ----------
    .. [1] Bagge, J., & Tornberg, A.-K. (2023).
        Fast Ewald summation for Stokes flow with arbitrary periodicity.
        Journal of Computational Physics, 493, 112473.
        https://doi.org/10.1016/j.jcp.2023.112473

    Examples
    --------
    >>> from mpi4py import MPI
    >>> import parkipy
    >>> import numpy as np
    >>> mpi_comm = MPI.COMM_WORLD
    >>> size = mpi_comm.Get_size()
    >>> am = parkipy.utils.get_array_module(
    ...     parkipy.utils.get_execution_space("Cuda")
    ... )
    >>> box = [size, 1, 1]
    >>> ns = nt = 10000
    >>> rc = np.ceil(ns / 224) ** (-1 / 3)
    >>> src     = am.random.rand(3, ns) * am.array(box).reshape(3, 1)
    >>> trg     = am.random.rand(3, nt) * am.array(box).reshape(3, 1)
    >>> dens_sl = am.random.randn(3, ns)
    >>> dens_dl = am.random.randn(3, ns)
    >>> norms   = am.random.randn(3, ns)
    >>> dens    = am.vstack((dens_sl, dens_dl))
    >>> pot, trg_out, timing = parkipy.distributed.ewald.stokes_comb(
    ...     trg, src, dens, norms, 1, box, 1e-4, "Cuda", rc=rc, time=True,
    ... )
    """

    pot = EwaldKernel(
        name="stokes_comb",
        dim_in=6,
        dim_out=3,
        kernel="stokes_comb",
        takes_normals=True,
        description="Stokes single and double layer potential.",
    )(*args, **kwargs)
    return pot
