"""
Module for PDE kernels and their
evaluation using ewald summation.

Contains the `pyse.ewald.EwaldKernel` class.
"""

import warnings
import numpy as np
import pykokkos as pk
from dataclasses import dataclass, field
from typing import Literal, List, Union

from parkipy.utils import get_array_module, get_execution_space
from ._ewald import p2p, p2g, fft, cnv, ifft, g2p
from ._prepare import DevicePre
from ._params import SEParams


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
        cell_size=None,
        rc=None,
        p2p_method="GM-1D",
        p2g_method="HYBRID",
        g2p_method="TARGET",
        p2p_threads_x=128,
        p2p_threads_y=1,
        p2g_threads=128,
        g2p_threads=128,
        buffer_size=None,
        return_walltime=False,
        return_params=False,
        fft_type="R2C",
        fourier_upsampling_factor_global=None,
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
                f"Wrong number of positional arguments, expected"
                f" {7 + self.takes_normals}"
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
            fourier_upsampling_factor_global=fourier_upsampling_factor_global,
        )
        # algorithm
        walltime = {}
        walltime["p2p"] = p2p(
            device_pre,
            method=p2p_method,
            threads_x=p2p_threads_x,
            threads_y=p2p_threads_y,
        )
        walltime["p2g"] = p2g(
            device_pre,
            method=p2g_method,
            threads=p2g_threads,
        )
        walltime["fft"] = fft(device_pre)
        walltime["cnv"] = cnv(device_pre)
        walltime["ifft"] = ifft(device_pre)
        device_pre.data.communicate_ghost_grid_cells()
        walltime["g2p"] = g2p(
            device_pre,
            method=g2p_method,
            threads=g2p_threads,
        )
        val = device_pre.near_potential + device_pre.far_potential
        val = val.squeeze()
        shape = self.get_shape_out(device_pre.data.targets.shape[-1])
        if val.shape != shape:
            raise ValueError(
                f"Kernel function returned wrong shape, found {val.shape}, expected {shape}."
            )
        out = [val]
        if return_walltime:
            out.append(walltime)
        if return_params:
            out.append(params)
        out = tuple(out)
        if len(out) == 1:
            out = out[0]
        return out

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


## kernel options class


@dataclass
class EwaldOptions:
    """
    Data class for providing options to the
    :func:`stokes_comb` and :func:`laplace` Ewald kernels.

    Parameters
    ----------
    box: List[float]
        Computational box lengths for Ewald summation.
        The computational box is centered at the origin with
        size lengths ``i=0,1,2`` given by ``box[i]``.

    periodicity: {'0','1','2','3'}
        Periodicity of the computational box. Periodicity ``j``
        means that ``box[i]`` is a periodic length for all ``i<j``,
        with ``j=0`` indicating free space.

    tolerance: float
        Tolerance for Ewald summation. Used to set internal Ewald parameters.

    execution_space: `pykokkos.ExecutionSpace` | {'CUDA', 'HIP', 'OPENMP'}
        Device for the Kokkos backend. May be pykokkos execution space type or a string.

    p2p_method: {'GM-1D', 'GM-2D', 'SM-1D', 'SM-2D'} , optional
        The algorithmic method for `p2p`,
        one of ``'GM-1D'``, ``'GM-2D'``, ``'SM-1D'``, or ``'SM-2D'``.
        All methods read both sources and targets in :class:`parkipy.CellList`
        ordering.
        ``'GM'``/``'SM'`` determines if the sources are read from
        global-memory/shared-memory, while ``'1D'``/``'2D'`` determines
        whether parallelisation is soley over the targets (1D)
        or over the targets and the sources (2D). The default is
        ``'GM-1D'``.

    p2g_method: {'BASE', 'SOURCE', 'GRID', 'HYBRID'}, optional
        The algorithmic method for `p2g`, one of ``'BASE'``,
        ``'SOURCE'``, ``'GRID'``, or ``'HYBRID'``. The ``'BASE'`` method
        does a simple OpenMP syle parallelization over the *unordered* source points,
        ``'SOURCE'`` does the same parallelization but with sources in a :class:`parkipy.CellList`
        ordering, ``'GRID'`` does a parallelization over
        the output grid points, and ``'HYBRID'`` does a parallel read of the sources,
        calls a synchronization barrier, and does a parallel write over the grid points.
        The default is ``'HYBRID'``.

    g2p_method: {'BASE', 'TARGET'}, optional
        The algorithmic method for `g2p`, one of ``'BASE'`` or ``'TARGET'``.
        Both ``'BASE'`` and ``'TARGET'`` are parallel over the output target points;
        the ``'BASE'`` method reads over the targets in an unordered fashion while
        the ``'TARGET'`` method reads over the targets in a :class:`parkipy.CellList`
        order. The default is ``'TARGET'``.

    fft_type: {'R2C', 'C2C'}, optional
        The type of fft to preform. Must be one of ``'R2C'`` or ``'C2C'``. The default is ``'R2C'``.

    p2p_threads_x: int, optional
        Number of threads per block used to parallelize over the targets in the p2p algorithm.
        Resets to ``1`` for the OpenMP execution space.
        The default is ``128``.

    p2p_threads_y: int, optional
        Number of threads per block used to parallelize over the sources in the p2p algorithm.
        Resets to ``1`` for `1D` methods and for the OpenMP execution space. The default is ``1``.

    p2g_threads: int, optional
        Number of threads per block used to parallelize the p2g method. The default is ``128``.

    g2p_threads: int, optional
        Number of threads per block used to parallelize the g2p method. The default is ``128``.

    cell_size: int | None, optional
        Near field cell size. If ``None``, ``rc`` must be provided. The defaults to ``None``.

    rc: float | None, optional
        Near field cutoff radius. If ``None``, ``cell_size`` must be provided. The defaults to ``None``.

    return_walltime: bool, optional
        Flag to return the walltime dict of Ewald stage wall-clock times. If true, the ``walltime`` dict
        will be the second item returned for a kernel call. The default is ``False``.

    return_params: bool, optional
        Flag to return the ``params`` struct for the parameters of the Ewald algorithm.
        If, true, the ``params`` struct will be the last item returned for a kernel call.
        The default is ``False``.
    """

    # required arguments
    box: List[float]
    periodicity: Literal[0, 1, 2, 3]
    tolerance: float
    execution_space: Union[pk.ExecutionSpace, Literal["CUDA", "HIP", "OPENMP"]]

    # default arguments
    p2p_method: Literal["GM-1D", "GM-2D", "SM-1D", "SM-2D"] = "GM-1D"
    p2g_method: Literal["BASE", "SOURCE", "GRID", "HYBRID"] = "HYBRID"
    g2p_method: Literal["BASE", "TARGET"] = "TARGET"
    fft_type: Literal["R2C", "C2C"] = "R2C"
    p2p_threads_x: int = 128
    p2p_threads_y: int = 1
    p2g_threads: int = 128
    g2p_threads: int = 128
    cell_size: int | None = None
    rc: float | None = None
    return_walltime: bool = False
    return_params: bool = False

    def __post_init__(self):
        if not isinstance(self.box, list) or not isinstance(self.box, np.ndarray):
            raise TypeError(
                f"box expected to be list of numpy array, got type {type(self.box)}."
            )
        for box_len in self.box:
            if box_len <= 0:
                raise ValueError("box lengths must be positive.")

        valid_periodicities = [0, 1, 2, 3]
        if self.periodicity not in valid_periodicities:
            raise ValueError(f"the periodicity must be one of {valid_periodicities}.")

        if not isinstance(self.tolerance, float) or self.tolerance <= 0:
            raise ValueError("tolerance must be a positive float.")

        if not isinstance(self.execution_space, pk.ExecutionSpace):
            valid_execution_spaces = ["CUDA", "HIP", "OPENMP"]
            if self.execution_space.upper() not in valid_execution_spaces:
                raise ValueError(
                    f"the execution space must be one of {valid_execution_spaces}."
                )

        valid_p2p_methods = ["GM-1D", "GM-2D", "SM-1D", "SM-2D"]
        if self.p2p_method is None:
            self.p2p_method = "GM-1D"
        if self.p2p_method.upper() not in valid_p2p_methods:
            raise ValueError(
                f"the value specified for p2p method must be one of {valid_p2p_methods}, got {self.p2p_method.upper()}."
            )

        valid_p2g_methods = ["BASE", "SOURCE", "GRID", "HYBRID"]
        if self.p2g_method is None:
            self.p2g_method = "HYBRID"
        if self.p2g_method.upper() not in valid_p2g_methods:
            raise ValueError(
                f"the value specified for p2g method must be one of {valid_p2g_methods}, got {self.p2g_method.upper()}."
            )

        valid_g2p_methods = ["BASE", "TARGET"]
        if self.g2p_method is None:
            self.g2p_method = "TARGET"
        if self.g2p_method.upper() not in valid_g2p_methods:
            raise ValueError(
                f"the value specified for g2p method must be one of {valid_g2p_methods}, got {self.g2p_method.upper()}."
            )

        valid_fft_types = ["R2C", "C2C"]
        if self.fft_type.upper() not in valid_fft_types:
            raise ValueError(
                f"the value specified for fft type must be one on {valid_fft_types}, got {self.fft_type.upper()}."
            )

        for threads in [
            self.p2p_threads_x,
            self.p2p_threads_y,
            self.p2g_threads,
            self.g2p_threads,
        ]:
            if not isinstance(threads, int) or threads <= 0:
                raise ValueError("thread count must be a positive int.")

        if self.cell_size is None and self.rc is None:
            raise ValueError("one of cell size or rc must be provided.")

        if not isinstance(self.return_walltime, bool):
            raise ValueError("walltime flag must be `True` or `False`.")

        if not isinstance(self.return_params, bool):
            raise ValueError("params flag must be `True` or `False`.")


## Implementations of specific kernels ##


def stokes_sl(trg, src, dens, options):
    """
    Compute the Stokes single layer potential
    """
    valid_periodicities = [0, 1, 2, 3]
    if options.periodicity not in valid_periodicities:
        raise NotImplementedError(
            f"stokes sl only supports periodicities {valid_periodicities}, got {options.periodicity}."
        )
    args = [
        trg,
        src,
        dens,
        options.periodicity,
        options.box,
        options.tolerance,
        options.execution_space,
    ]
    exclude = {"box", "tolerance", "periodicity", "execution_space"}
    kwargs = {k: v for k, v in options.__dict__.items() if k not in exclude}
    pot = EwaldKernel(
        name="stokes_sl",
        dim_in=3,
        dim_out=3,
        kernel="stokes_sl",
        takes_normals=False,
        description="Stokes single layer potential.",
    )(*args, **kwargs)
    return pot


def stokes_comb(trg, src, dens, normal, options):
    r"""
    Compute the combined Stokes single and double layer potential.

    .. math::

        \boldsymbol{u}(\boldsymbol{x}_i) = \sum_{j=1}^{N} \sum_{p \in P}
        \left(\left( \frac{\boldsymbol{q}_j}{\lVert \boldsymbol{r}_{ij} \rVert}
        +
        \frac{\boldsymbol{r}_{ij}}{\lVert \boldsymbol{r}_{ij} \rVert^3}
        (\boldsymbol{r}_{ij} \cdot \boldsymbol{q}_j)
        \right)
        + \left( \epsilon_{jlm} \frac{\boldsymbol{r}_m}{\|\boldsymbol{r}_{ij}\|^3}
        \boldsymbol{q}_l \boldsymbol{n}_m \right) \right),

    where :math:`\boldsymbol{r}_{ij} = \boldsymbol{x}_i - \boldsymbol{y}_j`
    and :math:`P` is the specified periodicity.

    .. warning:: Currently only supported for :math:`P=1,3`.

    Parameters
    __________
    trg: ndarray
        Array of target values :math:`\boldsymbol{x}_i` to compute the Stokes potential.
        Array should be shape ``(3,nt)`` with the default ordering (i.e., ``'C'`` ordering for
        `cupy` and ``'F'`` ordering for `numpy`.

    src: ndarray
        Array of source values :math:`\boldsymbol{y}_j` of shape ``(3, ns)`` with default ordering.

    dens: ndarray
        Array of density values of the source points of shape ``(6,ns)`` with default ordering where ``dens[:3]``
        represent the single layer density :math:`\boldsymbol{q}_j`
        and ``dens[3:]`` the double layer density :math:`\boldsymbol{q}_l`.

    normal: ndarray
        Array of normal vectors for the source points of shape ``(3,ns)`` with default ordering.

    options: EwaldOptions
        Dataclass specifying Ewald parameters.

    Returns
    _______
    potential: ndarray
        Array containing the Ewald approximation for the
        Stokes potential at the target points :math:`\boldsymbol{u}(\boldsymbol{x}_i)`.

    walltimes: dict, optional
        Dictionary containing the wall-time for each stage of the Ewald summation.
        Only returned if ``options.return_walltime == True``.

    params: SEParams, optional
        :class:`SEParams` dataclass containing derived Ewald parameters for the Ewald summation run.
        Only returned if ``options.return_params == True``.

    Raises
    ______
    NotImplementedError
        If ``options.periodicity`` is not ``1`` or ``3``.

    Notes
    -----
    Parameter selection based off [1]_.

    References
    __________
    .. [1] Bagge, J., & Tornberg, A.-K. (2023).
        Fast Ewald summation for Stokes flow with arbitrary periodicity.
        Journal of Computational Physics, 493, 112473. https://doi.org/10.1016/j.jcp.2023.112473


    Examples
    ________
    >>> import parkipy
    >>> import numpy as np
    >>> trg = src = np.random.rand(3, 100000)
    >>> dens_sl = np.random.randn(3, 100000)
    >>> dens_dl = np.random.randn(3, 100000)
    >>> norms = np.random.randn(3, 100000)
    >>> dens = np.vstack((dens_sl, dens_dl))  # stack densities for ewald call
    >>> options = parkipy.ewald.EwaldOptions(
    ... box=[1,1,1], periodicity=1, tolerance=1e-4,
    ... execution_space="openmp", cell_size=224, return_walltime=True
    ... )
    >>> pot, walltime = parkipy.ewald.stokes_comb(trg, src, dens, norms, options)
    >>> walltime
    {'p2p': {'args': 3.0994415283203125e-06, 'sort': 0.004508018493652344, 'kernel': 0.08536601066589355, 'tot': 0.08987712860107422}, 'p2g': {'args': 8.106231689453125e-06, 'sort': 0.002873659133911133, 'kernel': 0.06609272956848145, 'tot': 0.06897449493408203}, 'fft': {'tot': 0.05228924751281738}, 'cnv': {'tot': 0.025938749313354492}, 'ifft': {'tot': 0.010168075561523438}, 'g2p': {'args': 2.4318695068359375e-05, 'sort': 0.0019330978393554688, 'kernel': 0.0029044151306152344, 'adj': 0.0005638599395751953, 'tot': 0.005425691604614258}}

    """

    valid_periodicities = [0, 1, 2, 3]
    if options.periodicity not in valid_periodicities:
        raise NotImplementedError(
            f"stokes comb only supports periodicities {valid_periodicities}, got {options.periodicity}."
        )
    args = [
        trg,
        src,
        dens,
        normal,
        options.periodicity,
        options.box,
        options.tolerance,
        options.execution_space,
    ]
    exclude = {"box", "tolerance", "periodicity", "execution_space"}
    kwargs = {k: v for k, v in options.__dict__.items() if k not in exclude}
    pot = EwaldKernel(
        name="stokes_comb",
        dim_in=6,
        dim_out=3,
        kernel="stokes_comb",
        takes_normals=True,
        description="Stokes single and double layer potential.",
    )(*args, **kwargs)
    return pot


def laplace(trg, src, charge, options):
    r"""
    Compute the Laplace potential.

    .. math::

        \boldsymbol{u}(\boldsymbol{x}_i) = \sum_{j=1}^{N} \sum_{p \in P}
        \frac{q_j}{|\boldsymbol{x_j} - \boldsymbol{x_i} + P|}

    where `P` is the specified periodicity.

    We assume charge nutrality (i.e., :math:`\sum_j q_j = 0`).

    .. warning:: Currently only supported for :math:`P=3`.

    Parameters
    __________
    trg: ndarray
        Array of target values :math:`\boldsymbol{x}_i` to compute the Stokes potential.
        Array should be shape ``(3,nt)`` with the default ordering (i.e., ``'C'`` ordering for
        `cupy` and ``'F'`` ordering for `numpy`.

    src: ndarray
        Array of source values :math:`\boldsymbol{y}_j` of shape ``(3, ns)`` with default ordering.

    charge: ndarray
        Array of point charges for the source points of shape ``(ns)`` with default ordering.
        Our error estimates assume charge neutrality, i.e., ``charges.mean < eps``, where ``eps``
        is small.

    options: EwaldOptions
        Dataclass specifying Ewald parameters.

    Returns
    _______
    potential: ndarray
        Array containing the Ewald approximation for the
        Laplace potential at the target points :math:`\boldsymbol{u}(\boldsymbol{x}_i)`.

    walltimes: dict, optional
        Dictionary containing the wall-time for each stage of the Ewald summation.
        Only returned if ``options.return_walltime == True``.

    params: SEParams, optional
        :class:`SEParams` dataclass containing derived Ewald parameters for the Ewald summation run.
        Only returned if ``options.return_params == True``.

    Raises
    ______
    NotImplementedError
        If ``options.periodicity`` is not ``3``, if ``options.p2p_method`` is not ``'GM-1D'``,
        or if ``options.p2g_method`` is not ``'HYBRID'``.

    Notes
    -----
    Parameter selection based off [1]_.

    References
    __________
    .. [1] Shamshirgar, D. S., Bagge, J., & Tornberg, A.-K. (2021).
        Fast Ewald summation for electrostatic potentials with arbitrary periodicity.
        The Journal of Chemical Physics, 154(16), 164109. https://doi.org/10.1063/5.0044895

    Examples
    ________
    >>> import parkipy
    >>> import numpy as np
    >>> nt = ns = 10000
    >>> trg = src = np.random.rand(3, nt)
    >>> charge = np.random.randn(ns)
    >>> charge -= np.mean(charge)
    >>> options = parkipy.ewald.EwaldOptions(
    ... box=[1,1,1], periodicity=3, tolerance=1e-4,
    ... execution_space="openmp", cell_size=224, return_walltime=True
    ... )
    >>> pot, walltime = parkipy.ewald.laplace(trg, src, charge, options)
    >>> walltime
    {'p2p': {'args': 4.291534423828125e-06, 'sort': 0.008143424987792969, 'kernel': 0.2980661392211914, 'tot': 0.3062138557434082}, 'p2g': {'args': 6.604194641113281e-05, 'sort': 0.0052490234375, 'kernel': 0.35875630378723145, 'tot': 0.3640713691711426}, 'fft': {'tot': 0.0070722103118896484}, 'cnv': {'tot': 0.016383886337280273}, 'ifft': {'tot': 0.0004782676696777344}, 'g2p': {'args': 3.647804260253906e-05, 'sort': 0.0011799335479736328, 'kernel': 0.24137353897094727, 'adj': 4.0531158447265625e-06, 'tot': 0.24259400367736816}}

    """

    valid_periodicities = [0, 1, 2, 3]
    if options.periodicity not in valid_periodicities:
        raise NotImplementedError(
            f"laplace only supports periodicities {valid_periodicities}, got {options.periodicity}."
        )

    valid_p2p_methods = ["GM-1D"]
    if options.p2p_method.upper() not in valid_p2p_methods:
        raise NotImplementedError(
            f"laplace only supports p2p methods {valid_p2p_methods}, got {options.p2p_method.upper()}."
        )

    valid_p2g_methods = ["HYBRID"]
    if options.p2g_method.upper() not in valid_p2g_methods:
        raise NotImplementedError(
            f"laplace only supports p2g methods {valid_p2g_methods}, got {options.p2g_method.upper()}."
        )

    args = [
        trg,
        src,
        charge,
        options.periodicity,
        options.box,
        options.tolerance,
        options.execution_space,
    ]
    exclude = {"box", "tolerance", "periodicity", "execution_space"}
    kwargs = {k: v for k, v in options.__dict__.items() if k not in exclude}
    pot = EwaldKernel(
        "laplace",
        1,
        1,
        "laplace",
        takes_normals=False,
        description="Laplace layer potential.",
    )(*args, **kwargs)
    return pot
