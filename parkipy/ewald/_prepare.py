"""
Preallocated structs (dataclasses)
for the Spectral Ewald algorithm
"""

__all__ = ["Options", "SEPre", "DevicePre", "DeviceData"]

import os
import math
import time
import warnings
import pykokkos as pk
from typing import Any, Literal, List
from dataclasses import dataclass, field

from parkipy.utils import get_array_module, get_execution_space

from ._pk_kernels._p2g_workunits import (
    p2g_base_fp32,
    p2g_base_fp64,
    p2g_source_fp32,
    p2g_source_fp64,
    p2g_grid_fp32,
    p2g_grid_fp64,
    p2g_hybrid_fp32,
    p2g_hybrid_fp64,
    p2g_grid_sans_normals_fp32,
    p2g_grid_sans_normals_fp64,
    p2g_hybrid_sans_normals_fp32,
    p2g_hybrid_sans_normals_fp64,
)
from ._pk_kernels._g2p_workunits import (
    g2p_base_fp32,
    g2p_base_fp64,
    g2p_target_fp32,
    g2p_target_fp64,
)
from ._pk_kernels._cnv_workunits import (
    convolution_sum_sl_dl,
    convolution_sum_sl,
    laplace_convolution_kernel_fp64,
    laplace_convolution_kernel_fp32,
    stokeslet_convolution_kernel_fp64,
    stresslet_convolution_kernel_fp64,
    stokeslet_convolution_zero_kernel_fp64,
    stresslet_convolution_zero_kernel_fp64,
    stokeslet_convolution_kernel_fp32,
    stresslet_convolution_kernel_fp32,
    stokeslet_convolution_zero_kernel_fp32,
    stresslet_convolution_zero_kernel_fp32,
)

from ._params import se_params_stokes_comb, se_params_laplace, se_params_stokeslet


@dataclass
class Options:
    """
    Data class for configuring the Spectral Ewald algorithm and output options.

    Parameters
    ----------
    box : list
        Size of the primary cell [L1, L2, L3], containing all sources and targets (required).
    periodicity : {0, 1, 2, 3}
        The number of periodic spatial directions.
    xi : float
        Ewald decomposition parameter (required).
    rc : float
        Ewald cutoff radius (required).
    h: float
        Grid step size
    box_off : List[float]
        Extended primary cell offset in the x,y,z directions.
    grid_shape_ext : array_like
        Size of the extended grid [Mex1, Mex2, Mex3].
    grid_res : float
        Number of grid subintervals per unit length, i.e., 1/(grid step size) (required).
    window_P : int
        Window function support size, measured in number of grid subintervals (required).
    window : str, optional
        Window function (default: 'kaiser_poly'). Valid choices are 'kaiser_exact' and 'kaiser_poly'.
    window_shape_factor : float, optional
        Shape parameter (alpha or beta) divided by window_P (default: 2.5). Cannot be varied for the 'kaiser_poly' window.
    window_scaling_power : int, optional
        Exponent used in the convolution step (default: 2).
        Return Fourier coefficients after convolution step (default: False).
    stokeslet_k0_constant : int, optional
        Value of c in the 2D biharmonic Green's function -r^2 (log(r) - c), used in the Fourier space zero mode (default: 0).
    grid_shape_ext : array_like
        Size of the extended grid [Mex1, Mex2, Mex3].
    glb_grid_shape_ext : array_like
        Size of the extended grid [Mex1, Mex2, Mex3] for the global domain.
    glb_box_ext : array_like
        Size of the extended primary cell [Lex1, Lex2, Lex3] for the global domain.
    distributed : bool
        Flag used if part of a distributed setup. (default: False)

    Attributes
    ----------
    ghost_dist: float
        multiple of `h` used to determine number of ghost points.
        Defaults to`0` if `mpi_comm` is None.
    ghost_box : list
        Size of the primary cell [L1 + 2 * ghost_dist, L2, L3],
        containing all sources, ghost sources, and targets (required).
    ghost_grid_shape_ext : array_like
        Size of the extended grid
        when accounting for ghost points [Mex1 + 2 * grid_res * ghost_dist, Mex2, Mex3].

    """

    box: list
    periodicity: Literal[0, 1, 2, 3]
    xi: float
    rc: float
    h: int
    box_off: list[float]
    grid_shape_ext: list[int]
    grid_res: float
    window_P: int
    glb_box: list
    glb_box_ext: list
    glb_grid_shape_ext: list
    distributed: bool = False
    window: str = "kaiser_poly"
    window_shape_factor: float = 2.5
    window_scaling_power: int = 2
    stokeslet_k0_constant: int = 0
    actual_upsampling_global: list | None = None
    local_modes1: list | None = None

    def __post_init__(self):
        if self.distributed:
            from mpi4py import MPI

            mpi_comm = MPI.COMM_WORLD
            near_ghost_dist = math.ceil(self.rc / self.h / 2)
            # need to scale the ghost dist for p2g grid/hybrid cell lists to work
            scaled_near = int(
                math.ceil(near_ghost_dist / (self.window_P / 2)) * (self.window_P / 2)
            )
            self.ghost_dist = max(scaled_near, math.ceil(self.window_P / 2 / 2)) * (
                2 * self.h
            )
            self.k1_off = (
                round(self.grid_shape_ext[1] * float(self.actual_upsampling_global[-2]))
                // mpi_comm.size
            ) * mpi_comm.rank
        else:
            self.ghost_dist = 0
            self.k1_off = 0
        self.ghost_dist_grid = int(self.ghost_dist * self.grid_res)
        self.ghost_box = [self.box[0] + 2 * self.ghost_dist, self.box[1], self.box[2]]
        self.ghost_grid_shape_ext = [int(self.grid_shape_ext[i]) for i in range(3)]
        self.ghost_grid_shape_ext[0] += 2 * self.ghost_dist_grid

    @classmethod
    def from_params(cls, params, execution_space):
        """
        Construct an `Options` object from minimal input: `params`, `box`, and `local_modes1`.

        Parameters
        ----------
        params : Namespace or object
            Must contain attributes: periodicity, xi, rc, box_off, grid_res,
            grid_shape_ext, window_P, actual_upsampling, stokeslet_k0_constant.
        execution_space : Any
            Execution space enum (e.g., pk.ExecutionSpace.Cuda).

        Returns
        -------
        Options
            Fully initialized Options object.
        """
        am = get_array_module(execution_space)

        return cls(
            periodicity=params.periodicity,
            box=params.box,
            xi=params.xi,
            rc=params.rc,
            h=params.h,
            box_off=params.box_off,
            grid_res=params.grid_res,
            grid_shape_ext=params.grid_shape_ext,
            window_P=params.window_P,
            actual_upsampling_global=am.array(params.actual_upsampling),
            distributed=params.distributed,
            glb_box=am.array(params.glb_box),
            glb_box_ext=am.array(params.glb_box_ext),
            glb_grid_shape_ext=am.array(params.glb_grid_shape_ext, dtype=am.int32),
        )


@dataclass
class SEPre:
    """
    Data class for storing precomputed parameters used during Spectral Ewald setup.

    These values are derived from input parameters and used to define window scaling,
    grid extensions, and Green's function truncation in Fourier space.

    Parameters
    ----------
    box_ext : list of int
        Dimensions of the extended primary cell used to account for window support.
    window_scaling_power : int
        Exponent applied to the window function during scaling (e.g., 2 for squared Kaiser window).
    window_shape : float
        Window shape parameter (e.g., Kaiser–Bessel alpha or beta).
    kaiser_scaling : float
        Scaling constant for the Kaiser window function.
    grid_ext : list of int
        Dimensions of the extended uniform grid [Mex1, Mex2, Mex3].
    window_halfwidth : float
        Half the window support size, i.e., P / 2 * h.
    greens_truncation_R : float
        Truncation radius for the Green’s function in Fourier space.
    vico_var : int
        Internal flag for choosing a variant of the Vico quadrature or spectral method.
    """

    window_scaling_power: int
    window_shape: float
    kaiser_scaling: float
    window_halfwidth: float
    greens_truncation_R: float
    vico_var: int

    @classmethod
    def from_params(
        cls, params, execution_space, *, vico_var=2, window_scaling_power=2
    ):
        """
        Construct an `SEPre` object from minimal input: `params` and `execution_space`.

        Parameters
        ----------
        params : object
            Must contain the fields: box_ext, window_shape, kaiser_scaling,
            grid_shape_ext, h, window_P, greens_truncation_R.
        execution_space : Any
            The execution space (used to select array module for type compatibility).
        vico_var : int, optional
            Variant parameter for Vico quadrature method (default: 2).
        window_scaling_power : int, optional
            Scaling exponent for the window function (default: 2).

        Returns
        -------
        SEPre
            A fully initialized SEPre instance.
        """
        am = get_array_module(execution_space)
        return cls(
            window_scaling_power=window_scaling_power,
            window_shape=params.window_shape,
            kaiser_scaling=params.kaiser_scaling,
            window_halfwidth=params.h * (0.5 * params.window_P),
            greens_truncation_R=params.greens_truncation_R,
            vico_var=vico_var,
        )


@dataclass
class DeviceData:
    """
    Data class to manage device-side arrays and preallocated memory for the Spectral Ewald computation.

    This object holds particle data, grid information, upsampling metadata, and FFT working arrays
    used for GPU-accelerated execution. It also handles precision-specific casting of inputs.

    Parameters
    ----------
    sources : array_like
        Source point coordinates.
    targets : array_like
        Target point coordinates.
    forces : array_like
        Single-layer source strengths.
    normals : array_like
        Normal vectors at the source points. (Optional)
    dim_in : int
        Number of input fields per point (e.g., 12 for Stokeslet/stresslet).
    opt : Options
        Main configuration options.
    opt_sc_glob : SEPre
        Precomputed parameters for global upsampling.
    opt_sc_zero : SEPre
        Precomputed parameters for zero-mode upsampling.
    execution_space : str
        Identifier for the execution backend ("cuda", "openmp", etc.).
    fft_type: str
        Specify the fft type, either complex-to-complex (C2C) or real-to-complex (R2C)
    mpi_buffer_size: int
        Size of the mpi_buffers if `self.opt.distributed == True`.
        Defaults to `None`. If `None`, default to `2 * max(ns,nt)`.

    Attributes
    ----------
    dtype : type
        Data type to use (e.g., numpy.float64 or cupy.float64). Infered from x.dtype
    complex_dtype : type
        Complex type corresponding to `dtype`.
    fft_shape : list[int]
        Final shape used for real-to-complex FFT.
    Hg, Hl, H0 : array_like
        Main Fourier-space work arrays for global/local/zero modes.
    walltime: dict
        walltimes for different operations
    """

    sources: Any
    targets: Any
    forces: Any
    normals: Any
    dim_H: int
    opt: Any
    opt_sc_glob: Any
    opt_sc_zero: Any
    fft_type: str
    execution_space: str
    mpi_buffer_size: int = None
    box_dict: dict = None
    mpi_buffers: Any = None

    # Fields initialized in __post_init__
    dtype: Any = field(init=False)
    complex_dtype: Any = field(init=False)
    H: Any = field(init=False)
    Hg: Any = field(init=False)
    Hl: Any = None
    H0: Any = None
    fft_shape: list[int] = field(init=False)
    ghost_H: Any = field(init=False)
    walltime: dict = field(init=False)

    def __post_init__(self):
        self.walltime = {}
        am = get_array_module(self.execution_space)
        self.execution_space = get_execution_space(self.execution_space)

        # reshape forces if needed
        if len(self.forces.shape) == 1:
            self.forces = self.forces.reshape(1, -1)

        self.dtype = self.sources.dtype
        match self.dtype:
            case am.double:
                self.complex_dtype = am.complex128
            case am.single:
                self.complex_dtype = am.complex64
                self.opt.glb_box_ext = self.opt.glb_box_ext.astype(am.single)
                self.opt.xi = am.single(self.opt.xi)
                self.opt_sc_glob.window_shape = am.single(self.opt_sc_glob.window_shape)
                self.opt_sc_zero.window_shape = am.single(self.opt_sc_zero.window_shape)
                self.opt_sc_glob.kaiser_scaling = am.single(
                    self.opt_sc_glob.kaiser_scaling
                )
                self.opt_sc_zero.kaiser_scaling = am.single(
                    self.opt_sc_zero.kaiser_scaling
                )
                self.opt_sc_glob.window_halfwidth = am.single(
                    self.opt_sc_glob.window_halfwidth
                )
                self.opt_sc_zero.window_halfwidth = am.single(
                    self.opt_sc_zero.window_halfwidth
                )
                self.opt_sc_zero.greens_truncation_R = am.single(
                    self.opt_sc_zero.greens_truncation_R
                )
            case _:
                raise ValueError(
                    f"Expected dtype to be 'single' or 'double', received '{self.dtype}'."
                )
        self.opt.grid_res = self.opt.grid_res.astype(self.dtype)

        Mx, My, Mz = self.opt.grid_shape_ext

        L1 = self.opt.local_modes1

        match self.fft_type.upper():
            case "C2C":
                dim_z = round(Mz * float(self.opt.actual_upsampling_global[-1]))
            case "R2C":
                dim_z = (
                    round(Mz * float(self.opt.actual_upsampling_global[-1])) // 2 + 1
                )
            case _:
                raise NotImplementedError(
                    "FFT transform only implemented for type"
                    "'R2C' or 'C2C',"
                    f" got '{self.fft_type.upper()}'."
                )

        self.H = am.zeros(shape=(self.dim_H, Mx, My, Mz), dtype=self.dtype, order="C")

        if self.opt.distributed:
            from mpi4py import MPI

            mpi_comm = MPI.COMM_WORLD
            self.ghost_H = am.zeros(
                shape=(self.dim_H, *self.opt.ghost_grid_shape_ext),
                dtype=self.dtype,
                order="C",
            )
            # TODO: vectorize
            Mx_ups = int(self.opt.glb_grid_shape_ext[0])
            My_ups = round(
                int(self.opt.glb_grid_shape_ext[1])
                * float(self.opt.actual_upsampling_global[-2])
            )
            Mz_ups = round(
                int(self.opt.glb_grid_shape_ext[2])
                * float(self.opt.actual_upsampling_global[-1])
            )
            match self.fft_type.upper():
                case "C2C":
                    Hg_z = Mz_ups
                case "R2C":
                    Hg_z = Mz_ups // 2 + 1
                case _:
                    raise NotImplementedError(
                        "FFT transform only implemented for type"
                        "'R2C' or 'C2C',"
                        f" got '{self.fft_type.upper()}'."
                    )
            Hg_shape = (
                self.dim_H,
                Mx_ups,
                My_ups // mpi_comm.size,
                Hg_z,
                2,
            )  # Slab Y shape (when taken as [1:-1])
            if My_ups % mpi_comm.size != 0:
                raise ValueError(
                    f"mpi size must divide My_ups, but My_ups % mpi_comm.size={My_ups % mpi_comm.size}"
                )
            if Mx_ups // mpi_comm.size != Mx:
                raise ValueError(
                    f"Mx_ups // mpi_comm.size must equal Mx."
                    f"Mx_ups={Mx_ups}, mpi_comm.size={mpi_comm.size}, Mx={Mx}"
                )
            self.fft_shape = [
                Mx_ups // mpi_comm.size,
                My_ups,
                Mz_ups,
            ]  # Slab X shape
            if Mx_ups % mpi_comm.size != 0:
                raise ValueError(
                    f"mpi size must divide Mx_ups, but My_ups % mpi_comm.size={Mx_ups % mpi_comm.size}"
                )
        else:
            self.ghost_H = self.H  # dummy variable for g2p step
            Hg_shape = (
                self.dim_H,
                Mx,
                round(My * float(self.opt.actual_upsampling_global[-2])),
                dim_z,
                2,
            )
            self.fft_shape = [
                Mx,
                round(My * float(self.opt.actual_upsampling_global[-2])),
                round(Mz * float(self.opt.actual_upsampling_global[-1])),
            ]
        if not self.H.flags["C_CONTIGUOUS"]:
            raise TypeError("Expected H to be C_CONTIGUOUS!")

        self.Hg = am.empty(shape=Hg_shape, dtype=self.dtype)

        self.owned_ns = self.sources.shape[1]
        if self.opt.distributed:
            from mpi4py import MPI
            from parkipy.distributed._utils import (
                create_buffers,
                bucket_sort,
                communicate_ghost_points,
            )

            mpi_comm = MPI.COMM_WORLD
            ns = self.sources.shape[-1]
            nt = self.targets.shape[-1]
            ng = self.ghost_H[0].size
            if self.mpi_buffer_size == None:
                self.mpi_buffer_size = math.ceil(
                    (3)
                    * max(ns, nt, ng)  # FIXME: some theory to pick this would be nice
                )
            self.mpi_buffers = create_buffers(
                mpi_comm, self.execution_space, self.mpi_buffer_size
            )
            if self.box_dict == None:
                raise ValueError(
                    "box dict must be passed in to `DeviceData` for distributed Ewald."
                )
            # 0) slab sort arrays
            #   expand x, q, and n arrays to be buffer size
            extra_x = am.empty((3, self.mpi_buffer_size), dtype=self.dtype)
            extra_q = am.empty(
                (self.forces.shape[0], self.mpi_buffer_size), dtype=self.dtype
            )
            trg_buff = am.hstack((self.targets, extra_x), dtype=self.dtype)
            src_buff = am.hstack((self.sources, extra_x), dtype=self.dtype)
            frc_buff = am.hstack((self.forces, extra_q), dtype=self.dtype)

            # set up bucket sort and ghost points kwargs
            kwargs_bucket = {
                "src": src_buff.T,
                "ns": ns,
                "dens": frc_buff.T,
                "buffers": self.mpi_buffers,
            }
            if self.normals is not None:
                nrm_buff = am.hstack((self.normals, extra_x), dtype=self.dtype)
                kwargs_bucket["normal"] = nrm_buff.T

            #   bucket sort
            sort_start = time.time()
            nt_loc, ns_loc = bucket_sort(
                mpi_comm,
                self.execution_space,
                self.box_dict["box"],
                trg_buff.T,
                nt,
                **kwargs_bucket,
            )
            sort_end = time.time()
            self.walltime["mpi_sort"] = sort_end - sort_start
            self.owned_ns = ns_loc

            src_gst = am.empty_like(src_buff)
            frc_gst = am.empty_like(frc_buff)

            out_ghost = [src_gst.T, frc_gst.T]
            extra_arr_ghost = [frc_buff.T]

            if self.normals is not None:
                nrm_gst = am.empty_like(nrm_buff)
                out_ghost.append(nrm_gst.T)
                extra_arr_ghost.append(nrm_buff.T)

            # 1) distributed ghost sources
            ghost_start = time.time()
            ns_loc = communicate_ghost_points(
                mpi_comm,
                self.execution_space,
                self.box_dict,
                self.opt.ghost_dist,
                self.box_dict["box"][0],
                src_buff.T,
                ns_loc,
                *extra_arr_ghost,
                out=out_ghost,
                buffers=self.mpi_buffers,
            )
            ghost_end = time.time()
            self.walltime["mpi_ghost"] = ghost_end - ghost_start
            # 2) save to data
            #   offset particles to fit in slab box
            x_off = self.box_dict["left"] - (self.opt.ghost_dist)
            src_gst[0] -= x_off
            trg_buff[0] -= x_off

            #   save as contiguous for correct device array access
            self.targets = am.empty((3, nt_loc))
            self.sources = am.empty((3, ns_loc))
            self.forces = am.empty((self.forces.shape[0], ns_loc))
            self.targets[...] = trg_buff[:, :nt_loc]
            self.sources[...] = src_gst[:, :ns_loc]
            self.forces[...] = frc_gst[:, :ns_loc]
            if self.normals is not None:
                self.normals = am.empty((3, ns_loc))
                self.normals[...] = nrm_gst[:, :ns_loc]

    def communicate_ghost_grid_cells(self):
        if self.opt.distributed:
            am = get_array_module(self.execution_space)
            from mpi4py import MPI
            from parkipy.distributed._utils import communicate_grid_ghost_points

            mpi_comm = MPI.COMM_WORLD
            # communicate ghost cells to ghost grid
            self.ghost_H[
                :3, self.opt.ghost_dist_grid : -self.opt.ghost_dist_grid, ...
            ] = self.H[:3, :, :, :]
            H_view = self.ghost_H.transpose(1, 2, 3, 0)
            grid_start = time.time()
            n_ghost = communicate_grid_ghost_points(
                2 * self.opt.ghost_dist_grid,
                H_view,
                buffers=self.mpi_buffers,
                mpi_comm=mpi_comm,
            )
            grid_end = time.time()
            self.walltime["mpi_grid_ghost"] = grid_end - grid_start


@dataclass
class DevicePre:
    """
    Precomputes device-specific kernel selection, memory sizing, and geometry
    metadata for the Spectral Ewald method based on the target execution backend.

    Parameters
    ----------
    data : DeviceData
        Bundle containing all device-side geometry and configuration data.
    kernel : str
        Kernel mode to apply. Must be one of 'STOKES_SL', 'STOKES_DL', or 'STOKES_BOTH'.
    fs_code : str, optional
        Fourier space compute mode ('global' by default).

    Attributes
    ----------
    execution_space : ExecutionSpace
        Backend used for device execution (e.g. CUDA, HIP, or OpenMP).
    am : module
        Array module used (NumPy, CuPy, etc.).
    p2g_workunit, g2p_workunits : Dict[str, callable]
        Device kernel functions for P2P, P2G, and G2P stages.
    laplace_convolution_kernel, stokeslet_convolution_kernel, stresslet_convolution_kernel : callable
        PyKokkos kernels for convolutions
    convolution_sum_sl_dl : callable
        Summation kernel for combining SL and DL fields.
    has_sl, has_dl : bool
        Flags for which kernels (SL/DL) are present.
    p2g_max_cell_size : int
        Maximum shared memory cells available for the P2G stage.
    dim_in, dim_out : int
        Input/output data field dimensions.
    Ns, Nt, Ng : int
        Source, target, and grid sizes.
    near_potential : array_like
        Buffer for storing real-space potentials.
    far_potential : array_like
        Buffer for storing Fourier-space potentials.
    """

    data: Any
    kernel: str
    fs_code: str = "global"

    execution_space: Any = field(init=False)
    am: Any = field(init=False)
    p2g_workunit: Any = field(init=False)
    g2p_workunit: Any = field(init=False)
    stokeslet_convolution_kernel: Any = field(init=False)
    stresslet_convolution_kernel: Any = field(init=False)
    convolution_sum_sl_dl: Any = field(init=False)
    stokeslet_convolution_zero_kernel: Any = field(init=False)
    stresslet_convolution_zero_kernel: Any = field(init=False)
    p2g_max_cell_size: int = field(init=False)
    has_sl: bool = field(init=False)
    has_dl: bool = field(init=False)
    Ns: int = field(init=False)
    Nt: int = field(init=False)
    Ng: int = field(init=False)
    dim_in: int = field(init=False)
    dim_out: int = field(init=False)
    near_potential: Any = field(init=False)
    far_potential: Any = field(init=False)

    def __post_init__(self) -> None:
        self.execution_space = self.data.execution_space
        self.am = get_array_module(self.execution_space)

        # Determine which kernels are active
        match self.kernel.upper():
            case "LAPLACE":
                self.dim_in = 1
                self.dim_out = 1
            case "STOKES_COMB":
                self.dim_in = 12
                self.dim_out = 3
                self.has_sl = True
                self.has_dl = True
            case "STOKES_SL":
                self.dim_in = 3
                self.dim_out = 3
                self.has_sl = True
                self.has_dl = False
            case _:
                raise ValueError(
                    f"Kernel must be one of 'LAPLACE', 'STOKES_COMB', got '{self.kernel.upper()}'."
                )

        # Kernel and memory settings per backend
        match self.execution_space:
            case pk.ExecutionSpace.Cuda:
                max_shmem_block = (
                    self.am.cuda.Device(0).attributes["MaxSharedMemoryPerBlock"]
                    - 1024
                    - 3 * 1024
                )

            case pk.ExecutionSpace.HIP:
                print("WARNING: HIP shmem block size hardcoded to 64KB")
                max_shmem_block = 64000

            case pk.ExecutionSpace.OpenMP | pk.ExecutionSpace.DebugOpenMP:
                omp_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
                if omp_threads == 1:
                    warnings.warn(
                        "Only 1 OpenMP thread detected. Set OMP_NUM_THREADS to control parallelism.",
                        RuntimeWarning,
                    )
                max_shmem_block = 4.5 * 1024**2

            case _:
                raise ValueError(
                    f"Unsupported execution space '{self.execution_space}'."
                )

        # Choose kernel function variants based on dtype
        dtype_size = self.am.dtype(self.data.dtype).itemsize
        self.convolution_sum_sl_dl = convolution_sum_sl_dl
        self.convolution_sum_sl = convolution_sum_sl
        match self.data.dtype:
            case self.am.double:
                self.p2g_workunit = {
                    "BASE WITH NORMALS": p2g_base_fp64,
                    "SOURCE WITH NORMALS": p2g_source_fp64,
                    "GRID WITH NORMALS": p2g_grid_fp64,
                    "HYBRID WITH NORMALS": p2g_hybrid_fp64,
                    # without normals
                    "GRID WITHOUT NORMALS": p2g_grid_sans_normals_fp64,
                    "HYBRID WITHOUT NORMALS": p2g_hybrid_sans_normals_fp64,
                }
                self.g2p_workunit = {"BASE": g2p_base_fp64, "TARGET": g2p_target_fp64}
                self.laplace_convolution_kernel = laplace_convolution_kernel_fp64
                self.stokeslet_convolution_kernel = stokeslet_convolution_kernel_fp64
                self.stresslet_convolution_kernel = stresslet_convolution_kernel_fp64
                self.stokeslet_convolution_zero_kernel = (
                    stokeslet_convolution_zero_kernel_fp64
                )
                self.stresslet_convolution_zero_kernel = (
                    stresslet_convolution_zero_kernel_fp64
                )
            case self.am.single:
                self.p2g_workunit = {
                    "BASE WITH NORMALS": p2g_base_fp32,
                    "SOURCE WITH NORMALS": p2g_source_fp32,
                    "GRID WITH NORMALS": p2g_grid_fp32,
                    "HYBRID WITH NORMALS": p2g_hybrid_fp32,
                    # without normals
                    "GRID WITHOUT NORMALS": p2g_grid_sans_normals_fp32,
                    "HYBRID WITHOUT NORMALS": p2g_hybrid_sans_normals_fp32,
                }
                self.g2p_workunit = {"BASE": g2p_base_fp32, "TARGET": g2p_target_fp32}
                self.laplace_convolution_kernel = laplace_convolution_kernel_fp32
                self.stokeslet_convolution_kernel = stokeslet_convolution_kernel_fp32
                self.stresslet_convolution_kernel = stresslet_convolution_kernel_fp32
                self.stokeslet_convolution_zero_kernel = (
                    stokeslet_convolution_zero_kernel_fp32
                )
                self.stresslet_convolution_zero_kernel = (
                    stresslet_convolution_zero_kernel_fp32
                )
            case _:
                raise ValueError(f"Unsupported dtype '{self.data.dtype}'.")

        # flags
        self.has_normals = self.data.normals is not None
        dim_n = self.data.normals.shape[0] if self.has_normals else 0
        dim_f = self.data.forces.shape[0]

        # Compute max shared-memory-per-cell size for P2G
        int_size = self.am.dtype(int).itemsize
        iod_size = int_size / dtype_size
        p2g_cell_size = (max_shmem_block / dtype_size) / (
            3 * self.data.opt.window_P + 3 * iod_size + (dim_n + dim_f)
        )
        self.p2g_max_cell_size = max(int((p2g_cell_size // 32) * 32), 1)
        assert self.p2g_max_cell_size * dtype_size * (
            3 * self.data.opt.window_P + 3 * iod_size + (dim_n + dim_f)
        ) <= int(max_shmem_block)

        # Geometry and dimension metadata
        self.Ns = int(self.data.sources.shape[1])
        self.Nt = int(self.data.targets.shape[1])
        self.Ng = int(self.am.prod(self.data.opt.grid_shape_ext))

        # buffers for output potential
        self.near_potential = self.am.zeros(
            (self.dim_out, self.data.targets.shape[-1]), dtype=self.data.dtype
        )
        self.far_potential = self.am.zeros_like(self.near_potential)

    @classmethod
    def from_particles(
        cls,
        targets,
        sources,
        forces,
        normals,
        kernel,
        box,
        tolerance,
        periodicity,
        cell_size=None,
        rc=None,
        execution_space=None,
        fft_type="R2C",
        distributed=False,
        kill_fourier_grid=False,
    ):
        """
        Construct a `DevicePre` object from minimal input.

        If `kill_fourier_grid` is `True`, set params.actual_upsampling = [0,0].
        This is needed to test `p2g` independently, as we are
        interested in problem sizes that may require a Fourier grid
        too large for a single GPU.
        """
        box_dict = {"box": box}
        ns = ns_max = sources.shape[-1]
        data_kwargs = {}
        if distributed:
            from mpi4py import MPI
            from parkipy.distributed._utils import compute_slab

            mpi_comm = MPI.COMM_WORLD

            if cell_size is not None:
                raise ValueError(
                    "Cell size for distributed execution is not supported."
                )

            box_dict.update(compute_slab(box, mpi_comm))
            ns_max = mpi_comm.allreduce(ns, op=MPI.MAX)
            if box_dict["slab box"][0] != box[0] / mpi_comm.Get_size():
                raise ValueError(
                    "Expected each slab to be the same width"
                    " to ensure uniform Fourier grids."
                    f" Slab on rank {mpi_comm.Get_rank()} is of"
                    f" width {slab['width']},  but box[0]/mpi_comm.Get_size() "
                    f" is {box[0]/mpi_comm.Get_size()}."
                )
        # set cell_size based from tolerance
        if cell_size == None and rc == None:
            raise NotImplementedError(
                "If cell_size and rc are None, choose cell_size from performance model."
                " This functionality is not yet implemented."
            )
        # get the spectral Ewald parameters
        params_kwargs = {}
        match kernel.upper():
            case "STOKES_COMB":
                params_fun = se_params_stokes_comb
                dim_H = 12
            case "LAPLACE":
                params_fun = se_params_laplace
                params_kwargs["f"] = forces
                dim_H = 1
            case "STOKES_SL":
                params_fun = se_params_stokeslet
                dim_H = 3
            case _:
                raise NotImplementedError(
                    "Ewald Kernels only implemented for 'STOKES_COMB' and 'LAPLACE',"
                    f" got {kernel.upper()}."
                )
        params = params_fun(
            box_dict=box_dict,
            tolerance=tolerance,
            num_sources=ns_max,
            cell_size=cell_size,
            rc=rc,
            periodicity=periodicity,
            distributed=distributed,
            **params_kwargs,
        )
        # setup options and parameters
        if kill_fourier_grid:
            warnings.warn(
                "Fourier grid dimensions set to [0,0,0], only use to isolate kernel timings.",
                category=RuntimeWarning,
            )
            am = get_array_module(execution_space)
            params.actual_upsampling = am.array([0, 0, 0])
        opt = Options.from_params(params, execution_space)
        opt_sc = SEPre.from_params(params, execution_space)
        data = DeviceData(
            targets=targets,
            sources=sources,
            forces=forces,
            normals=normals,
            dim_H=dim_H,
            opt=opt,
            opt_sc_glob=opt_sc,
            opt_sc_zero=opt_sc,
            fft_type=fft_type,
            execution_space=execution_space,
            box_dict=box_dict,
        )
        device_pre = cls(
            data=data,
            kernel=kernel,
        )
        device_pre.data.opt_sc_glob = opt_sc
        device_pre.data.opt_sc_zero = opt_sc

        return device_pre, params
