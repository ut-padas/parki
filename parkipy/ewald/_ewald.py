"""
The spectral Ewald algorithm.

The near-field component of the
algorithm updates the
`device_pre.near_potentials`
array while the far field updates
the `device_pre.far_potentials`
array.

Each stage returns a `walltime` dict
contaings the **wall clock** execution times.
For accurate timing on a GPU, streaming
needs to be searialized.

----------
near-field
----------
    0) P2P

---------
far-field
---------
    1) P2G
    2) FFT
    3) CNV
    4) IFFT
    5) G2P
"""

import os
import time
import math
import pykokkos as pk
import numpy as np
import warnings

from ._utils import get_fftn, get_ifftn
from parkipy.utils import get_execution_space, get_array_module
from parkipy import CellList

# import P2P kernels
from ._pk_kernels._p2p_workunits import (
    p2p_stokes_sl_gm1d_fp32,
    p2p_stokes_sl_gm1d_fp64,
    p2p_stokes_comb_gm1d_fp32,
    p2p_stokes_comb_gm1d_fp64,
    p2p_stokes_comb_gm2d_fp32,
    p2p_stokes_comb_gm2d_fp64,
    p2p_stokes_comb_sm1d_fp32,
    p2p_stokes_comb_sm1d_fp64,
    p2p_stokes_comb_sm2d_fp32,
    p2p_stokes_comb_sm2d_fp64,
    p2p_laplace_gm1d_fp32,
    p2p_laplace_gm1d_fp64,
)

try:
    from cupy.cuda.nvtx import RangePush, RangePop
except ModuleNotFoundError:
    RangePush = lambda x: x


def p2p(
    device_pre,
    method="GM-1D",
    threads_x: int = 32,
    threads_y: int = 2,
    t_per_thread: int = 1,
    s_per_thread: int = 32,
    cell_pad: int = 1,
    kernel=None,  # for testing purposes only
) -> None:
    """
    Perform near-field particle-to-particle (P2P) interaction computation using a
    specified execution method and kernel configuration.

    This function sets up cell lists for both sources and targets, interprets the
    desired compute strategy (shared vs global memory, 1D vs 2D blocks), and launches
    the appropriate Kokkos-based P2P workload on the selected execution space.

    Parameters
    ----------
    device_pre : DevicePre
        Precomputed device configuration containing execution space, kernel bindings,
        and data references.
    method : str
        String of the form "<mem>-<blk>", where `<mem>` is either "SM" (shared memory)
        or "GM" (global memory), and `<blk>` is "1D" or "2D" for the thread block shape.
        Defaults to `'GM-1D'`.
    threads : int, default=32
        Number of threads per team (CUDA block or OpenMP group).
    t_per_thread : int, default=1
        Number of target particles processed per thread.
    s_per_thread : int, default=32
        Number of source particles processed per thread (1D kernels).
    threads_y : int, default=2
        Thread count per dimension for 2D kernels (i.e., total threads = threads_y²).
    cell_pad : int, default=1
        Padding applied to neighboring cells for interaction range.

    Returns
    -------
    dict
        Dictionary with wall clock timings (in seconds) for three stages:
        - "args": time spent parsing arguments and selecting method flags.
        - "sort": time spent constructing source/target cell lists.
        - "kernel": time spent executing the main Kokkos kernel.

    Raises
    ------
    NotImplementedError
        If the kernel name is unsupported.

    Notes
    -----
    This function is part of a Spectral Ewald implementation and handles only the
    near-field pairwise interactions. The far-field is typically handled separately
    in Fourier space. Cell lists are constructed on-the-fly and passed to the
    compiled Kokkos kernels along with configuration flags and simulation parameters.
    """
    if pk.is_host_execution_space(device_pre.execution_space) and threads_x != 1:
        warnings.warn(
            "'threads_x' argument is overridden to 1 in OpenMP execution space.",
            RuntimeWarning,
        )
        threads_x = 1
    args_start = time.time()

    # get kernel information
    kernel_flag = None
    has_sl = False
    has_dl = False
    has_ewald = False
    if kernel is None:
        kernel = device_pre.kernel.upper()
    match kernel.upper():
        case "STOKES_COMB":
            kernel_flag = 0
            has_sl = has_dl = has_ewald = True
        case "LAPLACE":
            kernel_flag = 3
        case "STOKES_SL":
            kernel_flag = 0
            has_sl = has_ewald = True
            has_dl = False
        case "DISTANCE / EWALD":  # NOTE: w/o Ewald, for testing only
            kernel_flag = 1
        case "LAPLACE / EWALD":
            kernel_flag = 2
        case "STOKES_SL / EWALD":
            kernel_flag = 0
            has_sl = True
        case _:
            raise NotImplementedError(
                "kernel expected to be one of"
                " 'STOKES_COMB', 'LAPLACE'"
                f" got {kernel.upper()}."
            )

    # get method information
    vector_size, method_flag = (None, None)
    try:
        mem, blk = method.strip().upper().split("-")
    except ValueError:
        raise ValueError(
            "method expected to be in the for `<mem>-<blk>`,"
            " where `<mem>` is the memory modle, either `SM` or `GM`,"
            " and `<blk>` is the block dimensions, either `1D` or `2D`,"
            f" got {method}."
        )
    if mem == "GM":
        method_flag = 0
        if blk == "1D":
            vector_size = 1
        elif blk == "2D":
            vector_size = threads_y  # TODO change based off execution space
        else:
            raise ValueError(
                "P2P block dimensions must be '1D' or '2D'," f" got {blk}."
            )
    elif mem == "SM":
        method_flag = 2
        if blk == "1D":
            vector_size = 1
        elif blk == "2D":
            vector_size = threads_y  # TODO change based off execution space
        else:
            raise ValueError(
                "P2P block dimensions must be '1D' or '2D'," f" got {blk}."
            )
    else:
        raise ValueError("P2P memory model must be 'GM' or 'SM'," f" got {mem}.")

    # set up workunit selector
    workunit = {
        "STOKES_COMB": {
            "GM-1D": (
                p2p_stokes_comb_gm1d_fp64
                if device_pre.data.dtype == np.float64
                else p2p_stokes_comb_gm1d_fp32
            ),
            "GM-2D": (
                p2p_stokes_comb_gm2d_fp64
                if device_pre.data.dtype == np.float64
                else p2p_stokes_comb_gm2d_fp32
            ),
            "SM-1D": (
                p2p_stokes_comb_sm1d_fp64
                if device_pre.data.dtype == np.float64
                else p2p_stokes_comb_sm1d_fp32
            ),
            "SM-2D": (
                p2p_stokes_comb_sm2d_fp64
                if device_pre.data.dtype == np.float64
                else p2p_stokes_comb_sm2d_fp32
            ),
        },
        "STOKES_SL": {
            "GM-1D": (
                p2p_stokes_sl_gm1d_fp64
                if device_pre.data.dtype == np.float64
                else p2p_stokes_sl_gm1d_fp32
            )
        },
        "LAPLACE": {
            "GM-1D": (
                p2p_laplace_gm1d_fp64
                if device_pre.data.dtype == np.float64
                else p2p_laplace_gm1d_fp32
            )
        },
    }

    sort_start = args_end = time.time()
    if device_pre.data.normals is not None:
        forces = (device_pre.data.forces, device_pre.data.normals)
    else:
        forces = device_pre.data.forces
    source_list = CellList(
        device_pre.data.sources,
        device_pre.data.opt.rc,
        device_pre.data.opt.ghost_box,
        execution_space=device_pre.execution_space,
        forces=forces,
    )
    target_list = CellList(
        device_pre.data.targets,
        device_pre.data.opt.rc,
        device_pre.data.opt.ghost_box,
        execution_space=device_pre.execution_space,
    )
    # call kokkos executable
    kernel_start = sort_end = time.time()
    RangePush("P2P-kernel")
    # set up kernel call
    target_threads = math.ceil(threads_x / vector_size)
    target_chunk_size = target_threads * t_per_thread
    target_cell_chunks = math.ceil(target_list.cell_size / target_chunk_size)
    teams = target_list.num_nonempty_cells * target_cell_chunks
    policy = pk.TeamPolicy(
        device_pre.execution_space,
        teams,
        target_threads,
    )
    kwargs = {
        "t_cell_chunk_size": target_chunk_size,
        "t_list2global": target_list.particle_index,
        "t_cell_size": target_list.cell_size,
        "nz2t_cell_map": target_list.nonempty_cells,
        "num_cells_shape": target_list.cell_grid_shape,
        "dim_out": int(device_pre.near_potential.shape[0]),
        "nnz_t_cells": target_list.num_nonempty_cells,
        "targets_list": target_list.particle_list,
        "s_counter": source_list.counter,
        "s2nz_cell_map": source_list.nonempty_cell_index,
        "s_cell_size": source_list.cell_size,
        "rc_squared": device_pre.data.opt.rc**2,
        "potentials": device_pre.near_potential,
        "sources_list": source_list.particle_list,
        "periodicity": (
            device_pre.data.opt.periodicity
            if not device_pre.data.opt.distributed
            else 0
        ),
        "box": device_pre.am.array(device_pre.data.opt.ghost_box),
        "xi": device_pre.data.opt.xi,
        "xi_squared": device_pre.data.opt.xi**2,
        "xi_two_inv_sqrt_pi": device_pre.data.opt.xi * 2 * 0.564189583547756286948079,
    }

    if device_pre.has_normals:
        kwargs["forces_list"] = source_list.force_list[0]
        kwargs["normals_list"] = source_list.force_list[1]
    else:
        kwargs["forces_list"] = source_list.force_list

    if method.upper() in ["GM-2D", "SM-1D", "SM-2D"]:
        kwargs["t_cell_chunks"] = math.ceil(
            target_list.cell_size / kwargs["t_cell_chunk_size"]
        )
        kwargs["t_counter"] = target_list.counter
    if method.upper() in ["GM-2D", "SM-2D"]:
        kwargs["vector_size"] = vector_size
    if method.upper() in ["SM-1D", "SM-2D"]:
        kwargs["s_cell_chunk_size"] = vector_size * s_per_thread
        kwargs["s_cell_chunks"] = math.ceil(
            source_list.cell_size / kwargs["s_cell_chunk_size"]
        )
    if method.upper() == "GM-2D":
        kwargs["s_cell_threads"] = math.ceil(source_list.cell_size / vector_size)
    if method.upper() == "SM-2D":
        kwargs["s_cell_chunk_threads"] = math.ceil(
            kwargs["s_cell_chunk_size"] / vector_size
        )
        kwargs["t_cell_chunk_threads"] = math.ceil(
            kwargs["t_cell_chunk_size"] / vector_size
        )

    pk.parallel_for(
        f"P2P-{kernel.upper()}-{method.upper()}",
        policy,
        workunit[kernel.upper()][method.upper()],
        **kwargs,
    )
    RangePop()
    kernel_end = time.time()
    walltime = {
        "args": args_end - args_start,
        "sort": sort_end - sort_start,
        "kernel": kernel_end - kernel_start,
        "tot": kernel_end - args_start,
    }
    return walltime


def p2g(
    device_pre,
    method="HYBRID",
    threads=128,
) -> None:
    """
    Perform the particle-to-grid (P2G) spreading operation using the Spectral Ewald method.

    This function maps particle source strengths (single-layer, double-layer, and normals)
    onto a uniform Fourier grid for subsequent FFT-based computation. The spreading is
    performed via a Pykokkos kernel, optionally using a cell-based sorting strategy to
    improve cache locality and parallel performance.

    Parameters
    ----------
    device_pre : DevicePre
        Precomputed device configuration containing source data, grid configuration,
        and references to Pykokkos kernels and execution space.

    method : str
        Spreading method to use. Must be one of:

        - `'BASE'`     : No sorting, baseline P2G kernel
        - `'SOURCE'`   : Sort sources into spatial cells and spread by cell
        - `'GRID'`     : Grid-major sorting
        - `'HYBRID'`   : Combines source and grid sorting heuristics

        Defaults to `'HYBRID'`.

    threads : int, optional
        Number of threads per team for the Pykokkos kernel (default: 128).
        Overridden to 1 in OpenMP execution mode.

    Returns
    -------
    walltime : dict
        Dictionary containing timing information for the different stages:

        - `'args'`   : Time spent preparing kernel arguments
        - `'sort'`   : Time spent sorting particles (if applicable)
        - `'kernel'` : Time spent executing the P2G kernel

    Raises
    ------
    ValueError
        If an invalid `method` string is passed.

    Notes
    -----
    - For methods other than `'BASE'`, a `CellList` is constructed to spatially sort the particles.
    - Particle positions are rescaled to grid coordinates and padded according to the window support.
    - If `'BASE'` is used, a dummy `source_list` with placeholder attributes is created to fulfill kernel args.
    - The kernel supports both single- and double-layer spreading with optional shared memory acceleration.
    - Grid shape and cell sizes are inferred from `device_pre` and depend on the execution space.
    """
    walltime = {}
    walltime["tot"] = time.time()
    if pk.is_host_execution_space(device_pre.execution_space) and threads != 1:
        warnings.warn(
            "'threads' argument is overridden to 1 in OpenMP execution space.",
            RuntimeWarning,
        )
        threads = 1

    if device_pre.data.opt.window_P > 14:
        raise ValueError(
            f"p2g only supported for window_P <= 14, "
            f"got window_P={device_pre.data.opt.window_P}. "
            f"Please adjust tolerance and cell size to decrease window_P."
        )

    # scale particles to Fourier grid
    offsets = device_pre.am.array(
        [
            (
                0.0
                if device_pre.data.opt.box_off[i] == 0
                else device_pre.data.opt.box_off[i] + 0.5 * device_pre.data.opt.h
            )
            for i in range(3)
        ],
        dtype=device_pre.data.dtype,
    ).reshape(3, 1)
    scaled_sources = (device_pre.data.sources - offsets) * device_pre.data.opt.grid_res

    # get method information
    method_flag = -1
    var_arr = method.strip().upper().split("-")
    match method.upper():
        case "BASE":
            method_flag = 0
            H_view = device_pre.data.ghost_H.transpose(1, 2, 3, 0).copy()

            # set up kernel call
            teams = math.ceil(device_pre.Ns / threads)
            policy = pk.TeamPolicy(device_pre.execution_space, teams, threads)
            kwargs = {
                "yj": scaled_sources,
                "qj": device_pre.data.forces,
                "nj": device_pre.data.normals,
                "H": H_view,
                "H_shape": device_pre.am.asarray(
                    device_pre.data.opt.ghost_grid_shape_ext
                ),
                "ny": device_pre.Ns,
                "window_P": device_pre.data.opt.window_P,
                "periodicity": (
                    device_pre.data.opt.periodicity
                    if not device_pre.data.opt.distributed
                    else 0
                ),
                "threads": threads,
            }
        case "SOURCE":
            method_flag = 1
            H_view = device_pre.data.ghost_H

            # sort sources
            walltime["sort"] = time.time()
            if device_pre.data.normals is not None:
                forces = (device_pre.data.forces, device_pre.data.normals)
            else:
                forces = device_pre.data.forces
            source_list = CellList(
                scaled_sources,
                (device_pre.data.opt.window_P / 2),
                device_pre.am.array(device_pre.data.opt.ghost_grid_shape_ext).astype(
                    device_pre.data.dtype
                ),
                execution_space=device_pre.execution_space,
                forces=forces,
            )
            cell_chunk_size: int = min(
                source_list.cell_size, device_pre.p2g_max_cell_size
            )
            walltime["sort"] = time.time() - walltime["sort"]

            # set up kernel call
            teams_per_cell = math.ceil(source_list.cell_size / threads)
            policy = pk.TeamPolicy(
                device_pre.execution_space,
                source_list.num_nonempty_cells * teams_per_cell,
                threads,
            )
            kwargs = {
                "yj": source_list.particle_list,
                "qj": source_list.force_list[0],
                "nj": source_list.force_list[1],
                "H": H_view,
                "H_shape": device_pre.am.asarray(
                    device_pre.data.opt.ghost_grid_shape_ext
                ),
                "ny": device_pre.Ns,
                "window_P": device_pre.data.opt.window_P,
                "periodicity": (
                    device_pre.data.opt.periodicity
                    if not device_pre.data.opt.distributed
                    else 0
                ),
                "cell_size": source_list.cell_size,
                "cell_teams": teams_per_cell,
                "threads": threads,
            }
        case "GRID":
            method_flag = 3
            H_view = device_pre.data.ghost_H.transpose(1, 2, 3, 0).copy()
            # sort sources
            walltime["sort"] = time.time()
            if device_pre.data.normals is not None:
                forces = (device_pre.data.forces, device_pre.data.normals)
            else:
                forces = device_pre.data.forces
            source_list = CellList(
                scaled_sources,
                (device_pre.data.opt.window_P / 2),
                device_pre.am.array(device_pre.data.opt.ghost_grid_shape_ext).astype(
                    device_pre.data.dtype
                ),
                execution_space=device_pre.execution_space,
                forces=forces,
            )
            cell_chunk_size: int = min(
                source_list.cell_size, device_pre.p2g_max_cell_size
            )
            walltime["sort"] = time.time() - walltime["sort"]

            # setup kernel call
            teams = math.ceil(device_pre.Ng / threads)
            policy = pk.TeamPolicy(device_pre.execution_space, teams, threads)
            kwargs = {
                "yj": source_list.particle_list,
                "H": H_view,
                "H_shape": device_pre.am.asarray(
                    device_pre.data.opt.ghost_grid_shape_ext
                ),
                "window_P": device_pre.data.opt.window_P,
                "periodicity": (
                    device_pre.data.opt.periodicity
                    if not device_pre.data.opt.distributed
                    else 0
                ),
                "cell_size": source_list.cell_size,
                "num_cells": source_list.cell_grid_shape,
                "nonempty_cell_index": source_list.nonempty_cell_index,
                "threads": threads,
            }
            if device_pre.has_normals:
                kwargs["qj"] = source_list.force_list[0]
                kwargs["nj"] = source_list.force_list[1]
            else:
                kwargs["qj"] = source_list.force_list
                kwargs["dim_f"] = kwargs["qj"].shape[0]
                kwargs["dim_H"] = device_pre.data.dim_H

        case "HYBRID":
            method_flag = 4
            H_view = device_pre.data.ghost_H
            # sort sources
            walltime["sort"] = time.time()
            if device_pre.data.normals is not None:
                forces = (device_pre.data.forces, device_pre.data.normals)
            else:
                forces = device_pre.data.forces
            source_list = CellList(
                scaled_sources,
                (device_pre.data.opt.window_P / 2),
                device_pre.am.array(device_pre.data.opt.ghost_grid_shape_ext).astype(
                    device_pre.data.dtype
                ),
                execution_space=device_pre.execution_space,
                forces=forces,
            )
            cell_chunk_size: int = min(
                source_list.cell_size, device_pre.p2g_max_cell_size
            )
            walltime["sort"] = time.time() - walltime["sort"]

            # setup kernel call
            chunks_per_cell = math.ceil(source_list.cell_size / cell_chunk_size)
            teams = source_list.num_nonempty_cells * chunks_per_cell
            policy = pk.TeamPolicy(device_pre.execution_space, teams, threads)
            kwargs = {
                "yj": source_list.particle_list,
                "H": H_view,
                "H_shape": device_pre.am.asarray(
                    device_pre.data.opt.ghost_grid_shape_ext
                ),
                "window_P": device_pre.data.opt.window_P,
                "periodicity": (
                    device_pre.data.opt.periodicity
                    if not device_pre.data.opt.distributed
                    else 0
                ),
                "cell_size": source_list.cell_size,
                "num_cells": source_list.cell_grid_shape,
                "nonempty_cell_index": source_list.nonempty_cell_index,
                "cell_index": source_list.nonempty_cells,
                "threads": threads,
                "dim_H": device_pre.data.dim_H,
                "cell_chunk_size": cell_chunk_size,
                "chunks_per_cell": chunks_per_cell,
            }
            if device_pre.has_normals:
                kwargs["qj"] = source_list.force_list[0]
                kwargs["nj"] = source_list.force_list[1]
            else:
                kwargs["qj"] = source_list.force_list

            kwargs["dim_f"] = kwargs["qj"].shape[0]

        case _:
            raise ValueError(
                "p2g method must be 'BASE', 'SOURCE', 'GRID', 'HYBRID',"
                f" got {method.upper()}."
            )

    if device_pre.data.normals is not None:
        forces = (device_pre.data.forces, device_pre.data.normals)
    else:
        forces = device_pre.data.forces

    # pykokkos kernel
    walltime["kernel"] = time.time()
    RangePush("P2G-kernel")
    # method == BASE | SOURCE | GRID | HYBRID
    method_name = method.upper() + (
        " WITH NORMALS" if device_pre.has_normals else " WITHOUT NORMALS"
    )
    if method_name not in device_pre.p2g_workunit.keys():
        raise NotImplementedError(
            f"P2G method '{method_name}' not implemented. Available methods: '{device_pre.p2g_workunit.keys()}'"
        )
    walltime["kernel"] = time.time()
    RangePush("P2G-kernel")
    pk.parallel_for(
        f"P2G-{method.upper()}",
        policy,
        device_pre.p2g_workunit[method_name],
        **kwargs,
    )
    RangePop()
    walltime["kernel"] = time.time() - walltime["kernel"]

    match method.upper():
        case "BASE":
            H_view = H_view.transpose(3, 0, 1, 2)
        case "SOURCE":
            pass
        case "GRID":
            H_view = H_view.transpose(3, 0, 1, 2)
        case "HYBRID":
            pass
        case _:
            raise ValueError(
                "p2g method must be 'BASE', 'SOURCE', 'GRID', 'HYBRID',"
                f" got {method.upper()}."
            )

    if device_pre.data.opt.distributed:
        device_pre.data.H[...] = H_view[
            :,
            device_pre.data.opt.ghost_dist_grid : -device_pre.data.opt.ghost_dist_grid,
            ...,
        ]
    else:
        device_pre.data.H[...] = H_view
    RangePop()
    kernel_end = time.time()
    walltime["tot"] = time.time() - walltime["tot"]
    return walltime


def fft(device_pre, options, fftmp_buffers=None):
    """
    Perform a multidimensional FFT on the gridded source data stored in `device_pre`.

    This function applies a real-to-complex n-dimensional FFT (rFFT) along the spatial axes
    of the input tensor `H`, storing the result in `Hg`.

    Parameters
    ----------
    device_pre : DevicePre
        The precomputed configuration and data buffers used for the Spectral Ewald method.
        Must include attributes like `H`, `Hg`, `fft_shape`, `fft_up`, and `fft_real_shape`,
        as well as the `execution_space`.
    options: EwaldOptions
        Ewald options data class.

    Returns
    -------
    walltime : dict
        Dictionary with the wall time spent in the FFT kernel under the key "kernel".

    Notes
    -----
    - If `fft_up` is True, the same upsampling factor is used globally in all directions.
    - If the execution space is OpenMP, the number of threads is set via `OMP_NUM_THREADS`.
    - The FFT is computed along axes (1, 2, 3), assuming the input has shape (channels, x, y, z).
    - The result is reshaped into the target layout (`Hg_real_shape`) after the FFT.

    Raises
    ------
    NotImplementedError
        If `fft_up` is False. Directional/local upsampling has not yet been implemented.
    """
    fft_start = time.time()
    if device_pre.data.opt.distributed:
        if fftmp_buffers is None:
            raise ValueError(
                "fftmp_buffers must be passed into fft for distributed execution."
            )
        if device_pre.execution_space == pk.ExecutionSpace.Cuda:
            import cupy as cp

            with cp.cuda.Device(fftmp_buffers.device_id):
                # Use stateful pre-planed FFT object 'f'.
                with fftmp_buffers.fft as f:

                    # loop over H dimensions
                    for d in range(device_pre.data.dim_H):
                        # copy to fft buffers
                        fftmp_buffers.fft_buff[
                            :,
                            : device_pre.data.H.shape[2],
                            : device_pre.data.H.shape[3],
                        ] = device_pre.data.H[d].astype(
                            np.complex128
                            if device_pre.data.fft_type.upper() == "C2C"
                            else np.float64
                        )
                        # Execute the FFT.
                        device_pre.data.Hg[d] = (
                            f.execute()
                            .view(device_pre.data.dtype)
                            .reshape(device_pre.data.Hg.shape[1:])
                        )

        else:
            raise NotImplementedError(
                "distributed fft only implemented for Cuda execution space,"
                f" got {device_pre.execution_space}."
            )

    else:
        RangePush("FFT-kernel")
        if options.torch_fft:
            import torch

            # get torch pointers for arrays
            Hg_ten = torch.as_tensor(
                device_pre.data.Hg.view(device_pre.data.complex_dtype).squeeze()
            )
            H_ten = torch.as_tensor(device_pre.data.H)

            # compute fft
            fftn = {"C2C": torch.fft.fftn, "R2C": torch.fft.rfftn}
            fftn[device_pre.data.fft_type](
                H_ten, s=device_pre.data.fft_shape, dim=(1, 2, 3), out=Hg_ten
            )

        else:
            extra_args = {}
            if pk.is_host_execution_space(device_pre.execution_space):
                extra_args["workers"] = int(os.environ.get("OMP_NUM_THREADS"))
            fftn = get_fftn(device_pre.execution_space, device_pre.data.fft_type)
            device_pre.data.Hg[...] = (
                fftn(
                    device_pre.data.H[...],
                    s=device_pre.data.fft_shape,
                    axes=(1, 2, 3),
                    overwrite_x=True,
                    **extra_args,
                )
                .view(device_pre.data.dtype)
                .reshape(device_pre.data.Hg.shape)
            )
        RangePop()
    fft_end = time.time()
    walltime = {"tot": fft_end - fft_start}
    return walltime


def cnv(device_pre, threads=128):
    """
    Apply the Fourier-space convolution kernel in the Spectral Ewald method.

    This function dispatches the convolution procedure depending on the Fourier-space code
    specified in `device_pre.fs_code`. Only the 'global' implementation is currently supported.

    Parameters
    ----------
    device_pre : DevicePre
        Object containing all buffers, kernel handles, and metadata needed for convolution.
    threads : int, optional
        Number of threads to use for parallel execution (default: 128).

    Returns
    -------
    walltime : dict
        Dictionary containing timing information for the convolution kernel.
    """
    cnv_start = time.time()
    if device_pre.fs_code == "global":
        _cnv_GLB(device_pre=device_pre, threads=threads)
    else:
        raise NotImplementedError("APT Scaling not yet implemented.")
        _cnv_APT(device_pre=device_pre)
    cnv_end = time.time()
    walltime = {"tot": cnv_end - cnv_start}
    return walltime


def _cnv_GLB(device_pre, threads):
    """
    Perform both global and zero-mode Fourier-space convolution.

    This function calls `_cnv_global` to apply the Fourier convolution on all non-zero modes
    and separately handles the zero-frequency mode using dedicated zero-mode convolution kernels.

    Parameters
    ----------
    device_pre : DevicePre
        Contains all data arrays and configuration metadata required for the convolution step.
    threads : int
        Number of threads per team for Kokkos execution.

    Notes
    -----
    - Uses different convolution kernels for zero and non-zero modes.
    - Handles both Stokeslet and Stresslet contributions if the corresponding force
      data (`f1`, `f2`) are present.
    - Uses `opt_sc_glob` and `opt_sc_zero` fields of `device_pre.data` for grid-specific
      spectral options.
    - The zero mode output is written into the first Fourier slice `Hg[:, 0]`.
    """
    if pk.is_host_execution_space(device_pre.execution_space) and threads != 1:
        warnings.warn(
            "'threads' argument is overridden to 1 in OpenMP execution space.",
            RuntimeWarning,
        )
        threads = 1
    # global domain
    _cnv_global(device_pre=device_pre, threads=threads)
    # zero mode
    match device_pre.data.opt.periodicity:
        case 1:
            zero_mode_shape = device_pre.data.Hg.shape[2:-1]
            freq_range = int(device_pre.am.prod(device_pre.am.array(zero_mode_shape)))
            freq_teams = math.ceil(freq_range / threads)
            shift = 0
            # stokeslet
            if device_pre.has_sl:
                G01 = [None] * 3
                for d in range(shift, shift + len(G01)):
                    G01[d - shift] = device_pre.data.Hg[d][0]
                RangePush("Sca-kernel")
                pk.parallel_for(
                    pk.TeamPolicy(device_pre.execution_space, freq_teams, threads),
                    device_pre.stokeslet_convolution_zero_kernel,
                    H1=G01[0],
                    H2=G01[1],
                    H3=G01[2],
                    grid_size_1=device_pre.data.Hg.shape[3],
                    box=device_pre.data.opt.glb_box_ext,
                    k1_off=device_pre.data.opt.k1_off,
                    xi=device_pre.data.opt.xi,
                    pw=device_pre.data.opt_sc_zero.window_scaling_power,
                    wsh=device_pre.data.opt_sc_zero.window_shape,
                    ksc=device_pre.data.opt_sc_zero.kaiser_scaling,
                    grid=device_pre.data.opt.glb_grid_shape_ext,
                    whw=device_pre.data.opt_sc_zero.window_halfwidth,
                    ups=device_pre.data.opt.actual_upsampling_global,
                    gR=device_pre.data.opt_sc_zero.greens_truncation_R,
                    vico_var=device_pre.data.opt_sc_zero.vico_var,
                    periodicity=device_pre.data.opt.periodicity,
                    freq_range=freq_range,
                    threads=threads,
                )
                RangePop()
                shift += 3
            if device_pre.has_dl:
                G02 = [[None for _ in range(3)] for _ in range(3)]
                for d in range(9):
                    G02[d % 3][d // 3] = device_pre.data.Hg[shift + d][0]
                RangePush("Sca-kernel")
                pk.parallel_for(
                    pk.TeamPolicy(device_pre.execution_space, freq_teams, threads),
                    device_pre.stresslet_convolution_zero_kernel,
                    H11=G02[0][0],
                    H21=G02[1][0],
                    H31=G02[2][0],
                    H12=G02[0][1],
                    H22=G02[1][1],
                    H32=G02[2][1],
                    H13=G02[0][2],
                    H23=G02[1][2],
                    H33=G02[2][2],
                    grid_size_1=device_pre.data.Hg.shape[3],
                    box=device_pre.data.opt.glb_box_ext,
                    k1_off=device_pre.data.opt.k1_off,
                    xi=device_pre.data.opt.xi,
                    pw=device_pre.data.opt_sc_zero.window_scaling_power,
                    wsh=device_pre.data.opt_sc_zero.window_shape,
                    ksc=device_pre.data.opt_sc_zero.kaiser_scaling,
                    grid=device_pre.data.opt.glb_grid_shape_ext,
                    whw=device_pre.data.opt_sc_zero.window_halfwidth,
                    ups=device_pre.data.opt.actual_upsampling_global,
                    gR=device_pre.data.opt_sc_zero.greens_truncation_R,
                    vico_var=device_pre.data.opt_sc_zero.vico_var,
                    periodicity=device_pre.data.opt.periodicity,
                    freq_range=freq_range,
                    threads=threads,
                )
                RangePop()
                shift += 9
            # sum
            for d in range(device_pre.dim_out):
                if device_pre.has_sl and device_pre.has_dl:
                    device_pre.data.Hg[d, 0] = G01[d] + G02[d][0]
                elif device_pre.has_sl:
                    device_pre.data.Hg[d, 0] = G01[d]
                else:
                    raise NotImplementedError(
                        "CNV summation not implemented for"
                        "stokes_sl or stokes_dl case."
                    )
        case 3:
            device_pre.data.Hg[:, 0, 0, 0, :] = 0
        case _:
            raise NotImplementedError(
                "CNV step is only implemented for periodicities 1 and 3,"
                f" got {device_pre.data.opt.periodicity}."
            )


def _cnv_global(device_pre, threads):
    """
    Perform global Fourier-space convolution on the non-zero modes.

    Applies convolution with the Green’s function in Fourier space for both Stokeslet
    and Stresslet kernels on all non-zero wavevector modes using Kokkos `parallel_for`.

    Parameters
    ----------
    device_pre : DevicePre
        Object holding input/output buffers and precomputed spectral metadata.
    threads : int
        Number of threads per team in Kokkos' TeamPolicy.

    Notes
    -----
    - If `device_pre.data.f1` is present, the Stokeslet kernel is applied.
    - If `device_pre.data.f2` is present, the Stresslet kernel is applied.
    - Results are written in-place into `device_pre.data.Hg`.
    - Final output for mixed kernels (both SL and DL) is summed into the first three
      channels of `Hg`.
    """
    if pk.is_host_execution_space(device_pre.execution_space) and threads != 1:
        warnings.warn(
            "'threads' argument is overridden to 1 in OpenMP execution space.",
            RuntimeWarning,
        )
        threads = 1
    shift = 0
    freq_offset = device_pre.am.array([0, 0, 0], dtype=device_pre.am.int32)
    if device_pre.data.opt.periodicity == 1:
        # ignore zero mode in the x (periodic) direction
        # this will be written over later
        freq_offset[0] = 1
    freq_range = int(
        device_pre.am.prod(
            device_pre.am.array(device_pre.data.Hg.shape[1:-1]) - freq_offset
        )
    )
    freq_teams = math.ceil(freq_range / threads)
    match device_pre.kernel.upper():
        case "STOKES_COMB" | "STOKES_SL":
            # Stokeslet
            if device_pre.has_sl:
                G01 = [None] * 3
                for d in range(shift, shift + len(G01)):
                    G01[d - shift] = device_pre.data.Hg[d]
                RangePush("Sca-kernel")
                pk.parallel_for(
                    pk.TeamPolicy(device_pre.execution_space, freq_teams, threads),
                    device_pre.stokeslet_convolution_kernel,
                    H1=G01[0],
                    H2=G01[1],
                    H3=G01[2],
                    grid_size_1=device_pre.data.Hg.shape[2],
                    grid_size_2=device_pre.data.Hg.shape[3],
                    box=device_pre.data.opt.glb_box_ext,
                    k1_off=device_pre.data.opt.k1_off,
                    xi=device_pre.data.opt.xi,
                    pw=device_pre.data.opt_sc_glob.window_scaling_power,
                    wsh=device_pre.data.opt_sc_glob.window_shape,
                    ksc=device_pre.data.opt_sc_glob.kaiser_scaling,
                    grid=device_pre.data.opt.glb_grid_shape_ext,
                    whw=device_pre.data.opt_sc_glob.window_halfwidth,
                    ups=device_pre.data.opt.actual_upsampling_global,
                    locals=device_pre.am.zeros(1, dtype=device_pre.am.int32),
                    num_locals=0,
                    freq_range=freq_range,
                    freq_offset=freq_offset,
                    threads=threads,
                )
                RangePop()
                shift += 3
            # Stresslet
            if device_pre.has_dl:
                G02 = [[None for _ in range(3)] for _ in range(3)]
                for d in range(9):
                    G02[d % 3][d // 3] = device_pre.data.Hg[shift + d]
                RangePush("Sca-kernel")
                pk.parallel_for(
                    pk.TeamPolicy(device_pre.execution_space, freq_teams, threads),
                    device_pre.stresslet_convolution_kernel,
                    H11=G02[0][0],
                    H21=G02[1][0],
                    H31=G02[2][0],
                    H12=G02[0][1],
                    H22=G02[1][1],
                    H32=G02[2][1],
                    H13=G02[0][2],
                    H23=G02[1][2],
                    H33=G02[2][2],
                    grid_size_1=device_pre.data.Hg.shape[2],
                    grid_size_2=device_pre.data.Hg.shape[3],
                    box=device_pre.data.opt.glb_box_ext,
                    k1_off=device_pre.data.opt.k1_off,
                    xi=device_pre.data.opt.xi,
                    pw=device_pre.data.opt_sc_glob.window_scaling_power,
                    wsh=device_pre.data.opt_sc_glob.window_shape,
                    ksc=device_pre.data.opt_sc_glob.kaiser_scaling,
                    grid=device_pre.data.opt.glb_grid_shape_ext,
                    whw=device_pre.data.opt_sc_glob.window_halfwidth,
                    ups=device_pre.data.opt.actual_upsampling_global,
                    locals=device_pre.am.zeros(1, dtype=device_pre.am.int32),
                    num_locals=0,
                    freq_range=freq_range,
                    freq_offset=freq_offset,
                    threads=threads,
                )
                RangePop()
                shift += 9
            # sum
            RangePush("Sca-kernel")
            if device_pre.has_sl and device_pre.has_dl:
                pk.parallel_for(
                    pk.TeamPolicy(device_pre.execution_space, freq_teams * 2, threads),
                    device_pre.convolution_sum_sl_dl,
                    H1=G01[0],
                    H2=G01[1],
                    H3=G01[2],
                    H11=G02[0][0],
                    H21=G02[1][0],
                    H31=G02[2][0],
                    D1=device_pre.data.Hg[0],
                    D2=device_pre.data.Hg[1],
                    D3=device_pre.data.Hg[2],
                    grid_size_1=device_pre.data.Hg.shape[2],
                    grid_size_2=device_pre.data.Hg.shape[3],
                    freq_range=freq_range,
                    freq_offset=freq_offset,
                    threads=threads,
                )
            elif device_pre.has_sl and not device_pre.has_dl:
                pk.parallel_for(
                    pk.TeamPolicy(device_pre.execution_space, freq_teams * 2, threads),
                    device_pre.convolution_sum_sl,
                    H1=G01[0],
                    H2=G01[1],
                    H3=G01[2],
                    D1=device_pre.data.Hg[0],
                    D2=device_pre.data.Hg[1],
                    D3=device_pre.data.Hg[2],
                    grid_size_1=device_pre.data.Hg.shape[2],
                    grid_size_2=device_pre.data.Hg.shape[3],
                    freq_range=freq_range,
                    freq_offset=freq_offset,
                    threads=threads,
                )
            else:
                raise NotImplementedError(
                    "CNV summation not implemented for stokes_sl or stokes_dl case."
                )
            RangePop()
        case "LAPLACE":
            RangePush("Sca-kernel")
            pk.parallel_for(
                pk.TeamPolicy(device_pre.execution_space, freq_teams, threads),
                device_pre.laplace_convolution_kernel,
                H=device_pre.data.Hg[0],
                grid_size_1=device_pre.data.Hg.shape[2],
                grid_size_2=device_pre.data.Hg.shape[3],
                box=device_pre.data.opt.glb_box_ext,
                k1_off=device_pre.data.opt.k1_off,
                xi=device_pre.data.opt.xi,
                pw=device_pre.data.opt_sc_glob.window_scaling_power,
                wsh=device_pre.data.opt_sc_glob.window_shape,
                ksc=device_pre.data.opt_sc_glob.kaiser_scaling,
                grid=device_pre.data.opt.glb_grid_shape_ext,
                whw=device_pre.data.opt_sc_glob.window_halfwidth,
                ups=device_pre.data.opt.actual_upsampling_global,
                locals=device_pre.am.zeros(1, dtype=device_pre.am.int32),
                num_locals=0,
                freq_range=freq_range,
                freq_offset=freq_offset,
                threads=threads,
            )
            RangePop()
        case _:
            raise NotImplementedError(
                "CNV global implemented for 'STOKES_COMB' and 'LAPLACE' kernels, "
                f" got {device_pre.kernel.upper()}."
            )


def ifft(device_pre, options, fftmp_buffers=None):
    """
    Perform the inverse FFT to transform scaled Fourier-space data back to physical space.

    This function applies an inverse real FFT (irfftn) to the output of the convolution step
    stored in `device_pre.data.Hg`, reshaping and truncating the result to match the original
    physical grid shape. Only the case with uniform upsampling is currently supported.

    Parameters
    ----------
    device_pre : DevicePre
        Object containing data buffers, FFT metadata, and execution space.
    options: EwaldOptions
        Ewald options data class.

    Returns
    -------
    walltime : dict
        Dictionary with timing information for the IFFT kernel, under the key 'kernel'.

    Raises
    ------
    NotImplementedError
        If local or zero-mode-specific IFFT logic is required (i.e., `Hl` or `H0` are not None),
        a `NotImplementedError` is raised as this path is not yet supported.

    Notes
    -----
    - The IFFT is performed using a backend-specific FFT function (`get_irfftn`).
    - For OpenMP, the number of FFT workers is read from `OMP_NUM_THREADS`.
    - The output is stored in `device_pre.data.H` and truncated to the original grid shape.
    """
    ifft_start = time.time()
    if device_pre.data.opt.distributed:
        if fftmp_buffers is None:
            raise ValueError(
                "fftmp_buffers must be passed into fft for distributed execution."
            )
        if device_pre.execution_space == pk.ExecutionSpace.Cuda:
            import cupy as cp

            with cp.cuda.Device(fftmp_buffers.device_id):
                # Use stateful pre-planed ifft object 'f'
                with fftmp_buffers.ifft as f:

                    # loop over H dimensions
                    for d in range(device_pre.dim_out):
                        # copy to ifft buffers
                        fftmp_buffers.ifft_buff[:] = (
                            device_pre.data.Hg[d].view(np.complex128).squeeze(-1)
                        )
                        # Execute the IFFT.
                        device_pre.data.H[d] = (
                            f.execute()[
                                : device_pre.data.opt.grid_shape_ext[0],
                                : device_pre.data.opt.grid_shape_ext[1],
                                : device_pre.data.opt.grid_shape_ext[2],
                            ].real
                            / fftmp_buffers.fft_size
                        )
        else:
            raise NotImplementedError(
                "distributed fft only implemented for Cuda execution space,"
                f" got {device_pre.execution_space}."
            )
    else:
        if device_pre.data.Hl is not None:  # G0 is also not None
            raise NotImplementedError("Local upsampling fft not yet implemented.")
        else:
            # TODO: torch IFFT
            extra_args = {}
            if pk.is_host_execution_space(device_pre.execution_space):
                extra_args["workers"] = int(os.environ.get("OMP_NUM_THREADS"))
            RangePush("IFFT-kernel")
            ifftn = get_ifftn(device_pre.execution_space, device_pre.data.fft_type)
            device_pre.data.H[: device_pre.dim_out, ...] = ifftn(
                device_pre.data.Hg[: device_pre.dim_out, ...]
                .view(dtype=device_pre.data.complex_dtype)
                .squeeze(-1),
                axes=(1, 2, 3),
                overwrite_x=True,
                **extra_args,
            )[
                ...,
                : device_pre.data.opt.grid_shape_ext[0],
                : device_pre.data.opt.grid_shape_ext[1],
                : device_pre.data.opt.grid_shape_ext[2],
            ].real
            RangePop()
    ifft_end = time.time()
    walltime = {"tot": ifft_end - ifft_start}
    return walltime


def g2p(device_pre, method="TARGET", threads=128):
    """
    Gather-to-particle (G2P) step for evaluating far-field contributions
    at particle locations from the uniform grid.

    This function interpolates grid data from the Fourier space convolution
    (stored in `device_pre.data.H`) to the target particle locations, using either
    direct particle access or cell-based sorting. It optionally applies the
    Stokeslet zero-mode correction for doubly periodic domains.

    Parameters
    ----------
    device_pre : DevicePre
        Data structure containing grid data, particle locations, execution space,
        kernel types, and configuration parameters for the Spectral Ewald algorithm.

    method : str
        Interpolation method. One of:
            - 'BASE'   : Perform interpolation directly using the target array.
            - 'TARGET' : Sort target particles into cells before interpolating.
        Defaults to `'TARGET'`.

    threads : int, optional
        Number of threads to use per team in the Kokkos parallel kernel.
        For OpenMP execution space, this is overridden to 1 (default: 128).

    Returns
    -------
    walltime : dict
        Dictionary containing walltime for various stages of the G2P step:
            - 'args'   : Time to process arguments and set up.
            - 'sort'   : Time spent sorting particles (if applicable).
            - 'kernel' : Time spent executing the interpolation kernel.
            - 'adj'    : Time spent applying the Stokeslet zero-mode adjustment.

    Raises
    ------
    ValueError
        If `method` is not one of 'BASE' or 'TARGET'.

    RuntimeWarning
        If the execution space is OpenMP and the `threads` argument is overridden.

    Notes
    -----
    - If `has_sl` is True, a zero-mode correction term is computed and subtracted.
    - Cell-based sorting is triggered for 'TARGET' method using a `CellList`.
    - The final far-field potentials are scaled by `1/(8π)` as part of the solution.
    """
    walltime = {}
    walltime["tot"] = time.time()
    if pk.is_host_execution_space(device_pre.execution_space) and threads != 1:
        warnings.warn(
            "'threads' argument is overridden to 1 in OpenMP execution space.",
            RuntimeWarning,
        )
        threads = 1

    # scale targets to the Fourier grid
    offsets = device_pre.am.array(
        [
            (
                0.0
                if device_pre.data.opt.box_off[i] == 0
                else device_pre.data.opt.box_off[i] + 0.5 * device_pre.data.opt.h
            )
            for i in range(3)
        ],
        dtype=device_pre.data.dtype,
    ).reshape(3, 1)
    scaled_targets = (device_pre.data.targets - offsets) * device_pre.data.opt.grid_res
    # method info
    sort_flag = False
    method_flag = -1
    match method.upper():
        case "BASE":
            method_flag = 0
            H_view = device_pre.data.ghost_H.transpose(1, 2, 3, 0).copy()

            walltime["kernel"] = time.time()
            # setup Kokkos workunit call
            teams = math.ceil(device_pre.Nt / threads)
            policy = pk.TeamPolicy(device_pre.execution_space, teams, threads)
            kwargs = {
                "u": device_pre.far_potential,
                "x": scaled_targets,
                "H": H_view,
                "window_P": device_pre.data.opt.window_P,
                "dim_out": device_pre.dim_out,
                "H_shape": device_pre.am.asarray(
                    device_pre.data.opt.ghost_grid_shape_ext
                ),
                "hhh": float(device_pre.data.opt.h) ** 3,
                "periodicity": (
                    device_pre.data.opt.periodicity
                    if not device_pre.data.opt.distributed
                    else 0
                ),
                "nt": device_pre.Nt,
                "threads": threads,
            }

        case "TARGET":
            method_flag = 1
            H_view = device_pre.data.ghost_H

            # sort target arrays
            walltime["sort"] = time.time()
            target_list = CellList(
                scaled_targets,
                device_pre.data.opt.window_P / 2,
                device_pre.am.array(device_pre.data.opt.ghost_grid_shape_ext).astype(
                    device_pre.data.dtype
                ),
                execution_space=device_pre.execution_space,
            )
            walltime["sort"] = time.time() - walltime["sort"]

            # setup Kokkos workunit call
            teams_per_cell = math.ceil(target_list.cell_size / threads)
            policy = pk.TeamPolicy(
                device_pre.execution_space,
                target_list.num_nonempty_cells * teams_per_cell,
                threads,
            )
            kwargs = {
                "u": device_pre.far_potential,
                "x": target_list.particle_list,
                "x_index": target_list.particle_index,
                "H": H_view,
                "window_P": device_pre.data.opt.window_P,
                "dim_out": device_pre.dim_out,
                "H_shape": device_pre.am.asarray(
                    device_pre.data.opt.ghost_grid_shape_ext
                ),
                "hhh": float(device_pre.data.opt.h) ** 3,
                "cell_size": target_list.cell_size,
                "periodicity": (
                    device_pre.data.opt.periodicity
                    if not device_pre.data.opt.distributed
                    else 0
                ),
                "threads": threads,
                "cell_teams": teams_per_cell,
            }
        case _:
            raise ValueError(
                f"g2p method expected to be one of 'BASE', 'TARGET', got '{method.upper()}'."
            )

    walltime["kernel"] = time.time()
    RangePush("G2P-kernel")
    pk.parallel_for(
        f"G2P-{method.upper()}",
        policy,
        device_pre.g2p_workunit[method.upper()],
        **kwargs,
    )
    RangePop()
    walltime["kernel"] = time.time() - walltime["kernel"]

    # Compute adjustment
    walltime["adj"] = time.time()
    match device_pre.kernel.upper():
        case "STOKES_COMB" | "STOKES_SL":
            if device_pre.has_sl:
                match device_pre.data.opt.periodicity:
                    case 1:
                        sl_sum = np.empty(3, dtype=device_pre.data.dtype)
                        try:  # TODO: replace with isdevicespace
                            sl_sum[...] = device_pre.am.sum(
                                device_pre.data.forces[:3, : device_pre.data.owned_ns],
                                axis=1,
                            ).get()
                        except AttributeError:
                            sl_sum[...] = device_pre.am.sum(
                                device_pre.data.forces[:3, : device_pre.data.owned_ns],
                                axis=1,
                            )
                        if device_pre.data.opt.distributed:
                            from mpi4py import MPI

                            mpi_comm = MPI.COMM_WORLD
                            sl_sum_total = np.zeros_like(sl_sum)
                            mpi_comm.Allreduce(sl_sum, sl_sum_total, op=MPI.SUM)
                            sl_sum = sl_sum_total

                        sl_sum = device_pre.am.asarray(sl_sum)
                        lB_out = device_pre.am.exp(
                            0
                        )  # stokeslet_k0_constant forced to 0.
                        lH_out = lB_out / device_pre.am.exp(1)
                        lB_in = (
                            device_pre.data.opt_sc_glob.greens_truncation_R
                            * device_pre.am.exp(1 / 2)
                        )
                        lH_in = device_pre.data.opt_sc_glob.greens_truncation_R

                        adj = device_pre.am.array([0, 0, 0])
                        if (
                            device_pre.data.opt_sc_glob.vico_var is None
                            or device_pre.data.opt_sc_glob.vico_var == 2
                        ):
                            adj = (
                                device_pre.am.log(lB_in / lB_out)
                                / device_pre.data.opt.glb_box[0]
                            ) * device_pre.am.array([0, 2, 2]) * sl_sum + (
                                device_pre.am.log(lH_in / lH_out)
                                / device_pre.data.opt.glb_box[0]
                            ) * device_pre.am.array(
                                [4, 0, 0]
                            ) * sl_sum
                        elif device_pre.data.opt_sc_glob.vico_var == 1:
                            adj = (
                                (
                                    device_pre.am.log(lB_in / lB_out)
                                    / device_pre.data.opt.glb_box[0]
                                )
                                * device_pre.am.array([4, 2, 2])
                                * sl_sum
                            )
                        device_pre.far_potential -= adj.reshape(3, 1)
                    case 3:
                        pass
                    case _:
                        raise NotImplementedError(
                            "Stokeslet adjustment is only implemented for periodicities 1 and 3,"
                            f" got {device_pre.data.opt.periodicity}."
                        )
        case "LAPLACE":
            pass
        case _:
            raise NotImplementedError(
                "G2P adjustment is only implemented for 'STOKES_COMB' and 'LAPLACE' kernels,"
                f" got {device_pre.kernel.upper()}."
            )
    walltime["adj"] = time.time() - walltime["adj"]

    walltime["tot"] = time.time() - walltime["tot"]
    return walltime
