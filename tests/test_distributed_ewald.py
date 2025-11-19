"""
Test the distributed execution of each Ewald stage.

>>> mpiexec -n 2 pytest tests/test_distributed_ewald.py
"""

import pytest
import numpy as np
import cupy as cp
from mpi4py import MPI

import parkipy


def setup_device_pre_stokes(
    box,
    tol,
    rc,
    periodicity,
    trg,
    src,
    dens,
    norms,
    execution_space,
    fft_type,
    kernel,
    distributed,
):
    device_pre, params = parkipy.ewald._prepare.DevicePre.from_particles(
        targets=trg,
        sources=src,
        forces=dens,
        normals=norms,
        kernel="STOKES_COMB",
        box=box,
        tolerance=tol,
        rc=rc,
        periodicity=periodicity,
        execution_space=execution_space,
        fft_type=fft_type,
        distributed=distributed,
    )

    return device_pre


def setup_stokes(nt, ns, execution_space):
    am = parkipy.utils.get_array_module(execution_space)

    mpi_comm = MPI.COMM_WORLD
    size = mpi_comm.Get_size()
    rank = mpi_comm.Get_rank()

    box = [size, 1, 1]
    tol = 1e-7

    trg = am.random.rand(3, nt) * am.array(box).reshape(3, 1)
    src = am.random.rand(3, ns) * am.array(box).reshape(3, 1)
    dens_sl = am.random.rand(3, ns)
    dens_dl = am.random.rand(3, ns)
    norms = am.random.rand(3, ns)
    dens = am.vstack((dens_sl, dens_dl))  # stack densities for ewald call

    device_pre_loc = setup_device_pre_stokes(
        box,
        tol,
        0.2,
        1,
        trg,
        src,
        dens,
        norms,
        "Cuda",
        "C2C",
        "stokes_comb",
        distributed=True,
    )

    nt_glb = mpi_comm.allreduce(nt, op=MPI.SUM)
    ns_glb = mpi_comm.allreduce(ns, op=MPI.SUM)
    # cpu arrays for gather
    trg_glb = np.empty((3, nt_glb))
    src_glb = np.empty((3, ns_glb))
    dens_sl_glb = np.empty((3, ns_glb))
    dens_dl_glb = np.empty((3, ns_glb))
    norms_glb = np.empty((3, ns_glb))
    trg_glb[:, :nt] = trg.get()
    src_glb[:, :ns] = src.get()
    dens_sl_glb[:, :ns] = dens_sl.get()
    dens_dl_glb[:, :ns] = dens_dl.get()
    norms_glb[:, :ns] = norms.get()

    parkipy.distributed._utils.gather_points(mpi_comm, "host", trg_glb.T, nt)
    parkipy.distributed._utils.gather_points(mpi_comm, "host", src_glb.T, ns)
    parkipy.distributed._utils.gather_points(mpi_comm, "host", dens_sl_glb.T, ns)
    parkipy.distributed._utils.gather_points(mpi_comm, "host", dens_dl_glb.T, ns)
    parkipy.distributed._utils.gather_points(mpi_comm, "host", norms_glb.T, ns)
    device_pre_glb = None

    if rank == 0:
        # send glb arrays back to gpu
        trg_glb = cp.asarray(trg_glb)
        src_glb = cp.asarray(src_glb)
        dens_sl_glb = cp.asarray(dens_sl_glb)
        dens_dl_glb = cp.asarray(dens_dl_glb)
        norms_glb = cp.asarray(norms_glb)
        dens_glb = cp.vstack((dens_sl_glb, dens_dl_glb))

        device_pre_glb = setup_device_pre_stokes(
            box,
            tol,
            0.2,
            1,
            trg_glb,
            src_glb,
            dens_glb,
            norms_glb,
            "Cuda",
            "C2C",
            "stokes_comb",
            distributed=False,
        )
    return device_pre_loc, device_pre_glb


@pytest.mark.parametrize("d", [3, 12])
def test_gather_points(d):
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.rank
    size = mpi_comm.size

    # each rank set up points
    n = 483
    loc = cp.arange(start=rank * (n * d), stop=(rank + 1) * (n * d)).reshape(d, -1)
    glb = np.empty((d, n * size))
    glb[:, :n] = loc.get()
    parkipy.distributed._utils.gather_points(mpi_comm, "host", glb.T, n)
    if rank == 0:
        ref = []
        for i in range(size):
            ref.append(
                np.arange(start=(i) * (n * d), stop=(i + 1) * (n * d)).reshape(d, -1)
            )
        ref = np.hstack(ref)
        np.testing.assert_allclose(glb, ref)


@pytest.mark.parametrize("method", ["GM-1D", "GM-2D", "SM-1D", "SM-2D"])
def test_p2p(method):
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.rank
    size = mpi_comm.size

    device_pre_loc, device_pre_glb = setup_stokes(996, 910, "cuda")
    if mpi_comm.rank == 0:
        parkipy.ewald._ewald.p2p(device_pre_glb)
        ref_sol = device_pre_glb.near_potential.get()
        ref_trg = device_pre_glb.data.targets.get()

    # get local solution
    parkipy.ewald._ewald.p2p(device_pre_loc, method=method)
    mpi_sol = np.empty((3, size * 996))
    mpi_trg = np.empty((3, size * 996))
    nt_loc = device_pre_loc.near_potential.shape[1]
    mpi_sol[:, :nt_loc] = device_pre_loc.near_potential.get()
    mpi_trg[:, :nt_loc] = device_pre_loc.data.targets.get()
    x_off = device_pre_loc.data.box_dict["left"] - (device_pre_loc.data.opt.ghost_dist)
    mpi_trg[0] += x_off

    # gather mpi_solution
    parkipy.distributed._utils.gather_points(mpi_comm, "host", mpi_sol.T, nt_loc)
    parkipy.distributed._utils.gather_points(mpi_comm, "host", mpi_trg.T, nt_loc)

    if mpi_comm.rank == 0:
        # sort the targets to compare potentials
        ref_args = np.argsort(ref_trg[0])
        mpi_args = np.argsort(mpi_trg[0])
        np.testing.assert_allclose(
            mpi_trg[:, mpi_args],
            ref_trg[:, ref_args],
            rtol=0,
            atol=1e-15,
            err_msg="sorted targets are unequal",
        )

        # check the potentials
        np.testing.assert_allclose(
            mpi_sol[:, mpi_args],
            ref_sol[:, ref_args],
            rtol=0,
            atol=1e-8,
            err_msg="sorted potentials are unequal",
        )


@pytest.mark.parametrize("method", ["BASE", "SOURCE", "GRID", "HYBRID"])
def test_p2g(method):
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.rank
    size = mpi_comm.size

    device_pre_loc, device_pre_glb = setup_stokes(996, 910, "cuda")

    if rank == 0:
        parkipy.ewald._ewald.p2g(device_pre_glb)
        ref_sol = device_pre_glb.data.H.get()

    parkipy.ewald._ewald.p2g(device_pre_loc, method)
    loc_sol = device_pre_loc.data.H.get()
    ng_loc = np.prod(loc_sol.shape[1:])

    mpi_sol = np.empty((12, size * loc_sol.shape[1], *loc_sol.shape[2:]))
    mpi_sol[:, : loc_sol.shape[1], ...] = loc_sol
    mpi_sol = mpi_sol.reshape(12, -1)

    parkipy.distributed._utils.gather_points(mpi_comm, "host", mpi_sol.T, ng_loc)

    if mpi_comm.rank == 0:
        mpi_sol = mpi_sol.reshape(ref_sol.shape)
        try:
            np.testing.assert_allclose(mpi_sol, ref_sol, rtol=0, atol=1e-13)
        except AssertionError as e:
            mismatch = ~np.isclose(mpi_sol[0], ref_sol[0], rtol=0, atol=1e-13)
            mismatch_indices = np.argwhere(mismatch)
            print("Mismatch detected at indices:")
            print(mismatch_indices)
            print("Values at mismatched indices:")
            for i, idx in enumerate(mismatch_indices):
                if i >= 5:
                    break
                print(
                    f"Index {tuple(idx)}: mpi_sol={mpi_sol[:, *tuple(idx)]}, ref_sol={ref_sol[:, *tuple(idx)]}"
                )
            raise e


def test_fft():
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.rank
    size = mpi_comm.size

    device_pre_loc, device_pre_glb = setup_stokes(996, 910, "cuda")

    if rank == 0:
        parkipy.ewald._ewald.p2g(device_pre_glb)
        parkipy.ewald._ewald.fft(device_pre_glb)
        ref_sol = device_pre_glb.data.Hg.get()

    parkipy.ewald._ewald.p2g(device_pre_loc)
    fftmp_buffers = parkipy.distributed.ewald._fft_utils.FFTMPBuffers(
        fft_shape=device_pre_loc.data.fft_shape,
        ifft_shape=device_pre_loc.data.Hg.shape[1:-1],
    )
    parkipy.ewald._ewald.fft(device_pre_loc, fftmp_buffers)
    del fftmp_buffers  # free symmetic heap if execution is distributed
    loc_sol = device_pre_loc.data.Hg.get()
    ng_loc = np.prod(loc_sol.shape[1:])

    mpi_sol_shape = [12, size * loc_sol.shape[2], loc_sol.shape[1], *loc_sol.shape[3:]]
    mpi_sol = np.empty(mpi_sol_shape)
    mpi_sol[:, : loc_sol.shape[2], ...] = loc_sol.transpose(0, 2, 1, 3, 4)
    mpi_sol = mpi_sol.reshape(12, -1)

    parkipy.distributed._utils.gather_points(mpi_comm, "host", mpi_sol.T, ng_loc)

    if mpi_comm.rank == 0:
        mpi_sol = mpi_sol.reshape(mpi_sol_shape).transpose(0, 2, 1, 3, 4)
        np.testing.assert_allclose(mpi_sol, ref_sol, rtol=0, atol=1e-11)


def test_cnv():
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.rank
    size = mpi_comm.size

    device_pre_loc, device_pre_glb = setup_stokes(996, 910, "cuda")

    if rank == 0:
        parkipy.ewald._ewald.p2g(device_pre_glb)
        parkipy.ewald._ewald.fft(device_pre_glb)
        parkipy.ewald._ewald.cnv(device_pre_glb)
        ref_sol = device_pre_glb.data.Hg.get()

    parkipy.ewald._ewald.p2g(device_pre_loc)
    fftmp_buffers = parkipy.distributed.ewald._fft_utils.FFTMPBuffers(
        fft_shape=device_pre_loc.data.fft_shape,
        ifft_shape=device_pre_loc.data.Hg.shape[1:-1],
    )
    parkipy.ewald._ewald.fft(device_pre_loc, fftmp_buffers)
    del fftmp_buffers  # free symmetic heap if execution is distributed
    parkipy.ewald._ewald.cnv(device_pre_loc)
    loc_sol = device_pre_loc.data.Hg.get()
    ng_loc = np.prod(loc_sol.shape[1:])

    mpi_sol_shape = [12, size * loc_sol.shape[2], loc_sol.shape[1], *loc_sol.shape[3:]]
    mpi_sol = np.empty(mpi_sol_shape)
    mpi_sol[:, : loc_sol.shape[2], ...] = loc_sol.transpose(0, 2, 1, 3, 4)
    mpi_sol = mpi_sol.reshape(12, -1)

    parkipy.distributed._utils.gather_points(mpi_comm, "host", mpi_sol.T, ng_loc)

    if mpi_comm.rank == 0:
        mpi_sol = mpi_sol.reshape(mpi_sol_shape).transpose(0, 2, 1, 3, 4)
        try:
            np.testing.assert_allclose(mpi_sol, ref_sol, rtol=1e-7, atol=1e-6)
        except AssertionError as e:
            mismatch = ~np.isclose(mpi_sol, ref_sol, rtol=1e-7, atol=1e-6)
            mismatch_indices = np.argwhere(mismatch)
            print(mismatch_indices)
            print("Values at mismatched indices:")
            for i, idx in enumerate(mismatch_indices):
                if i >= 5:
                    break
                print(
                    f"Index {tuple(idx)}:\n mpi_sol={mpi_sol[*tuple(idx)]},\n ref_sol={ref_sol[*tuple(idx)]}\n"
                )
            raise e


def test_ifft():
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.rank
    size = mpi_comm.size

    device_pre_loc, device_pre_glb = setup_stokes(996, 910, "cuda")

    if rank == 0:
        parkipy.ewald._ewald.p2g(device_pre_glb)
        parkipy.ewald._ewald.fft(device_pre_glb)
        parkipy.ewald._ewald.cnv(device_pre_glb)
        parkipy.ewald._ewald.ifft(device_pre_glb)
        ref_sol = device_pre_glb.data.H.get()

    parkipy.ewald._ewald.p2g(device_pre_loc)
    fftmp_buffers = parkipy.distributed.ewald._fft_utils.FFTMPBuffers(
        fft_shape=device_pre_loc.data.fft_shape,
        ifft_shape=device_pre_loc.data.Hg.shape[1:-1],
    )
    parkipy.ewald._ewald.fft(device_pre_loc, fftmp_buffers)
    parkipy.ewald._ewald.cnv(device_pre_loc)
    parkipy.ewald._ewald.ifft(device_pre_loc, fftmp_buffers)
    del fftmp_buffers  # free symmetic heap if execution is distributed
    loc_sol = device_pre_loc.data.H.get()
    ng_loc = np.prod(loc_sol.shape[1:])

    mpi_sol = np.empty((12, size * loc_sol.shape[1], *loc_sol.shape[2:]))
    mpi_sol[:, : loc_sol.shape[1], ...] = loc_sol
    mpi_sol = mpi_sol.reshape(12, -1)

    parkipy.distributed._utils.gather_points(mpi_comm, "host", mpi_sol.T, ng_loc)

    print(device_pre_loc.data.opt.glb_grid_shape_ext)

    if mpi_comm.rank == 0:
        mpi_sol = mpi_sol.reshape(ref_sol.shape)
        try:
            np.testing.assert_allclose(mpi_sol[:3], ref_sol[:3], rtol=0, atol=1e-5)
        except AssertionError as e:
            print(mpi_sol.shape, ref_sol.shape)
            mismatch = ~np.isclose(mpi_sol[0], ref_sol[0], rtol=0, atol=1e-5)
            mismatch_indices = np.argwhere(mismatch)
            print("Mismatch detected at indices:")
            print(mismatch_indices)
            print("Values at mismatched indices:")
            for i, idx in enumerate(mismatch_indices):
                if i >= 5:
                    break
                print(
                    f"Index {tuple(idx)}:\n mpi_sol={mpi_sol[:3, *tuple(idx)]},\n ref_sol={ref_sol[:3, *tuple(idx)]}\n"
                )
            print("Correct detected at indices:")
            correct_indices = np.argwhere(~mismatch)
            print(correct_indices)
            raise e


@pytest.mark.parametrize("method", ["BASE", "TARGET"])
def test_g2p(method):
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.rank
    size = mpi_comm.size

    device_pre_loc, device_pre_glb = setup_stokes(996, 910, "cuda")

    if rank == 0:
        parkipy.ewald._ewald.p2g(device_pre_glb)
        parkipy.ewald._ewald.fft(device_pre_glb)
        parkipy.ewald._ewald.cnv(device_pre_glb)
        parkipy.ewald._ewald.ifft(device_pre_glb)
        parkipy.ewald._ewald.g2p(device_pre_glb, method=method)
        ref_sol = device_pre_glb.far_potential.get()
        ref_trg = device_pre_glb.data.targets.get()

    parkipy.ewald._ewald.p2g(device_pre_loc)
    fftmp_buffers = parkipy.distributed.ewald._fft_utils.FFTMPBuffers(
        fft_shape=device_pre_loc.data.fft_shape,
        ifft_shape=device_pre_loc.data.Hg.shape[1:-1],
    )
    parkipy.ewald._ewald.fft(device_pre_loc, fftmp_buffers)
    parkipy.ewald._ewald.cnv(device_pre_loc)
    parkipy.ewald._ewald.ifft(device_pre_loc, fftmp_buffers)
    del fftmp_buffers  # free symmetic heap if execution is distributed
    device_pre_loc.data.communicate_ghost_grid_cells()
    parkipy.ewald._ewald.g2p(device_pre_loc, method=method)
    loc_sol = device_pre_loc.far_potential.get()
    nt_loc = np.prod(loc_sol.shape[1:])

    mpi_sol = np.empty((3, size * 996))
    mpi_trg = np.empty((3, size * 996))
    mpi_sol[:, :nt_loc] = loc_sol
    mpi_trg[:, :nt_loc] = device_pre_loc.data.targets.get()
    x_off = device_pre_loc.data.box_dict["left"] - (device_pre_loc.data.opt.ghost_dist)
    mpi_trg[0] += x_off

    parkipy.distributed._utils.gather_points(mpi_comm, "host", mpi_sol.T, nt_loc)
    parkipy.distributed._utils.gather_points(mpi_comm, "host", mpi_trg.T, nt_loc)

    if mpi_comm.rank == 0:
        ref_args = np.argsort(ref_trg[0])
        mpi_args = np.argsort(mpi_trg[0])
        np.testing.assert_allclose(
            mpi_trg[:, mpi_args],
            ref_trg[:, ref_args],
            rtol=0,
            atol=1e-15,
            err_msg="sorted targets are unequal",
        )
        mpi_sol = mpi_sol[:, mpi_args]
        ref_sol = ref_sol[:, ref_args]
        try:
            # check the potentials
            np.testing.assert_allclose(
                mpi_sol,
                ref_sol,
                rtol=0,
                atol=1e-10,
                err_msg="sorted potentials are unequal",
            )
        except AssertionError as e:
            mismatch = ~np.isclose(mpi_sol[0], ref_sol[0], rtol=0, atol=1e-10)
            mismatch_indices = np.argwhere(mismatch)
            print(mismatch_indices)
            print("Values at mismatched indices:")
            for i, idx in enumerate(mismatch_indices):
                if i >= 5:
                    break
                print(
                    f"Index {tuple(idx)}:\n mpi_sol={mpi_sol[:, *tuple(idx)]},\n ref_sol={ref_sol[:,*tuple(idx)]}\n"
                )
            print("Correct detected at indices:")
            correct_indices = np.argwhere(~mismatch)
            print(correct_indices)
            raise e
