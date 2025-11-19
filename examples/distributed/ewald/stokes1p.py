#!python3

"""
Ewald summation for the combined stokes single and double
layer potential with one periodic direction executed with
multiple processes.

>>> mpiexec -n 2 python examples/ewald/stokes1p_mp.py 
"""
import sys
import pprint
import argparse
import parkipy
import numpy as np
from mpi4py import MPI


def main(args):
    execution_space = parkipy.utils.get_execution_space(args.device)
    am = parkipy.utils.get_array_module(execution_space)

    # set up MPI COMM WORD
    mpi_comm = MPI.COMM_WORLD
    size = mpi_comm.Get_size()
    rank = mpi_comm.Get_rank()

    # set spectral Ewald box and tolerance
    box = [size, 1, 1]
    tol = 1e-4
    cell_size = 224

    # generate *local* sources, targets, densities, and normals
    nt = int(4e+6)
    ns = int(4e+6)

    rc = np.ceil(ns/cell_size) ** (-1 / 3)

    trg = am.random.rand(3, nt) * am.array(box).reshape(3, 1)
    src = am.random.rand(3, ns) * am.array(box).reshape(3, 1)
    dens_sl = am.random.randn(3, ns)
    dens_dl = am.random.randn(3, ns)
    norms = am.random.randn(3, ns)
    dens = am.vstack((dens_sl, dens_dl))  # stack densities for ewald call

    # call the PDE kernel
    pot, trg, dist_time = parkipy.distributed.ewald.stokes_comb(
        trg,
        src,
        dens,
        norms,
        1,
        box,
        tol,
        args.device,
        rc=rc,
        time=True,
    )

    if rank == 0:
        print(f"[{rank}]--------- dist runtimes ---------------", flush=True)
        pprint.pprint(dist_time)
        sys.stdout.flush()

    nt = trg.shape[1]

    nt_ref = mpi_comm.allreduce(nt, op=MPI.SUM)
    ns_ref = mpi_comm.allreduce(ns, op=MPI.SUM)
    # cpu arrays for gather
    pot_mpi = np.empty((3, nt_ref))
    trg_ref = np.empty((3, nt_ref))
    src_ref = np.empty((3, ns_ref))
    dens_sl_ref = np.empty((3, ns_ref))
    dens_dl_ref = np.empty((3, ns_ref))
    norms_ref = np.empty((3, ns_ref))
    pot_mpi[:, :nt] = pot.get()
    trg_ref[:, :nt] = trg.get()
    src_ref[:, :ns] = src.get()
    dens_sl_ref[:, :ns] = dens_sl.get()
    dens_dl_ref[:, :ns] = dens_dl.get()
    norms_ref[:, :ns] = norms.get()

    parkipy.distributed.gather_points(mpi_comm, "host", pot_mpi.T, nt)
    parkipy.distributed.gather_points(mpi_comm, "host", trg_ref.T, nt)
    parkipy.distributed.gather_points(mpi_comm, "host", src_ref.T, ns)
    parkipy.distributed.gather_points(mpi_comm, "host", dens_sl_ref.T, ns)
    parkipy.distributed.gather_points(mpi_comm, "host", dens_dl_ref.T, ns)
    parkipy.distributed.gather_points(mpi_comm, "host", norms_ref.T, ns)

    if rank == 0:
        # send ref arrays back to gpu
        trg_ref = am.asarray(trg_ref)
        src_ref = am.asarray(src_ref)
        dens_sl_ref = am.asarray(dens_sl_ref)
        dens_dl_ref = am.asarray(dens_dl_ref)
        norms_ref = am.asarray(norms_ref)
        dens_ref = am.vstack((dens_sl_ref, dens_dl_ref))

        options = parkipy.ewald.EwaldOptions(periodicity=1, box=box, tolerance=tol, execution_space=args.device, cell_size=224, return_walltime=True)

        pot_ref, ref_time = parkipy.ewald.stokes_comb(
            trg_ref,
            src_ref,
            dens_ref,
            norms_ref,
            options
        )
        print("--------- ref runtimes ---------------")
        pprint.pprint(ref_time)

        rtol = atol = 1e-5
        np.testing.assert_allclose(pot_ref.get(), pot_mpi, rtol=rtol, atol=atol)
        print(
            f"mpi and reference single-node potentials agree (at least) up to rtol={rtol}, atol={atol}."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined Stokes single and double layer potential with 1 periodic direction."
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="Cuda",
        help="Execution device.",
    )
    args = parser.parse_args()
    exit(main(args))
