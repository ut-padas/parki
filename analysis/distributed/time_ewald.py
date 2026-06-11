import os

""" the following try/execpt block is a hacky workaround to speedup scipt FFTs when conflicting with pykokkos"""
try:
    import numpy as np
    from scipy.fft import rfftn

    workers = int(os.environ.get("OMP_NUM_THREADS"))
    xfft = rfftn(np.ones((2, 2, 2, 2)), axes=(1, 2, 3), workers=workers)
except:
    pass

import pprint
import time
import argparse
import numpy as np
import pickle

import parkipy
from mpi4py import MPI


def main(args):
    """
    Main function. Takes `args` from the ArgumentParser at the bottom of this
    file. `args.nt` will be a list of integers, and we run `run()` with each of
    these integers as the number of targets.
    """
    nt_list = args.nt
    repeats = 5
    all_times = dict()
    all_memory = dict()
    if args.device.upper() == "OPENMP":
        threads = [1]
    else:
        threads = [32, 64, 128, 256, 512]  # only used for single node results
    for i, nt in enumerate(nt_list):
        args.nt = nt
        ns = args.nt * args.up
        print(f":: Running Nt={nt}")
        for j in range(repeats):
            tot_times = []
            for nthreads in threads:
                print(f":::: Running threads={nthreads}")
                args.threads = nthreads
                potential, times = run(args, time_every_step=True, verbosity=1)
                tot_times.append(times)
            print(f"   {j+1}th pass done (out of {repeats})")
            # Initialize arrays for storing runtimes and memory
            if i == 0 and j == 0:
                for key in times:
                    all_times[key] = np.full(
                        shape=[repeats, len(nt_list)], fill_value=np.inf
                    )
            # Store runtimes
            for times in tot_times:
                for key, val in times.items():
                    old_time = all_times[key][j, i]
                    if isinstance(val, dict):
                        new_time = val["tot"]
                    else:
                        new_time = val
                    if new_time < old_time:
                        all_times[key][j, i] = new_time
    # set up MPI COMM WORD
    mpi_comm = MPI.COMM_WORLD
    size = mpi_comm.Get_size()
    rank = mpi_comm.Get_rank()
    if rank == 0:
        save_times_to_disk(nt_list, repeats, all_times, args)
        pprint.pprint(all_times)
    return


def save_times_to_disk(nt, repeats, times, args):
    # set up MPI COMM WORD
    mpi_comm = MPI.COMM_WORLD
    size = mpi_comm.Get_size()
    now_str = time.strftime("%y%m%dT%H%M%S%Z")
    format_version = 1
    if args.device.upper() == "OPENMP":
        args.device = "host"
    if args.device.upper() == "HOST":
        import platform

        arch = platform.processor()
    else:
        import cupy as cp

        arch = cp.cuda.Device(0).compute_capability
    fname_base = (
        f"distributed_ewald_timing_result_N{size}_up{args.up}_clsz{args.cell_size}_tol{args.tolerance}"
        f"_dev{args.device.upper()}_arch{arch}_v{format_version}"
    )
    fname = fname_base + f"_{now_str}.pkl"
    fname_link = fname_base + "_latest.pkl"
    fpath = os.path.join(args.output_dir, fname)
    fpath_link = os.path.join(args.output_dir, fname_link)
    with open(fpath, "wb") as f:
        pickle.dump(
            {
                "nt": nt,
                "repeats": repeats,
                "times": times,
                "args": args,
            },
            f,
        )
    if os.path.lexists(fpath_link):
        os.remove(fpath_link)
    os.symlink(fname, fpath_link)
    print(f"Saved results to {fpath!r} and {fpath_link!r}")
    return


def run(args, time_every_step=False, verbosity=0) -> None:
    # set up MPI COMM WORD
    mpi_comm = MPI.COMM_WORLD
    size = mpi_comm.Get_size()
    rank = mpi_comm.Get_rank()

    # input arguments
    nt = args.nt
    box = np.array([size, 1.0, 1.0])

    ns = nt * args.up
    rc = np.ceil(ns / args.cell_size) ** (-1 / 3)

    cp = parkipy.utils.get_array_module(args.device)
    trg = cp.random.rand(3, nt) * cp.array(box).reshape(3, 1)
    src = cp.random.rand(3, ns) * cp.array(box).reshape(3, 1)
    dens_sl = cp.random.randn(3, ns)
    dens_dl = cp.random.randn(3, ns)
    norms = cp.random.randn(3, ns)
    dens = cp.vstack((dens_sl, dens_dl))  # stack densities for ewald call

    if size == 1:
        options = parkipy.ewald.EwaldOptions(
            periodicity=1,
            box=box,
            tolerance=args.tolerance,
            execution_space=args.device,
            cell_size=args.cell_size,
            return_walltime=True,
            p2g_threads=args.threads,
            g2p_threads=args.threads,
            p2p_threads_x=args.threads,
            fft_type=args.fft_type,
        )
        pot, walltimes = parkipy.ewald.stokes_comb(trg, src, dens, norms, options)
        walltimes = {
            "p2p": walltimes.time_p2p,
            "p2g": walltimes.time_p2g,
            "fft": walltimes.time_fft,
            "cnv": walltimes.time_cnv,
            "ifft": walltimes.time_ifft,
            "g2p": walltimes.time_g2p,
        }
    else:
        pot, trg, walltimes = parkipy.distributed.ewald.stokes_comb(
            trg,
            src,
            dens,
            norms,
            1,
            box,
            args.tolerance,
            args.device,
            rc=rc,
            time=True,
            fft_type=args.fft_type,
        )

    # Compute and store runtimes in dict
    return pot, walltimes


if __name__ == "__main__":
    slurm_script = """
#!/bin/bash
#SBATCH -A <PROJECT NAME>         # project name
#SBATCH -e %x.J%j.err             # error file name
#SBATCH -o %x.J%j.out             # output file name
#SBATCH -N <NUM GPUs>             # request nodes
#SBATCH --ntasks-per-node 1       # MPI tasks per node
#SBATCH -t 15                     # designate max run time (minutes)
#SBATCH -p <QUEUE>                # designate queue

echo "[SLURMJOB] parki Ewal Multi-GPU timing."

# Setup environment below

echo "[SLURMJOB] Setup done"


# Run the actual workload
ibrun python <PATH-TO-PARKI>/analysis/distributed/time_ewald.py --device "cuda" --fft_type "R2C"
    """
    parser = argparse.ArgumentParser(
        description="Weak scaling test for Ewald sum on multiple GPUs."
        "An example slurm script to launch on multiple GPUs:\n" + slurm_script,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    default_nt = [4000000]
    default_nt_str = " ".join([str(x) for x in default_nt])
    parser.add_argument(
        "--nt",
        dest="nt",
        type=int,
        nargs="+",
        default=default_nt,
        help=(
            "Set the number of target points, multiple values accepted"
            f" (default: {default_nt_str})"
        ),
    )
    parser.add_argument(
        "--up",
        dest="up",
        type=int,
        default=1,
        help="Set the upsampeling parameter (default: 16)",
    )
    parser.add_argument(
        "--cell_size",
        dest="cell_size",
        type=int,
        default=224,
        help="Set the max src per cell (default: 1024)",
    )
    parser.add_argument(
        "--fft_type",
        type=str,
        choices=["R2C", "C2C"],
        default="R2C",
        help="Type of FFT transform",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        required=True,
        help="Device to run code on",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="analysis/distributed/data",
        help="output directory for timing results (default: .)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Spectral Ewald tolerance",
    )

    args = parser.parse_args()
    main(args)
    exit()
