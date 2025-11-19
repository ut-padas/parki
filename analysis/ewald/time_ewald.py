import os
""" the following try/execpt block is a hacky workaround to speedup scipt FFTs when conflicting with pykokkos"""
try:
    import numpy as np
    from scipy.fft import rfftn
    workers = int(os.environ.get("OMP_NUM_THREADS"))
    xfft = rfftn(np.ones((2,2,2,2)), axes = (1, 2, 3), workers=workers)
except:
    pass

import time
import argparse
import numpy as np
import pickle

import parkipy

def main(args):
    """
    Main function. Takes `args` from the ArgumentParser at the bottom of this
    file. `args.nt` will be a list of integers, and we run `run()` with each of
    these integers as the number of targets.
    """
    nt_list = [250000, 1000000, 4000000]
    repeats = 5
    all_times = dict()
    all_memory = dict()
    if args.device.upper() == "OPENMP":
        threads = [1]
    else:
        threads = [32, 64, 128, 256, 512]
    for i, nt in enumerate(nt_list):
        args.nt = nt
        ns = args.nt * args.up
        print(f":: Running Nt={nt}")
        for j in range(repeats):
            tot_times = []
            for nthreads in threads:
                print(f":::: Running threads={nthreads}")
                args.threads = nthreads
                potential, times = run(
                    args, time_every_step=True, verbosity=1
                )
                tot_times.append(times)
            print(f"   {j+1}th pass done (out of {repeats})")
            # Initialize arrays for storing runtimes and memory
            if i == 0 and j == 0:
                for key in times:
                    all_times[key] = np.full(shape=[repeats, len(nt_list)], fill_value=np.inf)
            # Store runtimes
            for times in tot_times:
                for key, val in times.items():
                    old_time = all_times[key][j, i]
                    new_time = val['tot']
                    if new_time < old_time:
                        all_times[key][j, i] = val['tot']
    save_times_to_disk(nt_list, repeats, all_times, args)
    return


def save_times_to_disk(nt, repeats, times, args):
    now_str = time.strftime("%y%m%dT%H%M%S%Z")
    format_version = 1
    if args.device.upper() == "OPENMP":
        args.device = 'host'
    if args.device.upper() == "HOST":
        import platform
        arch = platform.processor()
    else:
        import cupy as cp
        arch = cp.cuda.Device(0).compute_capability
    fname_base = (
        f"ewald_timing_result_up{args.up}_clsz{args.cell_size}_tol{args.tolerance}"
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
    # input arguments
    nt = args.nt
    box = np.array([1.0, 1.0, 1.0])

    ns = nt * args.up

    cp = parkipy.utils.get_array_module(args.device)
    trg = cp.random.rand(3, nt) * cp.array(box).reshape(3,1)
    src = cp.random.rand(3, ns) * cp.array(box).reshape(3,1)
    dens_sl = cp.random.randn(3, ns)
    dens_dl = cp.random.randn(3, ns)
    norms = cp.random.randn(3, ns)
    dens = cp.vstack((dens_sl, dens_dl))  # stack densities for ewald call

    options = parkipy.ewald.EwaldOptions(
            periodicity=1, box=box, tolerance=args.tolerance, execution_space=args.device,
            cell_size=args.cell_size, return_walltime=True, p2g_threads=args.threads, g2p_threads=args.threads,
            p2p_threads_x=args.threads
            )
    pot, walltimes = parkipy.ewald.stokes_comb(
        trg, src, dens, norms, options 
    )

    # Compute and store runtimes in dict
    return pot, walltimes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaling test for the Spectral Ewald code on a GPU."
    )
    parser.add_argument(
        "--cell_size",
        dest="cell_size",
        type=int,
        default=224,
        help="Set the max src per cell (default: 1024)",
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
        default="analysis/ewald/data",
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
