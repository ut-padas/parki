import os
import time
import math
import argparse
import numpy as np
import pykokkos as pk
import pickle

import parkipy


def main(args):
    """
    Main function. Takes `args` from the ArgumentParser at the bottom of this
    file. `args.nt` will be a list of integers, and we run `run()` with each of
    these integers as the number of targets.
    """
    nt_list = [4000000]
    repeats = 3
    all_times = dict()
    all_memory = dict()
    threads = [32, 64, 128, 256]
    distributions = [
        "uniform-uniform",
        "gaussian-gaussian",
        "gaussian-uniform",
        "ellipsoid-ellipsoid",
        "ellipsoid-uniform",
    ]
    for distribution in distributions:
        print(f"::::::::: distribution={distribution}")
        all_times[distribution] = {}
        args.distribution = distribution
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
                        all_times[distribution][key] = np.full(
                            shape=[repeats, len(nt_list)], fill_value=np.inf
                        )
                # Store runtimes
                for times in tot_times:
                    for key, val in times.items():
                        old_time = all_times[distribution][key][j, i]
                        new_time = val["tot"]
                        if new_time < old_time:
                            all_times[distribution][key][j, i] = val["tot"]
    save_times_to_disk(nt_list, repeats, all_times, args)
    return


def save_times_to_disk(nt, repeats, times, args):
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
        f"distributions_timing_result_up{args.up}_clsz{args.cell_size}_tol{args.tolerance}"
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


def sample_gaussian(box, c, n):
    """
    rejects points outside of the box
    """
    am = parkipy.utils.get_array_module(args.device)
    lx, ly, lz = box
    center = am.array([lx / 2, ly / 2, lz / 2])
    lower = am.array([0.0, 0.0, 0.0])
    upper = am.array([lx, ly, lz])

    points = am.empty((3, n))
    i = 0
    num_accepted = 0
    while num_accepted < n:
        samples = am.random.normal(loc=0.0, scale=c, size=(3, n)) + center.reshape(3, 1)
        mask = am.all(
            (samples > lower.reshape(3, 1)) & (samples < upper.reshape(3, 1)), axis=0
        )
        new_samples = samples[:, mask]
        num_new_samples = min(new_samples.shape[1], n - num_accepted)
        points[:, num_accepted : num_accepted + num_new_samples] = new_samples[
            :, :num_new_samples
        ]
        num_accepted += num_new_samples
        i += 1

    return points


def sample_unit_sphere_surface(n):
    points = np.random.normal(size=(3, n))
    points /= np.linalg.norm(points, axis=0)
    return points


def run(args, time_every_step=False, verbosity=0) -> None:
    # input arguments
    am = parkipy.utils.get_array_module(args.device)
    nt = args.nt
    box = np.array([1.0, 1.0, 1.0])

    ns = nt * args.up

    cp = parkipy.utils.get_array_module(args.device)

    for i, dis in enumerate(args.distribution.split("-")):
        # 0 <- src_dis, 1 <- trg_dis
        n = None
        if i == 0:
            n = ns
        if i == 1:
            n = nt
        match dis.upper():
            case "GAUSSIAN":
                c = 0.3
                arr = sample_gaussian(box, c, n)
            case "ELLIPSOID":
                arr = sample_unit_sphere_surface(n)
                arr = cp.asarray(arr)
                # scale to be inside box
                arr *= 0.49
                arr += 0.5
                assert arr.shape[1] == n
            case "UNIFORM":
                arr = am.random.uniform(size=(3, n)) * am.array(box).reshape(3, 1)
            case _:
                raise ValueError(
                    f"Distribution must be one of 'GAUSSIAN', 'ELLIPSOID', or 'UNIFORM', got '{dis.upper()}'"
                )
        if i == 0:
            src = arr
        if i == 1:
            trg = arr
    dens_sl = cp.random.randn(3, ns)
    dens_dl = cp.random.randn(3, ns)
    norms = cp.random.randn(3, ns)
    dens = cp.vstack((dens_sl, dens_dl))  # stack densities for ewald call

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
    )
    pot, walltimes = parkipy.ewald.stokes_comb(
        trg,
        src,
        dens,
        norms,
        options,
    )
    times = {
        "p2p": walltimes.time_p2p,
        "p2g": walltimes.time_p2g,
        "fft": walltimes.time_fft,
        "cnv": walltimes.time_cnv,
        "ifft": walltimes.time_ifft,
        "g2p": walltimes.time_g2p,
    }

    # Compute and store runtimes in dict
    return pot, times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate timings for different levels of "
        "non-uniform particle distributions. "
        "Results are saved to `distributions_timing_result_up{args.up}"
        "_clsz{args.cell_size}_tol{args.tolerance}_dev{args.device.upper()}"
        "_arch{arch}_v{format_version}."
    )
    parser = argparse.ArgumentParser(
        description="Scaling test for the Spectral Ewald code on a GPU."
    )
    parser.add_argument(
        "--up",
        dest="up",
        type=int,
        default=1,
        help="Set the upsampeling parameter (default: 1)",
    )
    parser.add_argument(
        "--cell_size",
        dest="cell_size",
        type=int,
        default=224,
        help="Set the max src per cell (default: 224)",
    )
    parser.add_argument(
        "--device",
        dest="device",
        required=True,
        choices=("cuda", "hip", "host"),
        type=str,
        help="Device to run code on",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="analysis/ewald/data",
        help="output directory for timing results (default: analysis/ewald/data)",
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
