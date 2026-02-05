import enum
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
    nt_list = [250000, 1000000, 4000000]
    tols = [1e-1]
    cell_sizes = [64, 128, 256, 512]
    threadss = [32, 64, 128, 256]
    s_threads_2ds = [2, 8, 32]
    repeats = 3
    all_times = dict()
    all_threads = dict()

    for key in ["p2p"]:
        for l, kernel in enumerate(
            ["laplace / ewald", "stokes_sl / ewald", "stokes_comb"]
        ):
            for k, variant in enumerate(["GM-1D", "SM-1D"]):
                for cell_size in cell_sizes:
                    args.cell_size = cell_size
                    for tol in tols:
                        args.tolerance = tol
                        for i, nt in enumerate(nt_list):
                            args.nt = nt
                            if key not in all_times:
                                all_times[key] = {}  # Initialize as a dictionary
                                all_threads[key] = {}
                            if kernel not in all_times[key]:
                                all_times[key][kernel] = {}
                                all_threads[key][kernel] = {}
                            if variant not in all_times[key][kernel]:
                                all_times[key][kernel][variant] = {}
                                all_threads[key][kernel][variant] = {}
                            if cell_size not in all_times[key][kernel][variant]:
                                all_times[key][kernel][variant][cell_size] = {}
                                all_threads[key][kernel][variant][cell_size] = {}
                            if tol not in all_times[key][kernel][variant][cell_size]:
                                all_times[key][kernel][variant][cell_size][tol] = {}
                                all_threads[key][kernel][variant][cell_size][tol] = {}
                            if (
                                nt
                                not in all_times[key][kernel][variant][cell_size][tol]
                            ):
                                all_times[key][kernel][variant][cell_size][tol][nt] = {}
                                all_threads[key][kernel][variant][cell_size][tol][
                                    nt
                                ] = {}
                            all_times[key][kernel][variant][cell_size][tol][nt] = (
                                np.zeros(repeats)
                            )
                            all_threads[key][kernel][variant][cell_size][tol][nt] = (
                                np.zeros(4)
                            )
                            time_min = np.inf
                            time_min_repeats = np.zeros(repeats)
                            time_min_threads = np.zeros(4)
                            if variant == "GM-1D":
                                for threads in threadss:
                                    if threads > cell_size:
                                        continue
                                    t_chunk_size = threads
                                    t_per_thread = int(np.ceil(t_chunk_size / threads))
                                    # warm up
                                    _, times = run(
                                        args,
                                        variant,
                                        kernel,
                                        threads,
                                        t_per_thread,
                                        verbosity=1,
                                    )
                                    print(
                                        f"kernel {kernel} variant {variant} cell_size {cell_size} tol {tol} nt {nt}\n"
                                        + f"threads {threads} t_per_thread {t_per_thread} s_per_thread {np.nan} s_threads_2d {np.nan}\n"
                                        + "warm up"
                                    )
                                    time_temp = np.zeros(repeats)
                                    for j in range(repeats):
                                        _, times = run(
                                            args,
                                            variant,
                                            kernel,
                                            threads,
                                            t_per_thread,
                                            verbosity=1,
                                        )
                                        print(
                                            f"kernel {kernel} variant {variant} cell_size {cell_size} tol {tol} nt {nt}\n"
                                            + f"threads {threads} t_per_thread {t_per_thread} s_per_thread {np.nan} s_threads_2d {np.nan}\n"
                                            + f"{j+1}th pass done (out of {repeats})"
                                        )
                                        time_temp[j] = times[key]
                                    time_ave = np.average(time_temp)
                                    if time_ave < time_min:
                                        time_min = time_ave
                                        time_min_repeats = time_temp
                                        time_min_threads = np.array(
                                            [threads, t_per_thread, np.nan, np.nan]
                                        )
                                    print(
                                        f"kernel {kernel} variant {variant} cell_size {cell_size} tol {tol} nt {nt}\n"
                                        + f"threads {threads} t_per_thread {t_per_thread} s_per_thread {np.nan} s_threads_2d {np.nan}\n"
                                        + f"time_current {time_ave} time_min {time_min}"
                                    )
                            elif variant == "GM-2D":
                                for threads in threadss:
                                    if threads > cell_size:
                                        continue
                                    for s_threads_2d in s_threads_2ds:
                                        if s_threads_2d > threads:
                                            continue
                                        t_threads_2d = int(
                                            np.ceil(threads / s_threads_2d)
                                        )
                                        t_chunk_size = t_threads_2d
                                        t_per_thread = int(
                                            np.ceil(t_chunk_size / t_threads_2d)
                                        )
                                        # warm up
                                        _, times = run(
                                            args,
                                            variant,
                                            kernel,
                                            threads,
                                            t_per_thread,
                                            s_threads_2d=s_threads_2d,
                                            verbosity=1,
                                        )
                                        print(
                                            f"kernel {kernel} variant {variant} cell_size {cell_size} tol {tol} nt {nt}\n"
                                            + f"threads {threads} t_per_thread {t_per_thread} s_per_thread {np.nan} s_threads_2d {s_threads_2d}\n"
                                            + "warm up"
                                        )
                                        time_temp = np.zeros(repeats)
                                        for j in range(repeats):
                                            _, times = run(
                                                args,
                                                variant,
                                                kernel,
                                                threads,
                                                t_per_thread,
                                                s_threads_2d=s_threads_2d,
                                                verbosity=1,
                                            )
                                            print(
                                                f"kernel {kernel} variant {variant} cell_size {cell_size} tol {tol} nt {nt}\n"
                                                + f"threads {threads} t_per_thread {t_per_thread} s_per_thread {np.nan} s_threads_2d {s_threads_2d}\n"
                                                + f"{j+1}th pass done (out of {repeats})"
                                            )
                                            time_temp[j] = times[key]
                                        time_ave = np.average(time_temp)
                                        if time_ave < time_min:
                                            time_min = time_ave
                                            time_min_repeats = time_temp
                                            time_min_threads = np.array(
                                                [
                                                    threads,
                                                    t_per_thread,
                                                    np.nan,
                                                    s_threads_2d,
                                                ]
                                            )
                                        print(
                                            f"kernel {kernel} variant {variant} cell_size {cell_size} tol {tol} nt {nt}\n"
                                            + f"threads {threads} t_per_thread {t_per_thread} s_per_thread {np.nan} s_threads_2d {s_threads_2d}\n"
                                            + f"time_current {time_ave} time_min {time_min}"
                                        )
                            elif variant == "point-in":
                                for threads in threadss:
                                    if threads > cell_size:
                                        continue
                                    # warm up
                                    _, times = run(
                                        args, variant, kernel, threads, verbosity=1
                                    )
                                    print(
                                        f"kernel {kernel} variant {variant} cell_size {cell_size} tol {tol} nt {nt}\n"
                                        + f"threads {threads} t_per_thread {np.nan} s_per_thread {np.nan} s_threads_2d {np.nan}\n"
                                        + "warm up"
                                    )
                                    time_temp = np.zeros(repeats)
                                    for j in range(repeats):
                                        _, times = run(
                                            args, variant, kernel, threads, verbosity=1
                                        )
                                        print(
                                            f"kernel {kernel} variant {variant} cell_size {cell_size} tol {tol} nt {nt}\n"
                                            + f"threads {threads} t_per_thread {np.nan} s_per_thread {np.nan} s_threads_2d {np.nan}\n"
                                            + f"{j+1}th pass done (out of {repeats})"
                                        )
                                        time_temp[j] = times[key]
                                    time_ave = np.average(time_temp)
                                    if time_ave < time_min:
                                        time_min = time_ave
                                        time_min_repeats = time_temp
                                        time_min_threads = np.array(
                                            [threads, np.nan, np.nan, np.nan]
                                        )
                                    print(
                                        f"kernel {kernel} variant {variant} cell_size {cell_size} tol {tol} nt {nt}\n"
                                        + f"threads {threads} t_per_thread {np.nan} s_per_thread {np.nan} s_threads_2d {np.nan}\n"
                                        + f"time_current {time_ave} time_min {time_min}"
                                    )
                            elif variant == "SM-1D":
                                for threads in threadss:
                                    if threads > cell_size:
                                        continue
                                    t_chunk_size = threads
                                    t_per_thread = int(np.ceil(t_chunk_size / threads))
                                    s_chunk_size = threads
                                    s_per_thread = int(s_chunk_size)
                                    # warm up
                                    _, times = run(
                                        args,
                                        variant,
                                        kernel,
                                        threads,
                                        t_per_thread,
                                        s_per_thread,
                                        verbosity=1,
                                    )
                                    print(
                                        f"kernel {kernel} variant {variant} cell_size {cell_size} tol {tol} nt {nt}\n"
                                        + f"threads {threads} t_per_thread {t_per_thread} s_per_thread {s_per_thread} s_threads_2d {np.nan}\n"
                                        + "warm up"
                                    )
                                    time_temp = np.zeros(repeats)
                                    for j in range(repeats):
                                        _, times = run(
                                            args,
                                            variant,
                                            kernel,
                                            threads,
                                            t_per_thread,
                                            s_per_thread,
                                            verbosity=1,
                                        )
                                        print(
                                            f"kernel {kernel} variant {variant} cell_size {cell_size} tol {tol} nt {nt}\n"
                                            + f"threads {threads} t_per_thread {t_per_thread} s_per_thread {s_per_thread} s_threads_2d {np.nan}\n"
                                            + f"{j+1}th pass done (out of {repeats})"
                                        )
                                        time_temp[j] = times[key]
                                    time_ave = np.average(time_temp)
                                    if time_ave < time_min:
                                        time_min = time_ave
                                        time_min_repeats = time_temp
                                        time_min_threads = np.array(
                                            [
                                                threads,
                                                t_per_thread,
                                                s_per_thread,
                                                np.nan,
                                            ]
                                        )
                                    print(
                                        f"kernel {kernel} variant {variant} cell_size {cell_size} tol {tol} nt {nt}\n"
                                        + f"threads {threads} t_per_thread {t_per_thread} s_per_thread {s_per_thread} s_threads_2d {np.nan}\n"
                                        + f"time_current {time_ave} time_min {time_min}"
                                    )
                            elif variant == "SM-2D":
                                for threads in threadss:
                                    if threads > cell_size:
                                        continue
                                    for s_threads_2d in s_threads_2ds:
                                        if s_threads_2d > threads:
                                            continue
                                        t_threads_2d = int(
                                            np.ceil(threads / s_threads_2d)
                                        )
                                        t_chunk_size = t_threads_2d
                                        t_per_thread = int(
                                            np.ceil(t_chunk_size / t_threads_2d)
                                        )
                                        s_chunk_size = threads
                                        s_per_thread = int(
                                            np.ceil(s_chunk_size / s_threads_2d)
                                        )
                                        # warm up
                                        _, times = run(
                                            args,
                                            variant,
                                            kernel,
                                            threads,
                                            t_per_thread,
                                            s_per_thread,
                                            s_threads_2d,
                                            verbosity=1,
                                        )
                                        print(
                                            f"kernel {kernel} variant {variant} cell_size {cell_size} tol {tol} nt {nt}\n"
                                            + f"threads {threads} t_per_thread {t_per_thread} s_per_thread {s_per_thread} s_threads_2d {s_threads_2d}\n"
                                            + "warm up"
                                        )
                                        time_temp = np.zeros(repeats)
                                        for j in range(repeats):
                                            _, times = run(
                                                args,
                                                variant,
                                                kernel,
                                                threads,
                                                t_per_thread,
                                                s_per_thread,
                                                s_threads_2d,
                                                verbosity=1,
                                            )
                                            print(
                                                f"kernel {kernel} variant {variant} cell_size {cell_size} tol {tol} nt {nt}\n"
                                                + f"threads {threads} t_per_thread {t_per_thread} s_per_thread {s_per_thread} s_threads_2d {s_threads_2d}\n"
                                                + f"{j+1}th pass done (out of {repeats})"
                                            )
                                            time_temp[j] = times[key]
                                        time_ave = np.average(time_temp)
                                        if time_ave < time_min:
                                            time_min = time_ave
                                            time_min_repeats = time_temp
                                            time_min_threads = np.array(
                                                [
                                                    threads,
                                                    t_per_thread,
                                                    s_per_thread,
                                                    s_threads_2d,
                                                ]
                                            )
                                        print(
                                            f"kernel {kernel} variant {variant} cell_size {cell_size} tol {tol} nt {nt}\n"
                                            + f"threads {threads} t_per_thread {t_per_thread} s_per_thread {s_per_thread} s_threads_2d {s_threads_2d}\n"
                                            + f"time_current {time_ave} time_min {time_min}"
                                        )
                            all_times[key][kernel][variant][cell_size][tol][
                                nt
                            ] = time_min_repeats
                            all_threads[key][kernel][variant][cell_size][tol][
                                nt
                            ] = time_min_threads

    save_times_to_disk(repeats, all_times, all_threads, args)
    return


def save_times_to_disk(repeats, times, threads, args):
    now_str = time.strftime("%y%m%dT%H%M%S%Z")
    format_version = 1
    arch = cp.cuda.Device(0).compute_capability
    fname_base = (
        f"p2p_timing_workload_result_up{args.up}"
        f"_dev{args.device}_arch{arch}_v{format_version}"
    )
    fname = fname_base + f"_{now_str}.pkl"
    fname_link = fname_base + "_latest.pkl"
    fpath = os.path.join(args.output_dir, fname)
    fpath_link = os.path.join(args.output_dir, fname_link)
    with open(fpath, "wb") as f:
        pickle.dump(
            {
                "repeats": repeats,
                "times": times,
                "threads": threads,
                "args": args,
                "arch": arch,
            },
            f,
        )
    if os.path.lexists(fpath_link):
        os.remove(fpath_link)
    os.symlink(fname, fpath_link)
    print(f"Saved results to {fpath!r} and {fpath_link!r}")
    return


def bytes_to_gb(bytes_value):
    return bytes_value / (1024**3)


def rc_from_cell_size(ns, cell_size):
    # NOTE: assume that box is [1,1,1]
    num_cells = math.ceil(ns / cell_size)
    rc = num_cells ** (-1 / 3)
    return rc


def xi_from_rc(rc, factor):
    c = 4 * rc * factor
    xi = 1.0 / rc * math.sqrt(math.log(math.sqrt(c)))
    return xi


def grid_res_from_xi(xi, factor):
    c = factor
    grid_res = 2 * xi / np.pi * np.sqrt(np.log(c))
    return grid_res


def run(
    args,
    variant,
    kernel,
    threads=32,
    t_per_thread=1,
    s_per_thread=32,
    s_threads_2d=2,
    verbosity=0,
) -> None:
    # input arguments
    nt = args.nt
    box = np.array([1.0, 1.0, 1.0])

    # deterministic arguments
    ns = nt * args.up

    trg = cp.random.rand(3, nt) * cp.array(box).reshape(3, 1)
    src = cp.random.rand(3, ns) * cp.array(box).reshape(3, 1)
    dens_sl = cp.random.randn(3, ns)
    dens_dl = cp.random.randn(3, ns)
    dens = cp.vstack((dens_sl, dens_dl))
    normal = cp.random.randn(3, ns)

    device_pre, params = parkipy.ewald._prepare.DevicePre.from_particles(
        trg,
        src,
        dens,
        normal,
        "stokes_comb",
        box,
        args.tolerance,
        1,
        cell_size=args.cell_size,
        execution_space=args.device,
    )

    if verbosity >= 2:
        print("======Spectral Ewald Sum======")
    if verbosity >= 2:
        print("1) p2p")
    walltime = parkipy.ewald._ewald.p2p(
        device_pre,
        method=variant,
        threads_x=threads,
        threads_y=s_threads_2d,
        kernel=kernel,
    )
    runtimes = {}
    runtimes["p2p"] = walltime["kernel"]
    if verbosity >= 1:
        print("========Timing Results========")
        print("p2p:       %f [ms]" % (runtimes["p2p"] * 1e3))
    return None, runtimes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaling test for the Spectral Ewald code on a GPU."
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        help="Device to run code on",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="analysis/stokes1p/data",
        help="output directory for timing results (default: .)",
    )
    parser.add_argument(
        "--up",
        dest="up",
        type=int,
        default=1,
        help="Set the upsampeling parameter (default: 16)",
    )
    args = parser.parse_args()
    main(args)
    exit()
