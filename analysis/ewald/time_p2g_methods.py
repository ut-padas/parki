import os
import time
import argparse
import numpy as np
import cupy as cp
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
    _tols_s_ = [(1e-1, 160), (1e-4, 224), (1e-12, 1008)]
    threads = [32, 64, 128, 256, 512]
    methods = ["BASE", "SOURCE", "GRID", "HYBRID"]
    repeats = 5
    all_times = dict()
    all_params = dict()
    for i, nt in enumerate(nt_list):
        args.nt = nt
        ns = args.nt * args.up
        print(f":: Running Nt={nt}")
        for j in range(repeats):
            print(f":::: pass {j}")
            for method in methods:
                args.method = method
                for ep_s_pair in _tols_s_:
                    args.tolerance = tol = ep_s_pair[0]
                    args.cell_size = cell_size = ep_s_pair[1]
                    if tol == 1e-8 and cell_size <= 256 and nt >= 1000000:
                        continue  # fft will run out of memory
                    if tol == 1e-8 and cell_size <= 1024 and nt >= 4000000:
                        continue  # Hg is too big to fit in memory
                    if tol == 1e-3 and cell_size <= 128 and nt >= 8000000:
                        continue  # fft will run out of memory
                    for nthreads in threads:
                        args.threads = nthreads
                        print(
                            f"method {args.method} cell size {args.cell_size} tol {args.tolerance} nthreads {args.threads} nt {nt}"
                        )
                        cp.get_default_memory_pool().free_all_blocks()  # free gpu memory
                        _, times, params = run(args, verbosity=0)
                        cp.get_default_memory_pool().free_all_blocks()  # free gpu memory
                        # init alltimes
                        for key in times:
                            if key not in all_times:
                                all_times[key] = {}
                                all_params[key] = {}
                            if method not in all_times[key]:
                                all_times[key][method] = {}
                                all_params[key][method] = {}
                            if cell_size not in all_times[key][method]:
                                all_times[key][method][cell_size] = {}
                                all_params[key][method][cell_size] = {}
                            if tol not in all_times[key][method][cell_size]:
                                all_times[key][method][cell_size][tol] = {}
                                all_params[key][method][cell_size][tol] = {}
                            if nthreads not in all_times[key][method][cell_size][tol]:
                                all_times[key][method][cell_size][tol][nthreads] = (
                                    np.zeros([repeats, len(nt_list)])
                                )
                        # store alltimes
                        for key, val in times.items():
                            all_times[key][method][cell_size][tol][nthreads][j, i] = val

                        # store params (replace A_fun with None to make picklable)
                        if nt not in all_params[key][method][cell_size][tol]:
                            all_params[key][method][cell_size][tol][
                                nt
                            ] = params.__dict__
                            all_params[key][method][cell_size][tol][nt][
                                "A_fun"
                            ] = None  # turn off A fun to make pickleable
    save_times_to_disk(nt_list, repeats, all_times, all_params, args)
    return


def save_times_to_disk(nt, repeats, times, params, args):
    now_str = time.strftime("%y%m%dT%H%M%S%Z")
    format_version = 1
    arch = cp.cuda.Device(0).compute_capability
    fname_base = (
        f"p2g_timing_result_up{args.up}"
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
                "params": params,
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


def run(args, time_every_step=False, verbosity=0) -> None:
    # input arguments
    nt = args.nt
    box = np.array([1.0, 1.0, 1.0])

    # deterministic arguments
    ns = nt * args.up

    cp.random.seed(123)  # seed random numbers
    trg = cp.random.rand(3, nt) * cp.array(box).reshape(3, 1)
    src = cp.random.rand(3, ns) * cp.array(box).reshape(3, 1)
    dens_sl = cp.random.randn(3, ns)
    dens_dl = cp.random.randn(3, ns)
    dens = cp.vstack((dens_sl, dens_dl))
    norms = cp.random.randn(3, ns)

    device_pre, params = parkipy.ewald._prepare.DevicePre.from_particles(
        targets=trg,
        sources=src,
        forces=dens,
        normals=norms,
        kernel="STOKES_COMB",
        box=box,
        tolerance=args.tolerance,
        cell_size=args.cell_size,
        periodicity=1,
        execution_space=args.device,
        fourier_upsampling_factor_global=0,
    )
    walltime = parkipy.ewald._ewald.p2g(
        device_pre, method=args.method, threads=args.threads
    )
    runtimes = {"p2g": walltime["kernel"]}

    return None, runtimes, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaling test for the Spectral Ewald code on a GPU."
    )
    default_nt = [int(1e4), int(1e5), int(5e5), int(1e6), int(5e6)]
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
        "--device",
        dest="device",
        type=str,
        help="Device to run code on",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="analysis/ewald/data",
        help="output directory for timing results (default: .)",
    )
    args = parser.parse_args()
    main(args)
    exit()
