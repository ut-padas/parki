import os
import time
import math
import argparse
import numpy as np
import pykokkos as pk
import pickle
import pytest

from parkipy.utils import get_array_module, get_execution_space
from parkipy.ewald._prepare import DevicePre
from parkipy.ewald._ewald import p2p, p2g, fft, cnv, ifft, g2p


def main(args):
    """
    Main function. Takes `args` from the ArgumentParser at the bottom of this
    file. `args.nt` will be a list of integers, and we run `run()` with each of
    these integers as the number of targets.
    """
    am = get_array_module(args.device)
    SEED = 12345
    am.random.seed(SEED)
    nt_list = args.nt
    solution = dict()
    for i, nt in enumerate(nt_list):
        args.nt = nt
        ns = args.nt * args.up
        print(f":: Running Nt={nt}")
        potential, times, sol = run(
            args, time_every_step=True, verbosity=args.verbosity
        )
        solution[nt] = sol
    if args.create:
        save_solution_to_disk(solution, args)
    elif args.read:
        test_reference_from_disk(nt_list, solution, args)
    else:
        raise ValueError("Must be one of 'create' or 'read'")
    return


def test_reference_from_disk(nt_list, solution, args):
    am = get_array_module(args.device)
    now_str = time.strftime("%y%m%dT%H%M%S%Z")
    format_version = 1
    if args.device.upper() == "HOST":
        arch = None
    else:
        arch = am.cuda.Device(0).compute_capability
    fname_base = (
        f"reference{args.up}_clsz{args.cell_size}_tol{args.tolerance}"
        f"_dev{args.device.upper()}_arch{arch}_v{format_version}"
    )
    fname = fname_base + f"_latest.pkl"
    fpath = os.path.join("tests/data/", fname)
    with open(fpath, "rb") as f:
        reference = pickle.load(f)
    print(f"Read results from {fpath!r}")
    for i, nt in enumerate(nt_list):
        print(f":: Testing Nt={nt}")
        for stage in reference[nt].keys():
            try:
                am.testing.assert_allclose(solution[nt][stage], reference[nt][stage])
                print(f":::: Stage {stage} passed ✅")
            except AssertionError as e:
                print(f":::: Stage {stage} failed ❌")
                print(e)
    ref_sol = reference[nt]["p2p"] + reference[nt]["g2p"]
    acc_sol = solution[nt]["p2p"] + solution[nt]["g2p"]
    am.testing.assert_allclose(acc_sol, ref_sol)
    return


def save_solution_to_disk(solution, args):
    am = get_array_module(args.device)
    now_str = time.strftime("%y%m%dT%H%M%S%Z")
    format_version = 1
    if args.device.upper() == "HOST":
        arch = None
    else:
        arch = am.cuda.Device(0).compute_capability
    fname_base = (
        f"reference{args.up}_clsz{args.cell_size}_tol{args.tolerance}"
        f"_dev{args.device.upper()}_arch{arch}_v{format_version}"
    )
    fname = fname_base + f"_{now_str}.pkl"
    fname_link = fname_base + "_latest.pkl"
    fpath = os.path.join(args.output_dir, fname)
    fpath_link = os.path.join(args.output_dir, fname_link)
    with open(fpath, "wb") as f:
        pickle.dump(
            solution,
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


def run(args, time_every_step=False, verbosity=0) -> None:
    am = get_array_module(args.device)
    precision = "single"
    dtype = None
    match precision.upper():
        case "SINGLE":
            dtype = am.single
        case "DOUBLE":
            dtype = am.double
        case _:
            raise ValueError(
                f"only 'single' and 'double' precision supported, got {precision}"
            )
    solution = dict()
    # input arguments
    nt = args.nt
    box = np.array([1.0, 1.0, 1.0])

    # deterministic arguments
    ns = nt * args.up

    arr = am.linspace(1e-1, 0.99, nt, dtype=dtype).reshape(1, nt)
    arr = am.tile(arr, (3, 1))
    src = arr.copy()
    trg = arr.copy()

    match args.kernel.upper():
        case 'STOKES_COMB':
            dens_sl = am.random.randn(3, ns, dtype=dtype)
            dens_dl = am.random.randn(3, ns, dtype=dtype)
            dens = am.vstack((dens_sl, dens_dl))  # stack densities for ewald call
            normal = am.random.randn(3, ns, dtype=dtype)
        case 'LAPLACE':
            charges = am.random.randn(ns).astype(dtype)
            charges -= am.mean(charges).astype(dtype)
            dens = charges
            normal = None
        case _:
            raise NotImplementedError(f"kernel {args.kernel.upper()} not yet implemented.")


    if verbosity >= 2:
        print("======Spectral Ewald Sum======")
    # 0) device-pre
    if verbosity >= 2:
        print("device pre")
    device_pre_start = time.time()

    device_pre, _ = DevicePre.from_particles(
        targets=trg,
        sources=src,
        forces=dens,
        normals=normal,
        kernel=args.kernel,
        box=box,
        tolerance=args.tolerance,
        cell_size=args.cell_size,
        periodicity=args.periodicity,
        execution_space=args.device,
        fft_type="R2C",
        distributed=False,
    )
    pk.fence()
    device_pre_end = time.time()
    if verbosity >= 2:
        print("p2p")
    p2p_start = time.time()
    p2p(
        device_pre,
        args.p2p_method,
        threads_x=args.p2p_threads,
    )
    solution["p2p"] = device_pre.near_potential
    pk.fence()
    p2p_end = time.time()
    if verbosity >= 2:
        print("1) p2g")
    p2g_start = time.time()
    p2g(device_pre, method=args.p2g_method)
    solution["p2g"] = device_pre.data.H
    p2g_end = time.time()
    if verbosity >= 2:
        print("2) fft")
    fft_start = time.time()
    fft(device_pre=device_pre)
    solution["fft"] = device_pre.data.Hg
    fft_end = time.time()
    if verbosity >= 2:
        print("3) cnv")
    cnv_start = time.time()
    cnv(device_pre)
    solution["cnv"] = device_pre.data.Hg
    cnv_end = time.time()
    if verbosity >= 2:
        print("4) ifft")
    ifft_start = time.time()
    ifft(device_pre=device_pre)
    solution["ifft"] = device_pre.data.H
    ifft_end = time.time()
    if verbosity >= 2:
        print("5) g2p")
    g2p_start = time.time()
    g2p(device_pre, args.g2p_method, 128)
    solution["g2p"] = device_pre.far_potential
    g2p_end = time.time()

    # Compute and store runtimes in dict
    runtimes = {
        "p2p": p2p_end - p2p_start,
        "device_pre": device_pre_end - device_pre_start,
    }
    if time_every_step:
        runtimes["p2g"] = p2g_end - p2g_start
        runtimes["fft"] = fft_end - fft_start
        runtimes["cnv"] = cnv_end - cnv_start
        runtimes["ifft"] = ifft_end - ifft_start
        runtimes["g2p"] = g2p_end - g2p_start
    runtimes["fs_tot"] = g2p_end - p2g_start
    runtimes["p2p+fs_tot"] = runtimes["p2p"] + runtimes["fs_tot"]
    if verbosity >= 1:
        print("========Timing Results========")
        print("p2p:       %f [ms]" % (runtimes["p2p"] * 1e3))
        print("device_pre:   %f [ms]" % (runtimes["device_pre"] * 1e3))
        if time_every_step:
            print("p2g:       %f [ms]" % (runtimes["p2g"] * 1e3))
            print("fft:       %f [ms]" % (runtimes["fft"] * 1e3))
            print("cnv:       %f [ms]" % (runtimes["cnv"] * 1e3))
            print("ifft:      %f [ms]" % (runtimes["ifft"] * 1e3))
            print("g2p:       %f [ms]" % (runtimes["g2p"] * 1e3))
        print("fs (tot):  %f [ms]" % (runtimes["fs_tot"] * 1e3))
        print("p2p+fs:    %f [ms]" % (runtimes["p2p+fs_tot"] * 1e3))
    return (
        device_pre.near_potential + device_pre.far_potential,
        runtimes,
        solution,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaling test for the Spectral Ewald code on a GPU."
    )
    default_nt = [100000]
    default_nt_str = " ".join([str(x) for x in default_nt])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--create", action="store_true", help="Create the test files")
    group.add_argument("--read", action="store_true", help="Read the test files")
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
        default=1024,
        help="Set the max src per cell (default: 1024)",
    )
    parser.add_argument(
        "--per",
        dest="periodicity",
        type=int,
        default=1,
        help="SE Periodicity, either 0,1,2,3, (default:1)",
    )
    parser.add_argument(
        "--kernel",
        dest="kernel",
        type=str,
        default="stokes_comb",
        help="SE kernel (default:stokes_comb)",
    )
    parser.add_argument(
        "--window_P",
        dest="window_P",
        type=int,
        default=10,
        help="select the support for the window function",
    )
    parser.add_argument(
        "--p2g_method",
        dest="p2g_method",
        choices=("BASE", "SOURCE", "GRID", "HYBRID"),
        default="HYBRID",
        type=str,
        help="select the p2g method.",
    )
    parser.add_argument(
        "--g2p_method",
        dest="g2p_method",
        choices=(
            "BASE",
            "TARGET",
        ),
        default="TARGET",
        type=str,
        help="select the g2p method.",
    )

    parser.add_argument(
        "--device",
        dest="device",
        default="cuda",
        type=str,
        help="Device to run code on",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="tests/data/",
        help="output directory for timing results (default: .)",
    )
    parser.add_argument(
        "--p2p_method",
        dest="p2p_method",
        choices=("GM-1D", "GM-2D", "SM-1D", "SM-2D"),
        default="GM-1D",
        type=str,
        help="select the p2p variant",
    )
    parser.add_argument(
        "--p2p_threads",
        dest="p2p_threads",
        default=32,
        type=int,
        help="select the p2p threads.",
    )
    parser.add_argument(
        "--distribution",
        dest="distribution",
        choices=("uniform", "nonuniform"),
        default="uniform",
        type=str,
        help="particle distribution",
    )
    parser.add_argument(
        "--verbosity",
        dest="verbosity",
        type=int,
        default=1,
        help="Verbosity of Printed Output",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Spectral Ewald tolerance",
    )

    args = parser.parse_args()
    if not args.create and not args.read:
        args.read = True
    main(args)
    exit()
