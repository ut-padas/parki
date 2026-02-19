import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


plt.rc("font", family="serif")

import shutil

if shutil.which("latex") is not None:
    plt.rc("text", usetex=True)


def format_sig3(x):
    if x == 0:
        return "0.00"
    else:
        from math import log10, floor

        digits = 3
        exponent = floor(log10(abs(x)))
        decimals = digits - 1 - exponent
        return f"{x:.{max(decimals, 0)}f}"


OPERATION_CONSTANTS = {
    "a100": {
        "fadd": 2,
        "fmul": 2,
        "fsqrt": 29,
        "frsqrt": 19,
        "fdiv": 24,
        "fexpn": 45,
        "fsinh": 150,
        "ferf": 86,
    },
    "h200": {
        "fadd": 2,
        "fmul": 2,
        "fsqrt": 41,
        "frsqrt": 30,
        "fdiv": 35,
        "fexpn": 50,
        "fsinh": 361,
        "ferf": 155,
    },
}


DEVICE_CONSTANTS = {
    "a100": {
        "bandwidth": 1555,
        "gflops": 9.7e3,
        "bandwidth shmem": 20e3,
        "bandwidth l2": 6e3,
        "intensity": 9.7e3 / 1555,
        "peak flops": 9.7e3 * 1e9,
        "peak band": 1555 * 1e9,
    },
    "h200": {
        "bandwidth": 4000,
        "gflops": 33.5e3,
        "bandwidth shmem": 10 * 4000,
        "bandwidth l2": 4 * 4000,
        "intensity": 33.5e3 / 4000,
        "peak flops": 33.5e3 * 1e9,
        "peak band": 4000 * 1e9,
    },
}


def p2p_cnt_flop(op_cons, Nt, s):

    flops = 27 * Nt * s * (
        37 * op_cons["fmul"]
        + 17 * op_cons["fadd"]
        + 36
        + op_cons["frsqrt"]
        + op_cons["fdiv"]
        + op_cons["fexpn"]
        + op_cons["ferf"]
    ) * np.pi / 6 + 27 * Nt * s * (4 * op_cons["fadd"] + 4 + op_cons["fmul"])

    return flops


def p2p_cnt_mop(model, dev_cons, method, Nt, Ns, s, bt, bs, dp=True):
    if dp:
        dsize = 8
    else:
        dsize = 4

    m = 6 * Nt + 12 * Ns

    if method == "GM-1D":
        mg = Nt * (6 + 12 * 27 * s)
        ms = 0
    elif method == "SM-1D":
        mg = Nt * (6 + 12 * 27 * s / bt)
        ms = Nt * (3 + 12 * 27 * s)
    elif method == "GM-2D":
        mg = Nt * (3 * bs + 3 + 12 * 27 * s)
        ms = 0
    elif method == "SM-2D":
        mg = Nt * (6 + 12 * 27 * s / bt)
        ms = Nt * (3 * bs + 12 * 27 * s)

    m = m * dsize
    mg = mg * dsize
    ms = ms * dsize

    l1 = dev_cons["bandwidth"] / dev_cons["bandwidth shmem"]
    l2 = dev_cons["bandwidth"] / dev_cons["bandwidth l2"]

    if model == "zero cache":
        mop = mg + l1 * ms
    elif model == "inf cache":
        mop = m + l2 * mg + l1 * ms
    elif model == "pre fetch":
        mop = m + l1 * mg + l1 * ms

    return mop


def p2p_model_time(dev, arch, method, time, nt, ns, s, bt, bs, dp=True, both=False):

    dev_name = ""
    if dev.upper() == "CUDA":
        if int(arch) == 80:
            dev_name = "a100"
        elif int(arch) == 90:
            dev_name = "h200"
        else:
            raise ValueError(f"Unknown architecture {arch}")
    elif dev.upper() == "HIP":
        if int(arch) == 94:
            dev_name = "mi300a"
        else:
            raise ValueError(f"Unknown architecture {arch}")
    elif dev.upper() == "HOST":
        dev_name = "grace"
    else:
        raise ValueError(f"Unknown device {dev}")

    dev_cons = DEVICE_CONSTANTS[dev_name]
    op_cons = OPERATION_CONSTANTS[dev_name]

    string = ""

    models = ["zero cache", "inf cache", "pre fetch"]

    flop = p2p_cnt_flop(op_cons, nt, s)

    mops = []
    for model in models:
        mops.append(p2p_cnt_mop(model, dev_cons, method, nt, ns, s, bt, bs, dp))

    intens = flop / np.array(mops)

    string = [flop]

    for i, model in enumerate(models):
        string.append(intens[i])

    return string


def determine_degree(P):
    if P < 2:
        raise ValueError("P cannot be smaller than 2")
    elif P % 2 != 0:
        raise ValueError("P must be even")
    elif P <= 10:
        return P // 2 + 1
    else:
        return min(P // 2 + 2, 9)


def p2g_count_flops(op_cons, method, Ns, P):
    if method.upper() in ["BASE", "SOURCE", "HYBRID"]:
        return Ns * P**3 * (24 + 11 * op_cons["fmul"])
    elif method == "GRID":
        nu = determine_degree(P)
        return Ns * (
            27 * (P / 2) ** 3 * (24 + 11 * op_cons["fmul"] + 2 * op_cons["fadd"])
            + P**3 * 3 * 2 * nu
        )
    else:
        raise ValueError(f"method {method} not supported.")


def p2g_count_mops(model, dev_cons, method, Ns, Ng, P, b_fs, dp=True):
    if dp:
        dsize = 8
    else:
        dsize = 4

    isize = 4

    m = 12 * Ns + 12 * Ng
    m = m * dsize

    match method.upper():
        case "HYBRID":
            mg = 12 * Ns + 12 * 27 * Ns * (P / 2) ** 3 / b_fs
            mg = mg * dsize
            ms = 12 * Ns * P**3 * dsize + 3 * 27 * Ns * (P / 2) ** 3 * isize
        case "BASE":
            mg = 3 * Ns + 21 * Ns * P**3
            mg = mg * dsize
            ms = 0
        case "SOURCE":
            mg = 3 * Ns + 21 * Ns * P**3
            mg = mg * dsize
            ms = 0
        case "GRID":
            mg = 12 * 27 * Ns * (P / 2) ** 3 + 12 * Ng
            mg = mg * dsize
            ms = 0
        case _:
            raise ValueError(f"method {method.upper()} not supported.")

    l1 = dev_cons["bandwidth"] / dev_cons["bandwidth shmem"]
    l2 = dev_cons["bandwidth"] / dev_cons["bandwidth l2"]

    if model == "zero cache":
        mop = mg + l1 * ms
    elif model == "inf cache":
        mop = m + l2 * mg + l1 * ms
    elif model == "pre fetch":
        mop = m + l1 * mg + l1 * ms

    return mop


def p2g_model_intensity(dev, arch, method, time, ns, ng, P, fs_cell_size, dp_flag):
    dev_name = ""
    if dev.upper() == "CUDA":
        if arch == 80:
            dev_name = "a100"
        elif arch == 90:
            dev_name = "h200"
        else:
            raise ValueError(f"Unknown architecture {arch}")
    elif dev.upper() == "HIP":
        if arch == 94:
            dev_name = "mi300a"
        else:
            raise ValueError(f"Unknown architecture {arch}")
    elif dev.upper() == "HOST":
        dev_name = "grace"
    else:
        raise ValueError(f"Unknown device {dev}")

    dev_cons = DEVICE_CONSTANTS[dev_name]
    op_cons = OPERATION_CONSTANTS[dev_name]

    string = ""

    models = ["zero cache", "inf cache", "pre fetch"]

    flop = p2g_count_flops(op_cons, method, ns, P)

    mops = []
    for model in models:
        mops.append(
            p2g_count_mops(model, dev_cons, method, ns, ng, P, fs_cell_size, dp_flag)
        )

    intens = flop / np.array(mops)

    string = [flop]

    for i, model in enumerate(models):
        string.append(intens[i])

    return string


def g2p_count_flops(op_cons, Nt, P):
    return Nt * P**3 * (2 * 3 + op_cons["fmul"])


def g2p_count_mops(model, dev_cons, Nt, Ng, P, dp=True):
    if dp:
        dsize = 8
    else:
        dsize = 4

    m = 6 * Nt + 3 * Ng
    m = m * dsize

    mg = 3 * Nt * P**3
    mg = mg * dsize

    ms = 0

    l1 = dev_cons["bandwidth"] / dev_cons["bandwidth shmem"]
    l2 = dev_cons["bandwidth"] / dev_cons["bandwidth l2"]

    if model == "zero cache":
        mop = mg + l1 * ms
    elif model == "inf cache":
        mop = m + l2 * mg + l1 * ms
    elif model == "pre fetch":
        mop = m + l1 * mg + l1 * ms

    return mop


def g2p_model_intensity(dev, arch, method, time, nt, ng, P, dp_flag):
    dev_name = ""
    if dev.upper() == "CUDA":
        if arch == 80:
            dev_name = "a100"
        elif arch == 90:
            dev_name = "h200"
        else:
            raise ValueError(f"Unknown architecture {arch}")
    elif dev.upper() == "HIP":
        if arch == 94:
            dev_name = "mi300a"
        else:
            raise ValueError(f"Unknown architecture {arch}")
    elif dev.upper() == "HOST":
        dev_name = "grace"
    else:
        raise ValueError(f"Unknown device {dev}")

    dev_cons = DEVICE_CONSTANTS[dev_name]
    op_cons = OPERATION_CONSTANTS[dev_name]

    string = ""

    models = ["zero cache", "inf cache", "pre fetch"]

    flop = g2p_count_flops(op_cons, nt, P)

    mops = []
    for model in models:
        mops.append(g2p_count_mops(model, dev_cons, nt, ng, P, dp_flag))

    intens = flop / np.array(mops)

    string = [flop]

    for i, model in enumerate(models):
        string.append(intens[i])

    return string


def main(args):

    data_roofline = []

    nt_0 = 4000000
    cell_size_0 = 224

    p2p_model = "infcache"
    pg_model = "0cache"

    dev_name = ""
    if args.device.upper() == "CUDA":
        if int(args.arch) == 80:
            dev_name = "a100"
        elif int(args.arch) == 90:
            dev_name = "h200"

    kernels = ["p2p"]
    for kernel in kernels:
        data = load_times_from_disk(kernel, args, timestamp=args.timestamp)
        times = data["times"]["p2p"]
        methods = list(times.keys())
        cell_sizes = list(times[methods[0]].keys())
        tols = times[methods[0]][cell_sizes[0]].keys()
        nts = data["nt"]

        for i, tol in enumerate(tols):
            string = ""
            for j, nt in enumerate(nts):
                if nt not in [nt_0]:
                    continue
                string += "\\midrule\n"
                nt_mantissa, nt_exponent = f"{nt:.16e}".split("e")
                nt_mantissa = (
                    nt_mantissa.rstrip("0").rstrip(".")
                    if "." in nt_mantissa
                    else nt_mantissa
                )
                nt_exponent = int(nt_exponent)
                string += f"${nt_mantissa} \\times 10^{{{nt_exponent}}}$&"
                for k, method in enumerate(methods):
                    if method in ["GM-in"]:
                        continue
                    if k == 0:
                        string += f" {method}&"
                    else:
                        string += f"          & {method}&"
                    for l, cell_size in enumerate(cell_sizes):
                        if cell_size not in [cell_size_0]:
                            continue
                        threads_list = times[method][cell_size][tol].keys()
                        min_time = np.inf
                        min_threads = -1
                        for threads in threads_list:
                            time = times[method][cell_size][tol][threads][1:, j].mean()
                            if time < min_time:
                                min_time = time
                                min_threads = threads
                        time = min_time
                        threads = min_threads

                        string += f" ${threads}$& " + f"${time}$" + "& "
                        time_str = p2p_model_time(
                            args.device,
                            args.arch,
                            method,
                            time,
                            nt,
                            nt,
                            cell_size,
                            threads[0],
                            threads[1],
                            dp=True,
                            both=True,
                        )
                        string += f"${time_str[1]}$& ${time_str[2]}$& ${time_str[3]}$& ${time_str[0]}$&"

                        if p2p_model == "0cache":
                            time_temp = time_str[1]
                        elif p2p_model == "infcache":
                            time_temp = time_str[2]
                        elif p2p_model == "infcachepre":
                            time_temp = time_str[3]

                        data_temp = {
                            "name": kernel.upper() + "-" + method,
                            "flops": time_str[0] / time / 1e12,
                            "intensity": time_temp,
                        }

                        if method not in ["GM-2D", "SM-2D"]:
                            data_roofline.append(data_temp)

                    string += "\\\\\n"
            print(f"==========================begin {tol}==========================")
            string += "\\bottomrule"
            print(string)
            print(f"==========================end   {tol}==========================")

    kernels = ["p2g", "g2p"]
    for kernel in kernels:
        data = load_times_from_disk(kernel, args, timestamp=args.timestamp)
        df_times = pd.DataFrame(data["times"][kernel])
        if kernel == "p2g":
            df_params = pd.DataFrame(data["params"]["p2g"])
        else:
            df_params = pd.DataFrame(data["params"]["g2p"])
        nss = data["nt"] * args.up
        methods = df_times.keys()
        table = {}
        for k, method in enumerate(methods):
            for cell_size in df_times[method].keys():
                for tol in df_times[method][cell_size].keys():
                    if cell_size not in [cell_size_0]:
                        continue
                    threads_list = list(df_times[method][cell_size][tol].keys())
                    num_threads = len(threads_list)
                    a = np.empty(shape=(num_threads, len(nss)))
                    for i, threads in enumerate(threads_list):
                        a[i, :] = df_times[method][cell_size][tol][threads][1:].mean(
                            axis=0
                        )
                    for i, ns in enumerate(nss):
                        j = np.argmin(a[:, i])
                        threads = threads_list[j]
                        min_time = a[j, i]
                        if tol not in table:
                            table[tol] = {}
                        if cell_size not in table[tol]:
                            table[tol][cell_size] = {}
                        if ns not in table[tol][cell_size]:
                            table[tol][cell_size][ns] = np.empty(
                                shape=(len(methods), 2)
                            )
                        table[tol][cell_size][ns][k, :] = [threads, min_time]
        table = pd.DataFrame(table)
        print(table.columns)

        string = ""
        for j, ns in enumerate(nss):
            if ns not in [nt_0]:
                continue
            string += "\\midrule\n"
            nt_mantissa, nt_exponent = f"{ns:.16e}".split("e")
            nt_mantissa = (
                nt_mantissa.rstrip("0").rstrip(".")
                if "." in nt_mantissa
                else nt_mantissa
            )
            nt_exponent = int(nt_exponent)
            string += f"${nt_mantissa} \\times 10^{{{nt_exponent}}}$&"
            for k, method in enumerate(methods):

                for i, tol in enumerate(table.columns):
                    if i == 0:
                        if k == 0:
                            string += f" {method}&"
                        else:
                            string += f"                   & {method}&"
                    for l, cell_size in enumerate(table[tol].index):
                        if tol not in df_params[method][cell_size]:
                            continue
                        P = df_params[method][cell_size][tol][ns]["window_P"]
                        grid_shape = df_params[method][cell_size][tol][ns][
                            "grid_shape_ext"
                        ]
                        threads, time = table[tol][cell_size][ns][k]
                        threads = int(threads)
                        pps_string = f"${time}$"
                        string += f"({threads}, 1)& {pps_string}&"
                        fs_cell_size = np.ceil(
                            ns
                            / (
                                np.ceil(grid_shape[0] / P * 2)
                                * np.ceil(grid_shape[1] / P * 2) ** 2
                            )
                        )
                        if kernel == "p2g":
                            time_str = p2g_model_intensity(
                                args.device,
                                args.arch,
                                method,
                                time,
                                ns,
                                np.prod(grid_shape),
                                P,
                                fs_cell_size,
                                dp_flag=True,
                            )
                        else:
                            time_str = g2p_model_intensity(
                                args.device,
                                args.arch,
                                method,
                                time,
                                ns,
                                np.prod(grid_shape),
                                P,
                                dp_flag=True,
                            )
                        string += f"${time_str[1]}$& ${time_str[2]}$& ${time_str[3]}$& ${time_str[0]}$&"

                        if pg_model == "0cache":
                            time_temp = time_str[1]
                        elif pg_model == "infcache":
                            time_temp = time_str[2]
                        elif pg_model == "infcachepre":
                            time_temp = time_str[3]

                        data_temp = {
                            "name": kernel.upper() + "-" + method.upper(),
                            "flops": time_str[0] / time / 1e12,
                            "intensity": time_temp,
                        }

                        if method.upper() in ["GRID", "HYBRID", "TARGET"]:
                            data_roofline.append(data_temp)

                string += "\\\\\n"
        print(f"==========================begin {tol}==========================")
        string += "\\bottomrule"
        print(string)
        print(f"==========================end   {tol}==========================")

    plot_roofline(
        nt_0, cell_size_0, p2p_model, pg_model, dev_name, kernels=data_roofline
    )

    return


def plot_roofline(
    N,
    cell_size,
    p2p_model,
    pg_model,
    dev_name,
    kernels=None,  # list of dicts with 'name', 'flops', 'intensity'
):
    if kernels is None:
        kernels = [
            {"name": "P2P", "flops": 5, "intensity": 1000},
            {"name": "P2G", "flops": 1, "intensity": 10},
            {"name": "G2P", "flops": 0.5, "intensity": 8},
        ]

    peak_flops = DEVICE_CONSTANTS[dev_name]["gflops"] / 1e3
    peak_bandwidth = DEVICE_CONSTANTS[dev_name]["bandwidth"] / 1e3

    fig, ax = plt.subplots(figsize=(5, 3))

    # Limits and intersection
    x = np.logspace(-1, 4, 1000)
    bandwidth_line = peak_bandwidth * x
    flops_line = np.ones_like(x) * peak_flops

    intersection_x = peak_flops / peak_bandwidth

    # Truncated lines
    x_band = x[x <= intersection_x]
    x_flop = x[x >= intersection_x]

    ax.plot(
        x_band,
        peak_bandwidth * x_band,
        color="#a83b3b",
        linewidth=3,
        label="Bandwidth Limit",
    )
    ax.plot(
        x_flop,
        peak_flops * np.ones_like(x_flop),
        color="#3b6ea5",
        linewidth=2.5,
        label="Peak FLOP(64)/s",
    )

    # # Kernel points
    # marker_map = {'P2P': 'o', 'P2G': 's', 'G2P': 'D'}
    # color_map = {'P2P': "#00a9b7", 'P2G': "#f8971f", 'G2P': "#a6cd57"}

    colors = [
        "#a6cd57",  # light green
        "#f8971f",  # orange
        "#d6d2c4",  # light grey
        "#579d42",  # green
        "#ffd600",  # yellow
        "#005f86",  # blue
        "#00a9b7",  # teal
        "#9cadb7",  # light blue
        "#333f48",  # dark grey
    ]

    for i, kernel in enumerate(kernels):
        name = kernel["name"]
        flop = kernel["flops"]
        intensity = kernel["intensity"]
        ax.scatter(
            intensity, flop, label=name, marker="D", s=20, zorder=8
        )  # , color=colors[i%len(colors)])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (flop/byte)", fontsize=12)
    ax.set_ylabel("Performance (Tflop/s)", fontsize=12)
    ax.set_title(f"Roofline Model for {dev_name.upper()}", fontsize=14)
    ax.set_xlim(1e-1, 1e3)
    ax.set_ylim(1e-1, 1e2)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right")

    plt.tight_layout()
    fname = f"intensity_plot_cell{cell_size}_n{N}_p2pM{p2p_model}_pgM{pg_model}_dev{dev_name.upper()}.pdf"
    fpath = os.path.join(args.output_dir, fname)
    plt.savefig(fpath, format="pdf", bbox_inches="tight")


def load_times_from_disk(kernel, args, timestamp="latest", version=1):
    fname = (
        f"{kernel}_timing_result_up{args.up}"
        f"_dev{args.device.upper()}_arch{args.arch}_v{version}_{timestamp}.pkl"
    )
    fpath = os.path.join(args.input_dir, fname)
    try:
        with open(fpath, "rb") as f:
            data_dict = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            str(e)
            + f"\n please run 'analysis/ewald/time_{kernel}_methods.py' "
            + "with proper flags to generate the file"
        )
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the roofline mode for a given device. "
        "The device is specified by the `--device` and `--arch` flags. "
        "Timing results are generated by the `analysis/ewald/time_p2p_methods.py`, "
        "`analysis/ewald/time_p2g_methods.py` "
        "and the `analysis/ewald/time_g2p_methods.py` scripts."
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
        required=True,
        choices=("cuda", "hip", "host"),
        type=str,
        help="Device to run code on",
    )
    parser.add_argument(
        "--arch",
        dest="arch",
        choices=("80", "90", "94", None),
        type=str,
        help="Device compute architecture. `None` corresponds to the NVIDIA grace CPU, `80` the NVIDIA A100 GPU, `94` the NVIDIA GH200 GPU, and `94` the AMD MI300x GPU.",
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        default="latest",
        help="timestamp of result file to load (default: latest)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="analysis/ewald/plots",
        help="output directory for plots (default: analysis/ewald/plots)",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="analysis/ewald/data",
        help="input directory for timing results (default: analysis/ewald/data)",
    )
    args = parser.parse_args()
    if args.device.upper() in ["CUDA", "HIP"] and args.arch is None:
        raise ValueError(
            "arch must be passed for GPU devices, see `--help` for details"
        )
    main(args)
