import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Configure matplotlib to use LaTeX for text rendering and save plots as SVG
plt.rc("text", usetex=True)
plt.rc("font", family="serif")


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
    elif method.upper() == "GRID":
        nu = determine_degree(P)
        return Ns * (
            27 * (P / 2) ** 3 * (24 + 11 * op_cons["fmul"] + 2 * op_cons["fadd"])
            + P**3 * 3 * 2 * nu
        )
    else:
        raise ValueError(f"method {method.upper()} not supported.")


def p2g_count_mops(model, dev_cons, method, Ns, Ng, P, b_fs, dp=True):
    if dp:
        dsize = 8
    else:
        dsize = 4

    isize = 4

    m = 12 * Ns + 12 * Ng
    m = m * dsize

    if method == "cell-hybrid":
        mg = 12 * Ns + 12 * 27 * Ns * (P / 2) ** 3 / b_fs
        mg = mg * dsize
        ms = 12 * Ns * P**3 * dsize + 3 * 27 * Ns * (P / 2) ** 3 * isize
    elif method == "point-in" or "cell-in":
        mg = 3 * Ns + 21 * Ns * P**3
        mg = mg * dsize
        ms = 0
    elif method == "cell-out":
        mg = 12 * 27 * Ns * (P / 2) ** 3 + 12 * Ng
        mg = mg * dsize
        ms = 0
    else:
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


def p2g_model_time(
    dev, arch, method, time, ns, ng, P, fs_cell_size, dp_flag, ms_flag=True
):
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

    time_comp = flop / dev_cons["peak flops"]

    time_memo = np.array(mops) / dev_cons["peak band"]

    if ms_flag:
        fctr = 1e3
    else:
        fctr = 1

    string = [f"{format_sig3(time_comp*fctr)}"]

    mops = []
    for i, model in enumerate(models):
        string.append(f"{format_sig3((time_comp + time_memo[i])*fctr)}")

    return string


def main(args):
    """
    Main function. Takes `args` from the ArgumentParser at the bottom of this
    file.
    """

    ms_flag = True

    data = load_times_from_disk(args, timestamp=args.timestamp)
    df_times = pd.DataFrame(data["times"]["p2g"])
    df_params = pd.DataFrame(data["params"]["p2g"])
    nss = data["nt"] * args.up
    methods = df_times.keys()
    table = {}
    for k, method in enumerate(methods):
        for cell_size in df_times[method].keys():
            for tol in df_times[method][cell_size].keys():
                if cell_size not in [224]:
                    continue
                threads_list = list(df_times[method][cell_size][tol].keys())
                num_threads = len(threads_list)
                a = np.empty(shape=(num_threads, len(nss)))
                for i, threads in enumerate(threads_list):
                    a[i, :] = df_times[method][cell_size][tol][threads][1:].mean(axis=0)
                for i, ns in enumerate(nss):
                    j = np.argmin(a[:, i])
                    threads = threads_list[j]
                    min_time = a[j, i]
                    if tol not in table:
                        table[tol] = {}
                    if cell_size not in table[tol]:
                        table[tol][cell_size] = {}
                    if ns not in table[tol][cell_size]:
                        table[tol][cell_size][ns] = np.empty(shape=(len(methods), 2))
                    table[tol][cell_size][ns][k, :] = [threads, min_time]
    table = pd.DataFrame(table)

    if args.format.upper() == "LATEX":
        string = ""
        for j, ns in enumerate(nss):
            if ns not in [4000000]:
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
                        pps_string = f"${format_sig3(time*1e3)}$"
                        string += f"({threads}, 1)& {pps_string}&"
                        fs_cell_size = np.ceil(
                            ns
                            / (
                                np.ceil(grid_shape[0] / P * 2)
                                * np.ceil(grid_shape[1] / P * 2) ** 2
                            )
                        )
                        time_str = p2g_model_time(
                            args.device,
                            args.arch,
                            method,
                            time,
                            ns,
                            np.prod(grid_shape),
                            P,
                            fs_cell_size,
                            dp_flag=True,
                            ms_flag=ms_flag,
                        )
                        string += f"${time_str[1]}$& ${time_str[2]}$& ${time_str[3]}$& ${time_str[0]}$&"
                string += "\\\\\n"
        print(f"==========================begin {tol}==========================")
        string += "\\bottomrule"
        print(string)
        print(f"==========================end   {tol}==========================")

    elif args.format.upper() == "CL":
        for j, ns in enumerate(nss):
            if ns not in [4000000]:
                continue

            # Header for this block
            print("=" * 80)
            nt_mantissa, nt_exponent = f"{ns:.16e}".split("e")
            nt_mantissa = (
                nt_mantissa.rstrip("0").rstrip(".")
                if "." in nt_mantissa
                else nt_mantissa
            )
            nt_exponent = int(nt_exponent)
            print(f"Nt = {nt_mantissa}e{nt_exponent}")
            print("-" * 80)

            header = f"{'Method':<12} | {'Tol':<8} | {'Threads':>8} | {'T (ms)':>10} | {'T_0':>10} | {'T_inf':>10} | {'T_inf,l1':>10} | {'T_f':>10}"
            print(header)
            print("-" * len(header))

            for k, method in enumerate(methods):
                for i, tol in enumerate(table.columns):
                    for l, cell_size in enumerate(table[tol].index):
                        if tol not in df_params[method][cell_size]:
                            continue

                        P = df_params[method][cell_size][tol][ns]["window_P"]
                        grid_shape = df_params[method][cell_size][tol][ns][
                            "grid_shape_ext"
                        ]
                        threads, time = table[tol][cell_size][ns][k]
                        threads = int(threads)
                        pps_string = f"{format_sig3(time * 1e3)}"

                        fs_cell_size = np.ceil(
                            ns
                            / (
                                np.ceil(grid_shape[0] / P * 2)
                                * np.ceil(grid_shape[1] / P * 2) ** 2
                            )
                        )

                        time_str = p2g_model_time(
                            args.device,
                            args.arch,
                            method,
                            time,
                            ns,
                            np.prod(grid_shape),
                            P,
                            fs_cell_size,
                            dp_flag=True,
                            ms_flag=ms_flag,
                        )

                        print(
                            f"{method:<12} | {tol:<8} | {threads:>8} | {pps_string:>10} |"
                            f" {time_str[1]:>10} | {time_str[2]:>10} | {time_str[3]:>10} | {time_str[0]:>10}"
                        )
            print("=" * 80)
    else:
        raise ValueError

    if ms_flag:
        print("using millisecond!")


def load_times_from_disk(args, timestamp="latest", version=1):
    fname = (
        f"p2g_timing_result_up{args.up}"
        f"_dev{args.device.upper()}_arch{args.arch}_v{version}_{timestamp}.pkl"
    )
    fpath = os.path.join(args.output_dir, fname)
    with open(fpath, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results from scaling test.")
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
        default="cuda",
        type=str,
        help="Device to run code on",
    )
    parser.add_argument(
        "--arch",
        dest="arch",
        default=90,
        type=int,
        help="Device compute architecture",
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
        default="analysis/stokes1p/data",
        help="output directory for timing results (default: .)",
    )
    parser.add_argument(
        "--format", default="latex", help="output format, either 'latex' or 'cl'"
    )
    args = parser.parse_args()
    main(args)
