import os
import argparse
import numpy as np
import pandas as pd
import pickle

DEVICE_CONSTANTS = {
    "a100": {
        "bandwidth": 1555,
        "tflops": 9.7e3,
        "bandwidth shmem": 20e3,
        "intensity": 9.7e3 / 1555,
        "peak flops": 9.7e3 * 1e9,
        "peak band": 1555 * 1e9,
    },
    "h200": {
        "bandwidth": 3352,
        "tflops": 33.5e3,
        "bandwidth shmem": np.inf,
        "intensity": 33.5e3 / 3352,
        "peak flops": 33.5e3 * 1e9,
        "peak band": 3352 * 1e9,
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


def p2g_count_flops(method, Ns, P):
    if method.upper() == "HYBRID":
        return Ns * (P) ** 3 * 35
    elif method.upper() in ["BASE", "SOURCE"]:
        return Ns * (P) ** 3 * 37
    elif method.upper() == "GRID":
        nu = determine_degree(P)
        return Ns * (27 * P / 2**3 * 3 + P**3 * (3 * (2 * nu)) * 35)
    else:
        raise ValueError(f"method {method} not analyzed.")

    return 0


def p2g_count_mops_hbm(Ns, P, b_fs, dp=True):
    if dp:
        dsize = 8
    else:
        dsize = 4

    return 27 * Ns * (P / 2) ** 3 * 12 * dsize / b_fs


def p2g_count_mops_shmem(method, Ns, P, dp=True):
    if dp:
        dsize = 8
    else:
        dsize = 4

    isize = 4
    if method.upper() == "HYBRID":
        return Ns * (P) ** 3 * (12 * dsize + 3 * isize)
    return 0


def p2g_intensity(
    method,
    p2g_count_flops,
    p2g_count_hbm,
    p2g_count_shmem,
    P,
    b_fs,
    bandwidth_hbm,
    bandwidth_shmem,
    dp,
):

    return p2g_count_flops(method, 1, P) / (
        p2g_count_hbm(1, P, b_fs, dp)
        + p2g_count_shmem(method, 1, P, dp) * bandwidth_hbm / bandwidth_shmem
    )


def p2g_get_counts(
    method, ns, P, fs_cell_size, bandwidth_hbm, bandwidth_shmem, dp_flag=True
):
    flops = p2g_count_flops(method, ns, P)
    mops = (
        p2g_count_mops_hbm(ns, P, fs_cell_size, dp_flag)
        + p2g_count_mops_shmem(method, ns, P, dp_flag) * bandwidth_hbm / bandwidth_shmem
    )
    intensity = p2g_intensity(
        method,
        p2g_count_flops,
        p2g_count_mops_hbm,
        p2g_count_mops_shmem,
        P,
        fs_cell_size,
        bandwidth_hbm,
        bandwidth_shmem,
        dp_flag,
    )
    return intensity, flops, mops


def p2g_efficiency(dev, arch, method, time, ns, P, fs_cell_size, dp_flag):
    dev_name = ""
    if dev.upper() == "CUDA":
        if int(arch) == 80:
            dev_name = "a100"
        elif int(arch) == 90:
            dev_name = "h200"
        else:
            raise ValueError(f"Unknown architecture {arch}")
    dev_cons = DEVICE_CONSTANTS[dev_name]
    string = ""
    p2g_intensity, p2g_flops, p2g_mops = p2g_get_counts(
        method,
        ns,
        P,
        fs_cell_size,
        dev_cons["bandwidth"],
        dev_cons["bandwidth shmem"],
        dp_flag,
    )
    # print(f"method: {method} \tp2g_intensity: {p2g_intensity:.3f}", f"p2g_flops: {p2g_flops}, p2g_mops: {p2g_mops}")
    if p2g_intensity > dev_cons["intensity"]:
        string = f"${p2g_flops/dev_cons['peak flops']/time:.0%}$ (flops)"
    else:
        string = f"${p2g_mops/dev_cons['peak band']/time:.0%}$ (mops)"
    string = string.replace("%", "\\%")
    return string


def main(args):
    """
    Main function. Takes `args` from the ArgumentParser at the bottom of this
    file.
    """
    data = load_times_from_disk(args, timestamp=args.timestamp)
    df_times = pd.DataFrame(data["times"]["p2g"])
    df_params = pd.DataFrame(data["params"]["p2g"])
    nss = data["nt"] * args.up
    methods = df_times.keys()
    table = {}
    for k, method in enumerate(methods):
        for cell_size in df_times[method].keys():
            for tol in df_times[method][cell_size].keys():
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
                        particle_per_micro_second = ns / (time * 1e6)
                        string += (
                            f" ${threads}$& {round(particle_per_micro_second, 2)}&"
                        )
                        if k == 0:
                            string += f" ---&"
                        else:
                            speedup = table[tol][cell_size][ns][0, 1] / time
                            string += f" ${speedup:.2f}\\times$&"
                        fs_cell_size = np.ceil(
                            ns
                            / (
                                np.ceil(grid_shape[0] / P * 2)
                                * np.ceil(grid_shape[1] / P * 2) ** 2
                            )
                        )
                        eff_str = p2g_efficiency(
                            args.device,
                            args.arch,
                            method,
                            time,
                            ns,
                            P,
                            fs_cell_size,
                            dp_flag=True,
                        )
                        string += f"{eff_str}& "
                string += "\\\\\n"
        print(f"==========================begin {tol}==========================")
        string += "\\bottomrule"
        print(string)
        print(f"==========================end   {tol}==========================")
    elif args.format.upper() == "CL":
        string = ""
        for j, ns in enumerate(nss):
            string += "-" * 80 + "\n"
            nt_mantissa, nt_exponent = f"{ns:.16e}".split("e")
            nt_mantissa = (
                nt_mantissa.rstrip("0").rstrip(".")
                if "." in nt_mantissa
                else nt_mantissa
            )
            nt_exponent = int(nt_exponent)
            string += f"Nt = {nt_mantissa}e{nt_exponent}\n"

            for k, method in enumerate(methods):
                for i, tol in enumerate(table.columns):
                    if i == 0:
                        if k == 0:
                            string += f"{'Method':>12} | {'Tol':>8} | {'Threads':>8} | {'PPμS':>12} | {'Speedup':>8} | {'Efficiency':>10}\n"
                        string += "-" * 80 + "\n"
                    for l, cell_size in enumerate(table[tol].index):
                        if tol not in df_params[method][cell_size]:
                            continue
                        P = df_params[method][cell_size][tol][ns]["window_P"]
                        print(P)
                        grid_shape = df_params[method][cell_size][tol][ns][
                            "grid_shape_ext"
                        ]
                        threads, time = table[tol][cell_size][ns][k]
                        threads = int(threads)
                        particle_per_micro_second = ns / (time * 1e6)
                        ppms_string = f"{round(particle_per_micro_second, 2)}"

                        if k == 0:
                            speedup_str = "---"
                        else:
                            speedup = table[tol][cell_size][ns][0, 1] / time
                            speedup_str = f"{speedup:.2f}x"

                        fs_cell_size = np.ceil(
                            ns
                            / (
                                np.ceil(grid_shape[0] / P * 2)
                                * np.ceil(grid_shape[1] / P * 2) ** 2
                            )
                        )

                        eff_str = p2g_efficiency(
                            args.device,
                            args.arch,
                            method,
                            time,
                            ns,
                            P,
                            fs_cell_size,
                            dp_flag=True,
                        )

                        string += f"{method:>12} | {tol:<8} | {threads:>8} | {ppms_string:>12} | {speedup_str:>8} | {eff_str:>10}\n"
        print(string)


def load_times_from_disk(args, timestamp="latest", version=1):
    fname = (
        f"p2g_timing_result_up{args.up}"
        f"_dev{args.device.upper()}_arch{args.arch}_v{version}_{timestamp}.pkl"
    )
    fpath = os.path.join(args.input_dir, fname)
    try:
        with open(fpath, "rb") as f:
            try:
                data_dict = pickle.load(f)
            except Exception as e:
                raise RuntimeError(
                    "This file contains GPU arrays and requires a compatible CUDA installation to read. "
                    "Please ensure your CUDA driver is up to date and supports the required CUDA runtime version."
                ) from e
    except FileNotFoundError as e:
        raise FileNotFoundError(
            str(e)
            + f"\n please run 'analysis/ewald/time_p2g_methods.py' "
            + "with proper flags to generate the file"
        )
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare method variants "
        "for the P2G kernel on a given device. "
        "Timing results are generated by the "
        "'analysis/ewald/time_p2g_methods.py' file."
    )
    parser.add_argument(
        "--up",
        dest="up",
        type=int,
        default=1,
        help="Set the upsampeling parameter (default: 1)",
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
        "-i",
        "--input-dir",
        default="analysis/ewald/data",
        help="input directory for timing results (default: analysis/ewald/data)",
    )
    parser.add_argument(
        "--format", default="latex", help="output format, either 'latex' or 'cl'"
    )
    args = parser.parse_args()
    if args.device.upper() in ["CUDA", "HIP"] and args.arch is None:
        raise ValueError(
            "arch must be passed for GPU devices, see `--help` for details"
        )
    main(args)
