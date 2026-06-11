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

DEVICE_CONSTANTS = {
    "a100": {
        "bandwidth": 1555,
        "gflops": 9.7e3,
        "bandwidth shmem": 20e3,
        "intensity": 9.7e3 / 1555,
        "peak flops": 9.7e3 * 1e9,
        "peak band": 1555 * 1e9,
    },
    "h200": {
        "bandwidth": 4000,
        "gflops": 33.5e3,
        "bandwidth shmem": np.inf,
        "intensity": 33.5e3 / 4000,
        "peak flops": 33.5e3 * 1e9,
        "peak band": 4000 * 1e9,
    },
    "mi300a": {
        "bandwidth": 5300,
        "gflops": 61.3e3,
        "bandwidth shmem": np.inf,
        "intensity": 61.3e3 / 5300,
        "peak flops": 61.3e3 * 1e9,
        "peak band": 5300 * 1e9,
    },
    "grace": {
        "bandwidth": 1000 / 2,
        "gflops": 7.1e3 / 2,
        "bandwidth shmem": np.inf,
        "intensity": 7.1e3 / 1000,
        "peak flops": 7.1e3 * 1e9 / 2,
        "peak band": 1000 * 1e9 / 2,
    },
    "epyc": {
        "bandwidth": 204.8 * 2,
        "gflops": 5e3,
        "bandwidth shmem": np.inf,
        "intensity": 5e3 / (204.8 * 2),
        "peak flops": 5e3 * 1e9 / 2,
        "peak band": 204.8 * 2 * 1e9 / 2,
    },
}


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
    "mi300a": {
        "fadd": 2,
        "fmul": 2,
        "fsqrt": 41,
        "frsqrt": 30,
        "fdiv": 35,
        "fexpn": 50,
        "fsinh": 361,
        "ferf": 155,
    },
    "grace": {
        "fadd": 2,
        "fmul": 2,
        "fsqrt": 41,
        "frsqrt": 30,
        "fdiv": 35,
        "fexpn": 50,
        "fsinh": 361,
        "ferf": 155,
    },
    "epyc": {
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
        if arch == 0:
            dev_name = "grace"
        elif arch == 1:
            dev_name = "epyc"
        else:
            raise ValueError(f"Unknown architecture {arch}")
    else:
        raise ValueError(f"Unknown device {dev}")

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
        string = f"${p2g_flops/dev_cons['peak flops']/time:.0%}$\nflops"
    else:
        string = f"${p2g_mops/dev_cons['peak band']/time:.0%}$\nmops"
    string = string.replace("%", "\\%")
    return string


def get_dicts_for_method(METHOD):
    times_dict = {}
    eff_dict = {}
    params_dict = {}
    for device, arch in [
        ("CUDA", 90),
        ("HIP", 94),
        ("CUDA", 80),
        ("HOST", 0),
        ("HOST", 1),
    ]:
        args.device = device
        args.arch = arch
        data = load_times_from_disk(args, timestamp=args.timestamp)
        df_times = pd.DataFrame(data["times"]["p2g"])
        nss = data["nt"] * args.up

        params_dict[device + str(arch)] = pd.DataFrame(data["params"]["p2g"])
        methods = df_times.keys()

        table = {}
        table_eff = {}

        for k, method in enumerate(methods):
            if method != METHOD:
                continue
            for cell_size in df_times[method].keys():
                for tol in df_times[method][cell_size].keys():
                    args.tolerance = tol
                    args.cell_size = cell_size
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
                        table[ns] = min_time
                        loc_params = params_dict[device + str(arch)][method][cell_size][
                            tol
                        ][ns]
                        P = loc_params["window_P"]
                        grid_shape = loc_params["grid_shape_ext"]
                        fs_cell_size = np.ceil(
                            ns
                            / (
                                np.ceil(grid_shape[0] / P * 2)
                                * np.ceil(grid_shape[1] / P * 2) ** 2
                            )
                        )
                        eff = p2g_efficiency(
                            args.device,
                            args.arch,
                            method,
                            min_time,
                            ns,
                            P,
                            fs_cell_size,
                            True,
                        )
                        table_eff[ns] = eff
        times_dict[device + str(arch)] = table
        eff_dict[device + str(arch)] = table_eff
    return times_dict, eff_dict, nss


def main(args):
    """
    Main function. Takes `args` from the ArgumentParser at the bottom of this
    file.
    """

    key_mapping = {
        "CUDA80": "NVIDIA A100",
        "CUDA90": "NVIDIA H200",
        "HIP94": "AMD MI300a",
        "HOST0": "NVIDIA Grace",
        "HOST1": "AMD Epyc",
    }
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

    fig, axs = plt.subplots(2, 2, figsize=(14, 6 * len(args.p2g_methods)), sharex=True)
    label_size = 30
    font_size = 18
    tick_size = 25
    width = 1

    # Share y within each row only
    axs[0, 1].sharey(axs[0, 0])  # first row shares y-axis
    axs[1, 1].sharey(axs[1, 0])  # second row shares y-axis

    for p in range(2):
        for q, p2g_method in enumerate(args.p2g_methods):

            ax = axs[p, q]
            times_devs, effs_devs, nt_list = get_dicts_for_method(p2g_method)
            times_devs = {
                key_mapping.get(key, key): value for key, value in times_devs.items()
            }
            effs_devs = {
                key_mapping.get(key, key): value for key, value in effs_devs.items()
            }
            if p == 0:
                del times_devs["NVIDIA Grace"]
                del times_devs["AMD Epyc"]
            if p == 1:
                del times_devs["NVIDIA H200"]
                del times_devs["AMD MI300a"]
                del times_devs["NVIDIA A100"]

            for dev in times_devs:
                for nt in times_devs[dev]:
                    times_devs[dev][nt] = nt / times_devs[dev][nt]

            df_times = pd.DataFrame(times_devs)

            if args.ylim is not None:
                bars = df_times.plot(
                    kind="bar",
                    ax=ax,
                    color=colors[3 * p :],
                    alpha=1,
                    width=width,
                    edgecolor="black",
                    ylim=(0, args.ylim[p]),
                )
            else:
                bars = df_times.plot(
                    kind="bar",
                    ax=ax,
                    color=colors[3 * p :],
                    alpha=1,
                    width=width,
                    edgecolor="black",
                )

            for i, bar_group in enumerate(bars.containers):
                for j, bar in enumerate(bar_group):
                    eff_text = effs_devs[list(times_devs.keys())[i]][nt_list[j]]
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 2,
                        eff_text,
                        ha="center",
                        va="bottom",
                        fontsize=font_size,
                        rotation=0,
                    )

            ax.set_xlabel("$N=N_s=N_t$", fontsize=label_size)
            ax.set_ylabel(r"$N/\mathrm{\mu s}$", fontsize=label_size)
            if p == 0:
                ax.set_title(f"P2G-{p2g_method}", fontsize=label_size)

            x_labels = []
            for nt in nt_list:
                nt_mantissa, nt_exponent = f"{nt:.16e}".split("e")
                nt_mantissa = (
                    nt_mantissa.rstrip("0").rstrip(".")
                    if "." in nt_mantissa
                    else nt_mantissa
                )
                nt_exponent = int(nt_exponent)
                x_labels.append(f"${nt_mantissa} \\times 10^{{{nt_exponent}}}$")
            ax.set_xticks(range(len(nt_list)))
            ax.set_xticklabels(x_labels, rotation=0, fontsize=tick_size)

            y_labels = []
            for y in ax.get_yticks():
                if y == 0:
                    y_labels.append("0")
                    continue
                y_mantissa = f"{y/1e6:.2f}".rstrip("0").rstrip(".")
                string = f"${y_mantissa}$"
                y_labels.append(string)
            ax.set_yticklabels(y_labels, fontsize=tick_size, rotation=0)
            ax.legend(title="Device", fontsize=14, loc="upper left")

    plt.tight_layout()
    fname = f"p2g_portability_plot_cell{args.cell_size}_method{'_'.join(args.p2g_methods)}.pdf"
    fpath = os.path.join(args.output_dir, fname)
    plt.savefig(fpath, format="pdf", bbox_inches="tight")


def load_times_from_disk(args, timestamp="latest", version=1):
    fname = (
        f"p2g_timing_result_up{args.up}"
        f"_dev{args.device}_arch{args.arch}_v{version}_{timestamp}.pkl"
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
            + f"on a {args.device} arch {args.arch} device "
            + "with proper flags to generate the file"
        )
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare the P2G kernel "
        "on different devices. "
        "Timing results are generated by the "
        "'analysis/ewald/time_p2g_methods.py' file. "
        "NOTE: host arch 0 corresponds to the grace CPU, "
        "while host arch 1 corresponds to the epyc CPU."
    )
    parser.add_argument(
        "--up",
        dest="up",
        type=int,
        default=1,
        help="Set the upsampeling parameter (default: 1)",
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
        "-o",
        "--output-dir",
        default="analysis/ewald/plots",
        help="output directory for timing results (default: analysis/ewald/data)",
    )
    parser.add_argument(
        "--p2g_methods",
        dest="p2g_methods",
        type=str,
        default=["HYBRID", "GRID"],
        help="p2g method",
    )
    parser.add_argument(
        "--cell_size", dest="cell_size", type=str, default=256, help="p2g cell size"
    )
    parser.add_argument(
        "--tol", dest="tol", type=str, default=1e-4, help="ewald tolerance"
    )
    parser.add_argument(
        "--ylim", dest="ylim", default=[30e6, 1.9e6], help="matplotlib ylim for graph"
    )

    args = parser.parse_args()
    main(args)
