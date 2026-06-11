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
}


def p2p_cnt_flop(op_cons, kernel, Nt, s):
    match kernel.upper():
        case "DISTANCE":
            flops = 27 * Nt * s * (4 * op_cons["fadd"] + 3 * op_cons["fadd"])
        case "LAPLACE / EWALD":
            flops = 27 * Nt * s * (
                op_cons["fmul"] + op_cons["frsqrt"]
            ) * np.pi / 6 + 27 * Nt * s * (4 * op_cons["fadd"] + 4 + op_cons["fmul"])
        case "STOKES_SL / EWALD":
            flops = 27 * Nt * s * (
                4
                + 10 * op_cons["fmul"]
                + 9 * op_cons["fadd"]
                + op_cons["frsqrt"]
                + op_cons["fdiv"]
            ) * np.pi / 6 + 27 * Nt * s * (4 * op_cons["fadd"] + 4 + op_cons["fmul"])
        case "STOKES_COMB":
            flops = 27 * Nt * s * (
                37 * op_cons["fmul"]
                + 17 * op_cons["fadd"]
                + 36
                + op_cons["frsqrt"]
                + op_cons["fdiv"]
                + op_cons["fexpn"]
                + op_cons["ferf"]
            ) * np.pi / 6 + 27 * Nt * s * (4 * op_cons["fadd"] + 4 + op_cons["fmul"])
        case _:
            raise ValueError(f"performance model for {kernel.upper()} does not exist")

    return flops


def p2p_cnt_mop(dev_cons, kernel, method, Nt, s, bt, bs, dp=True):
    if dp:
        dsize = 8
    else:
        dsize = 4

    if kernel == "distance":
        if method == "GM-1D":
            mop = Nt * (3 + 27 * s * 3 + 3) * dsize
        else:
            print("performance model for {method} does not exist")
    elif kernel == "laplace":
        if method == "GM-1D":
            mop = Nt * (3 + 27 * s * 3 + 1) * dsize
        else:
            print("performance model for {method} does not exist")
    elif kernel == "stokes_sl":
        if method == "GM-1D":
            mop = Nt * (3 + 27 * s * 6 + 3) * dsize
        else:
            print("performance model for {method} does not exist")
    elif kernel == "stokes_both+ewald":
        if method == "GM-1D":
            mop = Nt * (3 + 27 * s * 12 + 3) * dsize
        elif method == "SM-1D":
            mop = (
                Nt / bt * (bt * 3 + 27 * s * 12 + bt * 3) * dsize
                + Nt
                * (3 + 27 * s * 12)
                * dsize
                * dev_cons["bandwidth"]
                / dev_cons["bandwidth shmem"]
            )
        elif method == "GM-2D":
            mop = Nt * bs * (3 + 27 * s * 12 / bs + 3) * dsize
        elif method == "SM-2D":
            mop = (
                Nt / bt * (bt * 3 + 27 * s * 12 + bt * 3) * dsize
                + Nt
                * bs
                * (3 + 27 * s * 12 / bs)
                * dsize
                * dev_cons["bandwidth"]
                / dev_cons["bandwidth shmem"]
            )
        else:
            raise ValueError(f"performance model for {method} does not exist")
    else:
        raise ValueError(f"performance model for {kernel.upper()} does not exist")

    return mop


def p2p_cnt_intns(op_cons, dev_cons, kernel, method, Nt, s, bt, bs, dp=True):

    return p2p_cnt_flop(op_cons, kernel, Nt, s) / p2p_cnt_mop(
        dev_cons, kernel, method, Nt, s, bt, bs, dp
    )


def p2p_efficiency(
    dev, arch, kernel, method, time, nt, s, bt, bs, dp=True, both=False, has_mop=True
):
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

    p2p = {}
    p2p["flop"] = p2p_cnt_flop(op_cons, kernel, nt, s)
    if both is not True:
        if has_mop:
            p2p["mop"] = p2p_cnt_mop(dev_cons, kernel, method, nt, s, bt, bs, dp)
            p2p["intense"] = p2p_cnt_intns(
                op_cons, dev_cons, kernel, method, nt, s, bt, bs, dp
            )

            print(
                f"method: {method} \tp2p_intensity: {p2p['intense']:.3f}",
                f"p2p_flops: {p2p['flop']}, p2g_mops: {p2p['mop']}",
            )
            if p2p["intense"] > dev_cons["intensity"]:
                string = f"${p2p['flop']/dev_cons['peak flops']/time:.0%}$ (flops)"
            else:
                string = f"${p2p['mop']/dev_cons['peak band']/time:.0%}$ (mops)"
            string = string.replace("%", "\\%")
        else:
            print("should have mop if print flop + mop together")
    if both:
        string_flops = f"${p2p['flop']/dev_cons['peak flops']/time:.0%}$"
        string_flops = string_flops.replace("%", "\\%")
        if has_mop:
            string_mops = f"${p2p['mop']/dev_cons['peak band']/time:.0%}$"
            string_mops = string_mops.replace("%", "\\%")
        else:
            string = [string_flops, None]
    return string


def get_time_eff_dicts(args, Dev, Arch, Nt, Variant, Tol):
    times_kernels = {}
    effs_kernels = {}

    args.device = Dev
    args.arch = Arch
    data = load_times_from_disk(args, timestamp=args.timestamp)
    times = pd.DataFrame(data["times"]["p2p"])
    threads_opt = data["threads"]["p2p"]

    methods = list(times[list(times.keys())[0]].keys())

    for kernel in times.keys():
        if kernel not in times_kernels:
            times_kernels[kernel] = {}
            effs_kernels[kernel] = {}
        for k, method in enumerate(methods):
            if method != Variant:
                print(f"skipping method {method}.")
                continue
            cell_size_list = list(times[kernel][method].keys())
            for cell_size in times[kernel][method].keys():
                for tol in times[kernel][method][cell_size].keys():
                    if tol != Tol:
                        print(f"skipping tol {tol}")
                        continue
                    for nt in times[kernel][method][cell_size][tol].keys():
                        if nt != Nt:
                            continue

                        time = np.average(times[kernel][method][cell_size][tol][nt])
                        times_kernels[kernel][cell_size] = time
                        threads, t_per_thread, s_per_thread, s_threads_2d = threads_opt[
                            kernel
                        ][method][cell_size][tol][nt]
                        if np.isnan(s_threads_2d):
                            s_threads = 1
                        else:
                            s_threads = int(s_threads_2d)
                        t_threads = int(threads / s_threads)
                        eff_str = p2p_efficiency(
                            args.device,
                            args.arch,
                            kernel,
                            method,
                            time,
                            nt,
                            cell_size,
                            t_threads,
                            s_threads,
                            dp=True,
                            both=True,
                            has_mop=False,
                        )
                        effs_kernels[kernel][cell_size] = eff_str[0]

    return times_kernels, effs_kernels, cell_size_list


def main(args):
    """
    Main function. Takes `args` from the ArgumentParser at the bottom of this
    file.
    """

    dev = args.device
    arch = args.arch
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

    key_mapping = {
        "laplace / ewald": "Poisson",
        "stokes_sl / ewald": "Stokes Single Layer",
        "stokes_comb": "Ewald",
    }
    fig, axs = plt.subplots(
        1,
        len(args.p2p_methods),
        figsize=(10 * len(args.p2p_methods), 4),
        sharey=True,
        sharex=True,
    )
    label_size = 25
    font_size = 17
    tick_size = 17
    width = 1

    colors = [
        # "#a6cd57", # light green
        # "#f8971f", # orange
        # "#d6d2c4", # light grey
        # "#579d42", # green
        # "#ffd600", # yellow
        "#005f86",  # blue
        "#00a9b7",  # teal
        "#9cadb7",  # light blue
        "#333f48",  # dark grey
    ]

    complementary_colors = {
        "#f8971f": "#005f86",  # orange and blue
        "#a6cd57": "#005f86",  # light green and blue
        "#579d42": "#00a9b7",  # green and teal
    }

    for idx, p2p_method in enumerate(args.p2p_methods):
        if isinstance(axs, list):
            ax = axs[idx]
        else:
            ax = axs

        times_kernels, effs_kernels, cell_size_list = get_time_eff_dicts(
            args, args.device, args.arch, args.nt, p2p_method, args.tol
        )
        print(times_kernels.keys())
        times_kernels = {
            key_mapping.get(key, key): value for key, value in times_kernels.items()
        }
        effs_kernels = {
            key_mapping.get(key, key): value for key, value in effs_kernels.items()
        }
        for kernel in times_kernels:
            for cell_size in times_kernels[kernel]:
                times_kernels[kernel][cell_size] = (
                    args.nt / times_kernels[kernel][cell_size]
                )
        df_times = pd.DataFrame(times_kernels)
        if args.ylim is not None:
            bars = df_times.plot(
                kind="bar",
                ax=ax,
                color=colors,
                alpha=1,
                width=width,
                edgecolor="black",
                ylim=(0, args.ylim),
            )
        else:
            bars = df_times.plot(
                kind="bar",
                ax=ax,
                color=colors,
                alpha=1,
                width=width,
                edgecolor="black",
            )
        for i, bar_group in enumerate(bars.containers):
            for j, bar in enumerate(bar_group):
                eff_text = effs_kernels[list(times_kernels.keys())[i]][
                    cell_size_list[j]
                ]
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    eff_text + "\nflops",
                    ha="center",
                    va="bottom",
                    fontsize=font_size,
                    rotation=0,
                )
        ax.set_xlabel("$s$", fontsize=label_size, labelpad=-15)
        ax.set_ylabel(r"$N/\mathrm{\mu s}$", fontsize=label_size)
        ax.set_title(f"P2P-{dev_name.upper()}-{p2p_method}", fontsize=label_size)

        x_labels = []
        for cell_size in cell_size_list:
            x_labels.append(f"${cell_size}$")
        ax.set_xticks(range(len(cell_size_list)))
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
        ax.legend(title="Workload", fontsize=16, loc="upper right")

    plt.tight_layout()
    fname = f"p2p_workload_plot_dev{dev_name}_method{'_'.join(args.p2p_methods)}.pdf"
    fpath = os.path.join(args.output_dir, fname)
    plt.savefig(fpath, format="pdf", bbox_inches="tight")
    # plt.show()


def load_times_from_disk(args, timestamp="latest", version=1):
    fname = (
        f"p2p_timing_workload_result_up{args.up}"
        f"_dev{args.device}_arch{args.arch}_v{version}_{timestamp}.pkl"
    )
    fpath = os.path.join(args.input_dir, fname)
    try:
        with open(fpath, "rb") as f:
            data_dict = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            str(e)
            + f"\n please run 'analysis/ewald/time_p2p_workloads.py' "
            + "with proper flags to generate the file"
        )
    return data_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compare workloads "
        "for the P2P kernel on a given device. "
        "Timing results are generated by the "
        "'analysis/ewald/time_p2p_workloads.py' file."
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
        "-o",
        "--output-dir",
        default="analysis/ewald/plots",
        help="output directory for timing results (default: analysis/ewald/data)",
    )
    parser.add_argument(
        "--p2p_methods",
        dest="p2p_methods",
        type=str,
        default=["GM-1D"],
        help="p2p method",
    )
    parser.add_argument(
        "--tol", dest="tol", type=str, default=1e-1, help="ewald tolerance"
    )
    parser.add_argument(
        "--nt", dest="nt", type=int, default=1000000, help="number of target points."
    )
    parser.add_argument(
        "--ylim", dest="ylim", default=2.85e8, help="matplotlib ylim for graph"
    )

    args = parser.parse_args()
    if args.device.upper() in ["CUDA", "HIP"] and args.arch is None:
        raise ValueError(
            "arch must be passed for GPU devices, see `--help` for details"
        )
    main(args)
