import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# # Configure matplotlib to use LaTeX for text rendering and save plots as SVG
# plt.rc("text", usetex=True)
# plt.rc("font", family="serif")

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


def p2p_cnt_mop(dev_cons, method, Nt, s, bt, bs, dp=True):
    if dp:
        dsize = 8
    else:
        dsize = 4

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
        print("performance model for {method} does not exist")

    return mop


def p2p_cnt_intns(op_cons, dev_cons, method, Nt, s, bt, bs, dp=True):

    return p2p_cnt_flop(op_cons, Nt, s) / p2p_cnt_mop(
        dev_cons, method, Nt, s, bt, bs, dp
    )


def p2p_efficiency(dev, arch, method, time, nt, s, bt, bs, dp=True, both=False):
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
    op_cons = OPERATION_CONSTANTS[dev_name]
    string = ""

    p2p = {}
    p2p["flop"] = p2p_cnt_flop(op_cons, nt, s)
    p2p["mop"] = p2p_cnt_mop(dev_cons, method, nt, s, bt, bs, dp)
    p2p["intense"] = p2p_cnt_intns(op_cons, dev_cons, method, nt, s, bt, bs, dp)

    print(
        f"method: {method} \tp2p_intensity: {p2p['intense']:.3f}",
        f"p2p_flops: {p2p['flop']}, p2g_mops: {p2p['mop']}",
    )
    if p2p["intense"] > dev_cons["intensity"]:
        string = f"${p2p['flop']/dev_cons['peak flops']/time:.0%}$ (flops)"
    else:
        string = f"${p2p['mop']/dev_cons['peak band']/time:.0%}$ (mops)"
    string = string.replace("%", "\\%")
    if both:
        string_flops = f"${p2p['flop']/dev_cons['peak flops']/time:.0%}$"
        string_flops = string_flops.replace("%", "\\%")
        string_mops = f"${p2p['mop']/dev_cons['peak band']/time:.0%}$"
        string_mops = string_mops.replace("%", "\\%")
        string = [string_flops, string_mops]
    return string


def get_time_eff_dicts(args, Variant, Cell_size, Tol):
    times_devs = {}
    effs_devs = {}

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
        times = pd.DataFrame(data["times"]["p2p"])
        nt_list = data["nt"]

        methods = list(times.keys())

        table = {}
        table_eff = {}

        for k, method in enumerate(methods):
            if method != Variant:
                continue
            for cell_size in times[method].keys():
                if cell_size != Cell_size:
                    continue
                for tol in times[method][cell_size].keys():
                    if tol != Tol:
                        continue
                    for i, nt in enumerate(nt_list):
                        min_time = np.inf
                        min_threads = -1
                        for threads in times[method][cell_size][tol].keys():
                            time = times[method][cell_size][tol][threads][1:, i].mean()
                            if time < min_time:
                                min_time = time
                                min_threads = threads

                        time = min_time
                        threads = min_threads

                        table[nt] = time
                        eff_str = p2p_efficiency(
                            args.device,
                            args.arch,
                            method,
                            time,
                            nt,
                            cell_size,
                            threads[0],
                            threads[1],
                            dp=True,
                            both=True,
                        )
                        table_eff[nt] = eff_str[0]

        times_devs[device + str(arch)] = table
        effs_devs[device + str(arch)] = table_eff

    return times_devs, effs_devs, nt_list


def main(args):
    """
    Main function. Takes `args` from the ArgumentParser at the bottom of this
    file.
    """

    key_mapping = {
        "CUDA80": "A100",
        "CUDA90": "H200",
        "HIP94": "MI300A",
        "HOST0": "Grace",
        "HOST1": "Epyc",
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

    fig, axs = plt.subplots(1, 2, figsize=(14, 5 * len(args.p2p_methods)), sharex=True)
    label_size = 30
    font_size = 18
    tick_size = 18
    width = 1

    print(axs.shape)
    for p in range(2):
        for q, p2p_method in enumerate(args.p2p_methods):

            if p2p_method == "point":
                method = "GM-1D"
            elif p2p_method == "point-2d":
                method = "GM-2D"
            elif p2p_method == "point-in":
                method = "GM-in"
            elif p2p_method == "cell":
                method = "SM-1D"
            elif p2p_method == "cell-2d":
                method = "SM-2D"

            ax = axs[p]
            times_devs, effs_devs, nt_list = get_time_eff_dicts(
                args, p2p_method, args.cell_size, args.tol
            )
            times_devs = {
                key_mapping.get(key, key): value for key, value in times_devs.items()
            }
            effs_devs = {
                key_mapping.get(key, key): value for key, value in effs_devs.items()
            }
            if p == 0:
                del times_devs["Grace"]
                del times_devs["Epyc"]
            if p == 1:
                del times_devs["H200"]
                del times_devs["MI300A"]
                del times_devs["A100"]

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
                        eff_text + "\nflops",
                        ha="center",
                        va="bottom",
                        fontsize=font_size,
                        rotation=0,
                    )

            ax.set_xlabel("$N=N_s=N_t$", fontsize=label_size)
            ax.set_ylabel(r"$N/\mathrm{\mu s}$", fontsize=label_size)

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

    axs[1].yaxis.get_label().set_visible(False)

    fig.suptitle(f"P2P-{p2p_method}", fontsize=label_size)

    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    fname = f"p2p_portability_plot_cell{args.cell_size}_method{'_'.join(args.p2p_methods)}.pdf"
    fpath = os.path.join(args.output_dir, fname)
    plt.savefig(fpath, format="pdf", bbox_inches="tight")


def load_times_from_disk(args, timestamp="latest", version=1):
    fname = (
        f"p2p_timing_result_up{args.up}"
        f"_dev{args.device}_arch{args.arch}_v{version}_{timestamp}.pkl"
    )
    fpath = os.path.join(args.input_dir, fname)
    try:
        with open(fpath, "rb") as f:
            data_dict = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            str(e)
            + f"\n please run 'analysis/ewald/time_p2p_methods.py' "
            + f"on a {args.device} arch {args.arch} device "
            + "with proper flags to generate the file"
        )
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare the P2P kernel "
        "on different devices. "
        "Timing results are generated by the "
        "'analysis/ewald/time_p2p_methods.py' file. "
        "NOTE: host arch 0 corresponds to the grace CPU, "
        "while host arch 1 corresponds to the epyc CPU."
    )
    parser.add_argument(
        "--up",
        dest="up",
        type=int,
        default=1,
        help="Set the upsampeling parameter (default: 16)",
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
        "--cell_size",
        dest="cell_size",
        type=str,
        default=256,
        help="near field cell size",
    )
    parser.add_argument(
        "--tol", dest="tol", type=str, default=1e-4, help="ewald tolerance"
    )
    parser.add_argument(
        "--ylim", dest="ylim", default=[30e6, 1.9e6], help="matplotlib ylim for graph"
    )

    args = parser.parse_args()
    main(args)
