import os, sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

sys.path.insert(
    0, os.path.join(os.path.split(__file__)[0], "..")
)  # for `_utils, _parmas`
from _utils import p2p_efficiency, p2g_efficiency, g2p_efficiency
from _params import se_params_stokes_comb


def get_eff(stage, args, time, nt):
    params = se_params_stokes_comb(
        {"box": [1, 1, 1]}, tolerance=args.tol, cell_size=args.cell_size, num_sources=nt
    )
    fs_cell_size = np.ceil(
        nt
        / (
            np.ceil(params.grid_shape_ext[0] / params.window_P * 2)
            * np.ceil(params.grid_shape_ext[1] / params.window_P * 2) ** 2
        )
    )
    match stage.upper():
        case "P2P":
            p2p_eff = p2p_efficiency(
                args.device,
                args.arch,
                "GM-1D",
                time,
                nt,
                args.cell_size,
                -1,
                -1,
                dp=True,
                both=True,
            )
            return p2p_eff[0]
        case "P2G":
            if args.device == "HIP":
                p2g_eff = p2g_efficiency(
                    args.device,
                    args.arch,
                    "GRID",
                    time,
                    nt,
                    params.window_P,
                    fs_cell_size,
                    True,
                )
            else:
                p2g_eff = p2g_efficiency(
                    args.device,
                    args.arch,
                    "HYBRID",
                    time,
                    nt,
                    params.window_P,
                    fs_cell_size,
                    True,
                )
            return p2g_eff.split("\n")[0]
        case "G2P":
            g2p_eff = g2p_efficiency(
                args.device,
                args.arch,
                "TARGET",
                time,
                nt,
                params.window_P,
                dp_flag=True,
            )
            return g2p_eff.split(" ")[0]
        case _:
            raise NotImplementedError(
                f"stage {stage.upper()} efficiency not implemented."
            )


def get_time_eff_dicts(args):
    times_dict = {}
    effs_dict = {}

    for device, arch in [
        ("CUDA", 90),
        ("HIP", 94),
        ("CUDA", 80),
        ("HOST", "aarch64"),
        ("HOST", "x86_64"),
    ]:
        args.device = device
        args.arch = arch
        data = load_times_from_disk(args, timestamp=args.timestamp)
        nt_list = data["nt"]

        times_dict[device + str(arch)] = {}
        effs_dict[device + str(arch)] = {}
        for i, nt in enumerate(nt_list):
            times_dict[device + str(arch)][nt] = {}
            effs_dict[device + str(arch)][nt] = {}
            for stage in data["times"].keys():
                times = np.mean(data["times"][stage][1:], axis=0)
                if stage in ["gpu_pre", "fs_tot", "p2p+fs_tot"]:
                    continue
                elif stage in ["fft", "cnv", "ifft", "sca"]:
                    if "fgc" not in times_dict[device + str(arch)][nt]:
                        times_dict[device + str(arch)][nt]["fgc"] = times[i]
                    else:
                        times_dict[device + str(arch)][nt]["fgc"] += times[i]
                else:
                    times_dict[device + str(arch)][nt][stage] = times[i]
                    eff_str = get_eff(stage, args, times[i], nt)
                    effs_dict[device + str(arch)][nt][stage] = eff_str

    return times_dict, effs_dict, nt_list


def main(args):
    """
    Main function. Takes `args` from the ArgumentParser at the bottom of this
    file.
    """

    key_mapping = {
        "CUDA80": "A100",
        "CUDA90": "H200",
        "HIP94": "MI300",
        "HOSTaarch64": "Grace",
        "HOSTx86_64": "Epyc",
    }
    ut_colors = ["#579d42", "#005f86", "#9cadb7", "#d6d2c4", "#333f48"]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    label_size = 30
    font_size = 18
    tick_size = 18
    width = 0.25

    for p in range(2):
        ax = axs[p]
        times_devs, effs_devs, nt_list = get_time_eff_dicts(args)
        times_devs = {
            key_mapping.get(key, key): value for key, value in times_devs.items()
        }
        effs_devs = {
            key_mapping.get(key, key): value for key, value in effs_devs.items()
        }
        print(times_devs.keys())
        if p == 0:
            del times_devs["Grace"]
            del times_devs["Epyc"]
        if p == 1:
            del times_devs["H200"]
            del times_devs["MI300"]
            del times_devs["A100"]
        devices = times_devs.keys()
        times_df = pd.DataFrame(times_devs, index=nt_list)

        # Flatten the dataframe into a multi-index DataFrame: nt x device x stage
        stacked_data = []

        for nt in times_df.index:
            for device in times_df.columns:
                for stage, time in times_df.loc[nt, device].items():
                    stacked_data.append(
                        {"nt": nt, "device": device, "stage": stage, "time": time}
                    )

        flat_df = pd.DataFrame(stacked_data)

        # Pivot it to get a format suitable for stacked bar plot
        pivot_df = flat_df.pivot_table(
            index=["nt", "device"], columns="stage", values="time"
        ).fillna(0)

        # For tracking bar bottoms (for stacking)
        D = len(devices)
        offsets = np.linspace(-width * (D - 1) / 2, width * (D - 1) / 2, D)
        for i, nt in enumerate(nt_list):
            bottom = {device: 0 for device in devices}
            for j, stage in enumerate(pivot_df.columns):
                for k, device in enumerate(devices):
                    time = pivot_df.loc[(nt, device), stage]
                    if stage != "fgc":
                        eff_str = effs_devs[device][nt][stage]
                    else:
                        eff_str = ""
                    spp = (time * 1e6) / nt
                    color = ut_colors[j % len(ut_colors)]
                    bar = ax.bar(
                        i + offsets[k],
                        spp,
                        width,
                        label=stage.upper() if i == 0 and k == 0 else "",
                        bottom=bottom[device],
                        color=color,
                        alpha=0.5,
                        edgecolor="black",
                    )
                    if stage == "g2p" or (device == "A100" and nt >= 4000000):
                        pass
                    else:
                        ax.text(
                            i + offsets[k],
                            bottom[device] + spp / 2,  # vertical middle of this stage
                            eff_str,
                            ha="center",
                            va="center",  # center horizontally and vertically
                            fontsize=10,
                            color="black",
                            fontweight="bold",
                        )
                    bottom[device] += spp

            # Add device name above each bar
            for k, device in enumerate(devices):
                if nt >= 4000000 and device == "A100":
                    continue
                else:
                    ax.text(
                        i + offsets[k],
                        bottom[device],
                        device,
                        ha="center",
                        va="bottom",
                        fontsize=12,
                    )

            ax.set_xlabel("$N=N_s=N_t$", fontsize=label_size)
            ax.set_ylabel("$\mathrm{\mu s}/N$", fontsize=label_size)

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

            if p == 0:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            if p == 1:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

            ax.legend(title="Stage", fontsize=12, loc="upper right", ncols=2)

    axs[1].yaxis.get_label().set_visible(False)

    fig.suptitle(f"Stokes Spectral Ewald Performance Portability", fontsize=label_size)

    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    fname = f"portability_plot_cell{args.cell_size}_tol{args.tol}.pdf"
    fpath = os.path.join(args.output_dir, fname)
    plt.savefig(fpath, format="pdf", bbox_inches="tight")
    plt.show()


def load_times_from_disk(args, timestamp="latest", version=1):
    fname = (
        f"ewald_timing_result_up{args.up}_clsz{args.cell_size}_tol{args.tol}"
        f"_dev{args.device.upper()}_arch{args.arch}_v{version}_{timestamp}.pkl"
    )
    fpath = os.path.join(args.input_dir, fname)
    try:
        with open(fpath, "rb") as f:
            data_dict = pickle.load(f)
    except:  # 0 data_dict if device data not found
        print(fpath)
        data_dict = {
            "nt": [250000, 1000000, 4000000],
            "times": {
                "p2p": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "p2g": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "fft": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "cnv": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "ifft": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "g2p": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            },
        }
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
        "-t",
        "--timestamp",
        default="latest",
        help="timestamp of result file to load (default: latest)",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="analysis/ewald/data",
        help="output directory for timing results (default: .)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="analysis/ewald/plots",
        help="output directory for timing results (default: .)",
    )
    parser.add_argument(
        "--cell_size",
        dest="cell_size",
        type=str,
        default=224,
    )
    parser.add_argument(
        "--tol",
        dest="tol",
        type=str,
        default=1e-4,
    )
    parser.add_argument(
        "--ylim",
        dest="ylim",
        # default=[30e6, 1.9e6],
        default=None,
    )

    args = parser.parse_args()
    main(args)
