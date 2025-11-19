import os
import argparse
import numpy as np
import pandas as pd
import pickle


def main(args):
    """
    Main function. Takes `args` from the ArgumentParser at the bottom of this
    file.
    """
    data = load_times_from_disk(args, timestamp=args.timestamp)
    distributions = data["times"].keys()
    stages = ["p2p", "p2g", "fft", "cnv", "ifft", "g2p", "total"]
    nindx = -1
    n = data["nt"][nindx]

    if args.format.upper() == "LATEX":
        string = "\\midrule\n"
        # list the distributions
        string += "& "
        for i, dist in enumerate(distributions):
            string += dist + " &"
            if i < len(distributions) - 1:
                string += " "
        string = string[:-2]
        string += "\\\\\n"
        # list the times per stage
        for stage in stages:
            string += f"{stage} &"
            for i, dist in enumerate(distributions):
                if stage == 'total':
                    time = 0
                    for _stage in stages:
                        if _stage == 'total':
                            continue
                        time += data["times"][dist][_stage][:, nindx][1:].mean()
                else:
                    time = data["times"][dist][stage][:, nindx][1:].mean()
                pps = n / (time * 1e6)
                string += f"{round(time*1e4, 2)}" + " &"
                if i < len(distributions) - 1:
                    string += " "
            string = string[:-2]
            string += "\\\\\n"
        string += "\\bottomrule"
        print(string)
    elif args.format.upper() == "CL":
        string = ""
        col_width = 12

        # Header
        header = f"{'Stage':<{col_width}} |"
        for dist in distributions:
            header += f" {dist:>{col_width}} |"
        string += "-" * len(header) + "\n"
        string += header + "\n"
        string += "-" * len(header) + "\n"

        # Data rows
        for stage in stages:
            label = stage
            if stage == "total":
                label = "p2p+fs_tot"
            row = f"{stage:<{col_width}} |"
            for dist in distributions:
                time = data["times"][dist][label][1:].mean()
                pps = n / (time * 1e6)
                row += f" {pps:>{col_width}.2f} |"
            string += row + "\n"

        string += "-" * len(header)
        print(string)


def load_times_from_disk(args, timestamp="latest", version=1):
    fname = (
        f"distributions_timing_result_up{args.up}_clsz{args.cell_size}_tol{args.tolerance}"
        f"_dev{args.device.upper()}_arch{args.arch}_v{version}_{timestamp}.pkl"
    )
    fpath = os.path.join(args.output_dir, fname)
    with open(fpath, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyize particle distribution timing script."
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
        required=True,
        help="Device to run code on",
    )
    parser.add_argument(
        "--arch",
        dest="arch",
        type=int,
        required=True,
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
        default="analysis/ewald/data",
        help="output directory for timing results (default: .)",
    )
    parser.add_argument(
        "--format", default="latex", help="output format, either 'latex' or 'cl'"
    )
    parser.add_argument(
        "--cell_size",
        type=int,
        default=224,
        help="Set the max src per cell (default: 1024)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Spectral Ewald tolerance",
    )
    args = parser.parse_args()
    main(args)
    exit()
