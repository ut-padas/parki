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
    ranks = [1, 2, 4, 8, 16, 32, 64]
    stages = [
        "p2p",
        "p2g",
        "fft",
        "cnv",
        "ifft",
        "g2p",
        "mpi_sort",
        "mpi_ghost",
        "mpi_grid_ghost",
    ]
    nindx = -1

    if args.format.upper() == "LATEX":
        string = "\\toprule\n"
        # list the distributions
        string += "& "
        for i, N in enumerate(ranks):
            string += f"{N}" + " &"
            if i < len(ranks) - 1:
                string += " "
        string = string[:-2]
        string += "\\\\\n"
        # list the times per stage
        for stage in stages:
            string += f"{stage} &"
            for N in ranks:
                try:
                    data = load_times_from_disk(args, N)
                except:
                    continue
                n = data["nt"][nindx]
                if stage == "total":
                    time = 0
                    for _stage in stages:
                        if _stage == "total":
                            continue
                        time += data["times"][_stage][:, nindx][1:].mean()
                        string += f"{round(time*1e3, 2)}" + " &"
                else:
                    if stage in stages[-3:] and N == 1:
                        time == "---"
                        string += f"---" + " &"
                    else:
                        time = data["times"][stage][:, nindx][1:].mean()
                        string += f"{round(time*1e3, 1)}" + " &"
                if i < len(ranks) - 1:
                    string += " "
            string = string[:-2]
            string += "\\\\\n"
        string += "\\bottomrule"
        print(string)
    elif args.format.upper() == "CL":
        raise NotImplementedError("CL format not yet implemented.")
        print(string)


def load_times_from_disk(args, N, timestamp="latest", version=1):
    fname = (
        f"distributed_ewald_timing_result_N{N}_up{args.up}_clsz{args.cell_size}_tol{args.tolerance}"
        f"_dev{args.device.upper()}_arch{args.arch}_v{version}_latest.pkl"
    )
    # print(f"Loading from {fname}")
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
        help="Device to run code on",
    )
    parser.add_argument(
        "--arch",
        dest="arch",
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
        default="analysis/distributed/data",
        help="output directory for timing results (default: .)",
    )
    parser.add_argument(
        "--format", default="latex", help="output format, either 'latex' or 'cl'"
    )
    parser.add_argument(
        "--cell_size",
        type=int,
        default=224,
        help="Set the max src per cell (default: 160)",
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
