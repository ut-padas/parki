import os
import argparse
import numpy as np
import pickle


def main(args):
    """
    Main function. Takes `args` from the ArgumentParser at the bottom of this
    file.
    """
    data = load_times_from_disk(args, timestamp=args.timestamp)
    dtypes = data["times"].keys()
    stages = ["p2p", "p2g", "fft", "cnv", "ifft", "g2p", "total"]
    nindx = -1
    n = data["nt"][nindx]

    if args.format.upper() == "LATEX":
        string = "\\midrule\n"
        # list the distributions
        string += "& "
        for i, dtype in enumerate(dtypes):
            string += dtype + " &"
            if i < len(dtypes) - 1:
                string += " "
        string = string[:-2]
        string += "\\\\\n"
        # list the times per stage
        for stage in stages:
            string += f"{stage} &"
            for i, dtype in enumerate(dtypes):
                if stage == "total":
                    time = 0
                    for _stage in stages:
                        if _stage == "total":
                            continue
                        time += data["times"][dtype][_stage][:, nindx][1:].mean()
                else:
                    time = data["times"][dtype][stage][:, nindx][1:].mean()
                print(n, stage, time)
                pps = n / (time * 1e6)
                string += f"{round(time*1e3, 2)}" + " &"
                if i < len(dtypes) - 1:
                    string += " "
            string = string[:-2]
            string += "\\\\\n"
        string += "\\bottomrule"
        print(string)
    elif args.format.upper() == "CL":
        raise NotImplementedError("CL format not yet implemented.")
        print(string)


def load_times_from_disk(args, timestamp="latest", version=1):
    fname = (
        f"dtypes_timing_result_up{args.up}_clsz{args.cell_size}_tol{args.tolerance}"
        f"_dev{args.device.upper()}_arch{args.arch}_v{version}_{timestamp}.pkl"
    )
    print(f"Loading from {fname}")
    fpath = os.path.join(args.input_dir, fname)
    try:
        with open(fpath, "rb") as f:
            data_dict = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            str(e)
            + f"\n please run 'analysis/ewald/time_dtypes.py' "
            + "with proper flags to generate the file"
        )
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyize Ewald timings for different floating point precisions. "
        "Run the 'analysis/ewald/time_dtypes.py' to generate that data for this script."
    )
    parser.add_argument(
        "--up",
        dest="up",
        type=int,
        default=1,
        help="Set the upsampeling parameter (default: 1)",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="analysis/ewald/data",
        help="input directory for timing results (default: analysis/ewald/data)",
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
        "--format", default="latex", help="output format, either 'latex' or 'cl'"
    )
    parser.add_argument(
        "--cell_size",
        type=int,
        default=160,
        help="Set the max src per cell (default: 160)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-1,
        help="Spectral Ewald tolerance (default: 1e-1)",
    )
    args = parser.parse_args()
    if args.device.upper() in ["CUDA", "HIP"] and args.arch is None:
        raise ValueError(
            "arch must be passed for GPU devices, see `--help` for details"
        )
    main(args)
    exit()
