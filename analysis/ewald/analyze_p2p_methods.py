import os
import argparse
import numpy as np
import pandas as pd
import pickle

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


def p2p_cnt_mop(dev_cons, variant, Nt, s, bt, bs, dp=True):
    if dp:
        dsize = 8
    else:
        dsize = 4

    if variant == "GM-1D":
        mop = Nt * (3 + 27 * s * 12 + 3) * dsize
    elif variant == "SM-1D":
        mop = (
            Nt / bt * (bt * 3 + 27 * s * 12 + bt * 3) * dsize
            + Nt
            * (3 + 27 * s * 12)
            * dsize
            * dev_cons["bandwidth"]
            / dev_cons["bandwidth shmem"]
        )
    elif variant == "GM-2D":
        mop = Nt * bs * (3 + 27 * s * 12 / bs + 3) * dsize
    elif variant == "SM-2D":
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
        print("performance model for {variant} does not exist")

    return mop


def p2p_cnt_intns(op_cons, dev_cons, variant, Nt, s, bt, bs, dp=True):

    return p2p_cnt_flop(op_cons, Nt, s) / p2p_cnt_mop(
        dev_cons, variant, Nt, s, bt, bs, dp
    )


def p2p_efficiency(dev, arch, variant, time, nt, s, bt, bs, dp=True, both=False):
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
    p2p["flop"] = p2p_cnt_flop(op_cons, nt, s)
    p2p["mop"] = p2p_cnt_mop(dev_cons, variant, nt, s, bt, bs, dp)
    p2p["intense"] = p2p_cnt_intns(op_cons, dev_cons, variant, nt, s, bt, bs, dp)

    print(
        f"method: {variant} \tp2p_intensity: {p2p['intense']:.3f}",
        f"p2p_flops: {p2p['flop']}, p2p_mops: {p2p['mop']}",
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


def main(args):
    """
    Main function. Takes `args` from the ArgumentParser at the bottom of this
    file.
    """
    data = load_times_from_disk(args, timestamp=args.timestamp)
    df_times = pd.DataFrame(data["times"]["p2p"])
    df_params = pd.DataFrame(data["params"]["p2p"])
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
                        table[tol][cell_size][ns] = np.empty(
                            shape=(len(methods), 2), dtype=object
                        )
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
                        eff_str = p2p_efficiency(
                            args.device,
                            args.arch,
                            method,
                            time,
                            ns,
                            cell_size,
                            threads[0],
                            threads[1],
                            dp=True,
                            both=True,
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
                        grid_shape = df_params[method][cell_size][tol][ns][
                            "grid_shape_ext"
                        ]
                        threads, time = table[tol][cell_size][ns][k]
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
                        eff_str = p2p_efficiency(
                            args.device,
                            args.arch,
                            method,
                            time,
                            ns,
                            cell_size,
                            threads[0],
                            threads[1],
                            dp=True,
                            both=True,
                        )

                        print(f"{eff_str}")
                        string += f"{method:>12} | {tol:<8} | {threads[0]:>4},{threads[1]:>3} | {ppms_string:>12} | {speedup_str:>8} | {eff_str[0]:>10}\n"
        print(string)


def load_times_from_disk(args, timestamp="latest", version=1):
    fname = (
        f"p2p_timing_result_up{args.up}"
        f"_dev{args.device.upper()}_arch{args.arch}_v{version}_{timestamp}.pkl"
    )
    fpath = os.path.join(args.input_dir, fname)
    try:
        with open(fpath, "rb") as f:
            data_dict = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            str(e)
            + f"\n please run 'analysis/ewald/time_p2p_methods.py' "
            + "with proper flags to generate the file"
        )
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare method variants "
        "for the P2P kernel on a given device. "
        "Timing results are generated by the "
        "'analysis/ewald/time_p2p_methods.py' file."
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
