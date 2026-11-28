import os
import argparse
import numpy as np
import pickle
import shutil


def sample_gaussian(box, c, n):
    """
    rejects points outside of the box
    """
    lx, ly, lz = box
    center = np.array([lx / 2, ly / 2, lz / 2])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([lx, ly, lz])

    points = np.empty((3, n))
    i = 0
    num_accepted = 0
    while num_accepted < n:
        samples = np.random.normal(loc=0.0, scale=c, size=(3, n)) + center.reshape(3, 1)
        mask = np.all(
            (samples > lower.reshape(3, 1)) & (samples < upper.reshape(3, 1)), axis=0
        )
        new_samples = samples[:, mask]
        num_new_samples = min(new_samples.shape[1], n - num_accepted)
        points[:, num_accepted : num_accepted + num_new_samples] = new_samples[
            :, :num_new_samples
        ]
        num_accepted += num_new_samples
        i += 1

    return points


def sample_unit_sphere_surface(n):
    points = np.random.normal(size=(3, n))
    points /= np.linalg.norm(points, axis=0)
    return points


def plot_distribution():
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    plt.rc("font", family="serif")
    if shutil.which("latex") is not None:
        plt.rc("text", usetex=True)
    n = 4_000
    label_size = 30
    tick_size = 18

    # Sample points
    sph_points = sample_unit_sphere_surface(n)
    sph_points /= 2
    sph_points += 0.5
    sph_x, sph_y, sph_z = sph_points

    u_x, u_y, u_z = np.random.uniform(size=(3, n))
    n_x, n_y, n_z = sample_gaussian([1, 1, 1], 0.3, n)

    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(14, 6))
    for ax in axs:
        ax.set_aspect("equal", adjustable="box")

    # Prepare grid for KDE evaluation
    def make_grid(x, z, bins=200):
        xi = np.linspace(np.min(x), np.max(x), bins)
        zi = np.linspace(np.min(z), np.max(z), bins)
        X, Z = np.meshgrid(xi, zi)
        positions = np.vstack([X.ravel(), Z.ravel()])
        return X, Z, positions

    # Plot each panel with 2D KDE
    datasets = [
        (u_x, u_z, r"Uniform (\textbf{U})"),
        (n_x, n_z, r" Truncated Gaussian (\textbf{N})"),
        (sph_x, sph_z, r"Sphere (\textbf{S})"),
    ]
    for ax, (x, z, title) in zip(axs, datasets):
        X, Z, positions = make_grid(x, z, bins=200)

        # KDE estimation
        kde = gaussian_kde(np.vstack([x, z]))
        density = kde(positions).reshape(X.shape) * 1e3

        # Plot smooth density
        im = ax.pcolormesh(X, Z, density, shading="auto")

        # Colorbar beneath each subplot
        cbar = plt.colorbar(
            im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.15
        )
        cbar.set_label("Density", fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        # cbar.ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))

        ax.set_title(title, fontsize=24)

    # Figure title
    fig.suptitle(
        "Particle Distributions Projected to the $x$-$z$ Plane", fontsize=label_size
    )

    # Save figure
    pname = "analysis/ewald/plots/distributions_kde.pdf"
    plt.rcParams["pdf.compression"] = 0
    plt.savefig(pname, format="pdf", bbox_inches="tight")
    plt.show()


def main(args):
    """
    Main function. Takes `args` from the ArgumentParser at the bottom of this
    file.
    """
    data = load_times_from_disk(args, timestamp=args.timestamp)
    if args.baseline_dir:
        baseline = load_times_from_disk(args, input_dir=args.baseline_dir)
        compare_layouts(baseline, data)
        return
    plot_distribution()
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
            string += f"{stage.upper()} &"
            for i, dist in enumerate(distributions):
                if stage == "total":
                    time = 0
                    for _stage in stages:
                        if _stage == "total":
                            continue
                        time += data["times"][dist][_stage][:, nindx][1:].mean()
                else:
                    time = data["times"][dist][stage][:, nindx][1:].mean()
                string += f"{round(time*1e3, 1)}" + " &"
                if i < len(distributions) - 1:
                    string += " "
            string = string[:-2]
            string += "\\\\\n"
        string += "\\bottomrule"
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
            row = f"{stage:<{col_width}} |"
            for dist in distributions:
                times = data["times"][dist]
                if stage == "total" and stage not in times:
                    time = sum(
                        times[key][1:, nindx].mean() for key in stages if key != "total"
                    )
                else:
                    time = times[stage][1:, nindx].mean()
                pps = n / (time * 1e6)
                row += f" {pps:>{col_width}.2f} |"
            string += row + "\n"

        string += "-" * len(header)
    print(string)


def compare_layouts(baseline, candidate):
    """Compare runs over identical inputs, discarding the first timed repeat."""
    settings = ("device", "up", "cell_size", "tolerance", "seed", "threads_list")
    if (
        baseline["nt"] != candidate["nt"]
        or baseline["repeats"] != candidate["repeats"]
        or baseline["times"].keys() != candidate["times"].keys()
        or any(
            getattr(baseline["args"], name) != getattr(candidate["args"], name)
            for name in settings
        )
    ):
        raise ValueError(
            "Baseline and candidate must use identical benchmark settings."
        )

    print("Cell list storage in MiB by stage (includes team maps where used).")
    print("Speedup = baseline / candidate time, averaged after warm-up.")
    print(
        f"Storage measured with {candidate['args'].threads_list[0]} threads per team."
    )
    for i, n in enumerate(candidate["nt"]):
        print(f"\nNt = {n:,}, Ns = {n * candidate['args'].up:,}")
        print(
            f"{'distribution':<21} {'stage':<5} {'baseline MiB':>12} "
            f"{'variable MiB':>12} {'saved %':>8} "
            f"{'baseline ms':>12} {'variable ms':>12} {'speedup':>8}"
        )
        for dist in candidate["times"]:
            for stage in ("p2p", "p2g", "g2p", "total"):
                t0 = baseline["times"][dist][stage][1:, i].mean()
                t1 = candidate["times"][dist][stage][1:, i].mean()
                m0 = baseline["memory"][dist][n]
                m1 = candidate["memory"][dist][n]
                if stage == "total":
                    b0, b1 = max(m0.values()), max(m1.values())
                else:
                    b0, b1 = m0[stage], m1[stage]
                print(
                    f"{dist:<21} {stage:<5} {b0 / 2**20:12.2f} "
                    f"{b1 / 2**20:12.2f} {100 * (1 - b1 / b0):8.2f} "
                    f"{t0 * 1000:12.2f} {t1 * 1000:12.2f} {t0 / t1:8.3f}"
                )
    print("The total row uses peak cell list storage across the three stages.")


def load_times_from_disk(args, timestamp="latest", version=1, input_dir=None):
    fname = (
        f"distributions_timing_result_up{args.up}_clsz{args.cell_size}_tol{args.tolerance}"
        f"_dev{args.device.upper()}_arch{args.arch}_v{version}_{timestamp}.pkl"
    )
    fpath = os.path.join(input_dir or args.input_dir, fname)
    try:
        with open(fpath, "rb") as f:
            data_dict = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            str(e)
            + f"\n please run 'analysis/ewald/time_particle_distributions.py' "
            + "with proper flags to generate the file"
        )
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyize particle distribution timing script. "
        "Data is generated by the 'analysis/ewald/time_particle_distributions.py' file."
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
        "--baseline-dir",
        help="Compare with the latest fixed-layout run in this directory.",
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
        default=224,
        help="Set the max src per cell (default:224)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Spectral Ewald tolerance",
    )
    args = parser.parse_args()
    if args.device.upper() in ["CUDA", "HIP"] and args.arch is None:
        raise ValueError(
            "arch must be passed for GPU devices, see `--help` for details"
        )
    main(args)
    exit()
