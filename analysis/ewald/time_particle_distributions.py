import os
import time
import argparse
import numpy as np
import pickle

import parkipy
from parkipy.ewald import _ewald


def measure_cell_lists(call):
    """Record logical allocations at each cell list construction."""
    cell_list = _ewald.CellList
    storage = []

    class MeasuredCellList(cell_list):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            forces = kwargs.get("forces")
            arrays = [
                self.particle_list,
                self.particle_index,
                self.counter,
                self.nonempty_cells,
                self.nonempty_cell_index,
                getattr(self, "cell_offsets", None),
            ]
            if forces is not None:
                arrays.extend(
                    self.force_list if isinstance(forces, tuple) else (self.force_list,)
                )
            self._measurement = [
                sum(array.nbytes for array in arrays if array is not None)
            ]
            storage.append(self._measurement)

        def _team_chunks(self, chunk_size):
            chunks = super()._team_chunks(chunk_size)
            self._measurement[0] += sum(array.nbytes for array in chunks)
            return chunks

    _ewald.CellList = MeasuredCellList
    try:
        result = call()
    finally:
        _ewald.CellList = cell_list
    return result, {
        "p2p": sum(storage[0] + storage[1]),
        "p2g": storage[2][0],
        "g2p": storage[3][0],
    }


def main(args):
    """
    Main function. Takes `args` from the ArgumentParser at the bottom of this
    file. `args.nt` will be a list of integers, and we run `run()` with each of
    these integers as the number of targets.
    """
    nt_list = args.nt
    repeats = args.repeats
    all_times = dict()
    all_memory = dict()
    distributions = [
        "uniform-uniform",
        "gaussian-gaussian",
        "gaussian-uniform",
        "ellipsoid-ellipsoid",
        "ellipsoid-uniform",
    ]
    for distribution in distributions:
        print(f"::::::::: distribution={distribution}")
        all_times[distribution] = {}
        all_memory[distribution] = {}
        args.distribution = distribution
        for i, nt in enumerate(nt_list):
            args.nt = nt
            print(f":: Running Nt={nt}")
            for j in range(repeats):
                tot_times = []
                for nthreads in args.threads_list:
                    print(f":::: Running threads={nthreads}")
                    args.threads = nthreads
                    if j == 0 and nthreads == args.threads_list[0]:
                        (_, times), storage = measure_cell_lists(lambda: run(args))
                        all_memory[distribution][nt] = storage
                    else:
                        _, times = run(args)
                    tot_times.append(times)
                print(f"   {j+1}th pass done (out of {repeats})")
                # Initialize arrays for storing runtimes and memory
                if i == 0 and j == 0:
                    for key in times:
                        all_times[distribution][key] = np.full(
                            shape=[repeats, len(nt_list)], fill_value=np.inf
                        )
                # Store runtimes
                for times in tot_times:
                    for key, val in times.items():
                        old_time = all_times[distribution][key][j, i]
                        new_time = val["tot"]
                        if new_time < old_time:
                            all_times[distribution][key][j, i] = val["tot"]
    save_times_to_disk(nt_list, repeats, all_times, all_memory, args)
    return


def save_times_to_disk(nt, repeats, times, memory, args):
    now_str = time.strftime("%y%m%dT%H%M%S%Z")
    format_version = 1
    if args.device.upper() == "HOST":
        arch = None
    else:
        import cupy as cp

        arch = cp.cuda.Device(0).compute_capability
    fname_base = (
        f"distributions_timing_result_up{args.up}_clsz{args.cell_size}_tol{args.tolerance}"
        f"_dev{args.device.upper()}_arch{arch}_v{format_version}"
    )
    fname = fname_base + f"_{now_str}.pkl"
    fname_link = fname_base + "_latest.pkl"
    fpath = os.path.join(args.output_dir, fname)
    fpath_link = os.path.join(args.output_dir, fname_link)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(fpath, "wb") as f:
        pickle.dump(
            {
                "nt": nt,
                "repeats": repeats,
                "times": times,
                "memory": memory,
                "args": args,
            },
            f,
        )
    if os.path.lexists(fpath_link):
        os.remove(fpath_link)
    os.symlink(fname, fpath_link)
    print(f"Saved results to {fpath!r} and {fpath_link!r}")
    return


def sample_gaussian(box, c, n, am):
    """
    rejects points outside of the box
    """
    lx, ly, lz = box
    center = am.array([lx / 2, ly / 2, lz / 2])
    lower = am.array([0.0, 0.0, 0.0])
    upper = am.array([lx, ly, lz])

    points = am.empty((3, n))
    num_accepted = 0
    while num_accepted < n:
        samples = am.random.normal(loc=0.0, scale=c, size=(3, n)) + center.reshape(3, 1)
        mask = am.all(
            (samples > lower.reshape(3, 1)) & (samples < upper.reshape(3, 1)), axis=0
        )
        new_samples = samples[:, mask]
        num_new_samples = min(new_samples.shape[1], n - num_accepted)
        points[:, num_accepted : num_accepted + num_new_samples] = new_samples[
            :, :num_new_samples
        ]
        num_accepted += num_new_samples

    return points


def sample_unit_sphere_surface(n):
    points = np.random.normal(size=(3, n))
    points /= np.linalg.norm(points, axis=0)
    return points


def run(args):
    # input arguments
    am = parkipy.utils.get_array_module(args.device)
    np.random.seed(args.seed)
    if am is not np:
        am.random.seed(args.seed)
    nt = args.nt
    box = np.array([1.0, 1.0, 1.0])

    ns = nt * args.up

    cp = parkipy.utils.get_array_module(args.device)

    for i, dis in enumerate(args.distribution.split("-")):
        # 0 <- src_dis, 1 <- trg_dis
        n = None
        if i == 0:
            n = ns
        if i == 1:
            n = nt
        match dis.upper():
            case "GAUSSIAN":
                c = 0.3
                arr = sample_gaussian(box, c, n, am)
            case "ELLIPSOID":
                arr = sample_unit_sphere_surface(n)
                arr = cp.asarray(arr)
                # scale to be inside box
                arr *= 0.49
                arr += 0.5
                assert arr.shape[1] == n
            case "UNIFORM":
                arr = am.random.uniform(size=(3, n)) * am.array(box).reshape(3, 1)
            case _:
                raise ValueError(
                    f"Distribution must be one of 'GAUSSIAN', 'ELLIPSOID', or 'UNIFORM', got '{dis.upper()}'"
                )
        if i == 0:
            src = arr
        if i == 1:
            trg = arr
    dens_sl = cp.random.randn(3, ns)
    dens_dl = cp.random.randn(3, ns)
    norms = cp.random.randn(3, ns)
    dens = cp.vstack((dens_sl, dens_dl))  # stack densities for ewald call

    options = parkipy.ewald.EwaldOptions(
        periodicity=1,
        box=box,
        tolerance=args.tolerance,
        execution_space=args.device,
        cell_size=args.cell_size,
        return_walltime=True,
        p2g_threads=args.threads,
        g2p_threads=args.threads,
        p2p_threads_x=args.threads,
    )
    if am is not np:
        am.cuda.Device().synchronize()
    start = time.perf_counter()
    pot, walltimes = parkipy.ewald.stokes_comb(
        trg,
        src,
        dens,
        norms,
        options,
    )
    if am is not np:
        am.cuda.Device().synchronize()
    times = {
        "p2p": walltimes.time_p2p,
        "p2g": walltimes.time_p2g,
        "fft": walltimes.time_fft,
        "cnv": walltimes.time_cnv,
        "ifft": walltimes.time_ifft,
        "g2p": walltimes.time_g2p,
        "total": {"tot": time.perf_counter() - start},
    }

    # Compute and store runtimes in dict
    return pot, times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate timings for different levels of "
        "non-uniform particle distributions. "
        "Results are saved to `distributions_timing_result_up{args.up}"
        "_clsz{args.cell_size}_tol{args.tolerance}_dev{args.device.upper()}"
        "_arch{arch}_v{format_version}."
    )
    parser.add_argument(
        "--up",
        dest="up",
        type=int,
        default=1,
        help="Set the upsampeling parameter (default: 1)",
    )
    parser.add_argument(
        "--cell_size",
        dest="cell_size",
        type=int,
        default=224,
        help="Set the max src per cell (default: 224)",
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
        "-o",
        "--output-dir",
        default="analysis/ewald/data",
        help="output directory for timing results (default: analysis/ewald/data)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Spectral Ewald tolerance",
    )
    parser.add_argument("--nt", nargs="+", type=int, default=[4_000_000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--threads",
        dest="threads_list",
        nargs="+",
        type=int,
        default=[32, 64, 128, 256],
    )
    parser.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args()
    if min(args.nt) < 1 or min(args.threads_list) < 1 or args.repeats < 2:
        parser.error(
            "Particle and thread counts must be positive, with at least two repeats."
        )
    main(args)
    exit()
