"""Compare adaptive and global Ewald errors across kernels and tolerances."""

import gc
import argparse
import json
import time
import numpy as np
from parkipy import ewald
from parkipy.utils import get_array_module


def relative(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts", nargs="+", type=int, default=[10000, 30000, 100000, 300000, 1000000]
    )
    parser.add_argument(
        "--tolerances", nargs="+", type=float, default=[1e-2, 1e-3, 1e-4, 1e-5]
    )
    parser.add_argument("--reference-tolerance", type=float, default=1e-7)
    parser.add_argument("--million-reference-tolerance", type=float, default=1e-6)
    parser.add_argument("--device", default="CUDA", help="CUDA or Serial execution")
    parser.add_argument(
        "--fft-type", choices=["R2C", "C2C"], nargs="+", default=["R2C", "C2C"]
    )
    parser.add_argument(
        "--periodicities", type=int, nargs="+", choices=[1, 2], default=[1, 2]
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=["stokes_comb", "stokes_sl", "laplace"],
        default=["stokes_comb", "stokes_sl", "laplace"],
    )
    parser.add_argument("--cell-size", type=int, default=224)
    args = parser.parse_args()
    am = get_array_module(args.device)
    gpu = am is not np
    rng = np.random.default_rng(2026)
    print(
        json.dumps(
            {
                "device": args.device,
                "counts": args.counts,
                "tolerances": args.tolerances,
                "reference_tolerance": args.reference_tolerance,
                "million_reference_tolerance": args.million_reference_tolerance,
                "dtype": "float64",
                "seed": 2026,
                "kernels": args.kernels,
                "periodicities": args.periodicities,
                "fft_type": args.fft_type,
                "cell_size": args.cell_size,
            }
        ),
        flush=True,
    )
    for kernel in args.kernels:
        for periodicity in args.periodicities:
            for fft_type in args.fft_type:
                for n in args.counts:
                    src = am.asarray(rng.random((3, n)))
                    trg = am.asarray(rng.random((3, n)))
                    density = am.asarray(rng.standard_normal((6, n)))
                    normals = am.asarray(rng.standard_normal((3, n)))

                    def solve(adaptive, tol):
                        opts = ewald.EwaldOptions(
                            box=[1.0] * 3,
                            periodicity=periodicity,
                            tolerance=tol,
                            execution_space=args.device,
                            cell_size=args.cell_size,
                            fft_type=fft_type,
                            adaptive_upsampling=adaptive,
                        )
                        if kernel == "stokes_comb":
                            result = ewald.stokes_comb(trg, src, density, normals, opts)
                        elif kernel == "stokes_sl":
                            result = ewald.stokes_sl(trg, src, density[:3], opts)
                        else:
                            charge = density[0] - density[0].mean()
                            result = ewald.laplace(trg, src, charge, opts)
                        if gpu:
                            am.cuda.Device().synchronize()
                        host = am.asnumpy(result) if gpu else result.copy()
                        del result
                        gc.collect()
                        if gpu:
                            am.fft.config.get_plan_cache().clear()
                            am.get_default_memory_pool().free_all_blocks()
                        return host

                    ref_tol = (
                        args.million_reference_tolerance
                        if n >= 1000000
                        else args.reference_tolerance
                    )
                    if ref_tol >= min(args.tolerances):
                        raise ValueError(
                            "reference tolerance must be tighter than every tested tolerance"
                        )
                    ref = solve(False, ref_tol)
                    check = None
                    if n == 10000:
                        check = relative(ref, solve(False, ref_tol / 10))
                    print(
                        json.dumps(
                            {
                                "n": n,
                                "kernel": kernel,
                                "periodicity": periodicity,
                                "fft_type": fft_type,
                                "reference_tolerance": ref_tol,
                                "reference_global_tenfold_tighter_difference": check,
                            }
                        ),
                        flush=True,
                    )
                    for tolerance in args.tolerances:
                        start = time.perf_counter()
                        glob = solve(False, tolerance)
                        adaptive = solve(True, tolerance)
                        err_global = relative(glob, ref)
                        err_adaptive = relative(adaptive, ref)
                        print(
                            json.dumps(
                                {
                                    "n": n,
                                    "kernel": kernel,
                                    "periodicity": periodicity,
                                    "fft_type": fft_type,
                                    "tolerance": tolerance,
                                    "reference_tolerance": ref_tol,
                                    "global_relative_l2_error": err_global,
                                    "adaptive_relative_l2_error": err_adaptive,
                                    "adaptive_over_global_error": (
                                        err_adaptive / err_global
                                        if err_global
                                        else None
                                    ),
                                    "adaptive_global_relative_l2_difference": relative(
                                        adaptive, glob
                                    ),
                                    "case_seconds": time.perf_counter() - start,
                                }
                            ),
                            flush=True,
                        )
                        del glob, adaptive
                    if n == 1000000:
                        check = solve(True, ref_tol / 10)
                        print(
                            json.dumps(
                                {
                                    "n": n,
                                    "kernel": kernel,
                                    "periodicity": periodicity,
                                    "fft_type": fft_type,
                                    "reference_crosscheck": "global reference versus tenfold tighter adaptive",
                                    "relative_l2_difference": relative(ref, check),
                                }
                            ),
                            flush=True,
                        )
                        del check
                    del src, trg, density, normals, ref
                    gc.collect()
                    if gpu:
                        am.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
