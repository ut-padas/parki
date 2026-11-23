"""Compare one-periodic global and adaptive Stokes Ewald on one GPU."""

import argparse
import json
import time

import cupy as cp
import numpy as np

from parkipy import ewald


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[10000, 30000, 100000, 300000]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    rng = np.random.default_rng(2026)
    print(
        json.dumps(
            {
                "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                "sizes": args.sizes,
                "repeats": args.repeats,
                "tolerance": args.tolerance,
            }
        ),
        flush=True,
    )
    for n in args.sizes:
        src = cp.asarray(rng.random((3, n)))
        trg = cp.asarray(rng.random((3, n)))
        density = cp.asarray(rng.standard_normal((6, n)))
        normals = cp.asarray(rng.standard_normal((3, n)))
        outputs, times = {}, {}
        for adaptive in (False, True):
            name = "adaptive" if adaptive else "global"
            options = ewald.EwaldOptions(
                box=[1.0] * 3,
                periodicity=1,
                tolerance=args.tolerance,
                execution_space="CUDA",
                cell_size=224,
                fft_type="C2C",
                adaptive_upsampling=adaptive,
            )

            def run():
                return ewald.stokes_comb(trg, src, density, normals, options)

            run()
            cp.cuda.Device().synchronize()
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                result = run()
                cp.cuda.Device().synchronize()
                samples.append(time.perf_counter() - start)
            outputs[name] = result
            times[name] = samples
        delta = cp.linalg.norm(
            outputs["adaptive"] - outputs["global"]
        ) / cp.linalg.norm(outputs["global"])
        print(
            json.dumps(
                {
                    "n": n,
                    "seconds": times,
                    "speedup_median": float(
                        np.median(times["global"]) / np.median(times["adaptive"])
                    ),
                    "relative_l2_difference": float(delta),
                }
            ),
            flush=True,
        )
        if not np.isfinite(float(delta)) or float(delta) > 5 * args.tolerance:
            raise RuntimeError("Adaptive/global difference exceeds study threshold")
        del src, trg, density, normals, outputs
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
