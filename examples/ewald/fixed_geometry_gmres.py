"""Reuse a fixed-geometry Ewald plan in a manufactured second-kind GMRES solve.

Run from the repository root:
    python examples/ewald/fixed_geometry_gmres.py --device Serial -N 64
    python examples/ewald/fixed_geometry_gmres.py --device GPU -N 512

This demonstrates operator reuse; it is not a physical boundary-integral
formulation and does not include singular quadrature corrections.
"""

import argparse
import time
import numpy as np
import pykokkos as pk
from parkipy import ewald
from parkipy.utils import get_array_module, get_execution_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="Serial")
    parser.add_argument("-N", type=int, default=64)
    args = parser.parse_args()
    if args.N < 1:
        parser.error("N must be positive")
    space = get_execution_space(args.device)
    am = get_array_module(space)
    if pk.is_host_execution_space(space):
        from scipy.sparse.linalg import LinearOperator, gmres
    else:
        from cupyx.scipy.sparse.linalg import LinearOperator, gmres

    rng = np.random.default_rng(17)
    points = am.asarray(rng.uniform(0.1, 0.9, size=(3, args.N)))
    normals = am.asarray(rng.normal(size=points.shape))
    normals /= am.linalg.norm(normals, axis=0)
    options = ewald.EwaldOptions(
        box=[1.0, 1.0, 1.0],
        periodicity=1,
        tolerance=1e-5,
        execution_space=space,
        cell_size=16,
    )
    plan = ewald.prepare_stokes_comb(points, points, normals, options)
    density = am.empty((6, args.N), dtype=points.dtype)
    potential = am.empty_like(points)
    scale = 1e-3 / args.N

    def matvec(vector):
        values = vector.reshape(3, args.N)
        density[:3] = values
        density[3:] = 0.1 * values
        plan.apply(density, out=potential)
        # Return independent storage: GMRES retains vectors across matvecs.
        return vector + scale * potential.reshape(-1)

    operator = LinearOperator(
        (3 * args.N, 3 * args.N), matvec=matvec, dtype=points.dtype
    )
    exact = am.asarray(rng.normal(size=3 * args.N))
    rhs = operator @ exact  # Also warms up kernels and FFT plans.

    # Check the prepared Ewald evaluation against the existing one-shot API.
    reference = ewald.stokes_comb(points, points, density, normals, options)
    am.testing.assert_allclose(potential, reference, rtol=1e-10, atol=1e-10)

    def synchronize():
        if not pk.is_host_execution_space(space):
            am.cuda.get_current_stream().synchronize()

    synchronize()
    start = time.perf_counter()
    solution, info = gmres(operator, rhs, rtol=1e-8, atol=0.0, restart=20, maxiter=100)
    synchronize()
    elapsed = time.perf_counter() - start
    residual = float(am.linalg.norm(operator @ solution - rhs) / am.linalg.norm(rhs))
    error = float(am.linalg.norm(solution - exact) / am.linalg.norm(exact))
    print(f"GMRES info={info}, solve={elapsed * 1000:.2f} ms")
    print(f"Relative residual={residual:.3e}, solution error={error:.3e}")
    if info != 0 or residual > 1e-7:
        raise RuntimeError("GMRES failed to converge")


if __name__ == "__main__":
    main()
