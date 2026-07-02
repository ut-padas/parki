#!python3

"""
Ewald summation for the combined stokes single and double
layer potential with all periodic directions.
"""

import argparse
import parkipy


def main(args):
    execution_space = parkipy.utils.get_execution_space(args.device)
    am = parkipy.utils.get_array_module(execution_space)

    # set spectral Ewald box and tolerance
    box = [1, 1, 1]
    tol = 1e-4

    # generate sources, targets, densities, and normals
    ns = nt = args.N
    trg = am.random.rand(3, nt) * am.array(box).reshape(3, 1)
    src = am.random.rand(3, ns) * am.array(box).reshape(3, 1)
    dens_sl = am.random.randn(3, ns)
    dens_dl = am.random.randn(3, ns)
    norms = am.random.randn(3, ns)
    dens = am.vstack((dens_sl, dens_dl))  # stack densities for ewald call

    # call the PDE kernel
    options = parkipy.ewald.EwaldOptions(
        periodicity=3,
        box=box,
        tolerance=tol,
        cell_size=224,
        execution_space=args.device,
        return_walltime=True,
    )
    __warmup__ = parkipy.ewald.stokes_comb(trg, src, dens, norms, options)
    pot, walltime = parkipy.ewald.stokes_comb(trg, src, dens, norms, options)
    print(walltime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined Stokes single and double layer potential with all periodic directions."
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="CUDA",
        help="Execution device.",
    )
    parser.add_argument(
        "-N",
        dest="N",
        type=int,
        default="100_000",
        help="Number of particles.",
    )
    args = parser.parse_args()
    exit(main(args))
