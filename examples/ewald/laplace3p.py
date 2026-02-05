#!python3

"""
Ewald summation for the Laplace potential using a Cuda execution space.
"""

import argparse
import parkipy


def main(args):
    execution_space = parkipy.utils.get_execution_space(args.device)
    am = parkipy.utils.get_array_module(execution_space)

    # set spectral Ewald box and tolerance
    box = [1, 1, 1]
    tol = 1e-7

    # generate sources, targets, densities, and normals
    nt = 100000
    ns = 100000
    trg = am.random.rand(3, nt) * am.array(box).reshape(3, 1)
    src = am.random.rand(3, ns) * am.array(box).reshape(3, 1)
    # generate nutral charges
    charges = am.random.randn(ns)
    charges -= am.mean(charges)

    # call the PDE kernel
    options = parkipy.ewald.EwaldOptions(
        periodicity=3,
        box=box,
        tolerance=tol,
        execution_space=execution_space,
        cell_size=512,
        return_walltime=True,
    )
    pot, walltime = parkipy.ewald.laplace(trg, src, charges, options)
    print(walltime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Laplace layer potential in a periodic box."
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="CUDA",
        help="Execution device.",
    )
    args = parser.parse_args()
    exit(main(args))
