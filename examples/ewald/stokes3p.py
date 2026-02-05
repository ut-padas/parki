#!python3

"""
Ewald summation for the combined stokes single and double
layer potential with one periodic direction.
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
    dens_sl = am.random.randn(3, ns)
    dens_dl = am.random.randn(3, ns)
    norms = am.random.randn(3, ns)
    dens = am.vstack((dens_sl, dens_dl))  # stack densities for ewald call

    # call the PDE kernel
    pot = parkipy.ewald.stokes_comb(
        trg, src, dens, norms, 3, box, tol, args.device, cell_size=512
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined Stokes single and double layer potential with 1 periodic direction."
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
