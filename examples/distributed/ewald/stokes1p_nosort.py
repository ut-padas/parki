"""
Ewald summation for the single layer stokes potential
**without** bucket sorting. That is, targets will always
lie on their initial rank. We allow for targets and sources
to be outside their partition by an amount called "fuzz factor".

>>> ibrun python examples/distributed/ewald/stokes1p_nosort.py
"""

import argparse
import parkipy
import cupy as cp
from mpi4py import MPI


def main(args):
    # MPI communicator
    comm = MPI.COMM_WORLD

    # get global box size
    box = [comm.size, 1, 1]

    # get local partition
    left = comm.rank
    right = comm.rank + 1

    # points are allowed to be fuzz factor amount outside the partition
    fuzz_factor = 0.0 * (right - left)

    # generate points
    rng = cp.random.default_rng()
    targets = cp.asarray(
        [
            rng.uniform(low=left - fuzz_factor, high=right + fuzz_factor, size=args.nt),
            rng.uniform(low=0, high=box[1], size=args.nt),
            rng.uniform(low=0, high=box[2], size=args.nt),
        ]
    )
    sources = cp.asarray(
        [
            rng.uniform(low=left - fuzz_factor, high=right + fuzz_factor, size=args.ns),
            rng.uniform(low=0, high=box[1], size=args.ns),
            rng.uniform(low=0, high=box[2], size=args.ns),
        ]
    )
    densities = cp.asarray(
        [
            rng.uniform(size=args.ns),
            rng.uniform(size=args.ns),
            rng.uniform(size=args.ns),
        ]
    )

    # call the distributed Ewald kernel
    options = parkipy.distributed.ewald.DistributedEwaldOptions(
        box=box,
        periodicity=1,
        tolerance=args.tolerance,
        execution_space="GPU",
        rc=args.rc,
        scatter=True,
    )
    pot, trg = parkipy.distributed.ewald.stokes_sl(
        targets % cp.asarray(box)[:, None],
        sources % cp.asarray(box)[:, None],
        densities,
        options,
    )

    cp.testing.assert_array_equal(trg, targets)

    """
    print(f"rank {comm.rank} targets  ", targets)
    print(f"rank {comm.rank} sources  ", sources)
    print(f"rank {comm.rank} densities", densities)
    print(f"rank {comm.rank} potential", pot)
    """

    # create buffers for single device arrays
    pot_mpi = cp.hstack([pot, pot])
    targets_mpi = cp.hstack([targets, targets])
    sources_mpi = cp.hstack([sources, sources])
    densities_mpi = cp.hstack([densities, densities])

    # gather points on GPU to test against single-device computation
    parkipy.distributed.gather_points(comm, "gpu", pot_mpi.T, args.nt)
    parkipy.distributed.gather_points(comm, "gpu", targets_mpi.T, args.nt)
    parkipy.distributed.gather_points(comm, "gpu", sources_mpi.T, args.ns)
    parkipy.distributed.gather_points(comm, "gpu", densities_mpi.T, args.ns)

    # gather potential on rank 0 and test against a serial computation
    if comm.rank == 0:
        """
        print(f"gathered targets  ", targets_mpi)
        print(f"gathered sources  ", sources_mpi)
        print(f"gathered densities", densities_mpi)
        print(f"gathered potential", pot_mpi)
        """
        options = parkipy.ewald.EwaldOptions(
            box=box,
            periodicity=1,
            tolerance=args.tolerance,
            execution_space="GPU",
            rc=args.rc,
        )
        pot_ref = parkipy.ewald.stokes_sl(
            targets_mpi, sources_mpi, densities_mpi, options
        )

        cp.testing.assert_array_equal(pot_mpi, pot_ref)
        print(f"Single- & Multi-rank results with {fuzz_factor} fuzz factor agree!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stokes single layer potential with 1 periodic direction."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Spectral Ewald tolerance (default: 1e-4)",
    )
    parser.add_argument(
        "--rc",
        type=float,
        default=0.2,
        help="Spectral Ewald cutoff radius",
    )
    parser.add_argument(
        "--nt",
        type=int,
        default=100_000,
        help="Number of target points",
    )
    parser.add_argument(
        "--ns",
        type=int,
        default=100_000,
        help="Number of source points",
    )
    args = parser.parse_args()
    exit(main(args))
