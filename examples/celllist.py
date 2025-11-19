#!python3

"""
CellList object creation using a Cuda execution space.
"""
import cupy as cp
import parkipy


def main():
    # set computational box and cell list radius
    box = [1, 1, 1]
    cutoff = 0.1

    # generate particles and forces
    n = 100000
    prt = cp.random.rand(3, n) * cp.array(box).reshape(3, 1)
    force1d = cp.random.rand(1, n)
    force2d = cp.random.rand(2, n)
    force3d = cp.random.rand(3, n)

    # get cell list object
    cell_list = parkipy.CellList(
        prt,
        cutoff,
        box,
        execution_space="Cuda",
        forces=(force1d, force2d, force3d),
    )


if __name__ == "__main__":
    main()
