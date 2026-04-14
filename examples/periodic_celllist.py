"""
Example of periodic cell lists.

Same Coulomb-like interaction as `celllist.py`, but the computational
box now uses fully periodic boundary conditions (`periodicity=3`).
Distances are computed with the minimum-image convention so that
each pair is counted through the nearest periodic image.
"""


def main(args):
    am = parkipy.utils.get_array_module(args.device)
    box = [1, 1, 1]
    cutoff = 0.1

    Nx = 312
    Ny = 773
    x = am.random.rand(3, Nx) * am.array(box).reshape(3, 1)
    y = am.random.rand(3, Ny) * am.array(box).reshape(3, 1)
    u = am.zeros(x.shape[1])

    # Build periodic cell lists (periodicity=3 → wrap all three axes)
    x_list = parkipy.CellList(
        x, cutoff, box, execution_space=args.device, periodicity=3
    )
    y_list = parkipy.CellList(
        y, cutoff, box, execution_space=args.device, periodicity=3
    )

    box_arr = am.array(box)

    # loop over nonempty x-cells
    for cell_ne in range(x_list.num_nonempty_cells):
        off_x = cell_ne * x_list.cell_size
        cell = x_list.nonempty_cells[cell_ne]

        # loop over x-particles within a cell
        for ii in range(off_x, off_x + x_list.cell_size):
            i = x_list.particle_index[ii]
            if i == -1:
                continue

            # loop over y-neighbors (including periodic images)
            for k in range(27):
                neighbor = y_list.nonempty_neighbors[cell, k].get()
                if neighbor == -1:
                    continue
                off_y = neighbor * y_list.cell_size

                # loop over y-particles within a neighbor
                for jj in range(off_y, off_y + y_list.cell_size):
                    j = y_list.particle_index[jj]
                    if j == -1:
                        continue
                    # minimum-image convention
                    dr = x[:, i] - y[:, j]
                    dr = dr - am.round(dr / box_arr) * box_arr
                    dist_sq = am.sum(dr**2)
                    if dist_sq == 0:
                        continue
                    dist = am.sqrt(dist_sq)
                    if dist < cutoff:
                        u[i] += 1 / dist


if __name__ == "__main__":
    main()
