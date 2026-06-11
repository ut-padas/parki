"""
Example of batched forces.

Forces are supplied as a 3-D array of shape `(r, k, N)`, where
`r` is the batch size and `k` is the per-force vector dimension.
The cell list reshuffles every slice `forces[b]` independently,
and the interaction loop accumulates a weighted sum for each
batch member separately.
"""


def main_batched_forces(args):
    am = parkipy.utils.get_array_module(args.device)
    box = [1, 1, 1]
    cutoff = 0.1

    Nx = 312
    Ny = 773
    r = 4  # batch size
    k = 2  # force vector dimension per batch member

    x = am.random.rand(3, Nx) * am.array(box).reshape(3, 1)
    y = am.random.rand(3, Ny) * am.array(box).reshape(3, 1)
    # batched forces: shape (r, k, Ny)
    q = am.random.rand(r, k, Ny)
    # output: shape (r, k, Nx)
    u = am.zeros((r, k, Nx))

    x_list = parkipy.CellList(x, cutoff, box, execution_space=args.device)
    # Pass batched forces — shape (r, k, N) — directly to CellList
    y_list = parkipy.CellList(y, cutoff, box, forces=q, execution_space=args.device)

    for cell_ne in range(x_list.num_nonempty_cells):
        off_x = cell_ne * x_list.cell_size
        cell = x_list.nonempty_cells[cell_ne]

        for ii in range(off_x, off_x + x_list.cell_size):
            i = x_list.particle_index[ii]
            if i == -1:
                continue

            for kk in range(27):
                neighbor = y_list.nonempty_neighbors[cell, kk].get()
                if neighbor == -1:
                    continue
                off_y = neighbor * y_list.cell_size

                for jj in range(off_y, off_y + y_list.cell_size):
                    j = y_list.particle_index[jj]
                    if j == -1:
                        continue
                    dist_sq = am.sum((x[:, i] - y[:, j]) ** 2)
                    if dist_sq == 0:
                        continue
                    dist = am.sqrt(dist_sq)
                    if dist < cutoff:
                        # y_list.force_list has shape (r, k, list_len)
                        u[:, :, i] += 1 / dist * y_list.force_list[:, :, jj]


if __name__ == "__main__":
    exit(main())
