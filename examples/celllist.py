#!python3

"""
Given two particle distributions x,y,
our goal is to compute `u = 1 / (x-y) if dist(x-y) < cutoff`
else u = 0`.

It is easy to do so in `O(n^2)` time:
    ```
    u = cp.zeros(x.shape[1])
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            dist_sq = cp.sum((x[:, i] - y[:, j]) ** 2)
            if dist_sq == 0:
                continue
            dist = cp.sqrt(dist_sq)
            if dist < cutoff:
                u[i] += 1 / dist
    return u
    ```

Our algorithm below does so in O(n) time
using the `parkipy.CellList` data structure
"""
import parkipy


def main(args):
    am = parkipy.utils.get_array_module(args.device)
    # set computational box and cell list radius
    box = [1, 1, 1]
    cutoff = 0.1

    # generate particles
    Nx = 312
    Ny = 773
    x = am.random.rand(3, Nx) * am.array(box).reshape(3, 1)
    y = am.random.rand(3, Ny) * am.array(box).reshape(3, 1)
    u = am.zeros(x.shape[1])

    # build cell lists
    x_list = parkipy.CellList(x, cutoff, box, execution_space=args.device)
    y_list = parkipy.CellList(y, cutoff, box, execution_space=args.device)

    # loop over nonempty x-cells
    for cell_ne in range(x_list.num_nonempty_cells):
        off_x = cell_ne * x_list.cell_size
        cell = x_list.nonempty_cells[cell_ne]

        # loop over x-particles within a cell
        for ii in range(off_x, off_x + x_list.cell_size):
            i = x_list.particle_index[ii]
            if i == -1:
                continue

            # loop over y-neighbors
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
                    dist_sq = am.sum((x[:, i] - y[:, j]) ** 2)
                    if dist_sq == 0:
                        continue
                    dist = am.sqrt(dist_sq)
                    if dist < cutoff:
                        u[i] += 1 / dist


if __name__ == "__main__":
    main()
