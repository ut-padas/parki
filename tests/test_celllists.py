import parkipy
import numpy as np


def reference(x, y, q, cutoff):
    u = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            dist_sq = np.sum((x[:, i] - y[:, j]) ** 2)
            if dist_sq == 0:
                continue
            dist = np.sqrt(dist_sq)
            if dist < cutoff:
                u[i] += 1 / dist * q[j]
    return u


def cell_list(x, y, q, cutoff, box):
    u = np.zeros(x.shape[1])
    x_list = parkipy.CellList(x, cutoff, box)
    y_list = parkipy.CellList(y, cutoff, box, forces=q)

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
                neighbor = y_list.nonempty_neighbors[cell, k]
                if neighbor == -1:
                    continue
                off_y = neighbor * y_list.cell_size

                # loop over y-particles within a neighbor
                for jj in range(off_y, off_y + y_list.cell_size):
                    j = y_list.particle_index[jj]
                    if j == -1:
                        continue
                    dist_sq = np.sum((x[:, i] - y[:, j]) ** 2)
                    if dist_sq == 0:
                        continue
                    dist = np.sqrt(dist_sq)
                    if dist < cutoff:
                        u[i] += 1 / dist * q[j]
    return u


def test_celllist(Nx=773, Ny=312, box=[1, 1, 1], cutoff=0.1):
    """
    Test particle-to-particle interactions
    within a box using cell-lists.

    The reference solution will be an O(n^2)
    double loop though the particles,
    while the tested solution will use
    cell-lists for an O(n) algorithm.
    """
    x = np.random.rand(3, Nx)
    y = np.random.rand(3, Ny)
    u = np.random.rand(Ny)

    u_ref = reference(x, y, u, cutoff)
    u_cl = cell_list(x, y, u, cutoff, box)

    np.testing.assert_allclose(u_cl, u_ref, rtol=1e-13, atol=1e-26)
