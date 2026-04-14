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
    x_list = parkipy.CellList(x, cutoff, box, execution_space="CPU")
    y_list = parkipy.CellList(y, cutoff, box, forces=q, execution_space="CPU")

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


def reference_periodic(x, y, q, cutoff, box):
    """O(n^2) reference with minimum-image convention."""
    box_arr = np.array(box)
    u = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            dr = x[:, i] - y[:, j]
            dr = dr - np.round(dr / box_arr) * box_arr
            dist_sq = np.sum(dr**2)
            if dist_sq == 0:
                continue
            dist = np.sqrt(dist_sq)
            if dist < cutoff:
                u[i] += 1 / dist * q[j]
    return u


def cell_list_periodic(x, y, q, cutoff, box):
    """Cell-list solution with periodic boundary conditions (periodicity=3)."""
    box_arr = np.array(box)
    u = np.zeros(x.shape[1])
    x_list = parkipy.CellList(x, cutoff, box, execution_space="CPU", periodicity=3)
    y_list = parkipy.CellList(
        y, cutoff, box, forces=q, execution_space="CPU", periodicity=3
    )

    for cell_ne in range(x_list.num_nonempty_cells):
        off_x = cell_ne * x_list.cell_size
        cell = x_list.nonempty_cells[cell_ne]

        for ii in range(off_x, off_x + x_list.cell_size):
            i = x_list.particle_index[ii]
            if i == -1:
                continue

            for k in range(27):
                neighbor = y_list.nonempty_neighbors[cell, k]
                if neighbor == -1:
                    continue
                off_y = neighbor * y_list.cell_size

                for jj in range(off_y, off_y + y_list.cell_size):
                    j = y_list.particle_index[jj]
                    if j == -1:
                        continue
                    dr = x[:, i] - y[:, j]
                    dr = dr - np.round(dr / box_arr) * box_arr
                    dist_sq = np.sum(dr**2)
                    if dist_sq == 0:
                        continue
                    dist = np.sqrt(dist_sq)
                    if dist < cutoff:
                        u[i] += 1 / dist * q[j]
    return u


def test_celllist_periodic(Nx=773, Ny=312, box=[1, 1, 1], cutoff=0.1):
    """
    Test periodic particle-to-particle interactions using cell-lists.

    Both the reference and cell-list solutions use the minimum-image
    convention so that each pair is evaluated through its nearest
    periodic image.  Correctness requires that the periodic neighbor
    lookup (periodicity=3) produces the same pairs as the brute-force
    double loop.
    """
    x = np.random.rand(3, Nx)
    y = np.random.rand(3, Ny)
    q = np.random.rand(Ny)

    u_ref = reference_periodic(x, y, q, cutoff, box)
    u_cl = cell_list_periodic(x, y, q, cutoff, box)

    np.testing.assert_allclose(u_cl, u_ref, rtol=1e-13, atol=1e-26)


def reference_batched(x, y, q, cutoff):
    """
    O(n^2) reference for batched forces.

    Parameters
    ----------
    x : ndarray, shape (3, Nx)
    y : ndarray, shape (3, Ny)
    q : ndarray, shape (r, k, Ny)  — batched force array
    cutoff : float

    Returns
    -------
    u : ndarray, shape (r, k, Nx)
    """
    r, k, Ny = q.shape
    Nx = x.shape[1]
    u = np.zeros((r, k, Nx))
    for i in range(Nx):
        for j in range(Ny):
            dist_sq = np.sum((x[:, i] - y[:, j]) ** 2)
            if dist_sq == 0:
                continue
            dist = np.sqrt(dist_sq)
            if dist < cutoff:
                u[:, :, i] += 1 / dist * q[:, :, j]
    return u


def cell_list_batched(x, y, q, cutoff, box):
    """
    Cell-list solution for batched forces of shape (r, k, Ny).

    Returns u of shape (r, k, Nx).
    """
    r, k, Ny = q.shape
    Nx = x.shape[1]
    u = np.zeros((r, k, Nx))

    x_list = parkipy.CellList(x, cutoff, box, execution_space="CPU")
    y_list = parkipy.CellList(y, cutoff, box, forces=q, execution_space="CPU")

    for cell_ne in range(x_list.num_nonempty_cells):
        off_x = cell_ne * x_list.cell_size
        cell = x_list.nonempty_cells[cell_ne]

        for ii in range(off_x, off_x + x_list.cell_size):
            i = x_list.particle_index[ii]
            if i == -1:
                continue

            for kk in range(27):
                neighbor = y_list.nonempty_neighbors[cell, kk]
                if neighbor == -1:
                    continue
                off_y = neighbor * y_list.cell_size

                for jj in range(off_y, off_y + y_list.cell_size):
                    j = y_list.particle_index[jj]
                    if j == -1:
                        continue
                    dist_sq = np.sum((x[:, i] - y[:, j]) ** 2)
                    if dist_sq == 0:
                        continue
                    dist = np.sqrt(dist_sq)
                    if dist < cutoff:
                        # force_list has shape (r, k, list_len)
                        u[:, :, i] += 1 / dist * y_list.force_list[:, :, jj]
    return u


def test_celllist_batched(Nx=773, Ny=312, box=[1, 1, 1], cutoff=0.1, r=3, k=2):
    """
    Test cell-list interactions with batched forces of shape (r, k, N).

    The reference solution is an O(n^2) double loop that accumulates
    all r*k force channels simultaneously.  The cell-list solution
    must produce an identical result within floating-point tolerance.
    """
    x = np.random.rand(3, Nx)
    y = np.random.rand(3, Ny)
    q = np.random.rand(r, k, Ny)  # batched forces

    u_ref = reference_batched(x, y, q, cutoff)
    u_cl = cell_list_batched(x, y, q, cutoff, box)

    np.testing.assert_allclose(u_cl, u_ref, rtol=1e-13, atol=1e-26)
