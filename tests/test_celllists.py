import parkipy
import numpy as np


def test_cell_capacity_pads_each_cell_separately():
    # The first and last of eight cells contain 3 and 35 particles.
    particles = np.repeat([[0.25, 0.75]] * 3, [3, 35], axis=1)
    forces = np.arange(2 * particles.shape[1], dtype=particles.dtype).reshape(2, 1, -1)
    common = dict(execution_space="CPU", forces=forces, padding_factor=32)
    cells = parkipy.CellList(particles, 0.5, [1, 1, 1], **common)
    np.testing.assert_array_equal(cells.cell_offsets, [0, 32, 96])
    np.testing.assert_array_equal(cells.nonempty_cells, [0, 7])
    np.testing.assert_array_equal(np.sort(cells.particle_index[:3]), np.arange(3))
    np.testing.assert_array_equal(
        np.sort(cells.particle_index[32:67]), np.arange(3, 38)
    )
    valid = cells.particle_index >= 0
    np.testing.assert_array_equal(
        cells.particle_list[:, valid], particles[:, cells.particle_index[valid]]
    )
    np.testing.assert_array_equal(
        cells.force_list[..., valid], forces[..., cells.particle_index[valid]]
    )
    assert np.all(cells.particle_list[:, ~valid] == -1)
    assert np.all(cells.force_list[..., ~valid] == 0)

    with_empty = parkipy.CellList(
        particles, 0.5, [1, 1, 1], skip_empty_cells=False, **common
    )
    np.testing.assert_array_equal(with_empty.cell_offsets, [0, *([32] * 7), 96])
    team_cells, cell_teams = with_empty._team_chunks(32)
    np.testing.assert_array_equal(team_cells, [0, 7, 7])
    np.testing.assert_array_equal(cell_teams, [0, *([1] * 7), 3])

    fixed = parkipy.CellList(
        particles, 0.5, [1, 1, 1], variable_cell_size=False, **common
    )
    np.testing.assert_array_equal(fixed.cell_offsets, [0, 64, 128])


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
        off_x, end_x = map(int, x_list.cell_offsets[cell_ne : cell_ne + 2])
        cell = x_list.nonempty_cells[cell_ne]

        # loop over x-particles within a cell
        for ii in range(off_x, end_x):
            i = x_list.particle_index[ii]
            if i == -1:
                continue

            # loop over y-neighbors
            for k in range(27):
                neighbor = y_list.nonempty_neighbors[cell, k]
                if neighbor == -1:
                    continue
                off_y, end_y = map(int, y_list.cell_offsets[neighbor : neighbor + 2])

                # loop over y-particles within a neighbor
                for jj in range(off_y, end_y):
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
        off_x, end_x = map(int, x_list.cell_offsets[cell_ne : cell_ne + 2])
        cell = x_list.nonempty_cells[cell_ne]

        for ii in range(off_x, end_x):
            i = x_list.particle_index[ii]
            if i == -1:
                continue

            for k in range(27):
                neighbor = y_list.nonempty_neighbors[cell, k]
                if neighbor == -1:
                    continue
                off_y, end_y = map(int, y_list.cell_offsets[neighbor : neighbor + 2])

                for jj in range(off_y, end_y):
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
        off_x, end_x = map(int, x_list.cell_offsets[cell_ne : cell_ne + 2])
        cell = x_list.nonempty_cells[cell_ne]

        for ii in range(off_x, end_x):
            i = x_list.particle_index[ii]
            if i == -1:
                continue

            for kk in range(27):
                neighbor = y_list.nonempty_neighbors[cell, kk]
                if neighbor == -1:
                    continue
                off_y, end_y = map(int, y_list.cell_offsets[neighbor : neighbor + 2])

                for jj in range(off_y, end_y):
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
