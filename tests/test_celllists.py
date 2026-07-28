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


class TestNearestNeighbors:
    N = 5
    box = [1, 1, 1]
    cutoff = 0.1

    def _dense_grid_and_tiled_queries(self):
        """Dataset fits snugly within the box (to test edge cases);
        queries tile the whole box."""
        dataset = (
            np.stack(
                np.meshgrid(
                    *[
                        np.linspace(
                            2 * self.cutoff, self.box[i] - 2 * self.cutoff, self.N
                        )
                        for i in range(3)
                    ]
                ),
                axis=-1,
            )
            .reshape(-1, 3)
            .T
        )
        queries = (
            np.stack(
                np.meshgrid(
                    *[
                        np.linspace(0, self.box[i], self.N, endpoint=False)
                        for i in range(3)
                    ]
                ),
                axis=-1,
            )
            .reshape(-1, 3)
            .T
        )
        return dataset, queries

    def _axis_pair_dataset_and_queries(self):
        """
        For each axis, self.N dataset/query pairs isolating the periodic
        wrap on just that axis: dataset points sit within `cutoff` of
        the low edge on axis i (other coordinates fixed at 0.5); their
        matching queries sit within `cutoff` of the high edge on that
        same axis. Unwrapped, each pair is ~1 apart (no match); wrapped
        on that axis, it's ~2*eps apart (a match). This makes a given
        `periodicity` value verifiable per-axis: pairs on axis i should
        match iff i < periodicity.

        Returns dataset, queries as (3, n) arrays, n = 3 * self.N.
        """
        eps = np.linspace(self.cutoff / 8, self.cutoff * 7 / 8, self.N)

        n = 3 * self.N
        dataset = np.full((3, n), 0.5)
        queries = np.full((3, n), 0.5)

        col = 0
        for axis in range(3):
            for e in eps:
                dataset[axis, col] = e
                queries[axis, col] = 1 - e
                col += 1

        return dataset, queries

    def _reference_nearest(self, dataset, queries, periodicity):
        """O(n^2) reference. Minimum-image convention is applied
        only to the first `periodicity` axes."""
        box_arr = np.array(self.box)
        distances = np.full(queries.shape[-1], fill_value=np.inf)
        indices = np.full(queries.shape[-1], fill_value=-1, dtype=np.int32)
        for qi in range(queries.shape[-1]):
            for xi in range(dataset.shape[-1]):
                dr = queries[:, qi] - dataset[:, xi]
                dr[:periodicity] -= (
                    np.round(dr[:periodicity] / box_arr[:periodicity])
                    * box_arr[:periodicity]
                )
                r = np.linalg.norm(dr)
                if r < self.cutoff and r < distances[qi]:
                    distances[qi] = r
                    indices[qi] = xi
        return distances, indices

    def _check_nearest(self, dataset, queries, periodicity):
        distances, indices = self._reference_nearest(dataset, queries, periodicity)

        kwargs = dict(execution_space="CPU")
        if periodicity:
            kwargs["periodicity"] = periodicity
        d_list = parkipy.CellList(dataset, self.cutoff, self.box, **kwargs)
        q_list = parkipy.CellList(queries, self.cutoff, self.box, **kwargs)

        dist, indx = d_list.nearest(q_list)

        np.testing.assert_allclose(
            desired=distances, actual=dist, err_msg="distances are incorrect :("
        )
        np.testing.assert_equal(
            desired=indices, actual=indx, err_msg="indices are incorrect :("
        )

    def test_free_space(self):
        """
        Test the nearest neighbors to points within a cell-list.

        If no neighbor is found within the cutoff, ensure that
        the returned distance is None and the returned index is None.
        """
        dataset, queries = self._dense_grid_and_tiled_queries()
        self._check_nearest(dataset, queries, periodicity=0)

    def test_periodic_1(self):
        """Only the x-axis wraps; only the axis-0 pair should be
        found — axis-1 and axis-2 pairs must NOT match."""
        dataset, queries = self._axis_pair_dataset_and_queries()
        self._check_nearest(dataset, queries, periodicity=1)

    def test_periodic_2(self):
        """x and y wrap; axis-0 and axis-1 pairs should match,
        axis-2 must NOT."""
        dataset, queries = self._axis_pair_dataset_and_queries()
        self._check_nearest(dataset, queries, periodicity=2)

    def test_periodic_3(self):
        """All three axes wrap; all three pairs should match."""
        dataset, queries = self._axis_pair_dataset_and_queries()
        self._check_nearest(dataset, queries, periodicity=3)
