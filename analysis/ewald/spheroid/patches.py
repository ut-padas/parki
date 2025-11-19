import numpy as np
import scipy.interpolate as sinterp

from numba import njit, prange

from time import time

n_patch = 6


def patches_ellipsoid_given_uv(ps, u, v, a, b, c):
    ps[0, :, 0] = a * np.sin(u) * np.cos(v)
    ps[0, :, 1] = b * np.sin(u) * np.sin(v)
    ps[0, :, 2] = c * np.cos(u)

    ps[1, :, 0] = -a * np.sin(u) * np.cos(v)
    ps[1, :, 1] = -b * np.sin(u) * np.sin(v)
    ps[1, :, 2] = c * np.cos(u)

    ps[2, :, 0] = a * np.sin(u) * np.sin(v)
    ps[2, :, 1] = -b * np.sin(u) * np.cos(v)
    ps[2, :, 2] = c * np.cos(u)

    ps[3, :, 0] = -a * np.sin(u) * np.sin(v)
    ps[3, :, 1] = b * np.sin(u) * np.cos(v)
    ps[3, :, 2] = c * np.cos(u)

    ps[4, :, 0] = a * np.sin(u) * np.cos(v)
    ps[4, :, 1] = -b * np.cos(u)
    ps[4, :, 2] = c * np.sin(u) * np.sin(v)

    ps[5, :, 0] = a * np.sin(u) * np.cos(v)
    ps[5, :, 1] = b * np.cos(u)
    ps[5, :, 2] = -c * np.sin(u) * np.sin(v)


def create_uv_patch(m, n):
    # uv_patch ((m - 1)*(n - 1), 2) matrix

    uv_patch = np.zeros(((m - 1) * (n - 1), 2))
    us = np.arange(1, m) * np.pi / m
    vs = np.arange(1, n) * np.pi / n
    uv_patch[:, 0] = np.tile(us, n - 1)
    uv_patch[:, 1] = np.tile(vs.reshape(-1, 1), (1, (m - 1))).ravel()

    return uv_patch


def create_patches_ellipsoid(m, n, a, b, c):
    # x_patches (n_patch, (m - 1)*(n - 1), 3) matrix

    uv_patch = create_uv_patch(m, n)

    x_patches = np.zeros((n_patch, (m - 1) * (n - 1), 3))
    patches_ellipsoid_given_uv(x_patches, uv_patch[:, 0], uv_patch[:, 1], a, b, c)

    return x_patches, uv_patch


def blend_patches(f_patches, m, n):
    # f_patches (n_patch, (m - 1)*(n - 1), k) matrix

    x_patches, _ = create_patches_ellipsoid(m, n, a=1, b=1, c=1)
    x = x_patches.reshape(-1, 3)
    nn = x.shape[0]

    pou_x_patches = np.zeros((n_patch, nn))
    for i in range(n_patch):
        pou_x_patches[i] = create_pou_given_points(x, i)
    pou_x_patches /= np.sum(pou_x_patches, axis=0)

    f_blended = np.zeros((nn, 3))
    for i in range(n_patch):
        uv_x = create_uv_given_points(x, i)
        idi = np.zeros(nn, dtype=np.bool_)
        idi[i * (m - 1) * (n - 1) : (i + 1) * (m - 1) * (n - 1)] = True
        f_blended[idi] += f_patches[i] * pou_x_patches[i, idi].reshape(-1, 1)
        for ik in range(3):
            f_blended[~idi, ik] += (
                interpolate_uv(
                    f_patches[i, :, ik].reshape(n - 1, m - 1), m, n, uv_x[~idi]
                )
                * pou_x_patches[i, ~idi]
            )

    return f_blended


def interpolate_patches_on_given_points(f_patches, x, m, n):
    # f_patches (n_patch, (m - 1)*(n - 1), k) matrix
    # x (nn, 3) matrix

    k = f_patches.shape[2]
    nn = x.shape[0]

    pou_x_patches = np.zeros((n_patch, nn))
    for i in range(n_patch):
        pou_x_patches[i] = create_pou_given_points(x, i)
    pou_x_patches /= np.sum(pou_x_patches, axis=0)

    f_interp = np.zeros((nn, k))
    for i in range(n_patch):
        uv_x = create_uv_given_points(x, i)
        for ik in range(k):
            f_interp[:, ik] += (
                interpolate_uv(f_patches[i, :, ik].reshape(n - 1, m - 1), m, n, uv_x)
                * pou_x_patches[i]
            )

    return f_interp


def interpolate_patches_on_given_points_bicubic_spline(f_patches, x, m, n):
    # f_patches (n_patch, (m - 1)*(n - 1), k) matrix
    # x (nn, 3) matrix

    k = f_patches.shape[2]
    nn = x.shape[0]

    pou_x_patches = np.zeros((n_patch, nn))
    for i in range(n_patch):
        pou_x_patches[i] = create_pou_given_points(x, i)
    pou_x_patches /= np.sum(pou_x_patches, axis=0)

    f_temp = np.zeros((n_patch, m - 1, n - 1, k))
    for ip in range(n_patch):
        for ik in range(k):
            f_temp[ip, :, :, ik] = f_patches[ip, :, ik].reshape(n - 1, m - 1).T

    f_interp = np.zeros((nn, k))
    for i in range(n_patch):
        uv_x = create_uv_given_points(x, i)
        f_interp += interpolate_uv_scipy_cubic(f_temp[i], m, n, uv_x) * pou_x_patches[
            i
        ].reshape(-1, 1)

    return f_interp


def interpolate_patches_on_given_points_bicubic_pchip(f_patches, x, m, n):
    # f_patches (n_patch, (m - 1)*(n - 1), k) matrix
    # x (nn, 3) matrix

    k = f_patches.shape[2]
    nn = x.shape[0]

    pou_x_patches = np.zeros((n_patch, nn))
    for i in range(n_patch):
        pou_x_patches[i] = create_pou_given_points(x, i)
    pou_x_patches /= np.sum(pou_x_patches, axis=0)

    f_temp = np.zeros((n_patch, m - 1, n - 1, k))
    for ip in range(n_patch):
        for ik in range(k):
            f_temp[ip, :, :, ik] = f_patches[ip, :, ik].reshape(n - 1, m - 1).T
    f_interp = np.zeros((nn, k))
    for i in range(n_patch):
        uv_x = create_uv_given_points(x, i)
        f_interp += interpolate_uv_scipy_pchip(f_temp[i], m, n, uv_x) * pou_x_patches[
            i
        ].reshape(-1, 1)

    return f_interp


def interpolate_patches_on_given_points_bicubic_convolution(f_patches, x, m, n):
    # f_patches (n_patch, (m - 1)*(n - 1), k) matrix
    # x (nn, 3) matrix

    k = f_patches.shape[2]
    nn = x.shape[0]

    pou_x_patches = np.zeros((n_patch, nn))
    for i in range(n_patch):
        pou_x_patches[i] = create_pou_given_points(x, i)
    pou_x_patches /= np.sum(pou_x_patches, axis=0)

    u = np.arange(1, m) * np.pi / m
    v = np.arange(1, n) * np.pi / n
    f_interp = np.zeros((nn, k))
    for i in range(n_patch):
        uv_x = create_uv_given_points(x, i)
        for ik in range(k):
            f_interp[:, ik] += (
                bicubic_interpolation(
                    f_patches[i, :, ik].reshape(n - 1, m - 1).T,
                    u,
                    v,
                    uv_x[:, 0],
                    uv_x[:, 1],
                )
                * pou_x_patches[i]
            )

    return f_interp


def interpolate_patches_on_given_points_bicubic_poly(f_patches, x, m, n):
    # f_patches (n_patch, (m - 1)*(n - 1), k) matrix
    # x (nn, 3) matrix

    k = f_patches.shape[2]
    nn = x.shape[0]

    pou_x_patches = np.zeros((n_patch, nn))
    for i in range(n_patch):
        pou_x_patches[i] = create_pou_given_points(x, i)
    pou_x_patches /= np.sum(pou_x_patches, axis=0)

    u = np.arange(1, m) * np.pi / m
    v = np.arange(1, n) * np.pi / n
    f_interp = np.zeros((nn, k))
    for i in range(n_patch):
        uv_x = create_uv_given_points(x, i)
        for ik in range(k):
            f_interp[:, ik] += (
                bicubic_interpolation_poly(
                    f_patches[i, :, ik].reshape(n - 1, m - 1).T,
                    u,
                    v,
                    uv_x[:, 0],
                    uv_x[:, 1],
                )
                * pou_x_patches[i]
            )

    return f_interp


@njit
def lagrange_basis(x, xi, xj, xk, xl):
    """Calculate the Lagrange basis polynomial."""
    return (x - xj) * (x - xk) * (x - xl) / ((xi - xj) * (xi - xk) * (xi - xl))


@njit
def cubic_polynomial_basis(x, x_grid):
    """Calculate the cubic polynomial basis for non-uniform grids."""
    L0 = lagrange_basis(x, x_grid[0], x_grid[1], x_grid[2], x_grid[3])
    L1 = lagrange_basis(x, x_grid[1], x_grid[0], x_grid[2], x_grid[3])
    L2 = lagrange_basis(x, x_grid[2], x_grid[0], x_grid[1], x_grid[3])
    L3 = lagrange_basis(x, x_grid[3], x_grid[0], x_grid[1], x_grid[2])
    return np.array([L0, L1, L2, L3])


@njit
def cubic_interpolate(grid, x, y, x_grid, y_grid):
    """Perform cubic interpolation using cubic polynomials."""
    x_basis = cubic_polynomial_basis(x, x_grid)
    y_basis = cubic_polynomial_basis(y, y_grid)
    result = 0.0
    for i in range(4):
        for j in range(4):
            result += grid[i, j] * x_basis[i] * y_basis[j]
    return result


@njit
def compute_fractional_index_and_weights(x, xi):
    """Compute the fractional index and weights for non-uniform grids."""
    n = len(x)
    index = np.searchsorted(x, xi) - 1

    # Clip index to be within valid range
    if index < 1:
        index = 1
    elif index > n - 3:
        index = n - 3

    return index


@njit
def bicubic_interpolation_poly(data, x, y, x_new, y_new):
    """Perform bicubic interpolation on a non-uniform grid."""
    z_new = np.zeros(len(x_new))

    for idx in range(len(x_new)):
        xi, yi = x_new[idx], y_new[idx]

        # Compute indices and fractional parts
        x0 = compute_fractional_index_and_weights(x, xi)
        y0 = compute_fractional_index_and_weights(y, yi)

        # Get the 4x4 grid of points surrounding (xi, yi)
        grid = data[x0 - 1 : x0 + 3, y0 - 1 : y0 + 3]
        x_grid = x[x0 - 1 : x0 + 3]
        y_grid = y[y0 - 1 : y0 + 3]

        # Perform bicubic interpolation using cubic polynomials
        z_new[idx] = cubic_interpolate(grid, xi, yi, x_grid, y_grid)

    return z_new


@njit
def cubic_convolution_kernel(t, a=-0.5):
    abs_t = np.abs(t)
    if abs_t <= 1:
        return (a + 2) * abs_t**3 - (a + 3) * abs_t**2 + 1
    elif abs_t <= 2:
        return a * abs_t**3 - 5 * a * abs_t**2 + 8 * a * abs_t - 4 * a
    else:
        return 0.0


@njit
def bicubic_interpolate(grid, dx, dy, a=-0.5):
    result = 0.0
    for i in range(4):
        for j in range(4):
            result += (
                grid[i, j]
                * cubic_convolution_kernel(i - 1 - dx, a)
                * cubic_convolution_kernel(j - 1 - dy, a)
            )
    return result


@njit
def bicubic_interpolation(data, x, y, x_new, y_new, a=-0.5):
    z_new = np.zeros(len(x_new))

    for idx in range(len(x_new)):
        xi, yi = x_new[idx], y_new[idx]

        # Find the indices of the four surrounding points
        x0 = np.searchsorted(x, xi) - 1
        y0 = np.searchsorted(y, yi) - 1

        # Clip indices to be within the valid range
        if x0 < 1:
            x0 = 1
        elif x0 > len(x) - 3:
            x0 = len(x) - 3
        if y0 < 1:
            y0 = 1
        elif y0 > len(y) - 3:
            y0 = len(y) - 3

        # Calculate the fractional part
        dx = (xi - x[x0]) / (x[x0 + 1] - x[x0])
        dy = (yi - y[y0]) / (y[y0 + 1] - y[y0])

        # Get the 4x4 grid of points surrounding (xi, yi)
        grid = data[x0 - 1 : x0 + 3, y0 - 1 : y0 + 3]

        # Perform bicubic interpolation using cubic convolution
        z_new[idx] = bicubic_interpolate(grid, dx, dy, a)

    return z_new


def interpolate_patches_on_given_points_with_precompute_numba(
    f_patches,
    m,
    n,
    uv_grid,
    pou_grid_patches,
    ixs_grid,
    iys_grid,
    Nxs_grid,
    Nys_grid,
    interp_method="spline",
    spline_mat=None,
    spline_self_l_pou=None,
    spline_self_u_pou=None,
):
    # f_patches (n_patch, (m - 1)*(n - 1), k) matrix

    if interp_method == "spline":
        if spline_mat is not None:
            # tic = time()
            f_interp = create_f_interp_with_spline_mat(f_patches, spline_mat)
            # toc = time()
            # print('time spline:', toc - tic)
        elif spline_self_l_pou is not None:
            k = f_patches.shape[2]
            f_patches_temp = np.zeros((n_patch, k, (m - 1) * (n - 1)))
            for i in range(n_patch):
                for ik in range(k):
                    f_patches_temp[i, ik] = (
                        f_patches[i, :, ik].reshape(n - 1, m - 1).T.ravel()
                    )

            coeffs = compute_coeffs_solve_triangular_banded_grid(
                n_patch, spline_self_l_pou, spline_self_u_pou, f_patches_temp, k
            )

            nn = uv_grid.shape[1]
            nk = n - 1
            f_interp = create_f_interp_given_pou_cubic_spline_with_precompute(
                coeffs,
                nk,
                k,
                nn,
                ixs_grid,
                iys_grid,
                Nxs_grid,
                Nys_grid,
                pou_grid_patches,
            )

        else:
            k = f_patches.shape[2]
            nn = uv_grid.shape[1]
            u = np.arange(1, m) * np.pi / m
            v = np.arange(1, n) * np.pi / n

            # tic = time()
            funcs = [
                [
                    sinterp.RectBivariateSpline(
                        u, v, f_patches[i, :, ik].reshape(n - 1, m - 1).T
                    )
                    for ik in range(k)
                ]
                for i in range(n_patch)
            ]
            # toc = time()
            # print('time spline generate forward:', toc - tic)

            # xknots = funcs[0][0].get_knots()[0]
            yknots = funcs[0][0].get_knots()[1]
            nk = yknots.shape[0] - 4  # cubic spline

            coeffs = np.zeros((n_patch, k) + funcs[0][0].get_coeffs().shape)
            for i in range(n_patch):
                for ik in range(k):
                    coeffs[i, ik] = funcs[i][ik].get_coeffs()

            # tic = time()
            f_interp = create_f_interp_given_pou_cubic_spline_with_precompute(
                coeffs,
                nk,
                k,
                nn,
                ixs_grid,
                iys_grid,
                Nxs_grid,
                Nys_grid,
                pou_grid_patches,
            )
            # toc = time()
            # print('time spline eval forward:', toc - tic)

    elif interp_method == "cubic":
        k = f_patches.shape[2]
        nn = uv_grid.shape[1]

        f_patches_temp = np.zeros((n_patch, k, m - 1, n - 1))
        for i in range(n_patch):
            for ik in range(k):
                f_patches_temp[i, ik] = f_patches[i, :, ik].reshape(n - 1, m - 1).T

        f_interp = create_f_interp_given_pou_cubic_poly_with_precompute(
            f_patches_temp,
            k,
            nn,
            ixs_grid,
            iys_grid,
            Nxs_grid,
            Nys_grid,
            pou_grid_patches,
        )

    return f_interp


@njit(parallel=True)
def compute_coeffs_solve_triangular_banded_grid(
    n_patch, spline_self_l_pou, spline_self_u_pou, f_patches_temp, k
):
    coeffs = np.zeros((n_patch, k, spline_self_l_pou.shape[1]))
    for i in prange(n_patch):
        for ik in prange(k):
            temp = solve_triangular_banded_save(
                spline_self_l_pou, f_patches_temp[i, ik], lower=True
            )
            coeffs[i, ik] = solve_triangular_banded_save(
                spline_self_u_pou, temp, lower=False
            )

    return coeffs


@njit
def solve_triangular_banded_save(lu, b, lower):
    n = b.size
    x = np.zeros(n)
    bandwidth = lu.shape[0] - 1

    if lower:
        # Forward substitution for lower triangular matrix
        for i in range(n):
            x[i] = b[i]
            for j in range(max(0, i - bandwidth), i):
                x[i] -= lu[i - j, i] * x[j]
            x[i] /= lu[0, i]
    else:
        # Backward substitution for upper triangular matrix
        for i in range(n - 1, -1, -1):
            x[i] = b[i]
            for j in range(i + 1, min(n, i + bandwidth + 1)):
                x[i] -= lu[j - i, i] * x[j]
            x[i] /= lu[0, i]

    return x


@njit
def solve_lower_triangular_banded(L, b, bandwidth):
    n = len(b)
    x = np.zeros(n)
    for i in range(n):
        sum = 0.0
        start_index = max(0, i - bandwidth)
        for j in range(start_index, i):
            sum += L[i][j] * x[j]
        x[i] = (b[i] - sum) / L[i][i]

    return x


@njit
def solve_upper_triangular_banded(U, b, bandwidth):
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum = 0.0
        end_index = min(n, i + bandwidth + 1)
        for j in range(i + 1, end_index):
            sum += U[i][j] * x[j]
        x[i] = (b[i] - sum) / U[i][i]

    return x


@njit(parallel=True)
def create_f_interp_with_spline_mat(f_patches, spline_mat):
    k = f_patches.shape[2]
    f_patches = f_patches.reshape(-1, k)
    n1, n2 = spline_mat.shape
    f_interp = np.zeros((n1, k))
    for i in prange(n1):
        for j in range(n2):
            for ik in range(k):
                f_interp[i, ik] += spline_mat[i, j] * f_patches[j, ik]

    return f_interp


@njit
def basis_function(t, i, k, x):
    """
    Evaluate the ith B-spline basis function of degree k at x.
    """
    if k == 0:
        return np.where((t[i] <= x) & (x < t[i + 1]), 1.0, 0.0)
    else:
        left = (x - t[i]) / (t[i + k] - t[i]) * basis_function(t, i, k - 1, x)
        right = (
            (t[i + k + 1] - x)
            / (t[i + k + 1] - t[i + 1])
            * basis_function(t, i + 1, k - 1, x)
        )
        left[np.isnan(left)] = 0
        right[np.isnan(right)] = 0

        return left + right


@njit
def construct_spline_matrix(x_data, y_data, tx, ty, degree=3):
    """
    Construct the system of equations to solve for the spline coefficients.
    """
    nx = len(tx) - degree - 1
    ny = len(ty) - degree - 1
    n = nx * ny

    # Initialize the coefficient matrix and RHS vector
    A = np.zeros((len(x_data) * len(y_data), n))

    # Precompute basis functions for all data points
    Bx = np.zeros((len(x_data), nx))
    By = np.zeros((len(y_data), ny))
    for i in range(nx):
        Bx[:, i] = basis_function(tx, i, degree, x_data)
    for j in range(ny):
        By[:, j] = basis_function(ty, j, degree, y_data)

    Bx[-1, -1] = 1
    By[-1, -1] = 1

    # Constructing the A matrix
    for i in range(len(x_data)):
        for j in range(len(y_data)):
            bi = i * len(y_data) + j
            for k in range(nx):
                for l in range(ny):
                    bk = k * ny + l
                    A[bi, bk] = Bx[i, k] * By[j, l]

    return A


@njit
def find_knot_span(t, n, x):
    if x <= t[3]:
        return 3  # The first valid span for cubic spline (degree 3)
    if x >= t[n + 1]:
        return n

    low, high = 3, n + 1
    mid = (low + high) // 2
    while x < t[mid] or x >= t[mid + 1]:
        if x < t[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2

    return mid


@njit
def basis_funs(N, left, right, t, x, i, degree):
    N[0] = 1.0

    for j in range(1, degree + 1):
        left[j] = x - t[i + 1 - j]
        right[j] = t[i + j] - x
        saved = 0.0
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved


@njit
def eval_spline(
    f_vec, ind_vec, Nx, Ny, xleft, yleft, xright, yright, tx, ty, c, x, y, degree=3
):
    m = len(tx) - degree - 1  # number of basis functions in x
    n = len(ty) - degree - 1  # number of basis functions in y

    # Handle extrapolation for x
    if x < tx[3]:
        x = tx[3]
    elif x > tx[m]:
        x = tx[m]

    # Handle extrapolation for y
    if y < ty[3]:
        y = ty[3]
    elif y > ty[n]:
        y = ty[n]

    # Find knot spans
    ix = find_knot_span(tx, m - 1, x)
    iy = find_knot_span(ty, n - 1, y)

    # Evaluate basis functions
    basis_funs(Nx, xleft, xright, tx, x, ix, degree)
    basis_funs(Ny, yleft, yright, ty, y, iy, degree)

    # Compute the spline value
    for i in range(degree + 1):
        for j in range(degree + 1):
            f_vec[ind_vec] += Nx[i] * Ny[j] * c[(ix - degree + i) * n + iy - degree + j]


@njit(parallel=True)
def create_f_interp_given_pou_cubic_poly_with_precompute(
    f_patches_temp, k, nn, ixs, iys, Nxs, Nys, pou_grid_patches
):
    f_interp = np.zeros((nn, k))

    for inn in prange(nn):
        for ik in range(k):
            for ip in range(n_patch):
                temp = 0
                for i in range(4):
                    for j in range(4):
                        temp += (
                            Nxs[ip, inn, i]
                            * Nys[ip, inn, j]
                            * f_patches_temp[
                                ip, ik, ixs[ip, inn] - 1 + i, iys[ip, inn] - 1 + j
                            ]
                        )
                f_interp[inn, ik] += temp * pou_grid_patches[ip, inn]

    return f_interp


@njit(parallel=True)
def create_f_interp_given_pou_cubic_spline_with_precompute(
    coeffs, nk, k, nn, ixs, iys, Nxs, Nys, pou_grid_patches
):
    f_interp = np.zeros((nn, k))

    for inn in prange(nn):
        for ik in range(k):
            for ip in range(n_patch):
                temp = 0
                for i in range(4):
                    for j in range(4):
                        temp += (
                            Nxs[ip, inn, i]
                            * Nys[ip, inn, j]
                            * coeffs[
                                ip,
                                ik,
                                (ixs[ip, inn] - 3 + i) * nk + iys[ip, inn] - 3 + j,
                            ]
                        )
                f_interp[inn, ik] += temp * pou_grid_patches[ip, inn]

    return f_interp


@njit(parallel=True)
def create_f_interp_patches_cubic_spline_with_precompute(
    coeffs, nk, k, nn, ixs, iys, Nxs, Nys
):
    f_interps = np.zeros((n_patch, k, nn))

    for ip in prange(n_patch):
        for inn in prange(nn):
            for ik in range(k):
                for i in range(4):
                    for j in range(4):
                        f_interps[ip, ik, inn] += (
                            Nxs[ip, inn, i]
                            * Nys[ip, inn, j]
                            * coeffs[
                                ip,
                                ik,
                                (ixs[ip, inn] - 3 + i) * nk + iys[ip, inn] - 3 + j,
                            ]
                        )

    return f_interps


@njit(parallel=True)
def create_f_interp_patches_cubic_spline(coeffs, xknots, yknots, uv_grid, k, nn, m, n):
    f_interps = np.zeros((n_patch, k, nn))

    mk = xknots.shape[0] - 4
    nk = yknots.shape[0] - 4

    xmin = xknots[3]
    xmax = xknots[mk]
    ymin = yknots[3]
    ymax = yknots[nk]

    for ip in prange(n_patch):
        Nxs = np.zeros((nn, 4))
        Nys = np.zeros((nn, 4))

        xlefts = np.zeros((nn, 4))
        ylefts = np.zeros((nn, 4))
        xrights = np.zeros((nn, 4))
        yrights = np.zeros((nn, 4))

        for inn in prange(nn):
            x = uv_grid[ip, inn, 0]
            y = uv_grid[ip, inn, 1]

            if x < xmin:
                x = xmin
            elif x > xmax:
                x = xmax

            # Handle extrapolation for y
            if y < ymin:
                y = ymin
            elif y > ymax:
                y = ymax

            ix = find_knot_span(xknots, mk - 1, x)
            iy = find_knot_span(yknots, nk - 1, y)

            basis_funs(Nxs[inn], xlefts[inn], xrights[inn], xknots, x, ix, degree=3)
            basis_funs(Nys[inn], ylefts[inn], yrights[inn], yknots, y, iy, degree=3)

            for ik in range(k):
                for i in range(4):
                    for j in range(4):
                        f_interps[ip, ik, inn] += (
                            Nxs[inn, i]
                            * Nys[inn, j]
                            * coeffs[ip, ik, (ix - 3 + i) * nk + iy - 3 + j]
                        )

    return f_interps


@njit(parallel=True)
def f_interps_multiplies_pou(f_interps, pou_grid_patches, nn, k):
    f_interp = np.zeros((nn, k))
    for inn in prange(nn):
        for ik in range(k):
            for i in range(n_patch):
                f_interp[inn, ik] += f_interps[i, ik, inn] * pou_grid_patches[i, inn]

    return f_interp


def interpolate_patches_on_given_points_with_precompute(
    f_patches, m, n, uv_grid, pou_grid_patches
):
    # f_patches (n_patch, (m - 1)*(n - 1), k) matrix

    k = f_patches.shape[2]
    nn = uv_grid.shape[1]
    f_interp = np.zeros((nn, k))
    for i in range(n_patch):
        for ik in range(k):
            f_interp[:, ik] += (
                interpolate_uv(
                    f_patches[i, :, ik].reshape(n - 1, m - 1), m, n, uv_grid[i]
                )
                * pou_grid_patches[i]
            )

    return f_interp


def interpolate_uv(f_uv, m, n, uv_query):
    u = np.arange(1, m) * np.pi / m
    v = np.arange(1, n) * np.pi / n
    f_func = sinterp.RectBivariateSpline(u, v, f_uv.T)
    # RectBivariateSpline follow indexing = 'ij', so need to transpose f_uv
    f = f_func(uv_query[:, 0], uv_query[:, 1], grid=False)

    return f


def interpolate_uv_scipy_cubic(f_uv, m, n, uv_query):
    u = np.arange(1, m) * np.pi / m
    v = np.arange(1, n) * np.pi / n
    f = sinterp.RegularGridInterpolator(
        (u, v), f_uv, method="cubic", bounds_error=False, fill_value=0
    )(uv_query)

    return f


def interpolate_uv_scipy_pchip(f_uv, m, n, uv_query):
    u = np.arange(1, m) * np.pi / m
    v = np.arange(1, n) * np.pi / n
    f = sinterp.RegularGridInterpolator(
        (u, v), f_uv, method="pchip", bounds_error=False, fill_value=0
    )(uv_query)

    return f


def create_uv_given_points(x, patch_id):
    # patch id in {0, 1, 2, 3, 4, 5}
    # x (n, 3) matrix
    # allow points ouside the patch (allow v > pi)

    r0s = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, -1, 0], [0, 1, 0]])
    bs = np.array([[1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]])
    idxs = np.array([1, 1, 0, 0, 2, 2])
    sgns = np.array([1, -1, 1, -1, 1, -1])

    i = patch_id
    r0 = r0s[i]
    b = bs[i]
    idx = idxs[i]
    sgn = sgns[i]
    u = angle_vec(x, r0)
    a = x - np.outer(np.linalg.norm(x, axis=1) * np.cos(u), r0)
    v = angle_vec(a, b)
    id_v = sgn * x[:, idx] < 0
    v[id_v] = 2 * np.pi - v[id_v]
    uv = np.column_stack((u, v))

    return uv


def create_pou_given_points(x, patch_id):
    # x (n, 3) matrix

    r0s = np.array(
        [[0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]]
    )

    lon, lat, _ = cart_to_lonlatrad(x)
    lon0, lat0, _ = cart_to_lonlatrad(r0s[patch_id].reshape(1, -1))
    d = (5 / 12) * np.pi
    t = great_circle_distance(lat0, lon0, lat, lon) / d  # vectorized over lat, lon
    val = np.zeros(t.shape)
    val[t == 0] = 1
    t_id = (t > 0) & (t < 1)
    val[t_id] = np.exp(
        (2 * np.exp(-1 / t[t_id])) / (t[t_id] - 1)
    )  # pou from Bruno paper

    return val


def angle_vec(x, y):
    # x, y (n, 3) matrices
    # or x (n, 3) matrix and y (3,) vector
    # or y (n, 3) matrix and x (3,) vector
    # angle in [0, pi]

    return (
        np.arctan2(np.linalg.norm(np.cross(x, y), axis=1), np.sum(x * y, axis=1))
        % np.pi
    )


def cart_to_lonlatrad(x):
    # x (n, 3) matrix
    # lon in [-pi, pi]
    # lat in [-pi/2, pi/2]

    lon = np.arctan2(x[:, 1], x[:, 0])
    lat = np.arctan2(x[:, 2], np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2))
    rad = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)

    return lon, lat, rad


def great_circle_distance(lat0, lon0, lat1, lon1, rad=1):
    # lon, lat, rad vectors
    # dist in [0, pi]

    lon_diff = lon1 - lon0

    dist = np.arctan2(
        np.sqrt(
            (np.cos(lat1) * np.sin(lon_diff)) ** 2
            + (
                np.cos(lat0) * np.sin(lat1)
                - np.sin(lat0) * np.cos(lat1) * np.cos(lon_diff)
            )
            ** 2
        ),
        np.sin(lat0) * np.sin(lat1) + np.cos(lat0) * np.cos(lat1) * np.cos(lon_diff),
    )
    dist = rad * dist

    return dist
