import numpy as np

n_patch = 6


def patches_ellipsoid_given_uv(ps, u, v, a, b, c):
    ps[0, ..., 0] = a * np.sin(u) * np.cos(v)
    ps[0, ..., 1] = b * np.sin(u) * np.sin(v)
    ps[0, ..., 2] = c * np.cos(u)

    ps[1, ..., 0] = -a * np.sin(u) * np.cos(v)
    ps[1, ..., 1] = -b * np.sin(u) * np.sin(v)
    ps[1, ..., 2] = c * np.cos(u)

    ps[2, ..., 0] = a * np.sin(u) * np.sin(v)
    ps[2, ..., 1] = -b * np.sin(u) * np.cos(v)
    ps[2, ..., 2] = c * np.cos(u)

    ps[3, ..., 0] = -a * np.sin(u) * np.sin(v)
    ps[3, ..., 1] = b * np.sin(u) * np.cos(v)
    ps[3, ..., 2] = c * np.cos(u)

    ps[4, ..., 0] = a * np.sin(u) * np.cos(v)
    ps[4, ..., 1] = -b * np.cos(u)
    ps[4, ..., 2] = c * np.sin(u) * np.sin(v)

    ps[5, ..., 0] = a * np.sin(u) * np.cos(v)
    ps[5, ..., 1] = b * np.cos(u)
    ps[5, ..., 2] = -c * np.sin(u) * np.sin(v)


def create_uv_patch(m, n):
    # uv_patch ((m - 1)*(n - 1), 2) matrix

    uv_patch = np.zeros(((m - 1) * (n - 1), 2))
    us = np.arange(1, m) * np.pi / m
    vs = np.arange(1, n) * np.pi / n
    uv_patch = np.meshgrid(us, vs, indexing="xy")

    return uv_patch


def create_patches_ellipsoid(m, n, a, b, c):
    # x_patches (n_patch, (m - 1)*(n - 1), 3) matrix

    uv_patch = create_uv_patch(m, n)

    x_patches = np.zeros((n_patch, (m - 1), (n - 1), 3))
    patches_ellipsoid_given_uv(x_patches, uv_patch[0], uv_patch[1], a, b, c)

    return x_patches, uv_patch
