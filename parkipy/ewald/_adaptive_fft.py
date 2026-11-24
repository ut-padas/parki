"""Building blocks for a one-periodic adaptive Fourier transform.

The arrays have a leading channel axis and the periodic axis is axis 1.
See Bagge and Tornberg, J. Comput. Phys. 493 (2023), 112473,
https://doi.org/10.1016/j.jcp.2023.112473.
"""


def fft_1p(
    values, local_indices, global_shape, local_shape, zero_shape, fft, real=False
):
    """Transform a physical grid into global, local, and zero-mode grids."""
    periodic = (fft.rfft if real else fft.fft)(values, axis=1)
    global_grid = fft.fftn(periodic, s=global_shape, axes=(2, 3))
    local_grid = None
    if len(local_indices):
        local_grid = fft.fftn(periodic[:, local_indices], s=local_shape, axes=(2, 3))
    zero_grid = fft.fftn(periodic[:, :1], s=zero_shape, axes=(2, 3))
    return global_grid, local_grid, zero_grid


def ifft_1p(global_grid, local_grid, zero_grid, local_indices, shape, fft, real=False):
    """Invert and assemble the three mode groups on the physical grid."""
    ny, nz = shape[1:]
    result = fft.ifftn(global_grid, axes=(2, 3))[:, :, :ny, :nz].copy()
    if len(local_indices):
        result[:, local_indices] = fft.ifftn(local_grid, axes=(2, 3))[:, :, :ny, :nz]
    result[:, :1] = fft.ifftn(zero_grid, axes=(2, 3))[:, :, :ny, :nz]
    return fft.irfft(result, n=shape[0], axis=1) if real else fft.ifft(result, axis=1)
