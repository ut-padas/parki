"""Building blocks for a one-periodic adaptive Fourier transform.

The arrays have a leading channel axis and the periodic axis is axis 1.
All transforms here are complex-to-complex; the inverse returns a complex
array so callers can inspect the imaginary residual before taking its real part.
"""


def fft_1p(values, local_indices, global_shape, local_shape, zero_shape, fft):
    """Transform a physical grid into global, local, and zero-mode grids."""
    periodic = fft.fft(values, axis=1)
    global_grid = fft.fftn(periodic, s=global_shape, axes=(2, 3))
    local_grid = fft.fftn(periodic[:, local_indices], s=local_shape, axes=(2, 3))
    zero_grid = fft.fftn(periodic[:, :1], s=zero_shape, axes=(2, 3))
    return global_grid, local_grid, zero_grid


def ifft_1p(global_grid, local_grid, zero_grid, local_indices, shape, fft):
    """Invert and assemble the three mode groups on the physical grid."""
    ny, nz = shape[1:]
    result = fft.ifftn(global_grid, axes=(2, 3))[:, :, :ny, :nz].copy()
    if len(local_indices):
        result[:, local_indices] = fft.ifftn(local_grid, axes=(2, 3))[:, :, :ny, :nz]
    result[:, :1] = fft.ifftn(zero_grid, axes=(2, 3))[:, :, :ny, :nz]
    return fft.ifft(result, axis=1)
