"""Adaptive Fourier transforms for one and two periodic directions.

The channel axis comes first, followed by the periodic and free axes.
See Bagge and Tornberg, J. Comput. Phys. 493 (2023), 112473,
https://doi.org/10.1016/j.jcp.2023.112473.
"""


def adaptive_fft(values, local_indices, local_shape, zero_shape, fft, real=False):
    """Transform the global, local, and zero-mode grids."""
    periodicity = len(local_indices)
    periodic = (fft.rfft if real else fft.fft)(values, axis=1)
    if periodicity == 2:
        periodic = fft.fft(periodic, axis=2)
    free_axes = tuple(range(periodicity + 1, 4))
    global_grid = fft.fftn(periodic, axes=free_axes)

    local_grid = None
    if len(local_indices[0]):
        selected = periodic[(slice(None), *local_indices)]
        # Keep the three spatial axes required by the convolution kernels.
        local_grid = fft.fftn(
            selected.reshape(
                selected.shape[0],
                -1,
                *(1,) * (periodicity - 1),
                *values.shape[periodicity + 1 :],
            ),
            s=local_shape,
            axes=free_axes,
        )
    zero = (slice(None),) + (slice(0, 1),) * periodicity
    zero_grid = fft.fftn(periodic[zero], s=zero_shape, axes=free_axes)
    return global_grid, local_grid, zero_grid


def adaptive_ifft(
    global_grid, local_grid, zero_grid, local_indices, shape, fft, real=False
):
    """Reassemble periodic modes after transforming and cropping the free axes."""
    periodicity = len(local_indices)
    free_axes = tuple(range(periodicity + 1, 4))
    crop = (Ellipsis,) + tuple(slice(n) for n in shape[periodicity:])
    result = fft.ifftn(global_grid, axes=free_axes)
    if len(local_indices[0]):
        local = fft.ifftn(local_grid, axes=free_axes)[crop]
        result[(slice(None), *local_indices)] = local.reshape(
            result.shape[0], -1, *shape[periodicity:]
        )
    zero = (slice(None),) + (slice(0, 1),) * periodicity
    result[zero] = fft.ifftn(zero_grid, axes=free_axes)[crop]
    if periodicity == 2:
        result = fft.ifft(result, axis=2)
    return fft.irfft(result, n=shape[0], axis=1) if real else fft.ifft(result, axis=1)
