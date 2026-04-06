import pykokkos as pk
import parkipy.utils as utils


def get_ifftn(execution_space, fft_type):
    """
    Return the appropriate `irfftn` callable
    given an execution_space.
    """
    fft_type = fft_type.upper()
    execution_space = utils.get_execution_space(execution_space)
    if not pk.is_host_execution_space(execution_space):
        from cupyx.scipy.fft import irfftn, ifftn
    else:
        from scipy.fft import irfftn, ifftn
    if fft_type == "R2C":
        return irfftn
    elif fft_type == "C2C":
        return ifftn
    else:
        raise ValueError(f"IFFT type must be one of 'R2C', 'C2C', got '{fft_type}'.")


def get_fftn(execution_space, fft_type):
    """
    Return the appropriate `rfftn` callable
    given an execution_space.
    """
    fft_type = fft_type.upper()
    execution_space = utils.get_execution_space(execution_space)
    if not pk.is_host_execution_space(execution_space):
        from cupyx.scipy.fft import rfftn, fftn
    else:
        from scipy.fft import rfftn, fftn
    if fft_type == "R2C":
        return rfftn
    elif fft_type == "C2C":
        return fftn
    else:
        raise ValueError(f"FFT type must be one of 'R2C', 'C2C', got '{fft_type}'.")
