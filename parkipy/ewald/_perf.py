import warnings
import numpy as np
from pykokkos.interface import is_host_execution_space

OPERATION_CONSTANTS = {
    None: {
        "fadd": 1,
        "fmul": 1,
        "fsqrt": 1,
        "frsqrt": 1,
        "fdiv": 1,
        "fexpn": 1,
        "fsinh": 1,
        "ferf": 1,
    },
    "a100": {
        "fadd": 2,
        "fmul": 2,
        "fsqrt": 29,
        "frsqrt": 19,
        "fdiv": 24,
        "fexpn": 45,
        "fsinh": 150,
        "ferf": 86,
    },
    "NVIDIA GH200 120GB": {
        "fadd": 2,
        "fmul": 2,
        "fsqrt": 41,
        "frsqrt": 30,
        "fdiv": 35,
        "fexpn": 50,
        "fsinh": 361,
        "ferf": 155,
    },
}


class PerfModel:
    """
    docstring
    """

    def __init__(
        self,
        kernel,
        N_out,
        N_in,
        fft_dim,
        ifft_dim,
        fft_shape,
        cell_size,
        window_P,
        execution_space,
    ):
        """
        docstring
        """

        device_name = None
        if not is_host_execution_space(execution_space):
            import cupy as cp

            device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        if device_name not in OPERATION_CONSTANTS.keys():
            warnings.warn(
                f"flop constants for device {device_name} not implemented, "
                "defaulting to flop=1 for all operations",
                UserWarning,
            )
            device_name = None

        # count flop
        self._flop_p2p = self._count_flop_p2p(kernel, N_out, cell_size, device_name)
        self._flop_p2g = self._count_flop_p2g(kernel, N_in, window_P, device_name)
        self._flop_fft = self._count_flop_fft(fft_dim, fft_shape)
        self._flop_cnv = self._count_flop_cnv(kernel, fft_shape, device_name)
        self._flop_ifft = self._count_flop_ifft(ifft_dim, fft_shape)
        self._flop_g2p = self._count_flop_g2p(ifft_dim, N_out, window_P, device_name)
        self._flop_ewald = (
            self.flop_p2p
            + self.flop_p2g
            + self.flop_fft
            + self.flop_cnv
            + self.flop_ifft
            + self.flop_g2p
        )

    @property
    def flop_p2p(self):
        """
        FLOP model for the P2P algorithm.
        Read-only.
        """
        return self._flop_p2p

    @property
    def flop_p2g(self):
        """
        FLOP model for the P2G algorithm.
        Read-only.
        """
        return self._flop_p2g

    @property
    def flop_fft(self):
        """
        FLOP model for the FFT algorithm.
        Read-only.
        """
        return self._flop_fft

    @property
    def flop_cnv(self):
        """
        FLOP model for the CNV algorithm.
        Read-only.
        """
        return self._flop_cnv

    @property
    def flop_ifft(self):
        """
        FLOP model for the IFFT algorithm.
        Read-only.
        """
        return self._flop_ifft

    @property
    def flop_g2p(self):
        """
        FLOP model for the G2P algorithm.
        Read-only.
        """
        return self._flop_g2p

    @property
    def flop_ewald(self):
        """
        FLOP model for the Ewald sum.
        Read-only.
        """
        return self._flop_ewald

    def _count_flop_p2p(self, kernel, N_out, cell_size, device_name):
        C_stokes_ewald = (
            14 * OPERATION_CONSTANTS[device_name]["fadd"]
            + 1 * OPERATION_CONSTANTS[device_name]["frsqrt"]
            + 19 * OPERATION_CONSTANTS[device_name]["fmul"]
            + 1 * OPERATION_CONSTANTS[device_name]["fdiv"]
            + 1 * OPERATION_CONSTANTS[device_name]["ferf"]
            + 1 * OPERATION_CONSTANTS[device_name]["fexpn"]
        )
        C_sl = (
            13 * OPERATION_CONSTANTS[device_name]["fmul"]
            + 11 * OPERATION_CONSTANTS[device_name]["fadd"]
        )
        C_dl = (
            33 * OPERATION_CONSTANTS[device_name]["fmul"]
            + 19 * OPERATION_CONSTANTS[device_name]["fadd"]
        )
        match kernel:
            case "stokes_sl":
                C_p2p = C_stokes_ewald + C_sl
            case "stokes_dl":
                C_p2p = C_stokes_ewald + C_dl
            case "stokes_comb":
                C_p2p = C_stokes_ewald + C_sl + C_dl
            case _:
                raise NotImplementedError(
                    f"P2P flop model not implemented for {kernel} kernel"
                )
        return 27 * N_out * cell_size * np.pi / 6.0 * C_p2p

    def _count_flop_p2g(self, kernel, N_in, window_P, device_name):
        match kernel:
            case "stokes_sl":
                C_p2g = (
                    5 * OPERATION_CONSTANTS[device_name]["fmul"]
                    + 6 * OPERATION_CONSTANTS[device_name]["fadd"]
                )
            case "stokes_comb":
                C_p2g = (
                    17 * OPERATION_CONSTANTS[device_name]["fmul"]
                    + 15 * OPERATION_CONSTANTS[device_name]["fadd"]
                )
            case _:
                raise NotImplementedError(
                    f"P2G flop model not implemented for {kernel} kernel"
                )
        return N_in * window_P**3 * C_p2g

    def _count_flop_fft(self, fft_dim, fft_shape):
        fft_size = np.array(fft_shape).prod()
        return 5 * fft_dim * fft_size * np.log2(fft_size)

    def _count_flop_cnv(self, kernel, fft_shape, device_name):
        C_kb = (
            1 * OPERATION_CONSTANTS[device_name]["fsqrt"]
            + 3 * OPERATION_CONSTANTS[device_name]["fmul"]
            + 1 * OPERATION_CONSTANTS[device_name]["fadd"]
            + 1 * OPERATION_CONSTANTS[device_name]["fdiv"]
            + 1 * OPERATION_CONSTANTS[device_name]["fsinh"]
        )
        C_stokes = (
            3 * OPERATION_CONSTANTS[device_name]["fdiv"] * C_kb
            + 8 * OPERATION_CONSTANTS[device_name]["fmul"]
            + 1 * OPERATION_CONSTANTS[device_name]["fexpn"]
            + 1 * OPERATION_CONSTANTS[device_name]["fadd"]
        )
        C_sl = (
            33 * OPERATION_CONSTANTS[device_name]["fmul"]
            + 12 * OPERATION_CONSTANTS[device_name]["fadd"]
            + 1 * C_stokes
            + 6 * OPERATION_CONSTANTS[device_name]["fdiv"]
        )
        C_dl = (
            96 * OPERATION_CONSTANTS[device_name]["fmul"]
            + 53 * OPERATION_CONSTANTS[device_name]["fadd"]
            + 9 * OPERATION_CONSTANTS[device_name]["fdiv"]
            + 1 * OPERATION_CONSTANTS[device_name]["fexpn"]
            + 3 * C_kb
        )
        match kernel:
            case "stokes_comb":
                C_cnv = C_sl + C_dl
            case _:
                raise NotImplementedError(
                    f"CNV flop model not implemented for {kernel} kernel"
                )
        fft_size = np.array(fft_shape).prod()
        return fft_size * C_cnv

    def _count_flop_ifft(self, ifft_dim, fft_shape):
        fft_size = np.array(fft_shape).prod()
        return 5 * ifft_dim * fft_size * np.log2(fft_size)

    def _count_flop_g2p(self, ifft_dim, N_out, window_P, device_name):
        return (
            N_out
            * window_P**3
            * (
                ifft_dim
                * (
                    1 * OPERATION_CONSTANTS[device_name]["fadd"]
                    + 1 * OPERATION_CONSTANTS[device_name]["fmul"]
                )
                + 1 * OPERATION_CONSTANTS[device_name]["fmul"]
            )
        )
