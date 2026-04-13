import pprint
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
        p2p_time,
        p2g_time,
        fft_time,
        cnv_time,
        ifft_time,
        g2p_time,
        kernel,
        N_out,
        N_in,
        N_grid,
        d_out,
        d_in,
        fft_dim,
        ifft_dim,
        fft_shape,
        cell_size,
        window_P,
        dtype,
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

        # store run parameters
        self._N_in = N_in
        self._N_out = N_out
        self._N_grid = N_grid
        self._N_fft = fft_shape

        # store time
        self._time_p2p = p2p_time
        self._time_p2g = p2g_time
        self._time_fft = fft_time
        self._time_cnv = cnv_time
        self._time_ifft = ifft_time
        self._time_g2p = g2p_time
        self._time_ewald = (
            self.time_p2p["tot"]
            + self.time_p2g["tot"]
            + self.time_fft["tot"]
            + self.time_cnv["tot"]
            + self.time_ifft["tot"]
            + self.time_g2p["tot"]
        )

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

        # count bytes
        N_fft = np.array(fft_shape).prod()
        d_fft = fft_dim
        d_ifft = ifft_dim
        self._mop_p2p = self._count_mop_p2p(N_out, N_in, d_out, d_in, dtype.itemsize)
        self._mop_p2g = self._count_mop_p2g(N_in, d_in, N_fft, d_fft, dtype.itemsize)
        self._mop_fft = self._count_mop_fft(N_fft, d_fft, 2 * dtype.itemsize)
        self._mop_cnv = self._count_mop_cnv(N_fft, d_fft, 2 * dtype.itemsize)
        self._mop_ifft = self._count_mop_ifft(N_fft, d_ifft, 2 * dtype.itemsize)
        self._mop_g2p = self._count_mop_g2p(N_fft, d_ifft, N_out, d_out, dtype.itemsize)
        self._mop_ewald = (
            self.mop_p2p
            + self.mop_p2g
            + self.mop_fft
            + self.mop_cnv
            + self.mop_ifft
            + self.mop_g2p
        )

    def __repr__(self):
        """pretty representation of the class"""
        BOLD = "\033[1m"
        RESET = "\033[0m"
        lines = []

        # Ewald Run Parameters
        lines.append("\n\n")
        lines.append(f"{BOLD}Ewald Parameters{RESET}")
        lines.append("-" * 60)
        lines.append(
            f"  N_in = {self.N_in:_}, N_out = {self.N_out:_}, N_grid = {self.N_grid:_}, N_fft = {self.N_fft:_}"
        )

        # Walltimes
        lines.append("\n\n")
        lines.append(f"{BOLD}Ewald Walltimes (seconds){RESET}")
        lines.append("-" * 60)
        lines.append(f"  P2P : ")
        lines.append(
            "\n".join("    " + l for l in pprint.pformat(self.time_p2p).splitlines())
        )
        lines.append(f"  P2G : ")
        lines.append(
            "\n".join("    " + l for l in pprint.pformat(self.time_p2g).splitlines())
        )
        lines.append(f"  FFT : ")
        lines.append(
            "\n".join("    " + l for l in pprint.pformat(self.time_fft).splitlines())
        )
        lines.append(f"  CNV : ")
        lines.append(
            "\n".join("    " + l for l in pprint.pformat(self.time_cnv).splitlines())
        )
        lines.append(f"  IFFT: ")
        lines.append(
            "\n".join("    " + l for l in pprint.pformat(self.time_ifft).splitlines())
        )
        lines.append(f"  G2P : ")
        lines.append(
            "\n".join("    " + l for l in pprint.pformat(self.time_g2p).splitlines())
        )
        lines.append(f"  TOTAL: {self.time_ewald}")

        lines.append("\n\n")
        lines.append(f"{BOLD}Ewald Cost Model{RESET}")
        lines.append("-" * 60)

        # FLOP (total number of floating point operations)
        lines.append("Throughput (FLOP/s):")
        lines.append(f"  P2P : {self.flop_p2p/self.time_p2p['tot']:12.3e}")
        lines.append(f"  P2G : {self.flop_p2g/self.time_p2g['tot']:12.3e}")
        lines.append(f"  FFT : {self.flop_fft/self.time_fft['tot']:12.3e}")
        lines.append(f"  CNV : {self.flop_cnv/self.time_cnv['tot']:12.3e}")
        lines.append(f"  IFFT: {self.flop_ifft/self.time_ifft['tot']:12.3e}")
        lines.append(f"  G2P : {self.flop_g2p/self.time_g2p['tot']:12.3e}")
        lines.append(f"  TOTAL: {self.flop_ewald/self.time_ewald:11.3e}")

        lines.append("")

        # Memory ops (bytes moved)
        lines.append("Bandwidth (bytes/s):")
        lines.append(f"  P2P : {self.mop_p2p/self.time_p2p['tot']:12.3e}")
        lines.append(f"  P2G : {self.mop_p2g/self.time_p2g['tot']:12.3e}")
        lines.append(f"  FFT : {self.mop_fft/self.time_fft['tot']:12.3e}")
        lines.append(f"  CNV : {self.mop_cnv/self.time_cnv['tot']:12.3e}")
        lines.append(f"  IFFT: {self.mop_ifft/self.time_ifft['tot']:12.3e}")
        lines.append(f"  G2P : {self.mop_g2p/self.time_g2p['tot']:12.3e}")
        lines.append(f"  TOTAL: {self.mop_ewald/self.time_ewald:11.3e}")
        lines.append("\n")

        return "\n".join(lines)

    @property
    def N_in(self):
        """
        Number of source particles.
        Read-only
        """
        return self._N_in

    @property
    def N_out(self):
        """
        Number of target particles.
        Read-only
        """
        return self._N_out

    @property
    def N_grid(self):
        """
        Number of regular grid points.
        Read-only
        """
        return self._N_grid

    @property
    def N_fft(self):
        """
        Number of regular grid points on the FFT (i.e., upsampled) grid.
        Read-only
        """
        return self._N_grid

    @property
    def time_p2p(self):
        """
        Execution time for the P2P algorithm.
        Read-only
        """
        return self._time_p2p

    @property
    def time_p2g(self):
        """
        Execution time for the P2G algorithm.
        Read-only
        """
        return self._time_p2g

    @property
    def time_fft(self):
        """
        Execution time for the FFT.
        Read-only
        """
        return self._time_fft

    @property
    def time_cnv(self):
        """
        Execution time for the CNV algorithm.
        Read-only
        """
        return self._time_cnv

    @property
    def time_ifft(self):
        """
        Execution time for the IFFT.
        Read-only
        """
        return self._time_ifft

    @property
    def time_g2p(self):
        """
        Execution time for the G2P algorithm.
        Read-only
        """
        return self._time_g2p

    @property
    def time_ewald(self):
        """
        Execution time for the Ewald sum.
        Read-only
        """
        return self._time_ewald

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

    @property
    def mop_p2p(self):
        """
        MOP model for the P2P algorithm.
        Read-only.
        """
        return self._mop_p2p

    @property
    def mop_p2g(self):
        """
        MOP model for the P2G algorithm.
        Read-only.
        """
        return self._mop_p2g

    @property
    def mop_fft(self):
        """
        MOP model for the FFT algorithm.
        Read-only.
        """
        return self._mop_fft

    @property
    def mop_cnv(self):
        """
        MOP model for the CNV algorithm.
        Read-only.
        """
        return self._mop_cnv

    @property
    def mop_ifft(self):
        """
        MOP model for the IFFT algorithm.
        Read-only.
        """
        return self._mop_ifft

    @property
    def mop_g2p(self):
        """
        MOP model for the G2P algorithm.
        Read-only.
        """
        return self._mop_g2p

    @property
    def mop_ewald(self):
        """
        MOP model for the Ewald sum.
        Read-only.
        """
        return self._mop_ewald

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
            case "laplace":
                C_p2p = (
                    12 * OPERATION_CONSTANTS[device_name]["fadd"]
                    + 9 * OPERATION_CONSTANTS[device_name]["fmul"]
                    + 1 * OPERATION_CONSTANTS[device_name]["fsqrt"]
                    + 1 * OPERATION_CONSTANTS[device_name]["fdiv"]
                    + 1 * OPERATION_CONSTANTS[device_name]["ferf"]
                )
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
                    + 3 * OPERATION_CONSTANTS[device_name]["fadd"]
                )
            case "stokes_comb":
                C_p2g = (
                    17 * OPERATION_CONSTANTS[device_name]["fmul"]
                    + 15 * OPERATION_CONSTANTS[device_name]["fadd"]
                )
            case "laplace":
                C_p2g = (
                    3 * OPERATION_CONSTANTS[device_name]["fmul"]
                    + 1 * OPERATION_CONSTANTS[device_name]["fadd"]
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
            case "stokes_sl":
                C_cnv = C_sl
            case "stokes_comb":
                C_cnv = C_sl + C_dl
            case "laplace":
                C_cnv = (
                    19 * OPERATION_CONSTANTS[device_name]["fmul"]
                    + 2 * OPERATION_CONSTANTS[device_name]["fadd"]
                    + 5 * OPERATION_CONSTANTS[device_name]["fdiv"]
                    + 1 * OPERATION_CONSTANTS[device_name]["fexpn"]
                    + 3 * C_kb
                )
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

    def _count_mop_p2p(self, N_out, N_in, d_out, d_in, real_bytes):
        return real_bytes * (N_out * d_out + N_in * d_in)

    def _count_mop_p2g(self, N_in, d_in, N_fft, d_fft, real_bytes):
        return real_bytes * (N_in * d_in + N_fft * d_fft)

    def _count_mop_fft(self, N_fft, d_fft, cmpx_bytes):
        return cmpx_bytes * (N_fft * d_fft)

    def _count_mop_cnv(self, N_fft, d_fft, cmpx_bytes):
        return cmpx_bytes * (N_fft * d_fft)

    def _count_mop_ifft(self, N_ifft, d_ifft, cmpx_bytes):
        return cmpx_bytes * (N_ifft * d_ifft)

    def _count_mop_g2p(self, N_ifft, d_ifft, N_out, d_out, real_bytes):
        return real_bytes * (N_out * d_out + N_ifft * d_ifft)
