import numpy as np
import cupy as cp
import nvmath.distributed
from mpi4py import MPI
from dataclasses import dataclass, field
from typing import Any, Literal, List


@dataclass
class FFTMPBuffers:
    """
    Prepare Ewald execution for multiple processes.
    """

    fft_shape: List[int]
    ifft_shape: List[int]
    fft_type: Literal["R2C", "C2C"]

    rank: int = field(init=False)
    nranks: int = field(init=False)
    device_id: int = field(init=False)
    fft_buff: Any = field(init=False)

    def __post_init__(self):
        if self.fft_type.upper() not in ["C2C", "R2C"]:
            raise ValueError(
                f"fft_type expected to be one of 'R2C', 'C2C', got {self.fft_type.upper()}"
            )
        elif self.fft_type.upper() == "C2C":
            self.fft_op = nvmath.distributed.fft.fft
            self.ifft_op = nvmath.distributed.fft.ifft
        elif self.fft_type.upper() == "R2C":
            self.fft_op = nvmath.distributed.fft.rfft
            self.ifft_op = nvmath.distributed.fft.irfft
        # Initialize nvmath.distributed.
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.nranks = comm.Get_size()
        self.device_id = self.rank % cp.cuda.runtime.getDeviceCount()
        try:
            nvmath.distributed.initialize(self.device_id, comm)
        except RuntimeError:
            pass

        # cuFFTMp uses the NVSHMEM PGAS model
        #   for distributed computation, which requires GPU
        #   operands to be on the symmetric heap.
        self.fft_buff = nvmath.distributed.fft.allocate_operand(
            self.fft_shape,
            cp,
            input_dtype=(cp.complex128 if self.fft_type.upper() == "C2C" else cp.float64), # TODO: change to device_pre data type
            distribution=nvmath.distributed.fft.Slab.X,
            memory_space="cuda",
            fft_type=self.fft_type, # TODO: change for single precision
        )
        self.fft_buff[:] = 0  # pad with zeros
        self.ifft_buff = nvmath.distributed.fft.allocate_operand(
            self.ifft_shape,
            cp,
            input_dtype=cp.complex128, # TODO: change to device_pre data type
            distribution=nvmath.distributed.fft.Slab.Y,
            memory_space="cuda",
            fft_type="Z2D", # TODO: change for single precision
        )
        self.fft_size = np.prod([self.fft_shape[0] * self.nranks, *self.fft_shape[1:]])

    def fft(self):
        out = self.fft_op(
            self.fft_buff,
            distribution=nvmath.distributed.fft.Slab.X,
            options={"reshape": False, "last_axis_parity": "even"},
        )
        return out

    def ifft(self):
        out = self.ifft_op(
            self.ifft_buff,
            distribution=nvmath.distributed.fft.Slab.Y,
            options={"reshape": False, "last_axis_parity": "even"},
        )
        out /= self.fft_size
        return out

    def __del__(self):
        nvmath.distributed.free_symmetric_memory(self.fft_buff)
        nvmath.distributed.free_symmetric_memory(self.ifft_buff)
