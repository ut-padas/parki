"""
Module for PDE kernels and their
evaluation using ewald summation.

Contains the `pyse.ewald.EwaldKernel` class.
"""

import warnings
import numpy as np
import pykokkos as pk
from dataclasses import dataclass, field

from parkipy.utils import get_array_module, get_execution_space
from parkipy.ewald import EwaldOptions
from parkipy.ewald._ewald import p2p, p2g, fft, cnv, ifft, g2p
from parkipy.ewald._prepare import DevicePre
from ._fft_utils import FFTMPBuffers

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None


class EwaldKernel:
    """
    Represents the Ewald split of a PDE kernel, i.e.,
    a fundamental solution (Green's function).

    Calling an instance of this class will evaluate the kernel
    using the spectral Ewald method.
    """

    def __init__(
        self, name, dim_in, dim_out, kernel, takes_normals=False, description=None
    ):
        """
        Create a new kernel.

        Input:
            - `name`: Name of the kernel (string). This will be used to build
              the `self.key` property and should therefore be unique.
            - `dim_in`: Input dimension, the number of dimensions of the layer
              density (or a point source).
            - `dim_out`: Output dimension, the number of dimensions of the layer
              potential (or potential from a point source).
            - `kernel`: The name of the kernel to be evaluated.
            - `takes_normals`: (bool) The kernel takes the normal vector as an
              argument (default: `False`).
            - `description`: Longer string describing the kernel (optional).
        """
        self._name = name
        self._dim_in = dim_in
        self._dim_out = dim_out
        self._kernel = kernel
        self._takes_normals = takes_normals
        self._description = description

        return

    @property
    def name(self):
        """
        Name of the kernel. Read-only.
        """
        return self._name

    @property
    def dim_in(self):
        """
        Input dimension. Read-only.

        See also `get_shape_in()`.
        """
        return self._dim_in

    @property
    def dim_out(self):
        """
        Output dimension. Read-only.

        See also `get_shape_out()`.
        """
        return self._dim_out

    @property
    def kernel(self):
        """
        Kernel name for evaluation. Read-only.
        """
        return self._kernel

    @property
    def takes_normals(self):
        """
        (bool) True if the kernel takes the normal vector as an argument.
        Read-only.
        """
        return self._takes_normals

    @property
    def description(self):
        """
        Longer string describing the kernel. Read-only.
        """
        return self._description

    @property
    def key(self):
        """
        Return a string suitable for use as a dict key, representing this
        kernel. Read-only.

        The key is formed as `'sph_kernel_' + self.name`.
        """
        return "sph_kernel_" + self.name

    def __call__(
        self,
        *args,
        cell_size=None,
        rc=None,
        p2p_method="GM-1D",
        p2g_method="HYBRID",
        g2p_method="TARGET",
        buffer_size=None,
        time=False,
        fft_type="R2C",
    ):
        """
        Evaluate the kernel using direct evaluation.

        Input:
            - `x_out`: Points to evaluate the potential at, shape `(3, N_out)`.
              May also be `(3,)` for a single point.
            - `x_in`: Source points, shape `(3, N_in)`.
            - `q_in`: Density values, shape `self.get_shape_in(N_in)`.
            - `n_in`: Only if `self.takes_normals==True`: Normal vectors, shape
              `(3, N_in)`.
            - `periodicity`:
                The number of periodic spatial directions.
            - `box`: Side lengths of computational domain.
               That is, `0 < min(x[i])`, `max(x[i]) < box[i]` for `i=0,1,2`
               and `x=x_in,x_out`.
            - `tol`: Spectral Ewald tolerance.
            - `execution_space`: PyKokkos execution space See
              https://kokkos.org/kokkos-core-wiki/API/core/execution_spaces.html
              for available execution spaces. Input is either a string
              (e.g., 'OpenMP') or a PyKokkos object (e.g., pk.OpenMP).
              Defaults to `pk.get_default_space()`
            - `cell_size`: Real space cell size. Defaults to `None`.
            - `p2p_method`: Must be one of `'GM-1D'`, `'GM-2D'`, `'SM-1D'`,
              or `'SM-2D'`. Defaults to `'GM-1D'`.
            - `p2g_method`: Must be one of `'BASE'`, `'SOURCE'`, `'GRID'`, `'HYBRID'`.
              Defaults to `'HYBRID'`.
            - `g2p_method`: Must be one of `'BASE'`, `'TARGET'`. Defaults to `'TARGET'`.
            - `mpi_comm`: MPI_COMM_WORLD for distributed execution.
            - `buffer_size`: Size of MPI buffers. Only relevant if `mpi_comm` is not `None`.
               Defaults to `2*MPI.Allreduce(N_in, MPI.MAX)`.
            - `time`: Flag to return `walltime` dictionary.

        Returns potential values at `x_out` as an array of shape
        `self.get_shape_out(N_out)`.
        """
        # parse arguments
        if len(args) != 7 + self.takes_normals:
            raise TypeError(
                f"Wrong number of positional arguments, "
                f"got {len(args)}, "
                f"expected {7 + self.takes_normals}"
            )
        x_out = args[0]
        if len(x_out.shape) == 1 and np.size(x_out) == 3:
            x_out = x_out.reshape((3, 1))
        N_out = x_out.shape[1]
        if x_out.shape != (3, N_out):
            raise ValueError(f"x_out must be of shape (3, N_out), found {x_out.shape}")
        x_in = args[1]
        N_in = x_in.shape[1]
        if x_in.shape != (3, N_in):
            raise ValueError(f"x_in must be of shape (3, N_in), found {x_in.shape}")
        q_in = args[2]
        if q_in.shape != self.get_shape_in(N_in):
            raise ValueError(
                f"q_in must be of shape {self.get_shape_in(N_in)}, found {q_in.shape}"
            )
        if q_in.dtype != x_in.dtype:
            raise TypeError(
                "q_in must have the same dtype as x_in,"
                f" got {q_in.dtype}, expected {x_in.dtype}."
            )
        n_in = None
        if self.takes_normals:
            n_in = args[3]
            if n_in.shape != (3, N_in):
                raise ValueError(f"n_in must be of shape (3, N_in), found {n_in.shape}")
        periodicity = args[3 + self.takes_normals]
        box = args[4 + self.takes_normals]
        if isinstance(box, list):
            box = np.asarray(box)
        tol = args[5 + self.takes_normals]
        execution_space = get_execution_space(args[6 + self.takes_normals])
        am = get_array_module(execution_space)
        dtype = x_in.dtype

        ### Spectral Ewald ###
        device_pre, params = DevicePre.from_particles(
            targets=x_out,
            sources=x_in,
            forces=q_in,
            normals=n_in,
            kernel=self.kernel,
            box=box,
            tolerance=tol,
            cell_size=cell_size,
            rc=rc,
            periodicity=periodicity,
            execution_space=execution_space,
            fft_type=fft_type,
            distributed=True,
        )
        # algorithm
        walltime = {}
        walltime["p2p"] = p2p(
            device_pre,
            method=p2p_method,
        )
        walltime["p2g"] = p2g(
            device_pre,
            method=p2g_method,
        )
        # TODO: allow for plan to be passed in if needed
        #   This will allow for speedups for repeated calls
        fft_buffers = FFTMPBuffers(
            fft_shape=device_pre.data.fft_shape,
            ifft_shape=device_pre.data.Hg.shape[1:-1],
            fft_type=device_pre.data.fft_type,
        )
        walltime["fft"] = fft(device_pre, None, fft_buffers)
        walltime["cnv"] = cnv(device_pre)
        walltime["ifft"] = ifft(device_pre, None, fft_buffers)
        del fft_buffers  # free symmetic heap if execution is distributed
        device_pre.data.communicate_ghost_grid_cells()
        walltime["g2p"] = g2p(
            device_pre,
            method=g2p_method,
        )
        walltime.update(
            device_pre.data.walltime
        )  # update walltimes with device_data comms
        val = device_pre.near_potential + device_pre.far_potential
        val = val.squeeze()
        shape = self.get_shape_out(device_pre.data.targets.shape[-1])
        if val.shape != shape:
            raise ValueError(
                f"Kernel function returned wrong shape, found {val.shape}, expected {shape}."
            )

        x_off = device_pre.data.box_dict["left"] - (device_pre.data.opt.ghost_dist)
        trg = device_pre.data.targets
        trg[0] += x_off
        if time:
            return val, trg, walltime
        else:
            return val, trg

    def get_shape_in(self, N_in):
        """
        Return the input shape expected by this kernel for `N_in` source points.

        This will be `(N_in,)` if `self.dim_in==1`, and `(self.dim_in, N_in)`
        otherwise.
        """
        if self.dim_in == 1:
            return (N_in,)
        return (self.dim_in, N_in)

    def get_shape_out(self, N_out):
        """
        Return the output shape expected by this kernel for `N_out` source points.

        This will be `(N_out,)` if `self.dim_out==1`, and `(self.dim_out,N_out)`
        otherwise.
        """
        if self.dim_out == 1:
            return (N_out,)
        return (self.dim_out, N_out)


@dataclass
class DistributedEwaldOptions(EwaldOptions):
    """
    Distributed Ewald Options
    """


## Implementations of specific kernels ##


def stokes_1p(*args, **kwargs):
    r"""
    Compute the Stokes single layer potential

    .. math::

        \boldsymbol{u}(\boldsymbol{x}_i) = \sum_{j=1}^{N} \sum_{p \in P}
        \left(\left( \frac{\boldsymbol{q}_j}{\lVert \boldsymbol{r}_{ij} \rVert}
        +
        \frac{\boldsymbol{r}_{ij}}{\lVert \boldsymbol{r}_{ij} \rVert^3}
        (\boldsymbol{r}_{ij} \cdot \boldsymbol{q}_j)
        \right)
        + \left( \epsilon_{jlm} \frac{\boldsymbol{r}_m}{\|\boldsymbol{r}_{ij}\|^3}
        \boldsymbol{q}_l \boldsymbol{n}_m \right) \right),

    where :math:`\boldsymbol{r}_{ij} = \boldsymbol{x}_i - \boldsymbol{y}_j`
    and :math:`P` is the specified periodicity.

    """

    pot = EwaldKernel(
        name="stokes_comb",
        dim_in=3,
        dim_out=3,
        kernel="stokes_sl",
        takes_normals=False,
        description="Stokes single layer potential.",
    )(*args, **kwargs)
    return pot


def stokes_comb(*args, **kwargs):
    r"""
    Compute the combined Stokes single and double layer potential

    .. math::

        \boldsymbol{u}(\boldsymbol{x}_i) = \sum_{j=1}^{N} \sum_{p \in P}
        \left(\left( \frac{\boldsymbol{q}_j}{\lVert \boldsymbol{r}_{ij} \rVert}
        +
        \frac{\boldsymbol{r}_{ij}}{\lVert \boldsymbol{r}_{ij} \rVert^3}
        (\boldsymbol{r}_{ij} \cdot \boldsymbol{q}_j)
        \right)
        + \left( \epsilon_{jlm} \frac{\boldsymbol{r}_m}{\|\boldsymbol{r}_{ij}\|^3}
        \boldsymbol{q}_l \boldsymbol{n}_m \right) \right),

    where :math:`\boldsymbol{r}_{ij} = \boldsymbol{x}_i - \boldsymbol{y}_j`
    and :math:`P` is the specified periodicity.

    """

    pot = EwaldKernel(
        name="stokes_comb",
        dim_in=6,
        dim_out=3,
        kernel="stokes_comb",
        takes_normals=True,
        description="Stokes single and double layer potential.",
    )(*args, **kwargs)
    return pot
