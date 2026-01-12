"""
Module for the parkipy.CellList class.
"""

import math
import pykokkos as pk

from .utils import get_execution_space, get_array_module
from ._pk_kernels._celllist import (
    count_particles_fp32,
    count_particles_fp64,
    reshuffle_particles_fp32,
    reshuffle_particles_fp64,
    reshuffle_forces_fp32,
    reshuffle_forces_fp64,
    get_nonempty_neighbors,
)


class CellList:
    """
    A class to generate cell lists given
    particle coordinates, a cutoff radius,
    and a computational box.

    This class automaticaly generates a CellList
    upon instantiation as well as relevant counters
    and mappings.

    Cell lists are performance portable, with
    target devices specified by a PyKokkos
    execution space object.
    """

    def __init__(
        self,
        particles,
        cutoff,
        box,
        execution_space=None,
        forces=None,
        padding_factor=1,
        skip_empty_cells=True,
    ):
        """
        Creates a new CellList object.

        Input:
            - `particles`. Particle coordinates. Expected to be of shape
              `(d, n)`, where `n` is the number of particles and `d`
              is the dimension.
            - `cutoff`: Cell cutoff radius. Radius must satisfy
              `cutoff < min(box)`.
            - `box`: Side lengths of the computational box.
              We expect `0 <= particles[d] <= box[d]` for all d,
              but we do not explicitly enforce this assumption.
            - `execution_space`: PyKokkos execution space See
              https://kokkos.org/kokkos-core-wiki/API/core/execution_spaces.html
              for available execution spaces. Input is either a string
              (e.g., 'OpenMP') or a PyKokkos object (e.g., pk.OpenMP).
              Defaults to `pk.get_default_space()`
            - `forces`: Optional force, or tuple of forces, ascribed to
              a particle, such that each force array is of shape `(k,n)`,
              where the dimension `k` is specific to the force.
            - `padding_factor`: Positive integer such that the common cell list
              size will be a multiple of `padding_factor`. Defualts to `1`.
            - `skip_empty_cells`: If false, every cell is treated as nonempty.

        Cell lists are of size `self.num_nonempty_cells * self.cell_size`,
        where the *nonempty* cell index ranges the slowest and the particles
        in cell index ranges the fastes.
        """
        # parse inputs
        self._dtype = particles.dtype
        self._execution_space = get_execution_space(execution_space)
        self._am = get_array_module(execution_space)
        self._particles = particles
        if not isinstance(self.particles, self.am.ndarray):
            raise ValueError(
                f"particles expected to be {self.am.ndarray} "
                "but are type {type(self.particles)}"
            )
        self._box = self.am.asarray(box, dtype=self.dtype)
        if self.box.size != 3:
            raise ValueError(f"expected box of shape (3,), got {box.shape}")
        self._cutoff = cutoff
        if not isinstance(self.cutoff, float) or not (
            0 <= self.cutoff <= self.box.min()
        ):
            raise ValueError(
                f"cutoff expected to be a float between {(0, self.box.min())}, "
                f"got {self.cutoff} of type {type(cutoff)}"
            )
        self._forces = forces
        self._skip_empty_cells = skip_empty_cells

        # get local variables and check for value errors
        if self.dtype is float and self.particles.min() < 1e-8:
            raise ValueError(
                "smallest particle coordinate {self.particles.min()} is less than machine precision 1e-8."
            )
        if self.particles.ndim != 2:
            raise ValueError(
                f"particles expected to have shape (d,n), got shape f{particles.shape}"
            )
        d, n = self.particles.shape
        if d != 3:
            raise ValueError(
                "CellList only supported for particles in dimension 3," f" got {d}."
            )
        if self.particles.dtype == self.am.float32:
            counter_workunit = count_particles_fp32
            reshuffle_particles_workunit = reshuffle_particles_fp32
            reshuffle_forces_workunit = reshuffle_forces_fp32
        elif self.particles.dtype == self.am.float64:
            counter_workunit = count_particles_fp64
            reshuffle_particles_workunit = reshuffle_particles_fp64
            reshuffle_forces_workunit = reshuffle_forces_fp64
        else:
            raise TypeError(
                "particles dtype must be `float32` or `float64`,"
                f" got {self.particles.dtype}."
            )
        if not isinstance(padding_factor, int) or padding_factor < 1:
            raise ValueError(
                "Expected `padding_factor` to be a positive integer,"
                f" got {padding_factor}."
            )

        # count particles in cells
        self._cell_grid_shape = [int(self.box[i] / self.cutoff) for i in range(3)]
        self._num_cells = int(self.am.prod(self.am.array(self.cell_grid_shape)))
        self._counter = self.am.zeros(shape=self.num_cells, dtype=self.am.int32)
        pk.parallel_for(
            "count particles in each cell",
            pk.RangePolicy(self.execution_space, 0, n),
            counter_workunit,
            counter=self._counter,
            p=self.particles,
            rc=self.cutoff,
            box=self.box,
        )

        # create lists
        max_cell_size = self.am.max(self.counter).item()
        self._cell_size = math.ceil(max_cell_size / padding_factor) * padding_factor
        self._nonempty_cells = self.am.nonzero(self.counter)[0].astype(self.am.int32)
        if not self.skip_empty_cells:
            self._nonempty_cells = self.am.arange(self.num_cells, dtype=self.am.int32)
        self._num_nonempty_cells = len(self.nonempty_cells)
        self._nonempty_cell_index = self.am.full(
            self.num_cells, -1, dtype=self.am.int32
        )
        self._nonempty_cell_index[self.nonempty_cells] = self.am.arange(
            self.num_nonempty_cells
        )
        self._create_nonempty_neighbors()
        list_len = self.num_nonempty_cells * self.cell_size
        self._particle_list = self.am.full(
            shape=(d, list_len), fill_value=-1, dtype=self.particles.dtype
        )
        self._particle_index = self.am.full(
            shape=list_len, fill_value=-1, dtype=self.am.int32
        )
        self._counter[:] = 0
        pk.parallel_for(
            "reshuffle particles into cells",
            pk.RangePolicy(self.execution_space, 0, n),
            reshuffle_particles_workunit,
            p_list=self._particle_list,
            counter=self._counter,
            l2g=self._particle_index,
            p=self.particles,
            cell2nz=self.nonempty_cell_index,
            rc=self.cutoff,
            box=self.box,
            cell_size=self.cell_size,
            dp=d,
        )
        self._force_list = None
        if isinstance(self.forces, tuple):
            out = []
            for force in forces:
                if not force.dtype is self.dtype:
                    raise TypeError(
                        f"force data type should be {self.dtype}, got {force.dtype}."
                    )
                if len(force) == 1:
                    force = force.reshape(1, -1)
                df, nf = force.shape
                if nf != n:
                    raise ValueError(
                        "force expected to have the same `n` dimension as"
                        f" particles {n}, got {nf}."
                    )
                force_list = self.am.zeros(
                    shape=(df, list_len), dtype=self.particles.dtype
                )
                pk.parallel_for(
                    "reshuffle particles and foces into cells",
                    pk.RangePolicy(self.execution_space, 0, list_len),
                    reshuffle_forces_workunit,
                    q_list=force_list,
                    l2g=self.particle_index,
                    q=force,
                    dq=df,
                )
                out.append(force_list)
            self._force_list = tuple(out)
        elif isinstance(self.forces, self.am.ndarray):
            if len(forces.shape) == 1:
                forces = forces.reshape(1, -1)
            df, nf = forces.shape
            if not forces.dtype is self.dtype:
                raise TypeError(
                    f"forces data type should be {self.dtype}, got {forces.dtype}."
                )
            if nf != n:
                raise ValueError(
                    "force expected to have the same `n` dimension as"
                    f" particles {n}, got {nf}"
                )
            force_list = self.am.zeros(shape=(df, list_len), dtype=self.particles.dtype)
            pk.parallel_for(
                "reshuffle particles and foces into cells",
                pk.RangePolicy(self.execution_space, 0, list_len),
                reshuffle_forces_workunit,
                q_list=force_list,
                l2g=self.particle_index,
                q=forces,
                dq=df,
            )
            self._force_list = force_list
        elif self.forces is not None:
            raise ValueError(
                "forces expected to be a tuple of `ndarray`s"
                f" or an `ndarray`, got {type(forces)}"
            )

    @property
    def dtype(self):
        """
        floating point data dtype,
        same as particles.dtype.
        Read-only.
        """
        return self._dtype

    @property
    def force_list(self):
        """
        If `self.forces` is a tuple,
        `self.force_list` is a tuple
        containg a cell list for each force
        in the tuple. Else, `self.force_list`
        is just a single force list. In either case,
        forces associated to "ghost" particles
        are set to 0.
        Read-only.
        """
        return self._force_list

    @property
    def particle_index(self):
        """
        Integer array such that the value of index `i` is the
        index of the particle at `self.particle_list[i]` in the
        origional array. If `-1`, then `self.particle_list[i]` is
        a 'ghost particle' and should be ignored. Read-only.
        """
        return self._particle_index

    @property
    def skip_empty_cells(self):
        """
        Boolean flag to skip empty cells in particle (and force) cell list.
        Read-only.
        """
        return self._skip_empty_cells

    @property
    def particle_list(self):
        """
        Particle cell list. Read-only.
        """
        return self._particle_list

    @property
    def nonempty_cell_index(self):
        """
        Array of size `num_cells` whoes value at
        index `i`, if nonnegative, corresponds to
        the nonempty cell index. If the value at
        index `i` is negative, cell `i` is empty.
        If `self.skip_empty_cells==False`, then
        every cell is treated as nonempty hence
        `self.nonempty_cell_index==self.am.arange(self.num_cells)`.
        Read-only.
        """
        return self._nonempty_cell_index

    @property
    def num_nonempty_cells(self):
        """
        The number of nonempty cells.
        If `self.skip_empty_cells==False`,
        then `self.num_nonempty_cells=self.num_cells`.
        Read-only.
        """
        return self._num_nonempty_cells

    @property
    def nonempty_cells(self):
        """
        Array of the nonzero indices of `self.counter`.
        Given an nonempty cell index `j`,
        `self.nonempty_cells[j]` returns `i`, the global
        cell index.
        If `self.skip_empty_cells==False`,
        then `self.num_nonempty_cells=np.arange(self.num_cells)`,
        i.e., nonempty and global cell indices are the same.
        Read-only.
        """
        return self._nonempty_cells

    @property
    def nonempty_neighbors(self):
        """
        Return an array of shape `(self.num_cells, 27)`
        such that given a (global) cell index `i`,
        return the (nonempty) cell indices of it's 27
        neighboring cells.
        Read-only
        """
        return self._nonempty_neighbors

    @property
    def cell_size(self):
        """
        Size of each cell. Read-only.
        """
        return self._cell_size

    @property
    def counter(self):
        """
        An array of size `n` containing the number of particles per cell,
        where `n` is the number of particles.
        Read-only.
        """
        return self._counter

    @property
    def cell_grid_shape(self):
        """
        The shape of the cell grid, i.e.,
        the total number of cells in each
        direction. Read-only.
        """
        return self._cell_grid_shape

    @property
    def num_cells(self):
        """
        The total number of cells. Read-only.
        """
        return self._num_cells

    @property
    def particles(self):
        """
        The particle array used by this module. Read-only.
        """
        return self._particles

    @property
    def cutoff(self):
        """
        The cell list cutoff radius. Read-only.
        """
        return self._cutoff

    @property
    def box(self):
        """
        The computational box side lengths. Read-only.
        """
        return self._box

    @property
    def forces(self):
        """
        List of forces for each particle. Read-only.
        """
        return self._forces

    @property
    def execution_space(self):
        """
        PyKokkos execution space. Read-only.
        """
        return self._execution_space

    @property
    def am(self):
        """
        Array module (NumPy/CuPu). Determined via `self.execution_space`. Read-only.
        """
        return self._am

    def _create_nonempty_neighbors(self):
        """
        For each (global) cell index, create a list of the
        nonempty cell index of the 27 3d neighbors.
        """
        grid_area = self.cell_grid_shape[1] * self.cell_grid_shape[2]
        nonempty_neighbors = self.am.full(
            shape=(self.num_cells, 3, 3, 3), fill_value=-1, dtype=self.am.int32
        )
        pk.parallel_for(
            "get nonempty neighbors",
            pk.RangePolicy(self.execution_space, 0, self.num_cells),
            get_nonempty_neighbors,
            grid_area=grid_area,
            cell_grid_shape_0=self.cell_grid_shape[0],
            cell_grid_shape_1=self.cell_grid_shape[1],
            cell_grid_shape_2=self.cell_grid_shape[2],
            nonempty_cell_index=self.nonempty_cell_index,
            nonempty_neighbors=nonempty_neighbors,
        )
        self._nonempty_neighbors = nonempty_neighbors.reshape(self.num_cells, 27)
