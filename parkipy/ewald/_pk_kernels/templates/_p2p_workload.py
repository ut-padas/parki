import math
import pykokkos as pk
from typing import List


# START TEMPLATE CELL
@pk.classtype
class Cell_CELL:
    def __init__(self):
        self.x: pk.double = 0.0
        self.y: pk.double = 0.0
        self.z: pk.double = 0.0

        self.x_shift: pk.double = 0.0
        self.y_shift: pk.double = 0.0
        self.z_shift: pk.double = 0.0

        self.inbounds: bool = True


# END TEMPLATE CELL


# START TEMPLATE REAL3D
@pk.classtype
class Real3d_REAL3D:
    def __init__(self):
        self.x: pk.double = 0.0
        self.y: pk.double = 0.0
        self.z: pk.double = 0.0


# END TEMPLATE REAL3D


# START TEMPLATE P2PWORKLOAD
class p2p_workload_P2PWORKLOAD:
    def __init__(
        self,
        potentials,  # (3, ns) LayoutRight
        t_counter,
        targets_list,  # (3, nt) LayoutRight
        s_counter,
        sources_list,  # (3, ns) LayoutRight
        forces_list,  # (3, ns) LayoutRight
        normals_list,  # (3, ns) LayoutRight
        t_list2global,
        nz2t_cell_map,
        nz2s_cell_map,
        t2nz_cell_map,
        s2nz_cell_map,
        box,
        periodicity,
        xi,
        rc,
        has_sl,
        has_dl,
        has_ewald,
        nnz_t_cells,
        nnz_s_cells,
        t_cell_size,
        s_cell_size,
        threads,
        t_per_thread,
        s_per_thread,
        vector_size,
        variant,
        kernel,
    ):
        # output array
        self.potentials: pk.View2D[pk.double] = potentials
        self.dim_out: int = self.potentials.shape[0]

        # input arrays
        self.t_counter: pk.View1D[int] = t_counter
        self.targets_list: pk.View2D[pk.double] = targets_list
        self.s_counter: pk.View1D[int] = s_counter
        self.sources_list: pk.View2D[pk.double] = sources_list
        self.forces_list: pk.View2D[pk.double] = forces_list
        self.normals_list: pk.View2D[pk.double] = normals_list
        self.t_list2global: pk.View1D[int] = t_list2global
        self.nz2t_cell_map: pk.View1D[int] = nz2t_cell_map
        self.nz2s_cell_map: pk.View1D[int] = nz2s_cell_map
        self.t2nz_cell_map: pk.View1D[int] = t2nz_cell_map
        self.s2nz_cell_map: pk.View1D[int] = s2nz_cell_map

        # input parameters
        self.box: List[pk.double] = box
        self.periodicity: int = periodicity
        self.xi: pk.double = xi
        self.rc: pk.double = rc
        self.has_sl: bool = has_sl
        self.has_dl: bool = has_dl
        self.has_ewald: bool = has_ewald

        # input arguments
        self.nnz_t_cells: int = nnz_t_cells
        self.nnz_s_cells: int = nnz_s_cells
        self.t_cell_size: int = t_cell_size
        self.s_cell_size: int = s_cell_size
        self.threads: int = threads
        self.t_per_thread: int = t_per_thread
        self.s_per_thread: int = s_per_thread
        self.vector_size: int = vector_size  # must be 1 if using 1d kernel
        self.variant: pk.uint8 = variant
        self.kernel: pk.uint8 = kernel

        # determined arguments
        self.rc_squared: pk.double = self.rc * self.rc
        self.xi_squared: pk.double = self.xi * self.xi
        self.box_x: pk.double = box[0]
        self.box_y: pk.double = box[1]
        self.box_z: pk.double = box[2]
        self.num_cells_x: int = int(self.box_x / rc)
        self.num_cells_y: int = int(self.box_y / rc)
        self.num_cells_z: int = int(self.box_z / rc)
        self.cell_grid_area: int = self.num_cells_y * self.num_cells_z
        self.num_cells: int = self.num_cells_x * self.cell_grid_area
        self.t_threads: int = math.ceil(self.threads / self.vector_size)
        self.t_cell_chunk_size: int = self.t_threads * self.t_per_thread
        self.s_cell_chunk_size: int = self.vector_size * self.s_per_thread
        self.s_cell_chunk_size_in: int = self.threads
        self.t_cell_chunks: int = math.ceil(self.t_cell_size / self.t_cell_chunk_size)
        self.s_cell_chunks: int = math.ceil(self.s_cell_size / self.s_cell_chunk_size)
        self.s_cell_chunks_in: int = math.ceil(
            self.s_cell_size / self.s_cell_chunk_size_in
        )
        self.t_cell_chunk_threads: int = math.ceil(
            self.t_cell_chunk_size / self.vector_size
        )
        self.s_cell_chunk_threads: int = math.ceil(
            self.s_cell_chunk_size / self.vector_size
        )
        self.s_cell_threads: int = math.ceil(self.s_cell_size / self.vector_size)

        # constants
        self.inv_sqrt_pi: pk.double = 0.564189583547756286948079  # = 1/sqrt(pi)
        self.two_inv_sqrt_pi: pk.double = 2 * self.inv_sqrt_pi
        self.xi_two_inv_sqrt_pi: pk.double = self.xi * self.two_inv_sqrt_pi
        self.m_xi_squared_2: pk.double = self.xi_squared * 2
        self.C_term1: pk.double = self.xi_squared * self.xi * self.inv_sqrt_pi
        self.inv_8_pi: pk.double = 0.0397887357729738339422209  # = 1/(8*pi)
        self.inv_4_pi: pk.double = 2 * self.inv_8_pi
        self.c1: pk.double = 1.3333333333333333
        self.c2: pk.double = 0.8
        self.c3: pk.double = 0.2857142857142857
        self.c4: pk.double = 0.07407407407407407
        self.m_c1_C_term1: pk.double = self.c1 * self.C_term1

    @pk.main
    def run(self):
        if self.variant == 0:
            if self.vector_size == 1:
                if self.kernel == 1:
                    pk.parallel_for(
                        "P2P-DIST-POINT",
                        pk.TeamPolicy(
                            self.nnz_t_cells * self.t_cell_chunks, self.t_threads
                        ),
                        self.p2p_point_distance,
                    )
                    return
                if self.kernel == 2:
                    pk.parallel_for(
                        "P2P-LAPLACE-POINT",
                        pk.TeamPolicy(
                            self.nnz_t_cells * self.t_cell_chunks, self.t_threads
                        ),
                        self.p2p_point_laplace,
                    )
                    return
                if not self.has_ewald and self.has_sl and not self.has_dl:
                    pk.parallel_for(
                        "P2P-SL-POINT",
                        pk.TeamPolicy(
                            self.nnz_t_cells * self.t_cell_chunks, self.t_threads
                        ),
                        self.p2p_point_stokes_sl,
                    )
                    return
                pk.parallel_for(
                    "P2P-GM-1D (stokes comb)",
                    pk.TeamPolicy(
                        self.nnz_t_cells * self.t_cell_chunks, self.t_threads
                    ),
                    self.p2p_gm_1d_stokes_comb,
                )
                return
                pk.parallel_for(
                    "P2P-POINT",
                    self.nnz_t_cells * self.t_cell_size,
                    self.p2p_point_range,
                )
            else:
                pk.parallel_for(
                    "P2P-GM-2D (stokes comb)",
                    pk.TeamPolicy(
                        self.nnz_t_cells * self.t_cell_chunks,
                        self.t_threads,
                        self.vector_size,
                    ),
                    self.p2p_gm_2d_stokes_comb,
                )
        elif self.variant == 1:
            pk.parallel_for(
                "P2P-POINT-IN",
                pk.TeamPolicy(self.nnz_s_cells * self.s_cell_chunks_in, self.threads),
                self.p2p_point_in,
            )
            return
            pk.parallel_for(
                "P2P-POINT-IN",
                self.nnz_s_cells * self.s_cell_size,
                self.p2p_point_in_range,
            )
        elif self.variant == 2:
            shmem_idt: int = pk.ScratchView1D[int].shmem_size((self.t_cell_chunk_size))
            shmem_t: int = pk.ScratchView1D[pk.double].shmem_size(
                (self.t_cell_chunk_size) * 3
            )
            shmem_s: int = pk.ScratchView1D[pk.double].shmem_size(
                (self.s_cell_chunk_size) * (3 + 3 * self.has_sl + 6 * self.has_dl)
            )
            if self.vector_size == 1:
                if self.kernel == 1:
                    pk.parallel_for(
                        "P2P-CELL-OUT",
                        pk.TeamPolicy(
                            self.nnz_t_cells * self.t_cell_chunks, self.t_threads
                        ).set_scratch_size(
                            0, pk.PerTeam(shmem_idt + shmem_t + shmem_s)
                        ),
                        self.p2p_cell_out_distance,
                    )
                    return
                if self.kernel == 2:
                    pk.parallel_for(
                        "P2P-CELL-OUT",
                        pk.TeamPolicy(
                            self.nnz_t_cells * self.t_cell_chunks, self.t_threads
                        ).set_scratch_size(
                            0, pk.PerTeam(shmem_idt + shmem_t + shmem_s)
                        ),
                        self.p2p_cell_out_laplace,
                    )
                    return
                if not self.has_ewald and self.has_sl and not self.has_dl:
                    pk.parallel_for(
                        "P2P-CELL-OUT",
                        pk.TeamPolicy(
                            self.nnz_t_cells * self.t_cell_chunks, self.t_threads
                        ).set_scratch_size(
                            0, pk.PerTeam(shmem_idt + shmem_t + shmem_s)
                        ),
                        self.p2p_cell_out_stokes_sl,
                    )
                    return
                pk.parallel_for(
                    "P2P-SM-1D (stokes comb)",
                    pk.TeamPolicy(
                        self.nnz_t_cells * self.t_cell_chunks, self.t_threads
                    ).set_scratch_size(0, pk.PerTeam(shmem_idt + shmem_t + shmem_s)),
                    self.p2p_sm_1d_stokes_comb,
                )
            else:
                pk.parallel_for(
                    "P2P-SM-2D (stokes comb)",
                    pk.TeamPolicy(
                        self.nnz_t_cells * self.t_cell_chunks,
                        self.t_threads,
                        self.vector_size,
                    ).set_scratch_size(0, pk.PerTeam(shmem_idt + shmem_t + shmem_s)),
                    self.p2p_sm_2d_stokes_comb,
                )
        else:
            pk.printf("Variant %d not implemented, returning!\n", self.variant)
            return

    @pk.function
    def dot_fp64(self, v: Real3d_fp64, w: Real3d_fp64) -> pk.double:
        return v.x * w.x + v.y * w.y + v.z * w.z

    @pk.function
    def get_source_cell(
        self, k: int, t_cell_x: int, t_cell_y: int, t_cell_z: int
    ) -> Cell_fp64:
        offsets: List[int] = [
            -1,-1,-1,
            -1,-1,0,
            -1,-1,1,
            -1,0,-1,
            -1,0,0,
            -1,0,1,
            -1,1,-1,
            -1,1,0,
            -1,1,1,
            0,-1,-1,
            0,-1,0,
            0,-1,1,
            0,0,-1,
            0,0,0,
            0,0,1,
            0,1,-1,
            0,1,0,
            0,1,1,
            1,-1,-1,
            1,-1,0,
            1,-1,1,
            1,0,-1,
            1,0,0,
            1,0,1,
            1,1,-1,
            1,1,0,
            1,1,1,
        ]
        dx: pk.double = offsets[k * 3]
        dy: pk.double = offsets[k * 3 + 1]
        dz: pk.double = offsets[k * 3 + 2]

        source_cell: Cell_fp64 = Cell_fp64()

        # x coord
        source_cell.x = t_cell_x + dx
        if source_cell.x < 0:
            if self.periodicity >= 1:
                source_cell.x += self.num_cells_x
                source_cell.x_shift = -self.box_x
            else:
                source_cell.inbounds = False
        if source_cell.x >= self.num_cells_x:
            if self.periodicity >= 1:
                source_cell.x -= self.num_cells_x
                source_cell.x_shift = self.box_x
            else:
                source_cell.inbounds = False

        # y coord
        source_cell.y = t_cell_y + dy
        if source_cell.y < 0:
            if self.periodicity >= 2:
                source_cell.y += self.num_cells_y
                source_cell.y_shift = -self.box_y
            else:
                source_cell.inbounds = False
        if source_cell.y >= self.num_cells_y:
            if self.periodicity >= 2:
                source_cell.y -= self.num_cells_y
                source_cell.y_shift = self.box_y
            else:
                source_cell.inbounds = False

        # z coord
        source_cell.z = t_cell_z + dz
        if source_cell.z < 0:
            if self.periodicity >= 3:
                source_cell.z += self.num_cells_z
                source_cell.z_shift = -self.box_z
            else:
                source_cell.inbounds = False
        if source_cell.z >= self.num_cells_z:
            if self.periodicity >= 3:
                source_cell.z -= self.num_cells_z
                source_cell.z_shift = self.box_z
            else:
                source_cell.inbounds = False

        return source_cell

    @pk.function
    def stokes_sl_ewald_fp64(
        self,
        u: Real3d_fp64,
        r: Real3d_fp64,
        f1: Real3d_fp64,
        d: pk.double,
        od: pk.double,
        od2: pk.double,
        A: pk.double,
        B: pk.double,
        C: pk.double,
    ) -> Real3d_fp64:
        # Punctured trapezoidal rule
        # Remove point where r==0 (1e-14 is ad hoc)
        s1: pk.double = (d >= 1e-14) * od
        s2: pk.double = (d >= 1e-14) * od * od2
        # Sum up all terms
        tmp: pk.double = s1 - B - A
        t1_x: pk.double = tmp * f1.x
        t1_y: pk.double = tmp * f1.y
        t1_z: pk.double = tmp * f1.z
        tmp = self.dot_fp64(r, f1) * (s2 - C)
        t2_x: pk.double = tmp * r.x
        t2_y: pk.double = tmp * r.y
        t2_z: pk.double = tmp * r.z
        u.x += t1_x + t2_x
        u.y += t1_y + t2_y
        u.z += t1_z + t2_z
        return u

    @pk.function
    def stokes_dl_ewald_fp64(
        self,
        u: Real3d_fp64,
        r: Real3d_fp64,
        f2: Real3d_fp64,
        n: Real3d_fp64,
        d: pk.double,
        od: pk.double,
        od2: pk.double,
        B: pk.double,
        C: pk.double,
    ) -> Real3d_fp64:
        r_dot_f2: pk.double = self.dot_fp64(r, f2)
        r_dot_n: pk.double = self.dot_fp64(r, n)
        f2_dot_n: pk.double = self.dot_fp64(f2, n)
        r_dot_f2_r_dot_n: pk.double = r_dot_f2 * r_dot_n
        # Singular part, i.e., terms coming from the full (free-space) stresslet
        # Punctured trapezoidal rule
        # Remove point where r==0 (1e-14 is ad hoc)
        s1: pk.double = (d >= 1e-14) * od * od2 * od2
        # Sum up all terms
        D: pk.double = self.m_xi_squared_2 * B
        # At r==0, this part will become zero (1e-14 is ad hoc)
        tmp: pk.double = (d >= 1e-14) * ((6 * C - 2 * D) * r_dot_f2_r_dot_n * od2)
        tmp += -6 * s1 * r_dot_f2_r_dot_n + D * (f2_dot_n)
        t1_x: pk.double = tmp * r.x
        t1_y: pk.double = tmp * r.y
        t1_z: pk.double = tmp * r.z
        tmp = D * r_dot_n
        t2_x: pk.double = tmp * f2.x
        t2_y: pk.double = tmp * f2.y
        t2_z: pk.double = tmp * f2.z
        tmp = D * r_dot_f2
        u.x += t1_x + t2_x + (tmp * n.x)
        u.y += t1_y + t2_y + (tmp * n.y)
        u.z += t1_z + t2_z + (tmp * n.z)
        return u

    @pk.function
    def stokes_comb_ewald_fp64(
        self,
        u: Real3d_fp64,
        r: Real3d_fp64,
        f1: Real3d_fp64,
        f2: Real3d_fp64,
        n: Real3d_fp64,
        d2: pk.double,
        d: pk.double,
        od: pk.double,
        od2: pk.double,
    ) -> Real3d_fp64:
        xid: pk.double = self.xi * d
        xid2: pk.double = self.xi_squared * d2
        # Replace A by its limit close to zero (relative error at 1e-14
        # should be around 4e-29, so this is very accurate)
        # Reformulation of if/else statement
        A: pk.double = (xid < 1e-14) * (self.xi_two_inv_sqrt_pi) + (xid >= 1e-14) * (
            pk.erf(xid) * od
        )
        B: pk.double = self.xi * pk.exp(-xid2) * self.two_inv_sqrt_pi
        # Replace C by its Taylor expansion close to zero
        # (relative error at most around 3e-13 for four terms)
        # NOTE: This might be a bit unnecessary, as C goes into terms
        # that will anyway go to zero as r -> 0. But we do it
        # anyway for good measure.
        # Reformulation of if/else statement
        term2: pk.double = xid2 * self.C_term1
        term3: pk.double = xid2 * term2
        C: pk.double = (xid < 4.75e-2) * (
            self.m_c1_C_term1
            - self.c2 * term2
            + self.c3 * term3
            - self.c4 * (xid2 * term3)
        ) + (xid >= 4.75e-2) * ((A - B) * od2)
        # single layer update
        u = self.stokes_sl_ewald_fp64(u, r, f1, d, od, od2, A, B, C)
        if self.has_dl:
            u = self.stokes_dl_ewald_fp64(u, r, f2, n, d, od, od2, B, C)
        return u

    @pk.function
    def laplace_ewald_fp64(self, u: Real3d_fp64, r: Real3d_fp64, f: Real3d_fp64, d2: pk.double) -> Real3d_fp64:
        TWO_OVER_RSQRT_PI: pk.double = 1.1283791670955126
        d: pk.double = pk.sqrt(d2)
        od: pk.double = 1.0 / d
        od = (od - od) + od
        od = pk.fmax(od, 0.0)
        xid: pk.double = self.xi * d
        ewald: pk.double = f.x * pk.erfc(xid) * od
        self: pk.double = (od == 0) * (-self.xi * TWO_OVER_RSQRT_PI * f.x)
        u.x += ewald + self
        return u



    @pk.workunit
    def p2p_point_distance(self, team_member: pk.TeamMember):
        t_off: int = team_member.league_rank() * self.t_cell_chunk_size

        def thread_loop(tid: int):
            t: int = t_off + tid
            if t >= self.nnz_t_cells * self.t_cell_size:
                return
            t_idx: int = self.t_list2global[t]
            if t_idx < 0:
                return
            t_x: pk.double = self.targets_list[0][t]
            t_y: pk.double = self.targets_list[1][t]
            t_z: pk.double = self.targets_list[2][t]
            nz_t_cell: int = t // self.t_cell_size
            t_cell: int = self.nz2t_cell_map[nz_t_cell]
            t_cell_x: int = t_cell // (self.cell_grid_area)
            t_cell_y: int = (t_cell % (self.cell_grid_area)) // self.num_cells_z
            t_cell_z: int = (t_cell % (self.cell_grid_area)) % self.num_cells_z
            pot_x: pk.double = 0
            pot_y: pk.double = 0
            pot_z: pk.double = 0
            for dx in range(-1, 2):
                s_cell_x: int = t_cell_x + dx
                x_shift: pk.double = 0
                # wrap in the periodic direction
                if s_cell_x < 0:
                    s_cell_x += self.num_cells_x
                    x_shift = -self.box_x
                elif s_cell_x >= self.num_cells_x:
                    s_cell_x -= self.num_cells_x
                    x_shift = self.box_x
                for dy in range(-1, 2):
                    s_cell_y: int = t_cell_y + dy
                    if s_cell_y < 0 or s_cell_y >= self.num_cells_y:
                        continue
                    for dz in range(-1, 2):
                        s_cell_z: int = t_cell_z + dz
                        if s_cell_z < 0 or s_cell_z >= self.num_cells_z:
                            continue
                        s_cell: int = (
                            s_cell_x * self.cell_grid_area
                            + s_cell_y * self.num_cells_z
                            + s_cell_z
                        )
                        ns_cell: int = self.s_counter[s_cell]
                        nz_s_cell: int = self.s2nz_cell_map[s_cell]
                        s_off: int = nz_s_cell * self.s_cell_size
                        for s in range(s_off, s_off + ns_cell):
                            r_x: pk.double = t_x - (self.sources_list[0][s]) - x_shift
                            r_y: pk.double = t_y - (self.sources_list[1][s])
                            r_z: pk.double = t_z - (self.sources_list[2][s])
                            pot_x += r_x
                            pot_y += r_y
                            pot_z += r_z

            self.potentials[0][t_idx] = pot_x
            self.potentials[1][t_idx] = pot_y
            self.potentials[2][t_idx] = pot_z

        pk.parallel_for(
            pk.TeamThreadRange(team_member, self.t_cell_chunk_size), thread_loop
        )

    @pk.workunit
    def p2p_point_laplace(self, team_member: pk.TeamMember):
        t_off: int = team_member.league_rank() * self.t_cell_chunk_size

        def thread_loop(tid: int):
            t: int = t_off + tid
            if t >= self.nnz_t_cells * self.t_cell_size:
                return
            t_idx: int = self.t_list2global[t]
            if t_idx < 0:
                return
            t_x: pk.double = self.targets_list[0][t]
            t_y: pk.double = self.targets_list[1][t]
            t_z: pk.double = self.targets_list[2][t]
            nz_t_cell: int = t // self.t_cell_size
            t_cell: int = self.nz2t_cell_map[nz_t_cell]
            t_cell_x: int = t_cell // (self.cell_grid_area)
            t_cell_y: int = (t_cell % (self.cell_grid_area)) // self.num_cells_z
            t_cell_z: int = (t_cell % (self.cell_grid_area)) % self.num_cells_z
            pot_x: pk.double = 0
            pot_y: pk.double = 0
            pot_z: pk.double = 0
            for dx in range(-1, 2):
                s_cell_x: int = t_cell_x + dx
                x_shift: pk.double = 0
                # wrap in the periodic direction
                if s_cell_x < 0:
                    s_cell_x += self.num_cells_x
                    x_shift = -self.box_x
                elif s_cell_x >= self.num_cells_x:
                    s_cell_x -= self.num_cells_x
                    x_shift = self.box_x
                for dy in range(-1, 2):
                    s_cell_y: int = t_cell_y + dy
                    if s_cell_y < 0 or s_cell_y >= self.num_cells_y:
                        continue
                    for dz in range(-1, 2):
                        s_cell_z: int = t_cell_z + dz
                        if s_cell_z < 0 or s_cell_z >= self.num_cells_z:
                            continue
                        s_cell: int = (
                            s_cell_x * self.cell_grid_area
                            + s_cell_y * self.num_cells_z
                            + s_cell_z
                        )
                        ns_cell: int = self.s_counter[s_cell]
                        nz_s_cell: int = self.s2nz_cell_map[s_cell]
                        s_off: int = nz_s_cell * self.s_cell_size
                        for s in range(s_off, s_off + ns_cell):
                            r_x: pk.double = t_x - (self.sources_list[0][s]) - x_shift
                            r_y: pk.double = t_y - (self.sources_list[1][s])
                            r_z: pk.double = t_z - (self.sources_list[2][s])
                            dist_squared: pk.double = r_x * r_x + r_y * r_y + r_z * r_z
                            # Check if source is within rc of target
                            if dist_squared > self.rc_squared:
                                continue
                            # TODO: change to George's method
                            over_dist: pk.double = pk.rsqrt(
                                (dist_squared != 0) * (dist_squared)
                                + (dist_squared == 0)
                            )
                            over_dist = (dist_squared != 0) * over_dist
                            pot_x += over_dist

            self.potentials[0][t_idx] = pot_x

        pk.parallel_for(
            pk.TeamThreadRange(team_member, self.t_cell_chunk_size), thread_loop
        )

    @pk.workunit
    def p2p_point_stokes_sl(self, team_member: pk.TeamMember):
        t_off: int = team_member.league_rank() * self.t_cell_chunk_size

        def thread_loop(tid: int):
            t: int = t_off + tid
            if t >= self.nnz_t_cells * self.t_cell_size:
                return
            t_idx: int = self.t_list2global[t]
            if t_idx < 0:
                return
            t_x: pk.double = self.targets_list[0][t]
            t_y: pk.double = self.targets_list[1][t]
            t_z: pk.double = self.targets_list[2][t]
            nz_t_cell: int = t // self.t_cell_size
            t_cell: int = self.nz2t_cell_map[nz_t_cell]
            t_cell_x: int = t_cell // (self.cell_grid_area)
            t_cell_y: int = (t_cell % (self.cell_grid_area)) // self.num_cells_z
            t_cell_z: int = (t_cell % (self.cell_grid_area)) % self.num_cells_z
            pot_x: pk.double = 0
            pot_y: pk.double = 0
            pot_z: pk.double = 0
            for dx in range(-1, 2):
                s_cell_x: int = t_cell_x + dx
                x_shift: pk.double = 0
                # wrap in the periodic direction
                if s_cell_x < 0:
                    s_cell_x += self.num_cells_x
                    x_shift = -self.box_x
                elif s_cell_x >= self.num_cells_x:
                    s_cell_x -= self.num_cells_x
                    x_shift = self.box_x
                for dy in range(-1, 2):
                    s_cell_y: int = t_cell_y + dy
                    if s_cell_y < 0 or s_cell_y >= self.num_cells_y:
                        continue
                    for dz in range(-1, 2):
                        s_cell_z: int = t_cell_z + dz
                        if s_cell_z < 0 or s_cell_z >= self.num_cells_z:
                            continue
                        s_cell: int = (
                            s_cell_x * self.cell_grid_area
                            + s_cell_y * self.num_cells_z
                            + s_cell_z
                        )
                        ns_cell: int = self.s_counter[s_cell]
                        nz_s_cell: int = self.s2nz_cell_map[s_cell]
                        s_off: int = nz_s_cell * self.s_cell_size
                        for s in range(s_off, s_off + ns_cell):
                            r_x: pk.double = t_x - (self.sources_list[0][s]) - x_shift
                            r_y: pk.double = t_y - (self.sources_list[1][s])
                            r_z: pk.double = t_z - (self.sources_list[2][s])
                            dist_squared: pk.double = r_x * r_x + r_y * r_y + r_z * r_z
                            # Check if source is within rc of target
                            if dist_squared > self.rc_squared:
                                continue
                            # TODO: change to George's method
                            over_dist: pk.double = pk.rsqrt(
                                (dist_squared != 0) * (dist_squared)
                                + (dist_squared == 0)
                            )
                            dist: pk.double = (dist_squared != 0) * (1 / over_dist)
                            over_dist = (dist_squared != 0) * over_dist
                            over_dist_squared: pk.double = over_dist * over_dist
                            # (kernel independent) ewald computations
                            A: pk.double = 0
                            B: pk.double = 0
                            C: pk.double = 0
                            f1_x: pk.double = self.forces_list[0][s]
                            f1_y: pk.double = self.forces_list[1][s]
                            f1_z: pk.double = self.forces_list[2][s]
                            # Punctured trapezoidal rule
                            # Remove point where r==0 (1e-14 is ad hoc)
                            s1: pk.double = (dist >= 1e-14) * over_dist
                            s2: pk.double = (
                                (dist >= 1e-14) * over_dist * over_dist_squared
                            )
                            # Sum up all terms
                            tmp: pk.double = s1 - B - A
                            t1_x: pk.double = tmp * f1_x
                            t1_y: pk.double = tmp * f1_y
                            t1_z: pk.double = tmp * f1_z
                            tmp = (r_x * f1_x + r_y * f1_y + r_z * f1_z) * (s2 - C)
                            t2_x: pk.double = tmp * r_x
                            t2_y: pk.double = tmp * r_y
                            t2_z: pk.double = tmp * r_z
                            pot_x += t1_x + t2_x
                            pot_y += t1_y + t2_y
                            pot_z += t1_z + t2_z

            self.potentials[0][t_idx] = pot_x
            self.potentials[1][t_idx] = pot_y
            self.potentials[2][t_idx] = pot_z

        pk.parallel_for(
            pk.TeamThreadRange(team_member, self.t_cell_chunk_size), thread_loop
        )

    @pk.workunit
    def p2p_gm_1d_stokes_comb(self, team_member: pk.TeamMember):
        t_off: int = team_member.league_rank() * self.t_cell_chunk_size

        def thread_loop(tid: int):
            t: int = t_off + tid
            if t >= self.nnz_t_cells * self.t_cell_size:
                return
            t_idx: int = self.t_list2global[t]
            if t_idx < 0:
                return
            trg: Real3d_fp64 = Real3d_fp64()
            trg.x = self.targets_list[0][t]
            trg.y = self.targets_list[1][t]
            trg.z = self.targets_list[2][t]
            t_x: pk.double = trg.x
            t_y: pk.double = trg.y
            t_z: pk.double = trg.z
            nz_t_cell: int = t // self.t_cell_size
            t_cell: int = self.nz2t_cell_map[nz_t_cell]
            t_cell_x: int = t_cell // (self.cell_grid_area)
            t_cell_y: int = (t_cell % (self.cell_grid_area)) // self.num_cells_z
            t_cell_z: int = (t_cell % (self.cell_grid_area)) % self.num_cells_z
            u: Real3d_fp64 = Real3d_fp64()
            for k in range(27):
                source_cell: Cell_fp64 = get_source_cell(
                    k, t_cell_x, t_cell_y, t_cell_z
                )
                if source_cell.inbounds == False:
                    continue
                s_cell: int = (
                    source_cell.x * self.cell_grid_area
                    + source_cell.y * self.num_cells_z
                    + source_cell.z
                )
                ns_cell: int = self.s_counter[s_cell]
                nz_s_cell: int = self.s2nz_cell_map[s_cell]
                s_off: int = nz_s_cell * self.s_cell_size
                for s in range(s_off, s_off + ns_cell):
                    r: Real3d_fp64 = Real3d_fp64()
                    r.x = trg.x - (self.sources_list[0][s]) - source_cell.x_shift
                    r.y = trg.y - (self.sources_list[1][s]) - source_cell.y_shift
                    r.z = trg.z - (self.sources_list[2][s]) - source_cell.z_shift
                    d2: pk.double = self.dot_fp64(r, r)
                    # Check if source is within rc of target
                    if d2 > self.rc_squared:
                        continue
                    # kernel dispatch
                    if kernel == 0: # stokes_comb_ewald
                        f1: Real3d_fp64 = Real3d_fp64()
                        f2: Real3d_fp64 = Real3d_fp64()
                        n: Real3d_fp64 = Real3d_fp64()
                        # TODO: change to George's method
                        od: pk.double = pk.rsqrt((d2 != 0) * (d2) + (d2 == 0))
                        d: pk.double = (d2 != 0) * (1 / od)
                        od = (d2 != 0) * od
                        od2: pk.double = od * od
                        f1.x = self.forces_list[0][s]
                        f1.y = self.forces_list[1][s]
                        f1.z = self.forces_list[2][s]
                        if self.has_dl:
                            f2.x = self.forces_list[3][s]
                            f2.y = self.forces_list[4][s]
                            f2.z = self.forces_list[5][s]
                            n.x = self.normals_list[0][s]
                            n.y = self.normals_list[1][s]
                            n.z = self.normals_list[2][s]
                        u = self.stokes_comb_ewald_fp64(u, r, f1, f2, n, d2, d, od, od2)
                    elif kernel == 3: # laplace_ewald
                        f: Real3d_fp64 = Real3d_fp64()
                        f.x = self.forces_list[0][s]
                        f.y = self.forces_list[1][s]
                        f.z = self.forces_list[2][s]
                        u = self.laplace_ewald_fp64(u, r, f, d2)
            u_lst: List[pk.double] = [u.x, u.y, u.z]
            for k in range(dim_out):
                self.potentials[k][t_idx] = u_lst[k]

        pk.parallel_for(
            pk.TeamThreadRange(team_member, self.t_cell_chunk_size), thread_loop
        )

    @pk.workunit
    def p2p_point_range(self, t: int):
        t_idx: int = self.t_list2global[t]
        if t_idx < 0:
            return
        t_x: pk.double = self.targets_list[0][t]
        t_y: pk.double = self.targets_list[1][t]
        t_z: pk.double = self.targets_list[2][t]
        nz_t_cell: int = t // self.t_cell_size
        t_cell: int = self.nz2t_cell_map[nz_t_cell]
        t_cell_x: int = t_cell // (self.cell_grid_area)
        t_cell_y: int = (t_cell % (self.cell_grid_area)) // self.num_cells_z
        t_cell_z: int = (t_cell % (self.cell_grid_area)) % self.num_cells_z
        pot_x: pk.double = 0
        pot_y: pk.double = 0
        pot_z: pk.double = 0
        for dx in range(-1, 2):
            s_cell_x: int = t_cell_x + dx
            x_shift: pk.double = 0
            # wrap in the periodic direction
            if s_cell_x < 0:
                s_cell_x += self.num_cells_x
                x_shift = -self.box_x
            elif s_cell_x >= self.num_cells_x:
                s_cell_x -= self.num_cells_x
                x_shift = self.box_x
            for dy in range(-1, 2):
                s_cell_y: int = t_cell_y + dy
                if s_cell_y < 0 or s_cell_y >= self.num_cells_y:
                    continue
                for dz in range(-1, 2):
                    s_cell_z: int = t_cell_z + dz
                    if s_cell_z < 0 or s_cell_z >= self.num_cells_z:
                        continue
                    s_cell: int = (
                        s_cell_x * self.cell_grid_area
                        + s_cell_y * self.num_cells_z
                        + s_cell_z
                    )
                    ns_cell: int = self.s_counter[s_cell]
                    nz_s_cell: int = self.s2nz_cell_map[s_cell]
                    s_off: int = nz_s_cell * self.s_cell_size
                    for s in range(s_off, s_off + ns_cell):
                        r_x: pk.double = t_x - (self.sources_list[0][s]) - x_shift
                        r_y: pk.double = t_y - (self.sources_list[1][s])
                        r_z: pk.double = t_z - (self.sources_list[2][s])
                        if self.kernel == 1:
                            pot_x += r_x
                            pot_y += r_y
                            pot_z += r_z
                            continue
                        dist_squared: pk.double = r_x * r_x + r_y * r_y + r_z * r_z
                        # Check if source is within rc of target
                        if dist_squared > self.rc_squared:
                            continue
                        # TODO: change to George's method
                        over_dist: pk.double = pk.rsqrt(
                            (dist_squared != 0) * (dist_squared) + (dist_squared == 0)
                        )
                        if self.kernel == 2:
                            over_dist = (dist_squared != 0) * over_dist
                            pot_x += over_dist
                            continue
                        dist: pk.double = (dist_squared != 0) * (1 / over_dist)
                        over_dist = (dist_squared != 0) * over_dist
                        over_dist_squared: pk.double = over_dist * over_dist
                        # (kernel independent) ewald computations
                        A: pk.double = 0
                        B: pk.double = 0
                        C: pk.double = 0
                        if self.has_ewald:
                            xir: pk.double = self.xi * dist
                            xir_squared: pk.double = self.xi_squared * dist_squared
                            # Replace A by its limit close to zero (relative error at 1e-14
                            # should be around 4e-29, so this is very accurate)
                            # Reformulation of if/else statement
                            A = (xir < 1e-14) * (self.xi_two_inv_sqrt_pi) + (
                                xir >= 1e-14
                            ) * (pk.erf(xir) * over_dist)
                            B = self.xi * pk.exp(-xir_squared) * self.two_inv_sqrt_pi
                            # Replace C by its Taylor expansion close to zero
                            # (relative error at most around 3e-13 for four terms)
                            # NOTE: This might be a bit unnecessary, as C goes into terms
                            # that will anyway go to zero as r -> 0. But we do it
                            # anyway for good measure.
                            # Reformulation of if/else statement
                            term2: pk.double = xir_squared * self.C_term1
                            term3: pk.double = xir_squared * term2
                            C = (xir < 4.75e-2) * (
                                self.m_c1_C_term1
                                - self.c2 * term2
                                + self.c3 * term3
                                - self.c4 * (xir_squared * term3)
                            ) + (xir >= 4.75e-2) * ((A - B) * over_dist_squared)
                        # kernel dependent computations
                        # stokes sl kernel
                        if self.has_sl:
                            f1_x: pk.double = self.forces_list[0][s]
                            f1_y: pk.double = self.forces_list[1][s]
                            f1_z: pk.double = self.forces_list[2][s]
                            # Punctured trapezoidal rule
                            # Remove point where r==0 (1e-14 is ad hoc)
                            s1: pk.double = (dist >= 1e-14) * over_dist
                            s2: pk.double = (
                                (dist >= 1e-14) * over_dist * over_dist_squared
                            )
                            # Sum up all terms
                            tmp: pk.double = s1 - B - A
                            t1_x: pk.double = tmp * f1_x
                            t1_y: pk.double = tmp * f1_y
                            t1_z: pk.double = tmp * f1_z
                            tmp = (r_x * f1_x + r_y * f1_y + r_z * f1_z) * (s2 - C)
                            t2_x: pk.double = tmp * r_x
                            t2_y: pk.double = tmp * r_y
                            t2_z: pk.double = tmp * r_z
                            pot_x += t1_x + t2_x
                            pot_y += t1_y + t2_y
                            pot_z += t1_z + t2_z

                        # stokes dl kernel
                        if self.has_dl:
                            f2_x: pk.double = self.forces_list[3][s]
                            f2_y: pk.double = self.forces_list[4][s]
                            f2_z: pk.double = self.forces_list[5][s]
                            r_dot_f2: pk.double = r_x * f2_x + r_y * f2_y + r_z * f2_z
                            n_x: pk.double = self.normals_list[0][s]
                            n_y: pk.double = self.normals_list[1][s]
                            n_z: pk.double = self.normals_list[2][s]
                            r_dot_n: pk.double = r_x * n_x + r_y * n_y + r_z * n_z
                            m_r_dot_f2_r_dot_n: pk.double = r_dot_f2 * r_dot_n
                            # Singular part, i.e., terms coming from the full (free-space) stresslet
                            # Punctured trapezoidal rule
                            # Remove point where r==0 (1e-14 is ad hoc)
                            s1: pk.double = (
                                (dist >= 1e-14)
                                * over_dist
                                * over_dist_squared
                                * over_dist_squared
                            )
                            # Sum up all terms
                            D: pk.double = self.m_xi_squared_2 * B
                            # At r==0, this part will become zero (1e-14 is ad hoc)
                            tmp: pk.double = (dist >= 1e-14) * (
                                (6 * C - 2 * D) * m_r_dot_f2_r_dot_n * over_dist_squared
                            )
                            tmp += -6 * s1 * m_r_dot_f2_r_dot_n + D * (
                                f2_x * n_x + f2_y * n_y + f2_z * n_z
                            )
                            t1_x: pk.double = tmp * r_x
                            t1_y: pk.double = tmp * r_y
                            t1_z: pk.double = tmp * r_z
                            tmp = D * r_dot_n
                            t2_x: pk.double = tmp * f2_x
                            t2_y: pk.double = tmp * f2_y
                            t2_z: pk.double = tmp * f2_z
                            tmp = D * r_dot_f2
                            pot_x += t1_x + t2_x + (tmp * n_x)
                            pot_y += t1_y + t2_y + (tmp * n_y)
                            pot_z += t1_z + t2_z + (tmp * n_z)

        self.potentials[0][t_idx] = pot_x
        self.potentials[1][t_idx] = pot_y
        self.potentials[2][t_idx] = pot_z

    @pk.workunit
    def p2p_point_in(self, team_member: pk.TeamMember):
        s_off: int = team_member.league_rank() * self.s_cell_chunk_size_in

        def thread_loop(tid: int):
            s: int = s_off + tid
            if s >= self.nnz_s_cells * self.s_cell_size:
                return
            s_x: pk.double = self.sources_list[0][s]
            s_y: pk.double = self.sources_list[1][s]
            s_z: pk.double = self.sources_list[2][s]
            if s_x < 0:
                return
            nz_s_cell: int = s // self.s_cell_size
            s_cell: int = self.nz2s_cell_map[nz_s_cell]
            s_cell_x: int = s_cell // (self.cell_grid_area)
            s_cell_y: int = (s_cell % self.cell_grid_area) // self.num_cells_z
            s_cell_z: int = (s_cell % self.cell_grid_area) % self.num_cells_z
            for dx in range(-1, 2):
                t_cell_x: int = s_cell_x + dx
                x_shift: pk.double = 0
                if t_cell_x < 0:
                    t_cell_x += self.num_cells_x
                    x_shift = self.box_x
                elif t_cell_x >= self.num_cells_x:
                    t_cell_x -= self.num_cells_x
                    x_shift = -self.box_x
                for dy in range(-1, 2):
                    t_cell_y: int = s_cell_y + dy
                    if t_cell_y < 0 or t_cell_y >= self.num_cells_y:
                        continue
                    for dz in range(-1, 2):
                        t_cell_z: int = s_cell_z + dz
                        if t_cell_z < 0 or t_cell_z >= self.num_cells_z:
                            continue
                        t_cell: int = (
                            t_cell_x * self.cell_grid_area
                            + t_cell_y * self.num_cells_z
                            + t_cell_z
                        )
                        nt_cell: int = self.t_counter[t_cell]
                        nz_t_cell: int = self.t2nz_cell_map[t_cell]
                        t_off: int = nz_t_cell * self.t_cell_size
                        for t in range(t_off, t_off + nt_cell):
                            t_idx: int = self.t_list2global[t]
                            if t_idx < 0:
                                continue
                            t_x: pk.double = self.targets_list[0][t]
                            t_y: pk.double = self.targets_list[1][t]
                            t_z: pk.double = self.targets_list[2][t]
                            r_x: pk.double = t_x - (s_x) - x_shift
                            r_y: pk.double = t_y - (s_y)
                            r_z: pk.double = t_z - (s_z)
                            if self.kernel == 1:
                                pk.atomic_add(self.potentials, [0, t_idx], r_x)
                                pk.atomic_add(self.potentials, [1, t_idx], r_y)
                                pk.atomic_add(self.potentials, [2, t_idx], r_z)
                                continue
                            dist_squared: pk.double = r_x * r_x + r_y * r_y + r_z * r_z
                            # Check if source is within rc of target
                            if dist_squared > self.rc_squared:
                                continue
                            # TODO: change to George's method
                            over_dist: pk.double = pk.rsqrt(
                                (dist_squared != 0) * (dist_squared)
                                + (dist_squared == 0)
                            )
                            if self.kernel == 2:
                                over_dist = (dist_squared != 0) * over_dist
                                pk.atomic_add(
                                    self.potentials,
                                    [0, t_idx],
                                    self.inv_4_pi * over_dist,
                                )
                                continue
                            dist: pk.double = (dist_squared != 0) * (1 / over_dist)
                            over_dist = (dist_squared != 0) * over_dist
                            over_dist_squared: pk.double = over_dist * over_dist
                            # (kernel independent) ewald computations
                            A: pk.double = 0
                            B: pk.double = 0
                            C: pk.double = 0
                            if self.has_ewald:
                                xir: pk.double = self.xi * dist
                                xir_squared: pk.double = self.xi_squared * dist_squared
                                # Replace A by its limit close to zero (relative error at 1e-14
                                # should be around 4e-29, so this is very accurate)
                                # Reformulation of if/else statement
                                A = (xir < 1e-14) * (self.xi_two_inv_sqrt_pi) + (
                                    xir >= 1e-14
                                ) * (pk.erf(xir) * over_dist)
                                B = (
                                    self.xi
                                    * pk.exp(-xir_squared)
                                    * self.two_inv_sqrt_pi
                                )
                                # Replace C by its Taylor expansion close to zero
                                # (relative error at most around 3e-13 for four terms)
                                # NOTE: This might be a bit unnecessary, as C goes into terms
                                # that will anyway go to zero as r -> 0. But we do it
                                # anyway for good measure.
                                # Reformulation of if/else statement
                                term2: pk.double = xir_squared * self.C_term1
                                term3: pk.double = xir_squared * term2
                                C = (xir < 4.75e-2) * (
                                    self.m_c1_C_term1
                                    - self.c2 * term2
                                    + self.c3 * term3
                                    - self.c4 * (xir_squared * term3)
                                ) + (xir >= 4.75e-2) * ((A - B) * over_dist_squared)
                            # kernel dependent computations
                            # stokes sl kernel
                            if self.has_sl:
                                f1_x: pk.double = self.forces_list[0][s]
                                f1_y: pk.double = self.forces_list[1][s]
                                f1_z: pk.double = self.forces_list[2][s]
                                # Punctured trapezoidal rule
                                # Remove point where r==0 (1e-14 is ad hoc)
                                s1: pk.double = (dist >= 1e-14) * over_dist
                                s2: pk.double = (
                                    (dist >= 1e-14) * over_dist * over_dist_squared
                                )
                                # Sum up all terms
                                tmp: pk.double = s1 - B - A
                                t1_x: pk.double = tmp * f1_x
                                t1_y: pk.double = tmp * f1_y
                                t1_z: pk.double = tmp * f1_z
                                tmp = (r_x * f1_x + r_y * f1_y + r_z * f1_z) * (s2 - C)
                                t2_x: pk.double = tmp * r_x
                                t2_y: pk.double = tmp * r_y
                                t2_z: pk.double = tmp * r_z
                                pk.atomic_add(
                                    self.potentials,
                                    [0, t_idx],
                                    self.inv_8_pi * (t1_x + t2_x),
                                )
                                pk.atomic_add(
                                    self.potentials,
                                    [1, t_idx],
                                    self.inv_8_pi * (t1_y + t2_y),
                                )
                                pk.atomic_add(
                                    self.potentials,
                                    [2, t_idx],
                                    self.inv_8_pi * (t1_z + t2_z),
                                )

                            # stokes dl kernel
                            if self.has_dl:
                                f2_x: pk.double = self.forces_list[3][s]
                                f2_y: pk.double = self.forces_list[4][s]
                                f2_z: pk.double = self.forces_list[5][s]
                                r_dot_f2: pk.double = (
                                    r_x * f2_x + r_y * f2_y + r_z * f2_z
                                )
                                n_x: pk.double = self.normals_list[0][s]
                                n_y: pk.double = self.normals_list[1][s]
                                n_z: pk.double = self.normals_list[2][s]
                                r_dot_n: pk.double = r_x * n_x + r_y * n_y + r_z * n_z
                                m_r_dot_f2_r_dot_n: pk.double = r_dot_f2 * r_dot_n
                                # Singular part, i.e., terms coming from the full (free-space) stresslet
                                # Punctured trapezoidal rule
                                # Remove point where r==0 (1e-14 is ad hoc)
                                s1: pk.double = (
                                    (dist >= 1e-14)
                                    * over_dist
                                    * over_dist_squared
                                    * over_dist_squared
                                )
                                # Sum up all terms
                                D: pk.double = self.m_xi_squared_2 * B
                                # At r==0, this part will become zero (1e-14 is ad hoc)
                                tmp: pk.double = (dist >= 1e-14) * (
                                    (6 * C - 2 * D)
                                    * m_r_dot_f2_r_dot_n
                                    * over_dist_squared
                                )
                                tmp += -6 * s1 * m_r_dot_f2_r_dot_n + D * (
                                    f2_x * n_x + f2_y * n_y + f2_z * n_z
                                )
                                t1_x: pk.double = tmp * r_x
                                t1_y: pk.double = tmp * r_y
                                t1_z: pk.double = tmp * r_z
                                tmp = D * r_dot_n
                                t2_x: pk.double = tmp * f2_x
                                t2_y: pk.double = tmp * f2_y
                                t2_z: pk.double = tmp * f2_z
                                tmp = D * r_dot_f2
                                pk.atomic_add(
                                    self.potentials,
                                    [0, t_idx],
                                    self.inv_8_pi * (t1_x + t2_x + (tmp * n_x)),
                                )
                                pk.atomic_add(
                                    self.potentials,
                                    [1, t_idx],
                                    self.inv_8_pi * (t1_y + t2_y + (tmp * n_y)),
                                )
                                pk.atomic_add(
                                    self.potentials,
                                    [2, t_idx],
                                    self.inv_8_pi * (t1_z + t2_z + (tmp * n_z)),
                                )

        pk.parallel_for(
            pk.TeamThreadRange(team_member, self.s_cell_chunk_size_in), thread_loop
        )

    @pk.workunit
    def p2p_point_in_range(self, s: int):
        s_x: pk.double = self.sources_list[0][s]
        s_y: pk.double = self.sources_list[1][s]
        s_z: pk.double = self.sources_list[2][s]
        if s_x < 0:
            return
        nz_s_cell: int = s // self.s_cell_size
        s_cell: int = self.nz2s_cell_map[nz_s_cell]
        s_cell_x: int = s_cell // (self.cell_grid_area)
        s_cell_y: int = (s_cell % self.cell_grid_area) // self.num_cells_z
        s_cell_z: int = (s_cell % self.cell_grid_area) % self.num_cells_z
        for dx in range(-1, 2):
            t_cell_x: int = s_cell_x + dx
            x_shift: pk.double = 0
            if t_cell_x < 0:
                t_cell_x += self.num_cells_x
                x_shift = self.box_x
            elif t_cell_x >= self.num_cells_x:
                t_cell_x -= self.num_cells_x
                x_shift = -self.box_x
            for dy in range(-1, 2):
                t_cell_y: int = s_cell_y + dy
                if t_cell_y < 0 or t_cell_y >= self.num_cells_y:
                    continue
                for dz in range(-1, 2):
                    t_cell_z: int = s_cell_z + dz
                    if t_cell_z < 0 or t_cell_z >= self.num_cells_z:
                        continue
                    t_cell: int = (
                        t_cell_x * self.cell_grid_area
                        + t_cell_y * self.num_cells_z
                        + t_cell_z
                    )
                    nt_cell: int = self.t_counter[t_cell]
                    nz_t_cell: int = self.t2nz_cell_map[t_cell]
                    t_off: int = nz_t_cell * self.t_cell_size
                    for t in range(t_off, t_off + nt_cell):
                        t_idx: int = self.t_list2global[t]
                        if t_idx < 0:
                            continue
                        t_x: pk.double = self.targets_list[0][t]
                        t_y: pk.double = self.targets_list[1][t]
                        t_z: pk.double = self.targets_list[2][t]
                        r_x: pk.double = t_x - (s_x) - x_shift
                        r_y: pk.double = t_y - (s_y)
                        r_z: pk.double = t_z - (s_z)
                        if self.kernel == 1:
                            pk.atomic_add(self.potentials, [0, t_idx], r_x)
                            pk.atomic_add(self.potentials, [1, t_idx], r_y)
                            pk.atomic_add(self.potentials, [2, t_idx], r_z)
                            continue
                        dist_squared: pk.double = r_x * r_x + r_y * r_y + r_z * r_z
                        # Check if source is within rc of target
                        if dist_squared > self.rc_squared:
                            continue
                        # TODO: change to George's method
                        over_dist: pk.double = pk.rsqrt(
                            (dist_squared != 0) * (dist_squared) + (dist_squared == 0)
                        )
                        if self.kernel == 2:
                            over_dist = (dist_squared != 0) * over_dist
                            pk.atomic_add(
                                self.potentials, [0, t_idx], self.inv_4_pi * over_dist
                            )
                            continue
                        dist: pk.double = (dist_squared != 0) * (1 / over_dist)
                        over_dist = (dist_squared != 0) * over_dist
                        over_dist_squared: pk.double = over_dist * over_dist
                        # (kernel independent) ewald computations
                        A: pk.double = 0
                        B: pk.double = 0
                        C: pk.double = 0
                        if self.has_ewald:
                            xir: pk.double = self.xi * dist
                            xir_squared: pk.double = self.xi_squared * dist_squared
                            # Replace A by its limit close to zero (relative error at 1e-14
                            # should be around 4e-29, so this is very accurate)
                            # Reformulation of if/else statement
                            A = (xir < 1e-14) * (self.xi_two_inv_sqrt_pi) + (
                                xir >= 1e-14
                            ) * (pk.erf(xir) * over_dist)
                            B = self.xi * pk.exp(-xir_squared) * self.two_inv_sqrt_pi
                            # Replace C by its Taylor expansion close to zero
                            # (relative error at most around 3e-13 for four terms)
                            # NOTE: This might be a bit unnecessary, as C goes into terms
                            # that will anyway go to zero as r -> 0. But we do it
                            # anyway for good measure.
                            # Reformulation of if/else statement
                            term2: pk.double = xir_squared * self.C_term1
                            term3: pk.double = xir_squared * term2
                            C = (xir < 4.75e-2) * (
                                self.m_c1_C_term1
                                - self.c2 * term2
                                + self.c3 * term3
                                - self.c4 * (xir_squared * term3)
                            ) + (xir >= 4.75e-2) * ((A - B) * over_dist_squared)
                        # kernel dependent computations
                        # stokes sl kernel
                        if self.has_sl:
                            f1_x: pk.double = self.forces_list[0][s]
                            f1_y: pk.double = self.forces_list[1][s]
                            f1_z: pk.double = self.forces_list[2][s]
                            # Punctured trapezoidal rule
                            # Remove point where r==0 (1e-14 is ad hoc)
                            s1: pk.double = (dist >= 1e-14) * over_dist
                            s2: pk.double = (
                                (dist >= 1e-14) * over_dist * over_dist_squared
                            )
                            # Sum up all terms
                            tmp: pk.double = s1 - B - A
                            t1_x: pk.double = tmp * f1_x
                            t1_y: pk.double = tmp * f1_y
                            t1_z: pk.double = tmp * f1_z
                            tmp = (r_x * f1_x + r_y * f1_y + r_z * f1_z) * (s2 - C)
                            t2_x: pk.double = tmp * r_x
                            t2_y: pk.double = tmp * r_y
                            t2_z: pk.double = tmp * r_z
                            pk.atomic_add(
                                self.potentials,
                                [0, t_idx],
                                self.inv_8_pi * (t1_x + t2_x),
                            )
                            pk.atomic_add(
                                self.potentials,
                                [1, t_idx],
                                self.inv_8_pi * (t1_y + t2_y),
                            )
                            pk.atomic_add(
                                self.potentials,
                                [2, t_idx],
                                self.inv_8_pi * (t1_z + t2_z),
                            )

                        # stokes dl kernel
                        if self.has_dl:
                            f2_x: pk.double = self.forces_list[3][s]
                            f2_y: pk.double = self.forces_list[4][s]
                            f2_z: pk.double = self.forces_list[5][s]
                            r_dot_f2: pk.double = r_x * f2_x + r_y * f2_y + r_z * f2_z
                            n_x: pk.double = self.normals_list[0][s]
                            n_y: pk.double = self.normals_list[1][s]
                            n_z: pk.double = self.normals_list[2][s]
                            r_dot_n: pk.double = r_x * n_x + r_y * n_y + r_z * n_z
                            m_r_dot_f2_r_dot_n: pk.double = r_dot_f2 * r_dot_n
                            # Singular part, i.e., terms coming from the full (free-space) stresslet
                            # Punctured trapezoidal rule
                            # Remove point where r==0 (1e-14 is ad hoc)
                            s1: pk.double = (
                                (dist >= 1e-14)
                                * over_dist
                                * over_dist_squared
                                * over_dist_squared
                            )
                            # Sum up all terms
                            D: pk.double = self.m_xi_squared_2 * B
                            # At r==0, this part will become zero (1e-14 is ad hoc)
                            tmp: pk.double = (dist >= 1e-14) * (
                                (6 * C - 2 * D) * m_r_dot_f2_r_dot_n * over_dist_squared
                            )
                            tmp += -6 * s1 * m_r_dot_f2_r_dot_n + D * (
                                f2_x * n_x + f2_y * n_y + f2_z * n_z
                            )
                            t1_x: pk.double = tmp * r_x
                            t1_y: pk.double = tmp * r_y
                            t1_z: pk.double = tmp * r_z
                            tmp = D * r_dot_n
                            t2_x: pk.double = tmp * f2_x
                            t2_y: pk.double = tmp * f2_y
                            t2_z: pk.double = tmp * f2_z
                            tmp = D * r_dot_f2
                            pk.atomic_add(
                                self.potentials,
                                [0, t_idx],
                                self.inv_8_pi * (t1_x + t2_x + (tmp * n_x)),
                            )
                            pk.atomic_add(
                                self.potentials,
                                [1, t_idx],
                                self.inv_8_pi * (t1_y + t2_y + (tmp * n_y)),
                            )
                            pk.atomic_add(
                                self.potentials,
                                [2, t_idx],
                                self.inv_8_pi * (t1_z + t2_z + (tmp * n_z)),
                            )

    @pk.workunit
    def p2p_cell_out_distance(self, team_member: pk.TeamMember):
        nz_t_cell: int = team_member.league_rank() // self.t_cell_chunks
        t_cell_chunk: int = team_member.league_rank() % self.t_cell_chunks
        t_off: int = nz_t_cell * self.t_cell_size + (
            t_cell_chunk * self.t_cell_chunk_size
        )
        t_cell: int = self.nz2t_cell_map[nz_t_cell]
        nt_cell: int = self.t_counter[t_cell]
        t_cell_x: int = t_cell // (self.cell_grid_area)
        t_cell_y: int = (t_cell % (self.cell_grid_area)) // self.num_cells_z
        t_cell_z: int = (t_cell % (self.cell_grid_area)) % self.num_cells_z
        # declare shmem arrays
        shmem_idt: pk.ScratchView1D[int] = pk.ScratchView1D(
            team_member.team_scratch(0), self.t_cell_chunk_size
        )
        shmem_t: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.t_cell_chunk_size, 3
        )
        shmem_s: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3
        )
        shmem_sl: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3 * self.has_sl
        )
        shmem_dl: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3 * self.has_dl
        )
        shmem_n: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3 * self.has_dl
        )

        def load_shmem_t(ii: int):
            if ii + (t_cell_chunk * self.t_cell_chunk_size) >= nt_cell:
                return
            t: int = t_off + ii
            t_idx: int = self.t_list2global[t]
            shmem_idt[ii] = t_idx
            if t_idx < 0:
                return
            for d in range(3):
                shmem_t[ii][d] = self.targets_list[d][t]

        pk.parallel_for(
            pk.TeamThreadRange(team_member, self.t_cell_chunk_size), load_shmem_t
        )

        def load_shmem_s(ii: int):
            s: int = s_off + ii
            if s - (nz_s_cell * self.s_cell_size) >= ns_cell:
                return
            for d in range(3):
                shmem_s[ii][d] = self.sources_list[d][s]
                if self.has_sl:
                    shmem_sl[ii][d] = self.forces_list[d][s]
                if self.has_dl:
                    shmem_dl[ii][d] = self.forces_list[d + 3][s]
                    shmem_n[ii][d] = self.normals_list[d][s]

        def target_loop(jj: int):
            if jj + (t_cell_chunk * self.t_cell_chunk_size) >= nt_cell:
                return
            t_idx: int = shmem_idt[jj]
            if t_idx < 0:
                return
            t_x: pk.double = shmem_t[jj][0]
            t_y: pk.double = shmem_t[jj][1]
            t_z: pk.double = shmem_t[jj][2]
            pot_x: pk.double = 0
            pot_y: pk.double = 0
            pot_z: pk.double = 0
            for ii in range(self.s_cell_chunk_size):
                s: int = s_off + ii
                if s - (nz_s_cell * self.s_cell_size) >= ns_cell:
                    continue
                r_x: pk.double = t_x - (shmem_s[ii][0]) - x_shift
                r_y: pk.double = t_y - (shmem_s[ii][1])
                r_z: pk.double = t_z - (shmem_s[ii][2])
                pot_x += r_x
                pot_y += r_y
                pot_z += r_z

            self.potentials[0][t_idx] += pot_x
            self.potentials[1][t_idx] += pot_y
            self.potentials[2][t_idx] += pot_z

        for dx in range(-1, 2):
            s_cell_x: int = t_cell_x + dx
            x_shift: pk.double = 0
            # wrap in the periodic direction
            if s_cell_x < 0:
                s_cell_x += self.num_cells_x
                x_shift = -self.box_x
            elif s_cell_x >= self.num_cells_x:
                s_cell_x -= self.num_cells_x
                x_shift = self.box_x
            for dy in range(-1, 2):
                s_cell_y: int = t_cell_y + dy
                if s_cell_y < 0 or s_cell_y >= self.num_cells_y:
                    continue
                for dz in range(-1, 2):
                    s_cell_z: int = t_cell_z + dz
                    if s_cell_z < 0 or s_cell_z >= self.num_cells_z:
                        continue
                    s_cell: int = (
                        s_cell_x * self.cell_grid_area
                        + s_cell_y * self.num_cells_z
                        + s_cell_z
                    )
                    ns_cell: int = self.s_counter[s_cell]
                    nz_s_cell: int = self.s2nz_cell_map[s_cell]
                    s_off: int = nz_s_cell * self.s_cell_size
                    for _ in range(self.s_cell_chunks):
                        pk.parallel_for(
                            pk.TeamThreadRange(team_member, self.s_cell_chunk_size),
                            load_shmem_s,
                        )
                        team_member.team_barrier()
                        pk.parallel_for(
                            pk.TeamThreadRange(team_member, self.t_cell_chunk_size),
                            target_loop,
                        )
                        team_member.team_barrier()
                        s_off += self.s_cell_chunk_size

    @pk.workunit
    def p2p_cell_out_laplace(self, team_member: pk.TeamMember):
        nz_t_cell: int = team_member.league_rank() // self.t_cell_chunks
        t_cell_chunk: int = team_member.league_rank() % self.t_cell_chunks
        t_off: int = nz_t_cell * self.t_cell_size + (
            t_cell_chunk * self.t_cell_chunk_size
        )
        t_cell: int = self.nz2t_cell_map[nz_t_cell]
        nt_cell: int = self.t_counter[t_cell]
        t_cell_x: int = t_cell // (self.cell_grid_area)
        t_cell_y: int = (t_cell % (self.cell_grid_area)) // self.num_cells_z
        t_cell_z: int = (t_cell % (self.cell_grid_area)) % self.num_cells_z
        # declare shmem arrays
        shmem_idt: pk.ScratchView1D[int] = pk.ScratchView1D(
            team_member.team_scratch(0), self.t_cell_chunk_size
        )
        shmem_t: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.t_cell_chunk_size, 3
        )
        shmem_s: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3
        )
        shmem_sl: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3 * self.has_sl
        )
        shmem_dl: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3 * self.has_dl
        )
        shmem_n: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3 * self.has_dl
        )

        def load_shmem_t(ii: int):
            if ii + (t_cell_chunk * self.t_cell_chunk_size) >= nt_cell:
                return
            t: int = t_off + ii
            t_idx: int = self.t_list2global[t]
            shmem_idt[ii] = t_idx
            if t_idx < 0:
                return
            for d in range(3):
                shmem_t[ii][d] = self.targets_list[d][t]

        pk.parallel_for(
            pk.TeamThreadRange(team_member, self.t_cell_chunk_size), load_shmem_t
        )

        def load_shmem_s(ii: int):
            s: int = s_off + ii
            if s - (nz_s_cell * self.s_cell_size) >= ns_cell:
                return
            for d in range(3):
                shmem_s[ii][d] = self.sources_list[d][s]
                if self.has_sl:
                    shmem_sl[ii][d] = self.forces_list[d][s]
                if self.has_dl:
                    shmem_dl[ii][d] = self.forces_list[d + 3][s]
                    shmem_n[ii][d] = self.normals_list[d][s]

        def target_loop(jj: int):
            if jj + (t_cell_chunk * self.t_cell_chunk_size) >= nt_cell:
                return
            t_idx: int = shmem_idt[jj]
            if t_idx < 0:
                return
            t_x: pk.double = shmem_t[jj][0]
            t_y: pk.double = shmem_t[jj][1]
            t_z: pk.double = shmem_t[jj][2]
            pot_x: pk.double = 0
            pot_y: pk.double = 0
            pot_z: pk.double = 0
            for ii in range(self.s_cell_chunk_size):
                s: int = s_off + ii
                if s - (nz_s_cell * self.s_cell_size) >= ns_cell:
                    continue
                r_x: pk.double = t_x - (shmem_s[ii][0]) - x_shift
                r_y: pk.double = t_y - (shmem_s[ii][1])
                r_z: pk.double = t_z - (shmem_s[ii][2])
                dist_squared: pk.double = r_x * r_x + r_y * r_y + r_z * r_z
                # Check if source is within rc of target
                if dist_squared > self.rc_squared:
                    continue
                # TODO: change to George's method
                over_dist: pk.double = pk.rsqrt(
                    (dist_squared != 0) * (dist_squared) + (dist_squared == 0)
                )
                over_dist = (dist_squared != 0) * over_dist
                pot_x += over_dist

            self.potentials[0][t_idx] = pot_x

        for dx in range(-1, 2):
            s_cell_x: int = t_cell_x + dx
            x_shift: pk.double = 0
            # wrap in the periodic direction
            if s_cell_x < 0:
                s_cell_x += self.num_cells_x
                x_shift = -self.box_x
            elif s_cell_x >= self.num_cells_x:
                s_cell_x -= self.num_cells_x
                x_shift = self.box_x
            for dy in range(-1, 2):
                s_cell_y: int = t_cell_y + dy
                if s_cell_y < 0 or s_cell_y >= self.num_cells_y:
                    continue
                for dz in range(-1, 2):
                    s_cell_z: int = t_cell_z + dz
                    if s_cell_z < 0 or s_cell_z >= self.num_cells_z:
                        continue
                    s_cell: int = (
                        s_cell_x * self.cell_grid_area
                        + s_cell_y * self.num_cells_z
                        + s_cell_z
                    )
                    ns_cell: int = self.s_counter[s_cell]
                    nz_s_cell: int = self.s2nz_cell_map[s_cell]
                    s_off: int = nz_s_cell * self.s_cell_size
                    for _ in range(self.s_cell_chunks):
                        pk.parallel_for(
                            pk.TeamThreadRange(team_member, self.s_cell_chunk_size),
                            load_shmem_s,
                        )
                        team_member.team_barrier()
                        pk.parallel_for(
                            pk.TeamThreadRange(team_member, self.t_cell_chunk_size),
                            target_loop,
                        )
                        team_member.team_barrier()
                        s_off += self.s_cell_chunk_size

    @pk.workunit
    def p2p_cell_out_stokes_sl(self, team_member: pk.TeamMember):
        nz_t_cell: int = team_member.league_rank() // self.t_cell_chunks
        t_cell_chunk: int = team_member.league_rank() % self.t_cell_chunks
        t_off: int = nz_t_cell * self.t_cell_size + (
            t_cell_chunk * self.t_cell_chunk_size
        )
        t_cell: int = self.nz2t_cell_map[nz_t_cell]
        nt_cell: int = self.t_counter[t_cell]
        t_cell_x: int = t_cell // (self.cell_grid_area)
        t_cell_y: int = (t_cell % (self.cell_grid_area)) // self.num_cells_z
        t_cell_z: int = (t_cell % (self.cell_grid_area)) % self.num_cells_z
        # declare shmem arrays
        shmem_idt: pk.ScratchView1D[int] = pk.ScratchView1D(
            team_member.team_scratch(0), self.t_cell_chunk_size
        )
        shmem_t: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.t_cell_chunk_size, 3
        )
        shmem_s: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3
        )
        shmem_sl: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3 * self.has_sl
        )
        shmem_dl: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3 * self.has_dl
        )
        shmem_n: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3 * self.has_dl
        )

        def load_shmem_t(ii: int):
            if ii + (t_cell_chunk * self.t_cell_chunk_size) >= nt_cell:
                return
            t: int = t_off + ii
            t_idx: int = self.t_list2global[t]
            shmem_idt[ii] = t_idx
            if t_idx < 0:
                return
            for d in range(3):
                shmem_t[ii][d] = self.targets_list[d][t]

        pk.parallel_for(
            pk.TeamThreadRange(team_member, self.t_cell_chunk_size), load_shmem_t
        )

        def load_shmem_s(ii: int):
            s: int = s_off + ii
            if s - (nz_s_cell * self.s_cell_size) >= ns_cell:
                return
            for d in range(3):
                shmem_s[ii][d] = self.sources_list[d][s]
                if self.has_sl:
                    shmem_sl[ii][d] = self.forces_list[d][s]
                if self.has_dl:
                    shmem_dl[ii][d] = self.forces_list[d + 3][s]
                    shmem_n[ii][d] = self.normals_list[d][s]

        def target_loop(jj: int):
            if jj + (t_cell_chunk * self.t_cell_chunk_size) >= nt_cell:
                return
            t_idx: int = shmem_idt[jj]
            if t_idx < 0:
                return
            t_x: pk.double = shmem_t[jj][0]
            t_y: pk.double = shmem_t[jj][1]
            t_z: pk.double = shmem_t[jj][2]
            pot_x: pk.double = 0
            pot_y: pk.double = 0
            pot_z: pk.double = 0
            for ii in range(self.s_cell_chunk_size):
                s: int = s_off + ii
                if s - (nz_s_cell * self.s_cell_size) >= ns_cell:
                    continue
                r_x: pk.double = t_x - (shmem_s[ii][0]) - x_shift
                r_y: pk.double = t_y - (shmem_s[ii][1])
                r_z: pk.double = t_z - (shmem_s[ii][2])
                dist_squared: pk.double = r_x * r_x + r_y * r_y + r_z * r_z
                # Check if source is within rc of target
                if dist_squared > self.rc_squared:
                    continue
                # TODO: change to George's method
                over_dist: pk.double = pk.rsqrt(
                    (dist_squared != 0) * (dist_squared) + (dist_squared == 0)
                )
                dist: pk.double = (dist_squared != 0) * (1 / over_dist)
                over_dist = (dist_squared != 0) * over_dist
                over_dist_squared: pk.double = over_dist * over_dist
                # (kernel independent) ewald computations
                A: pk.double = 0
                B: pk.double = 0
                C: pk.double = 0

                f1_x: pk.double = shmem_sl[ii][0]
                f1_y: pk.double = shmem_sl[ii][1]
                f1_z: pk.double = shmem_sl[ii][2]
                # Punctured trapezoidal rule
                # Remove point where r==0 (1e-14 is ad hoc)
                s1: pk.double = (dist >= 1e-14) * over_dist
                s2: pk.double = (dist >= 1e-14) * over_dist * over_dist_squared
                # Sum up all terms
                tmp: pk.double = s1 - B - A
                t1_x: pk.double = tmp * f1_x
                t1_y: pk.double = tmp * f1_y
                t1_z: pk.double = tmp * f1_z
                tmp = (r_x * f1_x + r_y * f1_y + r_z * f1_z) * (s2 - C)
                t2_x: pk.double = tmp * r_x
                t2_y: pk.double = tmp * r_y
                t2_z: pk.double = tmp * r_z
                pot_x += t1_x + t2_x
                pot_y += t1_y + t2_y
                pot_z += t1_z + t2_z

            self.potentials[0][t_idx] += pot_x
            self.potentials[1][t_idx] += pot_y
            self.potentials[2][t_idx] += pot_z

        for dx in range(-1, 2):
            s_cell_x: int = t_cell_x + dx
            x_shift: pk.double = 0
            # wrap in the periodic direction
            if s_cell_x < 0:
                s_cell_x += self.num_cells_x
                x_shift = -self.box_x
            elif s_cell_x >= self.num_cells_x:
                s_cell_x -= self.num_cells_x
                x_shift = self.box_x
            for dy in range(-1, 2):
                s_cell_y: int = t_cell_y + dy
                if s_cell_y < 0 or s_cell_y >= self.num_cells_y:
                    continue
                for dz in range(-1, 2):
                    s_cell_z: int = t_cell_z + dz
                    if s_cell_z < 0 or s_cell_z >= self.num_cells_z:
                        continue
                    s_cell: int = (
                        s_cell_x * self.cell_grid_area
                        + s_cell_y * self.num_cells_z
                        + s_cell_z
                    )
                    ns_cell: int = self.s_counter[s_cell]
                    nz_s_cell: int = self.s2nz_cell_map[s_cell]
                    s_off: int = nz_s_cell * self.s_cell_size
                    for _ in range(self.s_cell_chunks):
                        pk.parallel_for(
                            pk.TeamThreadRange(team_member, self.s_cell_chunk_size),
                            load_shmem_s,
                        )
                        team_member.team_barrier()
                        pk.parallel_for(
                            pk.TeamThreadRange(team_member, self.t_cell_chunk_size),
                            target_loop,
                        )
                        team_member.team_barrier()
                        s_off += self.s_cell_chunk_size

    @pk.workunit
    def p2p_sm_1d_stokes_comb(self, team_member: pk.TeamMember):
        nz_t_cell: int = team_member.league_rank() // self.t_cell_chunks
        t_cell_chunk: int = team_member.league_rank() % self.t_cell_chunks
        t_off: int = nz_t_cell * self.t_cell_size + (
            t_cell_chunk * self.t_cell_chunk_size
        )
        t_cell: int = self.nz2t_cell_map[nz_t_cell]
        nt_cell: int = self.t_counter[t_cell]
        t_cell_x: int = t_cell // (self.cell_grid_area)
        t_cell_y: int = (t_cell % (self.cell_grid_area)) // self.num_cells_z
        t_cell_z: int = (t_cell % (self.cell_grid_area)) % self.num_cells_z
        # declare shmem arrays
        shmem_idt: pk.ScratchView1D[int] = pk.ScratchView1D(
            team_member.team_scratch(0), self.t_cell_chunk_size
        )
        shmem_t: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.t_cell_chunk_size, 3
        )
        shmem_s: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3
        )
        shmem_sl: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3 * self.has_sl
        )
        shmem_dl: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3 * self.has_dl
        )
        shmem_n: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3 * self.has_dl
        )

        def load_shmem_t(ii: int):
            if ii + (t_cell_chunk * self.t_cell_chunk_size) >= nt_cell:
                return
            t: int = t_off + ii
            t_idx: int = self.t_list2global[t]
            shmem_idt[ii] = t_idx
            if t_idx < 0:
                return
            for d in range(3):
                shmem_t[ii][d] = self.targets_list[d][t]

        pk.parallel_for(
            pk.TeamThreadRange(team_member, self.t_cell_chunk_size), load_shmem_t
        )

        def load_shmem_s(ii: int):
            s: int = s_off + ii
            if s - (nz_s_cell * self.s_cell_size) >= ns_cell:
                return
            for d in range(3):
                shmem_s[ii][d] = self.sources_list[d][s]
                if self.has_sl:
                    shmem_sl[ii][d] = self.forces_list[d][s]
                if self.has_dl:
                    shmem_dl[ii][d] = self.forces_list[d + 3][s]
                    shmem_n[ii][d] = self.normals_list[d][s]

        def target_loop(jj: int):
            if jj + (t_cell_chunk * self.t_cell_chunk_size) >= nt_cell:
                return
            t_idx: int = shmem_idt[jj]
            if t_idx < 0:
                return
            t_x: pk.double = shmem_t[jj][0]
            t_y: pk.double = shmem_t[jj][1]
            t_z: pk.double = shmem_t[jj][2]
            pot_x: pk.double = 0
            pot_y: pk.double = 0
            pot_z: pk.double = 0
            for ii in range(self.s_cell_chunk_size):
                s: int = s_off + ii
                if s - (nz_s_cell * self.s_cell_size) >= ns_cell:
                    continue
                r_x: pk.double = t_x - (shmem_s[ii][0]) - source_cell.x_shift
                r_y: pk.double = t_y - (shmem_s[ii][1]) - source_cell.y_shift
                r_z: pk.double = t_z - (shmem_s[ii][2]) - source_cell.z_shift
                if self.kernel == 1:
                    pot_x += r_x
                    pot_y += r_y
                    pot_z += r_z
                    continue
                dist_squared: pk.double = r_x * r_x + r_y * r_y + r_z * r_z
                # Check if source is within rc of target
                if dist_squared > self.rc_squared:
                    continue
                # TODO: change to George's method
                over_dist: pk.double = pk.rsqrt(
                    (dist_squared != 0) * (dist_squared) + (dist_squared == 0)
                )
                if self.kernel == 2:
                    over_dist = (dist_squared != 0) * over_dist
                    pot_x += over_dist
                    continue
                dist: pk.double = (dist_squared != 0) * (1 / over_dist)
                over_dist = (dist_squared != 0) * over_dist
                over_dist_squared: pk.double = over_dist * over_dist
                # (kernel independent) ewald computations
                A: pk.double = 0
                B: pk.double = 0
                C: pk.double = 0
                if self.has_ewald:
                    xir: pk.double = self.xi * dist
                    xir_squared: pk.double = self.xi_squared * dist_squared
                    # Replace A by its limit close to zero (relative error at 1e-14
                    # should be around 4e-29, so this is very accurate)
                    # Reformulation of if/else statement
                    A = (xir < 1e-14) * (self.xi_two_inv_sqrt_pi) + (xir >= 1e-14) * (
                        pk.erf(xir) * over_dist
                    )
                    B = self.xi * pk.exp(-xir_squared) * self.two_inv_sqrt_pi
                    # Replace C by its Taylor expansion close to zero
                    # (relative error at most around 3e-13 for four terms)
                    # NOTE: This might be a bit unnecessary, as C goes into terms
                    # that will anyway go to zero as r -> 0. But we do it
                    # anyway for good measure.
                    # Reformulation of if/else statement
                    term2: pk.double = xir_squared * self.C_term1
                    term3: pk.double = xir_squared * term2
                    C = (xir < 4.75e-2) * (
                        self.m_c1_C_term1
                        - self.c2 * term2
                        + self.c3 * term3
                        - self.c4 * (xir_squared * term3)
                    ) + (xir >= 4.75e-2) * ((A - B) * over_dist_squared)
                # kernel dependent computations
                # stokes sl kernel
                if self.has_sl:
                    f1_x: pk.double = shmem_sl[ii][0]
                    f1_y: pk.double = shmem_sl[ii][1]
                    f1_z: pk.double = shmem_sl[ii][2]
                    # Punctured trapezoidal rule
                    # Remove point where r==0 (1e-14 is ad hoc)
                    s1: pk.double = (dist >= 1e-14) * over_dist
                    s2: pk.double = (dist >= 1e-14) * over_dist * over_dist_squared
                    # Sum up all terms
                    tmp: pk.double = s1 - B - A
                    t1_x: pk.double = tmp * f1_x
                    t1_y: pk.double = tmp * f1_y
                    t1_z: pk.double = tmp * f1_z
                    tmp = (r_x * f1_x + r_y * f1_y + r_z * f1_z) * (s2 - C)
                    t2_x: pk.double = tmp * r_x
                    t2_y: pk.double = tmp * r_y
                    t2_z: pk.double = tmp * r_z
                    pot_x += t1_x + t2_x
                    pot_y += t1_y + t2_y
                    pot_z += t1_z + t2_z

                # stokes dl kernel
                if self.has_dl:
                    f2_x: pk.double = shmem_dl[ii][0]
                    f2_y: pk.double = shmem_dl[ii][1]
                    f2_z: pk.double = shmem_dl[ii][2]
                    r_dot_f2: pk.double = r_x * f2_x + r_y * f2_y + r_z * f2_z
                    n_x: pk.double = shmem_n[ii][0]
                    n_y: pk.double = shmem_n[ii][1]
                    n_z: pk.double = shmem_n[ii][2]
                    r_dot_n: pk.double = r_x * n_x + r_y * n_y + r_z * n_z
                    m_r_dot_f2_r_dot_n: pk.double = r_dot_f2 * r_dot_n
                    # Singular part, i.e., terms coming from the full (free-space) stresslet
                    # Punctured trapezoidal rule
                    # Remove point where r==0 (1e-14 is ad hoc)
                    s1: pk.double = (
                        (dist >= 1e-14)
                        * over_dist
                        * over_dist_squared
                        * over_dist_squared
                    )
                    # Sum up all terms
                    D: pk.double = self.m_xi_squared_2 * B
                    # At r==0, this part will become zero (1e-14 is ad hoc)
                    tmp: pk.double = (dist >= 1e-14) * (
                        (6 * C - 2 * D) * m_r_dot_f2_r_dot_n * over_dist_squared
                    )
                    tmp += -6 * s1 * m_r_dot_f2_r_dot_n + D * (
                        f2_x * n_x + f2_y * n_y + f2_z * n_z
                    )
                    t1_x: pk.double = tmp * r_x
                    t1_y: pk.double = tmp * r_y
                    t1_z: pk.double = tmp * r_z
                    tmp = D * r_dot_n
                    t2_x: pk.double = tmp * f2_x
                    t2_y: pk.double = tmp * f2_y
                    t2_z: pk.double = tmp * f2_z
                    tmp = D * r_dot_f2
                    pot_x += t1_x + t2_x + (tmp * n_x)
                    pot_y += t1_y + t2_y + (tmp * n_y)
                    pot_z += t1_z + t2_z + (tmp * n_z)

            self.potentials[0][t_idx] += pot_x
            self.potentials[1][t_idx] += pot_y
            self.potentials[2][t_idx] += pot_z

        for k in range(27):
            source_cell: Cell_fp64 = get_source_cell(k, t_cell_x, t_cell_y, t_cell_z)
            if source_cell.inbounds == False:
                continue
            s_cell: int = (
                source_cell.x * self.cell_grid_area
                + source_cell.y * self.num_cells_z
                + source_cell.z
            )
            ns_cell: int = self.s_counter[s_cell]
            nz_s_cell: int = self.s2nz_cell_map[s_cell]
            s_off: int = nz_s_cell * self.s_cell_size
            for _ in range(self.s_cell_chunks):
                pk.parallel_for(
                    pk.TeamThreadRange(team_member, self.s_cell_chunk_size),
                    load_shmem_s,
                )
                team_member.team_barrier()
                pk.parallel_for(
                    pk.TeamThreadRange(team_member, self.t_cell_chunk_size), target_loop
                )
                team_member.team_barrier()
                s_off += self.s_cell_chunk_size

    @pk.workunit
    def p2p_sm_2d_stokes_comb(self, team_member: pk.TeamMember):
        nz_t_cell: int = team_member.league_rank() // self.t_cell_chunks
        t_cell_chunk: int = team_member.league_rank() % self.t_cell_chunks
        t_off: int = nz_t_cell * self.t_cell_size + (
            t_cell_chunk * self.t_cell_chunk_size
        )
        t_cell: int = self.nz2t_cell_map[nz_t_cell]
        nt_cell: int = self.t_counter[t_cell]
        t_cell_x: int = t_cell // (self.cell_grid_area)
        t_cell_y: int = (t_cell % (self.cell_grid_area)) // self.num_cells_z
        t_cell_z: int = (t_cell % (self.cell_grid_area)) % self.num_cells_z
        # declare shmem arrays
        shmem_idt: pk.ScratchView1D[int] = pk.ScratchView1D(
            team_member.team_scratch(0), self.t_cell_chunk_size
        )
        shmem_t: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.t_cell_chunk_size, 3
        )
        shmem_s: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3
        )
        shmem_sl: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3 * self.has_sl
        )
        shmem_dl: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3 * self.has_dl
        )
        shmem_n: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
            team_member.team_scratch(0), self.s_cell_chunk_size, 3 * self.has_dl
        )

        # def load_shmem_t(ii: int):
        #     if ii + (t_cell_chunk * self.t_cell_chunk_size) >= nt_cell:
        #         return
        #     t: int = t_off + ii
        #     t_idx:int  = self.t_list2global[t]
        #     shmem_idt[ii] = t_idx
        #     if t_idx < 0:
        #         return
        #     for d in range(3):
        #         shmem_t[ii][d] = self.targets_list[d][t]
        # pk.parallel_for(pk.TeamThreadRange(team_member, self.t_cell_chunk_size), load_shmem_t)
        def load_shmem_t(kk: int):
            def vector_loop(jj: int):
                ii: int = jj + kk * self.vector_size
                if ii + (t_cell_chunk * self.t_cell_chunk_size) >= nt_cell:
                    return
                t: int = t_off + ii
                t_idx: int = self.t_list2global[t]
                shmem_idt[ii] = t_idx
                if t_idx < 0:
                    return
                for d in range(3):
                    shmem_t[ii][d] = self.targets_list[d][t]

            pk.parallel_for(
                pk.ThreadVectorRange(team_member, self.vector_size), vector_loop
            )

        pk.parallel_for(
            pk.TeamThreadRange(team_member, self.t_cell_chunk_threads), load_shmem_t
        )

        def load_shmem_s(kk: int):
            def vector_loop(jj: int):
                ii: int = jj + kk * self.vector_size
                if ii >= self.s_cell_chunk_size:
                    return
                s: int = s_off + ii
                for d in range(3):
                    shmem_s[ii][d] = self.sources_list[d][s]
                    if self.has_sl:
                        shmem_sl[ii][d] = self.forces_list[d][s]
                    if self.has_dl:
                        shmem_dl[ii][d] = self.forces_list[d + 3][s]
                        shmem_n[ii][d] = self.normals_list[d][s]

            pk.parallel_for(
                pk.ThreadVectorRange(team_member, self.vector_size), vector_loop
            )

        def target_loop(jj: int):
            if jj + (t_cell_chunk * self.t_cell_chunk_size) >= nt_cell:
                return
            t_idx: int = shmem_idt[jj]
            if t_idx < 0:
                return
            t_x: pk.double = shmem_t[jj][0]
            t_y: pk.double = shmem_t[jj][1]
            t_z: pk.double = shmem_t[jj][2]
            pot_x: pk.double = 0
            pot_y: pk.double = 0
            pot_z: pk.double = 0

            def vector_loop(ii: int):
                s: int = s_off + ii
                if s - (nz_s_cell * self.s_cell_size) >= ns_cell:
                    return
                r_x: pk.double = t_x - (shmem_s[ii][0]) - source_cell.x_shift
                r_y: pk.double = t_y - (shmem_s[ii][1]) - source_cell.y_shift
                r_z: pk.double = t_z - (shmem_s[ii][2]) - source_cell.z_shift
                if self.kernel == 1:
                    pot_x += r_x
                    pot_y += r_y
                    pot_z += r_z
                    return
                dist_squared: pk.double = r_x * r_x + r_y * r_y + r_z * r_z
                # Check if source is within rc of target
                if dist_squared > self.rc_squared:
                    return
                # TODO: change to George's method
                over_dist: pk.double = pk.rsqrt(
                    (dist_squared != 0) * (dist_squared) + (dist_squared == 0)
                )
                if self.kernel == 2:
                    over_dist = (dist_squared != 0) * over_dist
                    pot_x += over_dist
                    return
                dist: pk.double = (dist_squared != 0) * (1 / over_dist)
                over_dist = (dist_squared != 0) * over_dist
                over_dist_squared: pk.double = over_dist * over_dist
                # (kernel independent) ewald computations
                A: pk.double = 0
                B: pk.double = 0
                C: pk.double = 0
                if self.has_ewald:
                    xir: pk.double = self.xi * dist
                    xir_squared: pk.double = self.xi_squared * dist_squared
                    # Replace A by its limit close to zero (relative error at 1e-14
                    # should be around 4e-29, so this is very accurate)
                    # Reformulation of if/else statement
                    A = (xir < 1e-14) * (self.xi_two_inv_sqrt_pi) + (xir >= 1e-14) * (
                        pk.erf(xir) * over_dist
                    )
                    B = self.xi * pk.exp(-xir_squared) * self.two_inv_sqrt_pi
                    # Replace C by its Taylor expansion close to zero
                    # (relative error at most around 3e-13 for four terms)
                    # NOTE: This might be a bit unnecessary, as C goes into terms
                    # that will anyway go to zero as r -> 0. But we do it
                    # anyway for good measure.
                    # Reformulation of if/else statement
                    term2: pk.double = xir_squared * self.C_term1
                    term3: pk.double = xir_squared * term2
                    C = (xir < 4.75e-2) * (
                        self.m_c1_C_term1
                        - self.c2 * term2
                        + self.c3 * term3
                        - self.c4 * (xir_squared * term3)
                    ) + (xir >= 4.75e-2) * ((A - B) * over_dist_squared)
                # kernel dependent computations
                # stokes sl kernel
                if self.has_sl:
                    f1_x: pk.double = shmem_sl[ii][0]
                    f1_y: pk.double = shmem_sl[ii][1]
                    f1_z: pk.double = shmem_sl[ii][2]
                    # Punctured trapezoidal rule
                    # Remove point where r==0 (1e-14 is ad hoc)
                    s1: pk.double = (dist >= 1e-14) * over_dist
                    s2: pk.double = (dist >= 1e-14) * over_dist * over_dist_squared
                    # Sum up all terms
                    tmp: pk.double = s1 - B - A
                    t1_x: pk.double = tmp * f1_x
                    t1_y: pk.double = tmp * f1_y
                    t1_z: pk.double = tmp * f1_z
                    tmp = (r_x * f1_x + r_y * f1_y + r_z * f1_z) * (s2 - C)
                    t2_x: pk.double = tmp * r_x
                    t2_y: pk.double = tmp * r_y
                    t2_z: pk.double = tmp * r_z
                    pot_x += t1_x + t2_x
                    pot_y += t1_y + t2_y
                    pot_z += t1_z + t2_z

                # stokes dl kernel
                if self.has_dl:
                    f2_x: pk.double = shmem_dl[ii][0]
                    f2_y: pk.double = shmem_dl[ii][1]
                    f2_z: pk.double = shmem_dl[ii][2]
                    r_dot_f2: pk.double = r_x * f2_x + r_y * f2_y + r_z * f2_z
                    n_x: pk.double = shmem_n[ii][0]
                    n_y: pk.double = shmem_n[ii][1]
                    n_z: pk.double = shmem_n[ii][2]
                    r_dot_n: pk.double = r_x * n_x + r_y * n_y + r_z * n_z
                    m_r_dot_f2_r_dot_n: pk.double = r_dot_f2 * r_dot_n
                    # Singular part, i.e., terms coming from the full (free-space) stresslet
                    # Punctured trapezoidal rule
                    # Remove point where r==0 (1e-14 is ad hoc)
                    s1: pk.double = (
                        (dist >= 1e-14)
                        * over_dist
                        * over_dist_squared
                        * over_dist_squared
                    )
                    # Sum up all terms
                    D: pk.double = self.m_xi_squared_2 * B
                    # At r==0, this part will become zero (1e-14 is ad hoc)
                    tmp: pk.double = (dist >= 1e-14) * (
                        (6 * C - 2 * D) * m_r_dot_f2_r_dot_n * over_dist_squared
                    )
                    tmp += -6 * s1 * m_r_dot_f2_r_dot_n + D * (
                        f2_x * n_x + f2_y * n_y + f2_z * n_z
                    )
                    t1_x: pk.double = tmp * r_x
                    t1_y: pk.double = tmp * r_y
                    t1_z: pk.double = tmp * r_z
                    tmp = D * r_dot_n
                    t2_x: pk.double = tmp * f2_x
                    t2_y: pk.double = tmp * f2_y
                    t2_z: pk.double = tmp * f2_z
                    tmp = D * r_dot_f2
                    pot_x += t1_x + t2_x + (tmp * n_x)
                    pot_y += t1_y + t2_y + (tmp * n_y)
                    pot_z += t1_z + t2_z + (tmp * n_z)

            def potential_reduce_x(_: int, acc: pk.Acc[pk.double]):
                acc += pot_x

            def potential_reduce_y(_: int, acc: pk.Acc[pk.double]):
                acc += pot_y

            def potential_reduce_z(_: int, acc: pk.Acc[pk.double]):
                acc += pot_z

            def write_output():
                self.potentials[0][t_idx] += trg_pot_x
                self.potentials[1][t_idx] += trg_pot_y
                self.potentials[2][t_idx] += trg_pot_z

            pk.parallel_for(
                pk.ThreadVectorRange(team_member, self.s_cell_chunk_size), vector_loop
            )
            trg_pot_x: pk.double = pk.parallel_reduce(
                pk.ThreadVectorRange(team_member, self.vector_size),
                potential_reduce_x,
                0,
            )
            trg_pot_y: pk.double = pk.parallel_reduce(
                pk.ThreadVectorRange(team_member, self.vector_size),
                potential_reduce_y,
                0,
            )
            trg_pot_z: pk.double = pk.parallel_reduce(
                pk.ThreadVectorRange(team_member, self.vector_size),
                potential_reduce_z,
                0,
            )
            pk.single(pk.PerThread(team_member), write_output)

        for k in range(27):
            source_cell: Cell_fp64 = get_source_cell(k, t_cell_x, t_cell_y, t_cell_z)
            if source_cell.inbounds == False:
                continue
            s_cell: int = (
                source_cell.x * self.cell_grid_area
                + source_cell.y * self.num_cells_z
                + source_cell.z
            )
            ns_cell: int = self.s_counter[s_cell]
            nz_s_cell: int = self.s2nz_cell_map[s_cell]
            s_off: int = nz_s_cell * self.s_cell_size
            for _ in range(self.s_cell_chunks):
                pk.parallel_for(
                    pk.TeamThreadRange(team_member, self.s_cell_chunk_threads),
                    load_shmem_s,
                )
                team_member.team_barrier()
                pk.parallel_for(
                    pk.TeamThreadRange(team_member, self.t_cell_chunk_size), target_loop
                )
                team_member.team_barrier()
                s_off += self.s_cell_chunk_size

    @pk.workunit
    def p2p_gm_2d_stokes_comb(self, team_member: pk.TeamMember):
        nz_t_cell: int = team_member.league_rank() // self.t_cell_chunks
        t_cell_chunk: int = team_member.league_rank() % self.t_cell_chunks
        t_off: int = nz_t_cell * self.t_cell_size + (
            t_cell_chunk * self.t_cell_chunk_size
        )
        t_cell: int = self.nz2t_cell_map[nz_t_cell]
        nt_cell: int = self.t_counter[t_cell]
        t_cell_x: int = t_cell // (self.cell_grid_area)
        t_cell_y: int = (t_cell % (self.cell_grid_area)) // self.num_cells_z
        t_cell_z: int = (t_cell % (self.cell_grid_area)) % self.num_cells_z

        def target_loop(ll: int):
            t: int = t_off + ll
            if ll + (t_cell_chunk * self.t_cell_chunk_size) >= nt_cell:
                return
            t_idx: int = self.t_list2global[t]
            if t_idx < 0:
                return

            t_x: pk.double = self.targets_list[0][t]
            t_y: pk.double = self.targets_list[1][t]
            t_z: pk.double = self.targets_list[2][t]
            pot_x: pk.double = 0
            pot_y: pk.double = 0
            pot_z: pk.double = 0

            def vector_loop(kk: int):
                for k in range(27):
                    source_cell: Cell_fp64 = get_source_cell(
                        k, t_cell_x, t_cell_y, t_cell_z
                    )
                    if source_cell.inbounds == False:
                        continue
                    s_cell: int = (
                        source_cell.x * self.cell_grid_area
                        + source_cell.y * self.num_cells_z
                        + source_cell.z
                    )
                    ns_cell: int = self.s_counter[s_cell]
                    nz_s_cell: int = self.s2nz_cell_map[s_cell]
                    s_off: int = nz_s_cell * self.s_cell_size
                    for jj in range(self.s_cell_threads):
                        ii: int = jj + kk * self.s_cell_threads
                        if ii >= self.s_cell_size:
                            break
                        s: int = s_off + ii
                        if s - (nz_s_cell * self.s_cell_size) >= ns_cell:
                            break
                        r_x: pk.double = (
                            t_x - (self.sources_list[0][s]) - source_cell.x_shift
                        )
                        r_y: pk.double = (
                            t_y - (self.sources_list[1][s]) - source_cell.y_shift
                        )
                        r_z: pk.double = (
                            t_z - (self.sources_list[2][s]) - source_cell.z_shift
                        )
                        if self.kernel == 1:
                            pot_x += r_x
                            pot_y += r_y
                            pot_z += r_z
                            continue
                        dist_squared: pk.double = r_x * r_x + r_y * r_y + r_z * r_z
                        # Check if source is within rc of target
                        if dist_squared > self.rc_squared:
                            continue
                        # TODO: change to George's method
                        over_dist: pk.double = pk.rsqrt(
                            (dist_squared != 0) * (dist_squared) + (dist_squared == 0)
                        )
                        if self.kernel == 2:
                            over_dist = (dist_squared != 0) * over_dist
                            pot_x += over_dist
                            continue
                        dist: pk.double = (dist_squared != 0) * (1 / over_dist)
                        over_dist = (dist_squared != 0) * over_dist
                        over_dist_squared: pk.double = over_dist * over_dist
                        # (kernel independent) ewald computations
                        A: pk.double = 0
                        B: pk.double = 0
                        C: pk.double = 0
                        if self.has_ewald:
                            xir: pk.double = self.xi * dist
                            xir_squared: pk.double = self.xi_squared * dist_squared
                            # Replace A by its limit close to zero (relative error at 1e-14
                            # should be around 4e-29, so this is very accurate)
                            # Reformulation of if/else statement
                            A = (xir < 1e-14) * (self.xi_two_inv_sqrt_pi) + (
                                xir >= 1e-14
                            ) * (pk.erf(xir) * over_dist)
                            B = self.xi * pk.exp(-xir_squared) * self.two_inv_sqrt_pi
                            # Replace C by its Taylor expansion close to zero
                            # (relative error at most around 3e-13 for four terms)
                            # NOTE: This might be a bit unnecessary, as C goes into terms
                            # that will anyway go to zero as r -> 0. But we do it
                            # anyway for good measure.
                            # Reformulation of if/else statement
                            term2: pk.double = xir_squared * self.C_term1
                            term3: pk.double = xir_squared * term2
                            C = (xir < 4.75e-2) * (
                                self.m_c1_C_term1
                                - self.c2 * term2
                                + self.c3 * term3
                                - self.c4 * (xir_squared * term3)
                            ) + (xir >= 4.75e-2) * ((A - B) * over_dist_squared)
                        # kernel dependent computations
                        # stokes sl kernel
                        if self.has_sl:
                            f1_x: pk.double = self.forces_list[0][s]
                            f1_y: pk.double = self.forces_list[1][s]
                            f1_z: pk.double = self.forces_list[2][s]
                            # Punctured trapezoidal rule
                            # Remove point where r==0 (1e-14 is ad hoc)
                            s1: pk.double = (dist >= 1e-14) * over_dist
                            s2: pk.double = (
                                (dist >= 1e-14) * over_dist * over_dist_squared
                            )
                            # Sum up all terms
                            tmp: pk.double = s1 - B - A
                            t1_x: pk.double = tmp * f1_x
                            t1_y: pk.double = tmp * f1_y
                            t1_z: pk.double = tmp * f1_z
                            tmp = (r_x * f1_x + r_y * f1_y + r_z * f1_z) * (s2 - C)
                            t2_x: pk.double = tmp * r_x
                            t2_y: pk.double = tmp * r_y
                            t2_z: pk.double = tmp * r_z
                            pot_x += t1_x + t2_x
                            pot_y += t1_y + t2_y
                            pot_z += t1_z + t2_z

                        # stokes dl kernel
                        if self.has_dl:
                            f2_x: pk.double = self.forces_list[3][s]
                            f2_y: pk.double = self.forces_list[4][s]
                            f2_z: pk.double = self.forces_list[5][s]
                            r_dot_f2: pk.double = r_x * f2_x + r_y * f2_y + r_z * f2_z
                            n_x: pk.double = self.normals_list[0][s]
                            n_y: pk.double = self.normals_list[1][s]
                            n_z: pk.double = self.normals_list[2][s]
                            r_dot_n: pk.double = r_x * n_x + r_y * n_y + r_z * n_z
                            m_r_dot_f2_r_dot_n: pk.double = r_dot_f2 * r_dot_n
                            # Singular part, i.e., terms coming from the full (free-space) stresslet
                            # Punctured trapezoidal rule
                            # Remove point where r==0 (1e-14 is ad hoc)
                            s1: pk.double = (
                                (dist >= 1e-14)
                                * over_dist
                                * over_dist_squared
                                * over_dist_squared
                            )
                            # Sum up all terms
                            D: pk.double = self.m_xi_squared_2 * B
                            # At r==0, this part will become zero (1e-14 is ad hoc)
                            tmp: pk.double = (dist >= 1e-14) * (
                                (6 * C - 2 * D) * m_r_dot_f2_r_dot_n * over_dist_squared
                            )
                            tmp += -6 * s1 * m_r_dot_f2_r_dot_n + D * (
                                f2_x * n_x + f2_y * n_y + f2_z * n_z
                            )
                            t1_x: pk.double = tmp * r_x
                            t1_y: pk.double = tmp * r_y
                            t1_z: pk.double = tmp * r_z
                            tmp = D * r_dot_n
                            t2_x: pk.double = tmp * f2_x
                            t2_y: pk.double = tmp * f2_y
                            t2_z: pk.double = tmp * f2_z
                            tmp = D * r_dot_f2
                            pot_x += t1_x + t2_x + (tmp * n_x)
                            pot_y += t1_y + t2_y + (tmp * n_y)
                            pot_z += t1_z + t2_z + (tmp * n_z)

            pk.parallel_for(
                pk.ThreadVectorRange(team_member, self.vector_size), vector_loop
            )

            def reduce_x(_: int, acc: pk.Acc[pk.double]):
                acc += pot_x

            def reduce_y(_: int, acc: pk.Acc[pk.double]):
                acc += pot_y

            def reduce_z(_: int, acc: pk.Acc[pk.double]):
                acc += pot_z

            reduced_pot_x: pk.double = pk.parallel_reduce(
                pk.ThreadVectorRange(team_member, self.vector_size), reduce_x
            )
            reduced_pot_y: pk.double = pk.parallel_reduce(
                pk.ThreadVectorRange(team_member, self.vector_size), reduce_y
            )
            reduced_pot_z: pk.double = pk.parallel_reduce(
                pk.ThreadVectorRange(team_member, self.vector_size), reduce_z
            )

            def write():
                self.potentials[0][t_idx] = reduced_pot_x
                self.potentials[1][t_idx] = reduced_pot_y
                self.potentials[2][t_idx] = reduced_pot_z

            pk.single(pk.PerThread(team_member), write)

        pk.parallel_for(
            pk.TeamThreadRange(team_member, self.t_cell_chunk_size), target_loop
        )


# END TEMPLATE P2PWORKLOAD

# START APPLICATION CELL _
# END APPLICATION CELL _

# START APPLICATION REAL3D _
# END APPLICATION REAL3D _

# START APPLICATION P2PWORKLOAD single_force_cuda
@pk.workload(
    potentials=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    t_counter=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    targets_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.CudaSpace
    ),
    s_counter=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    sources_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.CudaSpace
    ),
    forces_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    normals_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    t_list2global=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    nz2t_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    nz2s_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    t2nz_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    s2nz_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
)
# END APPLICATION P2PWORKLOAD single_force_cuda
# START APPLICATION P2PWORKLOAD single_force_hip
@pk.workload(
    potentials=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    t_counter=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    targets_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HIPSpace
    ),
    s_counter=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    sources_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HIPSpace
    ),
    forces_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    normals_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    t_list2global=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    nz2t_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    nz2s_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    t2nz_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    s2nz_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
)
# END APPLICATION P2PWORKLOAD single_force_hip
# START APPLICATION P2PWORKLOAD single_force_host
@pk.workload(
    potentials=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HostSpace
    ),
    t_counter=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    targets_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    s_counter=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    sources_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    forces_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HostSpace
    ),
    normals_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HostSpace
    ),
    t_list2global=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    nz2t_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    nz2s_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    t2nz_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    s2nz_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
)
# END APPLICATION P2PWORKLOAD single_force_host
# START APPLICATION P2PWORKLOAD multi_forces_cuda
@pk.workload(
    potentials=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.CudaSpace
    ),
    t_counter=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    targets_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.CudaSpace
    ),
    s_counter=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    sources_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.CudaSpace
    ),
    forces_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.CudaSpace
    ),
    normals_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.CudaSpace
    ),
    t_list2global=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    nz2t_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    nz2s_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    t2nz_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    s2nz_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
)
# END APPLICATION P2PWORKLOAD multi_forces_cuda
# START APPLICATION P2PWORKLOAD multi_forces_hip
@pk.workload(
    potentials=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HIPSpace
    ),
    t_counter=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    targets_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HIPSpace
    ),
    s_counter=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    sources_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HIPSpace
    ),
    forces_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HIPSpace
    ),
    normals_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HIPSpace
    ),
    t_list2global=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    nz2t_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    nz2s_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    t2nz_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    s2nz_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
)
# END APPLICATION P2PWORKLOAD multi_forces_hip
# START APPLICATION P2PWORKLOAD multi_forces_host
@pk.workload(
    potentials=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    t_counter=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    targets_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    s_counter=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    sources_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    forces_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    normals_list=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    t_list2global=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    nz2t_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    nz2s_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    t2nz_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    s2nz_cell_map=pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
)
# END APPLICATION P2PWORKLOAD multi_forces_host
