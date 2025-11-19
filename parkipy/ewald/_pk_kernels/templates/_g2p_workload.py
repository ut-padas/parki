__all__ = [
    "g2p_workload_cuda_fp64",
    "g2p_workload_hip_fp64",
    "g2p_workload_host_fp64",
    "g2p_workload_cuda_fp32",
    "g2p_workload_hip_fp32",
    "g2p_workload_host_fp32",
]

import math
import pykokkos as pk
from typing import List


# START TEMPLATE G2PWORKLOAD
class g2p_workload_G2PWORKLOAD:
    def __init__(
        self,
        potentials: pk.View2D[pk.double],
        targets: pk.View2D[pk.double],
        t_l2g: pk.View1D[int],
        H: pk.View4D[pk.double],
        window_P: int,
        dim_out: int,
        grid_shape: List[int],
        h: pk.double,
        num_cells: int,
        nnz_cells: int,
        cell_size: int,
        variant_flag: pk.uint8,
        periodicity: int,
        threads: int,
    ):
        # input arrays
        self.potentials: pk.View2D[pk.double] = potentials
        self.targets: pk.View2D[pk.double] = targets
        self.t_l2g: pk.View1D[int] = t_l2g
        self.H: pk.View4D[pk.double] = H
        # input arguments
        self.window_P: int = window_P
        self.dim_out: int = dim_out
        self.H_dim_x: int = grid_shape[0]
        self.H_dim_y: int = grid_shape[1]
        self.H_dim_z: int = grid_shape[2]
        self.h: pk.double = h
        self.num_cells: int = num_cells
        self.nnz_cells: int = nnz_cells
        self.cell_size: int = cell_size
        self.periodicity: int = periodicity
        self.threads: int = threads
        # flags
        self.variant_flag: pk.uint8 = variant_flag
        # determined arguments
        self.hhh: pk.double = h * h * h
        self.po2: int = self.window_P // 2
        self.po2_cbd: int = self.po2 * self.po2 * self.po2
        self.nt: int = self.potentials.shape[1]
        ## thread specific variables
        self.target_teams: int = math.ceil(self.nt / self.threads)
        self.cell_teams: int = math.ceil(self.cell_size / self.threads)

    @pk.main
    def run(self):
        if self.variant_flag == 0:
            pk.parallel_for(
                "G2P-BASE",
                pk.TeamPolicy(self.target_teams, self.threads),
                self.g2p_base,
            )
            return
            pk.parallel_for("G2P-BASE (range)", self.nt, self.g2p_base_range)
        elif self.variant_flag == 1:
            pk.parallel_for(
                "G2P-TARGET",
                pk.TeamPolicy(self.nnz_cells * self.cell_teams, self.threads),
                self.g2p_target,
            )
            return
            pk.parallel_for(
                "G2P-TARGET (range)",
                self.nnz_cells * self.cell_size,
                self.g2p_target_range,
            )
        elif self.variant_flag == 2:
            pk.parallel_for(
                "G2P-GRID (depreciated)",
                pk.TeamPolicy(self.num_cells, "auto"),
                self.g2p_grid,
            )
        else:
            pk.printf("Variant flag %i not supported, returning!\n", self.variant_flag)

    @pk.workunit
    def g2p_base(self, team_member: pk.TeamMember):
        t_off: int = team_member.league_rank() * self.threads

        def thread_loop(tid: int):
            t: int = t_off + tid
            if t >= self.nt:
                return
            # get grid index and distance
            target_i: pk.double = self.targets[0][t]
            target_j: pk.double = self.targets[1][t]
            target_k: pk.double = self.targets[2][t]
            grid_i: int = int(target_i)
            grid_j: int = int(target_j)
            grid_k: int = int(target_k)
            d_x: pk.double = target_i - grid_i
            d_y: pk.double = target_j - grid_j
            d_z: pk.double = target_k - grid_k
            # get the ftl corner of the p-cube
            a_x: int = grid_i - (self.po2 - 1)
            a_y: int = grid_j - (self.po2 - 1)
            a_z: int = grid_k - (self.po2 - 1)
            # allocate registers for window function
            w_x: List[pk.double] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,]  # NOTE: placeholder for max windowP of 14, would like to be [0] * self.window_P
            w_y: List[pk.double] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,]  # NOTE: placeholder for max windowP of 14, would like to be [0] * self.window_P
            w_z: List[pk.double] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,]  # NOTE: placeholder for max windowP of 14, would like to be [0] * self.window_P
            # compute window function
            for r in range(self.window_P):
                if self.window_P == 14:
                    w_x[r] = self._basic_kaiser_poly_p14(d_x, r)
                    w_y[r] = self._basic_kaiser_poly_p14(d_y, r)
                    w_z[r] = self._basic_kaiser_poly_p14(d_z, r)
                elif self.window_P == 12:
                    w_x[r] = self._basic_kaiser_poly_p12(d_x, r)
                    w_y[r] = self._basic_kaiser_poly_p12(d_y, r)
                    w_z[r] = self._basic_kaiser_poly_p12(d_z, r)
                elif self.window_P == 10:
                    w_x[r] = self._basic_kaiser_poly_p10(d_x, r)
                    w_y[r] = self._basic_kaiser_poly_p10(d_y, r)
                    w_z[r] = self._basic_kaiser_poly_p10(d_z, r)
                if self.window_P == 8:
                    w_x[r] = self._basic_kaiser_poly_p8(d_x, r)
                    w_y[r] = self._basic_kaiser_poly_p8(d_y, r)
                    w_z[r] = self._basic_kaiser_poly_p8(d_z, r)
                elif self.window_P == 6:
                    w_x[r] = self._basic_kaiser_poly_p6(d_x, r)
                    w_y[r] = self._basic_kaiser_poly_p6(d_y, r)
                    w_z[r] = self._basic_kaiser_poly_p6(d_z, r)
                elif self.window_P == 4:
                    w_x[r] = self._basic_kaiser_poly_p4(d_x, r)
                    w_y[r] = self._basic_kaiser_poly_p4(d_y, r)
                    w_z[r] = self._basic_kaiser_poly_p4(d_z, r)
                elif self.window_P == 2:
                    w_x[r] = self._basic_kaiser_poly_p2(d_x, r)
                    w_y[r] = self._basic_kaiser_poly_p2(d_y, r)
                    w_z[r] = self._basic_kaiser_poly_p2(d_z, r)

            # loop over p-cube
            wx: pk.double = 0.0
            wx_wy: pk.double = 0.0
            w: pk.double = 0.0
            l: int = 0
            m: int = 0
            n: int = 0
            potj: List[pk.double] = [0, 0, 0]
            for i in range(self.window_P):
                wx = w_x[i]
                l = a_x + i
                if self.periodicity >= 1:
                    l -= (l >= self.H_dim_x) * self.H_dim_x
                    l += (l < 0) * self.H_dim_x
                else:
                    if l >= self.H_dim_x or l < 0:
                        continue
                for j in range(self.window_P):
                    wx_wy = wx * w_y[j]
                    m = a_y + j
                    if self.periodicity >= 2:
                        m -= (m >= self.H_dim_y) * self.H_dim_y
                        m += (m < 0) * self.H_dim_y
                    else:
                        if m >= self.H_dim_y or m < 0:
                            continue
                    for k in range(self.window_P):
                        n = a_z + k
                        if self.periodicity >= 3:
                            n -= (n >= self.H_dim_z) * self.H_dim_z
                            n += (n < 0) * self.H_dim_z
                        else:
                            if n >= self.H_dim_z or n < 0:
                                continue
                        w = wx_wy * w_z[k]
                        for d in range(self.dim_out):
                            potj[d] += w * self.H[l][m][n][d]
            for d in range(self.dim_out):
                self.potentials[d][t] = self.hhh * potj[d]

        pk.parallel_for(pk.TeamThreadRange(team_member, self.threads), thread_loop)

    @pk.workunit
    def g2p_base_range(self, t: int):
        # get grid index and distance
        target_i: pk.double = self.targets[0][t]
        target_j: pk.double = self.targets[1][t]
        target_k: pk.double = self.targets[2][t]
        grid_i: int = int(target_i)
        grid_j: int = int(target_j)
        grid_k: int = int(target_k)
        d_x: pk.double = target_i - grid_i
        d_y: pk.double = target_j - grid_j
        d_z: pk.double = target_k - grid_k
        # get the ftl corner of the p-cube
        a_x: int = grid_i - (self.po2 - 1)
        a_y: int = grid_j - (self.po2 - 1)
        a_z: int = grid_k - (self.po2 - 1)
        # allocate registers for window function
        w_x: List[pk.double] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,]  # NOTE: placeholder for max windowP of 14, would like to be [0] * self.window_P
        w_y: List[pk.double] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,]  # NOTE: placeholder for max windowP of 14, would like to be [0] * self.window_P
        w_z: List[pk.double] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,]  # NOTE: placeholder for max windowP of 14, would like to be [0] * self.window_P
        # compute window function
        for r in range(self.window_P):
            if self.window_P == 14:
                w_x[r] = self._basic_kaiser_poly_p14(d_x, r)
                w_y[r] = self._basic_kaiser_poly_p14(d_y, r)
                w_z[r] = self._basic_kaiser_poly_p14(d_z, r)
            elif self.window_P == 12:
                w_x[r] = self._basic_kaiser_poly_p12(d_x, r)
                w_y[r] = self._basic_kaiser_poly_p12(d_y, r)
                w_z[r] = self._basic_kaiser_poly_p12(d_z, r)
            elif self.window_P == 10:
                w_x[r] = self._basic_kaiser_poly_p10(d_x, r)
                w_y[r] = self._basic_kaiser_poly_p10(d_y, r)
                w_z[r] = self._basic_kaiser_poly_p10(d_z, r)
            if self.window_P == 8:
                w_x[r] = self._basic_kaiser_poly_p8(d_x, r)
                w_y[r] = self._basic_kaiser_poly_p8(d_y, r)
                w_z[r] = self._basic_kaiser_poly_p8(d_z, r)
            elif self.window_P == 6:
                w_x[r] = self._basic_kaiser_poly_p6(d_x, r)
                w_y[r] = self._basic_kaiser_poly_p6(d_y, r)
                w_z[r] = self._basic_kaiser_poly_p6(d_z, r)
            elif self.window_P == 4:
                w_x[r] = self._basic_kaiser_poly_p4(d_x, r)
                w_y[r] = self._basic_kaiser_poly_p4(d_y, r)
                w_z[r] = self._basic_kaiser_poly_p4(d_z, r)
            elif self.window_P == 2:
                w_x[r] = self._basic_kaiser_poly_p2(d_x, r)
                w_y[r] = self._basic_kaiser_poly_p2(d_y, r)
                w_z[r] = self._basic_kaiser_poly_p2(d_z, r)

        # loop over p-cube
        wx: pk.double = 0.0
        wx_wy: pk.double = 0.0
        w: pk.double = 0.0
        l: int = 0
        m: int = 0
        n: int = 0
        potj: List[pk.double] = [0, 0, 0]
        for i in range(self.window_P):
            wx = w_x[i]
            l = a_x + i
            if self.periodicity >= 1:
                l -= (l >= self.H_dim_x) * self.H_dim_x
                l += (l < 0) * self.H_dim_x
            else:
                if l >= self.H_dim_x or l < 0:
                    continue
            for j in range(self.window_P):
                wx_wy = wx * w_y[j]
                m = a_y + j
                if self.periodicity >= 2:
                    m -= (m >= self.H_dim_y) * self.H_dim_y
                    m += (m < 0) * self.H_dim_y
                else:
                    if m >= self.H_dim_y or m < 0:
                        continue
                for k in range(self.window_P):
                    n = a_z + k
                    if self.periodicity >= 3:
                        n -= (n >= self.H_dim_z) * self.H_dim_z
                        n += (n < 0) * self.H_dim_z
                    else:
                        if n >= self.H_dim_z or n < 0:
                            continue
                    w = wx_wy * w_z[k]
                    for d in range(self.dim_out):
                        potj[d] += w * self.H[l][m][n][d]
        for d in range(self.dim_out):
            self.potentials[d][t] = self.hhh * potj[d]

    @pk.workunit
    def g2p_target(self, team_member: pk.TeamMember):
        nz_cell: int = team_member.league_rank() // self.cell_teams
        cell_team: int = team_member.league_rank() % self.cell_teams
        cell_off: int = cell_team * self.threads
        t_off: int = nz_cell * self.cell_size + cell_off

        def thread_loop(tid: int):
            if cell_off + tid >= self.cell_size:
                return
            t: int = t_off + tid
            t_glb: int = self.t_l2g[t]
            if t_glb < 0:
                return
            t_x: pk.double = self.targets[0][t]
            t_y: pk.double = self.targets[1][t]
            t_z: pk.double = self.targets[2][t]
            # get grid index and distance
            g_x: int = int(t_x)
            g_y: int = int(t_y)
            g_z: int = int(t_z)
            d_x: pk.double = t_x - g_x
            d_y: pk.double = t_y - g_y
            d_z: pk.double = t_z - g_z
            # get the ftl corner of the p-cube (i.e., the anchors)
            a_x: int = g_x - (self.po2 - 1)
            a_y: int = g_y - (self.po2 - 1)
            a_z: int = g_z - (self.po2 - 1)
            # allocate registers for window function
            w_x: List[pk.double] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,]  # NOTE: placeholder for max windowP of 14, would like to be [0] * self.window_P
            w_y: List[pk.double] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,]  # NOTE: placeholder for max windowP of 14, would like to be [0] * self.window_P
            w_z: List[pk.double] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,]  # NOTE: placeholder for max windowP of 14, would like to be [0] * self.window_P
            # compute window function
            for r in range(self.window_P):
                if self.window_P == 14:
                    w_x[r] = self._basic_kaiser_poly_p14(d_x, r)
                    w_y[r] = self._basic_kaiser_poly_p14(d_y, r)
                    w_z[r] = self._basic_kaiser_poly_p14(d_z, r)
                elif self.window_P == 12:
                    w_x[r] = self._basic_kaiser_poly_p12(d_x, r)
                    w_y[r] = self._basic_kaiser_poly_p12(d_y, r)
                    w_z[r] = self._basic_kaiser_poly_p12(d_z, r)
                elif self.window_P == 10:
                    w_x[r] = self._basic_kaiser_poly_p10(d_x, r)
                    w_y[r] = self._basic_kaiser_poly_p10(d_y, r)
                    w_z[r] = self._basic_kaiser_poly_p10(d_z, r)
                if self.window_P == 8:
                    w_x[r] = self._basic_kaiser_poly_p8(d_x, r)
                    w_y[r] = self._basic_kaiser_poly_p8(d_y, r)
                    w_z[r] = self._basic_kaiser_poly_p8(d_z, r)
                elif self.window_P == 6:
                    w_x[r] = self._basic_kaiser_poly_p6(d_x, r)
                    w_y[r] = self._basic_kaiser_poly_p6(d_y, r)
                    w_z[r] = self._basic_kaiser_poly_p6(d_z, r)
                elif self.window_P == 4:
                    w_x[r] = self._basic_kaiser_poly_p4(d_x, r)
                    w_y[r] = self._basic_kaiser_poly_p4(d_y, r)
                    w_z[r] = self._basic_kaiser_poly_p4(d_z, r)
                elif self.window_P == 2:
                    w_x[r] = self._basic_kaiser_poly_p2(d_x, r)
                    w_y[r] = self._basic_kaiser_poly_p2(d_y, r)
                    w_z[r] = self._basic_kaiser_poly_p2(d_z, r)

            # loop over p-cube
            wx: pk.double = 0.0
            wx_wy: pk.double = 0.0
            w: pk.double = 0.0
            l: int = 0
            m: int = 0
            n: int = 0
            potj: List[pk.double] = [0, 0, 0]
            for i in range(self.window_P):
                wx = w_x[i]
                l = a_x + i
                if self.periodicity >= 1:
                    l -= (l >= self.H_dim_x) * self.H_dim_x
                    l += (l < 0) * self.H_dim_x
                else:
                    if l >= self.H_dim_x or l < 0:
                        continue
                for j in range(self.window_P):
                    wx_wy = wx * w_y[j]
                    m = a_y + j
                    if self.periodicity >= 2:
                        m -= (m >= self.H_dim_y) * self.H_dim_y
                        m += (m < 0) * self.H_dim_y
                    else:
                        if m >= self.H_dim_y or m < 0:
                            continue
                    for k in range(self.window_P):
                        n = a_z + k
                        if self.periodicity >= 3:
                            n -= (n >= self.H_dim_z) * self.H_dim_z
                            n += (n < 0) * self.H_dim_z
                        else:
                            if n >= self.H_dim_z or n < 0:
                                continue

                        w = wx_wy * w_z[k]
                        for d in range(self.dim_out):
                            potj[d] += w * self.H[d][l][m][n]
            for d in range(self.dim_out):
                self.potentials[d][t_glb] = self.hhh * potj[d]

        pk.parallel_for(pk.TeamThreadRange(team_member, self.threads), thread_loop)

    @pk.workunit
    def g2p_target_range(self, t: int):
        t_glb: int = self.t_l2g[t]
        if t_glb < 0:
            return
        t_x: pk.double = self.targets[0][t]
        t_y: pk.double = self.targets[1][t]
        t_z: pk.double = self.targets[2][t]
        # get grid index and distance
        g_x: int = int(t_x)
        g_y: int = int(t_y)
        g_z: int = int(t_z)
        d_x: pk.double = t_x - g_x
        d_y: pk.double = t_y - g_y
        d_z: pk.double = t_z - g_z
        # get the ftl corner of the p-cube (i.e., the anchors)
        a_x: int = g_x - (self.po2 - 1)
        a_y: int = g_y - (self.po2 - 1)
        a_z: int = g_z - (self.po2 - 1)
        # allocate registers for window function
        w_x: List[pk.double] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,]  # NOTE: placeholder for max windowP of 14, would like to be [0] * self.window_P
        w_y: List[pk.double] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,]  # NOTE: placeholder for max windowP of 14, would like to be [0] * self.window_P
        w_z: List[pk.double] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,]  # NOTE: placeholder for max windowP of 14, would like to be [0] * self.window_P
        # compute window function
        for r in range(self.window_P):
            if self.window_P == 14:
                w_x[r] = self._basic_kaiser_poly_p14(d_x, r)
                w_y[r] = self._basic_kaiser_poly_p14(d_y, r)
                w_z[r] = self._basic_kaiser_poly_p14(d_z, r)
            elif self.window_P == 12:
                w_x[r] = self._basic_kaiser_poly_p12(d_x, r)
                w_y[r] = self._basic_kaiser_poly_p12(d_y, r)
                w_z[r] = self._basic_kaiser_poly_p12(d_z, r)
            elif self.window_P == 10:
                w_x[r] = self._basic_kaiser_poly_p10(d_x, r)
                w_y[r] = self._basic_kaiser_poly_p10(d_y, r)
                w_z[r] = self._basic_kaiser_poly_p10(d_z, r)
            if self.window_P == 8:
                w_x[r] = self._basic_kaiser_poly_p8(d_x, r)
                w_y[r] = self._basic_kaiser_poly_p8(d_y, r)
                w_z[r] = self._basic_kaiser_poly_p8(d_z, r)
            elif self.window_P == 6:
                w_x[r] = self._basic_kaiser_poly_p6(d_x, r)
                w_y[r] = self._basic_kaiser_poly_p6(d_y, r)
                w_z[r] = self._basic_kaiser_poly_p6(d_z, r)
            elif self.window_P == 4:
                w_x[r] = self._basic_kaiser_poly_p4(d_x, r)
                w_y[r] = self._basic_kaiser_poly_p4(d_y, r)
                w_z[r] = self._basic_kaiser_poly_p4(d_z, r)
            elif self.window_P == 2:
                w_x[r] = self._basic_kaiser_poly_p2(d_x, r)
                w_y[r] = self._basic_kaiser_poly_p2(d_y, r)
                w_z[r] = self._basic_kaiser_poly_p2(d_z, r)

        # loop over p-cube
        wx: pk.double = 0.0
        wx_wy: pk.double = 0.0
        w: pk.double = 0.0
        l: int = 0
        m: int = 0
        n: int = 0
        potj: List[pk.double] = [0, 0, 0]
        for i in range(self.window_P):
            wx = w_x[i]
            l = a_x + i
            if self.periodicity >= 1:
                l -= (l >= self.H_dim_x) * self.H_dim_x
                l += (l < 0) * self.H_dim_x
            else:
                if l >= self.H_dim_x or l < 0:
                    continue
            for j in range(self.window_P):
                wx_wy = wx * w_y[j]
                m = a_y + j
                if self.periodicity >= 2:
                    m -= (m >= self.H_dim_y) * self.H_dim_y
                    m += (m < 0) * self.H_dim_y
                else:
                    if m >= self.H_dim_y or m < 0:
                        continue
                for k in range(self.window_P):
                    n = a_z + k
                    if self.periodicity >= 3:
                        n -= (n >= self.H_dim_z) * self.H_dim_z
                        n += (n < 0) * self.H_dim_z
                    else:
                        if n >= self.H_dim_z or n < 0:
                            continue
                    w = wx_wy * w_z[k]
                    for d in range(self.dim_out):
                        potj[d] += w * self.H[d][l][m][n]
        for d in range(self.dim_out):
            self.potentials[d][t_glb] = self.hhh * potj[d]

    @pk.workunit
    def g2p_grid(self, team_member: pk.TeamMember):
        cell: int = team_member.league_rank()

        def load_shmem(ll: int):
            return

        # pk.parallel_for(team_member, self.po2_cbd)
        team_member.team_barrier()

    # NOTE: These functions are generated by ``generate_basic_kaiser_poly_functions.py``.
    # They should not be modified by hand.
    @pk.function
    def _basic_kaiser_poly_p2(x: pk.double, i: int) -> float:
        """
        Evaluate the basic kaiser polynomial of support 2 and degree 2.

        Parameters
        ----------
        x : pk.double,
            the point to be evaluated
        i : int
            the fourier grid positioning of the
            point where the 0 index is the
            front/top/left and the 9 index
            is the back/bottom/right corner
            of a p-cube

        Returns
        -------
        pk.double
            output of the basic kaiser polynomial
        """
        z: pk.double = 2.0 * x - 1.0
        c0: pk.double = 0.0
        c1: pk.double = 0.0
        c2: pk.double = 0.0

        if i == 0:
            c0 = 5.5285176969913241e-01
            c1 = -5.2944029903301548e-01
            c2 = -2.8456480645416164e-02
        else:  # i == 1
            c0 = 5.5285176969913241e-01
            c1 = 5.2944029903301548e-01
            c2 = -2.8456480645416164e-02

        return c0 + z * (c1 + z * c2)

    @pk.function
    def _basic_kaiser_poly_p4(x: pk.double, i: int) -> float:
        """
        Evaluate the basic kaiser polynomial of support 4 and degree 3.

        Parameters
        ----------
        x : pk.double,
            the point to be evaluated
        i : int
            the fourier grid positioning of the
            point where the 0 index is the
            front/top/left and the 9 index
            is the back/bottom/right corner
            of a p-cube

        Returns
        -------
        pk.double
            output of the basic kaiser polynomial
        """
        z: pk.double = 2.0 * x - 1.0
        c0: pk.double = 0.0
        c1: pk.double = 0.0
        c2: pk.double = 0.0
        c3: pk.double = 0.0

        if i == 0:
            c0 = 4.2449397753344893e-02
            c1 = -1.1014994798585261e-01
            c2 = 9.9342620436954909e-02
            c3 = -3.1323192806224052e-02
        elif i == 1:
            c0 = 7.3973436857626695e-01
            c1 = -4.5107441408235749e-01
            c2 = -9.9127437865195789e-02
            c3 = 9.3305063138205149e-02
        elif i == 2:
            c0 = 7.3973436857626695e-01
            c1 = 4.5107441408235749e-01
            c2 = -9.9127437865195858e-02
            c3 = -9.3305063138205149e-02
        else:  # i == 3
            c0 = 4.2449397753344872e-02
            c1 = 1.1014994798585256e-01
            c2 = 9.9342620436954951e-02
            c3 = 3.1323192806224094e-02

        return c0 + z * (c1 + z * (c2 + z * c3))

    @pk.function
    def _basic_kaiser_poly_p6(x: pk.double, i: int) -> float:
        """
        Evaluate the basic kaiser polynomial of support 6 and degree 4.

        Parameters
        ----------
        x : pk.double,
            the point to be evaluated
        i : int
            the fourier grid positioning of the
            point where the 0 index is the
            front/top/left and the 9 index
            is the back/bottom/right corner
            of a p-cube

        Returns
        -------
        pk.double
            output of the basic kaiser polynomial
        """
        z: pk.double = 2.0 * x - 1.0
        c0: pk.double = 0.0
        c1: pk.double = 0.0
        c2: pk.double = 0.0
        c3: pk.double = 0.0
        c4: pk.double = 0.0

        if i == 0:
            c0 = 1.6540370225928969e-03
            c1 = -5.7277155781439949e-03
            c2 = 8.5348088270022649e-03
            c3 = -6.9934428864010332e-03
            c4 = 2.5617798426059981e-03
        elif i == 1:
            c0 = 1.4423601720594059e-01
            c1 = -2.0064113179630325e-01
            c2 = 9.4632600070752290e-02
            c3 = -5.2234361541604619e-03
            c4 = -7.6482720120769265e-03
        elif i == 2:
            c0 = 8.1657974641970854e-01
            c1 = -3.3189172963931191e-01
            c2 = -1.0317396726045011e-01
            c3 = 5.0627398974012756e-02
            c4 = 5.0894078772892436e-03
        elif i == 3:
            c0 = 8.1657974641970843e-01
            c1 = 3.3189172963931207e-01
            c2 = -1.0317396726044990e-01
            c3 = -5.0627398974012909e-02
            c4 = 5.0894078772892436e-03
        elif i == 4:
            c0 = 1.4423601720594048e-01
            c1 = 2.0064113179630322e-01
            c2 = 9.4632600070752693e-02
            c3 = 5.2234361541604940e-03
            c4 = -7.6482720120771607e-03
        else:  # i == 5
            c0 = 1.6540370225928977e-03
            c1 = 5.7277155781439732e-03
            c2 = 8.5348088270022181e-03
            c3 = 6.9934428864010115e-03
            c4 = 2.5617798426060038e-03

        return c0 + z * (c1 + z * (c2 + z * (c3 + z * c4)))

    @pk.function
    def _basic_kaiser_poly_p8(x: pk.double, i: int) -> float:
        """
        Evaluate the basic kaiser polynomial of support 8 and degree 5.

        Parameters
        ----------
        x : pk.double,
            the point to be evaluated
        i : int
            the fourier grid positioning of the
            point where the 0 index is the
            front/top/left and the 9 index
            is the back/bottom/right corner
            of a p-cube

        Returns
        -------
        pk.double
            output of the basic kaiser polynomial
        """
        z: pk.double = 2.0 * x - 1.0
        c0: pk.double = 0.0
        c1: pk.double = 0.0
        c2: pk.double = 0.0
        c3: pk.double = 0.0
        c4: pk.double = 0.0
        c5: pk.double = 0.0

        if i == 0:
            c0 = 4.8578020255329445e-05
            c1 = -2.0473311653615309e-04
            c2 = 3.6500015171070258e-04
            c3 = -3.9515382923689755e-04
            c4 = 2.9285217772518528e-04
            c5 = -1.0719423279770371e-04
        elif i == 1:
            c0 = 1.4093074597497146e-02
            c1 = -2.7296566196224156e-02
            c2 = 2.2076485169361951e-02
            c3 = -9.0587423929823149e-03
            c4 = 1.4356460491201392e-03
            c5 = 1.6757024091231378e-04
        elif i == 2:
            c0 = 2.4145586160286991e-01
            c1 = -2.3749999374500885e-01
            c2 = 7.0809793325569650e-02
            c3 = 4.4203335381892200e-03
            c4 = -6.0644287988766959e-03
            c5 = 6.6216983861179990e-04
        elif i == 3:
            c0 = 8.5823242657076015e-01
            c1 = -2.6341440579190539e-01
            c2 = -9.3251248511389051e-02
            c3 = 3.4799176677449861e-02
            c4 = 4.3359086970709724e-03
            c5 = -2.0739456785273101e-03
        elif i == 4:
            c0 = 8.5823242657075993e-01
            c1 = 2.6341440579190545e-01
            c2 = -9.3251248511389689e-02
            c3 = -3.4799176677449944e-02
            c4 = 4.3359086970715666e-03
            c5 = 2.0739456785273235e-03
        elif i == 5:
            c0 = 2.4145586160286983e-01
            c1 = 2.3749999374500880e-01
            c2 = 7.0809793325569845e-02
            c3 = -4.4203335381890240e-03
            c4 = -6.0644287988769179e-03
            c5 = -6.6216983861192415e-04
        elif i == 6:
            c0 = 1.4093074597497150e-02
            c1 = 2.7296566196224142e-02
            c2 = 2.2076485169361931e-02
            c3 = 9.0587423929823375e-03
            c4 = 1.4356460491201494e-03
            c5 = -1.6757024091232399e-04
        else:  # i == 7
            c0 = 4.8578020255329879e-05
            c1 = 2.0473311653615341e-04
            c2 = 3.6500015171069737e-04
            c3 = 3.9515382923689196e-04
            c4 = 2.9285217772519390e-04
            c5 = 1.0719423279771294e-04

        return c0 + z * (c1 + z * (c2 + z * (c3 + z * (c4 + z * c5))))

    @pk.function
    def _basic_kaiser_poly_p10(x: pk.double, i: int) -> float:
        """
        Evaluate the basic kaiser polynomial of support 10 and degree 6.

        Parameters
        ----------
        x : pk.double,
            the point to be evaluated
        i : int
            the fourier grid positioning of the
            point where the 0 index is the
            front/top/left and the 9 index
            is the back/bottom/right corner
            of a p-cube

        Returns
        -------
        pk.double
            output of the basic kaiser polynomial
        """
        z: pk.double = 2.0 * x - 1.0
        c0: pk.double = 0.0
        c1: pk.double = 0.0
        c2: pk.double = 0.0
        c3: pk.double = 0.0
        c4: pk.double = 0.0
        c5: pk.double = 0.0
        c6: pk.double = 0.0

        if i == 0:
            c0 = 1.1444139894604897e-06
            c1 = -5.7109581755672432e-06
            c2 = 1.2278267964705377e-05
            c3 = -1.5003044716551718e-05
            c4 = 1.2921143664270225e-05
            c5 = -8.6833898998692485e-06
            c6 = 3.0638201762096556e-06
        elif i == 1:
            c0 = 9.3404383654471725e-04
            c1 = -2.2234946772728670e-03
            c2 = 2.3402638135243542e-03
            c3 = -1.4137605675357805e-03
            c4 = 5.2107789749180461e-04
            c5 = -1.0494834350720240e-04
            c6 = 5.6022316099124393e-06
        elif i == 2:
            c0 = 3.7755086353646147e-02
            c1 = -5.3221014661241697e-02
            c2 = 3.0458292112371211e-02
            c3 = -8.1430479799957343e-03
            c4 = 4.4803646178746468e-04
            c5 = 3.1974377755057670e-04
            c6 = -7.3993233113948034e-05
        elif i == 3:
            c0 = 3.2378023294326008e-01
            c1 = -2.4916690147223192e-01
            c2 = 5.0327375262011614e-02
            c3 = 9.0023521503664015e-03
            c4 = -4.5997734587431109e-03
            c5 = 1.3109734014771798e-04
            c6 = 1.5771694832438472e-04
        elif i == 4:
            c0 = 8.8446244968820342e-01
            c1 = -2.1771131331760402e-01
            c2 = -8.3138209766243240e-02
            c3 = 2.3739526096388130e-02
            c4 = 3.6177389765869568e-03
            c5 = -1.1792176658678142e-03
            c6 = -9.2390500449356287e-05
        elif i == 5:
            c0 = 8.8446244968820331e-01
            c1 = 2.1771131331760363e-01
            c2 = -8.3138209766243629e-02
            c3 = -2.3739526096386278e-02
            c4 = 3.6177389765874920e-03
            c5 = 1.1792176658662991e-03
            c6 = -9.2390500449356287e-05
        elif i == 6:
            c0 = 3.2378023294326003e-01
            c1 = 2.4916690147223222e-01
            c2 = 5.0327375262011323e-02
            c3 = -9.0023521503688510e-03
            c4 = -4.5997734587404767e-03
            c5 = -1.3109734014548306e-04
            c6 = 1.5771694832209836e-04
        elif i == 7:
            c0 = 3.7755086353646147e-02
            c1 = 5.3221014661241697e-02
            c2 = 3.0458292112371242e-02
            c3 = 8.1430479799956389e-03
            c4 = 4.4803646178726925e-04
            c5 = -3.1974377755047208e-04
            c6 = -7.3993233113747972e-05
        elif i == 8:
            c0 = 9.3404383654471703e-04
            c1 = 2.2234946772728731e-03
            c2 = 2.3402638135243811e-03
            c3 = 1.4137605675358007e-03
            c4 = 5.2107789749178813e-04
            c5 = 1.0494834350717542e-04
            c6 = 5.6022316099021681e-06
        else:  # i == 9
            c0 = 1.1444139894605024e-06
            c1 = 5.7109581755673889e-06
            c2 = 1.2278267964705636e-05
            c3 = 1.5003044716551524e-05
            c4 = 1.2921143664269732e-05
            c5 = 8.6833898998693095e-06
            c6 = 3.0638201762098940e-06

        return c0 + z * (c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * c6)))))

    @pk.function
    def _basic_kaiser_poly_p12(x: pk.double, i: int) -> float:
        """
        Evaluate the basic kaiser polynomial of support 12 and degree 8.

        Parameters
        ----------
        x : pk.double,
            the point to be evaluated
        i : int
            the fourier grid positioning of the
            point where the 0 index is the
            front/top/left and the 9 index
            is the back/bottom/right corner
            of a p-cube

        Returns
        -------
        pk.double
            output of the basic kaiser polynomial
        """
        z: pk.double = 2.0 * x - 1.0
        c0: pk.double = 0.0
        c1: pk.double = 0.0
        c2: pk.double = 0.0
        c3: pk.double = 0.0
        c4: pk.double = 0.0
        c5: pk.double = 0.0
        c6: pk.double = 0.0
        c7: pk.double = 0.0
        c8: pk.double = 0.0

        if i == 0:
            c0 = 2.4000869177589145e-08
            c1 = -1.3165411022430414e-07
            c2 = 3.2556973918858509e-07
            c3 = -4.8177614434634580e-07
            c4 = 4.7338360924466619e-07
            c5 = -3.2428381026128102e-07
            c6 = 1.6569803565965060e-07
            c7 = -6.7724562122442271e-08
            c8 = 1.6797773443477222e-08
        elif i == 1:
            c0 = 4.7824764029482742e-05
            c1 = -1.3210963439974909e-04
            c2 = 1.6594105667277210e-04
            c3 = -1.2466422784728479e-04
            c4 = 6.1740613696053450e-05
            c5 = -2.0784779413243903e-05
            c6 = 4.5914893479062664e-06
            c7 = -5.0875618285466670e-07
            c8 = -1.9684788067570534e-08
        elif i == 2:
            c0 = 3.9738148632413013e-03
            c1 = -6.9868611719839119e-03
            c2 = 5.3912787096484114e-03
            c3 = -2.3337438232604812e-03
            c4 = 5.8109587927006974e-04
            c5 = -6.3319612173018367e-05
            c6 = -7.1084554053383563e-06
            c7 = 3.3237789418683282e-06
            c8 = -3.3452620549359216e-07
        elif i == 3:
            c0 = 6.8555270927532691e-02
            c1 = -7.7101687470241029e-02
            c2 = 3.4057664727189768e-02
            c3 = -6.1792584626924199e-03
            c4 = -2.8792409251734860e-04
            c5 = 3.1734188868718879e-04
            c6 = -3.9126640860095966e-05
            c7 = -4.3114676234924414e-06
            c8 = 1.3782753994837844e-06
        elif i == 4:
            c0 = 3.9205672959747978e-01
            c1 = -2.4867649569508812e-01
            c2 = 3.4706258957121240e-02
            c3 = 1.0371392756068808e-02
            c4 = -3.2473102915772530e-03
            c5 = -5.8859280062558573e-05
            c6 = 1.1163119416643705e-04
            c7 = -5.9862315408341454e-06
            c8 = -2.1833052948529252e-06
        elif i == 5:
            c0 = 9.0249609230319705e-01
            c1 = -1.8549336896701540e-01
            c2 = -7.4321469018942016e-02
            c3 = 1.7245598664074900e-02
            c4 = 2.8919244855335003e-03
            c5 = -7.7626463543078387e-04
            c6 = -7.0153241543290364e-05
            c7 = 2.1573606009848492e-05
            c8 = 1.1424184356591698e-06
        elif i == 6:
            c0 = 9.0249609230319705e-01
            c1 = 1.8549336896701718e-01
            c2 = -7.4321469018944764e-02
            c3 = -1.7245598664096067e-02
            c4 = 2.8919244855602180e-03
            c5 = 7.7626463548661605e-04
            c6 = -7.0153241608204137e-05
            c7 = -2.1573606046841067e-05
            c8 = 1.1424184772044353e-06
        elif i == 7:
            c0 = 3.9205672959747973e-01
            c1 = 2.4867649569508843e-01
            c2 = 3.4706258957121372e-02
            c3 = -1.0371392756070238e-02
            c4 = -3.2473102915762777e-03
            c5 = 5.8859280064598689e-05
            c6 = 1.1163119416384790e-04
            c7 = 5.9862315399508323e-06
            c8 = -2.1833052933691657e-06
        elif i == 8:
            c0 = 6.8555270927532441e-02
            c1 = 7.7101687470240640e-02
            c2 = 3.4057664727189435e-02
            c3 = 6.1792584626939205e-03
            c4 = -2.8792409251496981e-04
            c5 = -3.1734188869180467e-04
            c6 = -3.9126640865938901e-05
            c7 = 4.3114676271516161e-06
            c8 = 1.3782754035641229e-06
        elif i == 9:
            c0 = 3.9738148632413013e-03
            c1 = 6.9868611719839770e-03
            c2 = 5.3912787096484375e-03
            c3 = 2.3337438232598345e-03
            c4 = 5.8109587926932250e-04
            c5 = 6.3319612174373539e-05
            c6 = -7.1084554035456196e-06
            c7 = -3.3237789426874137e-06
            c8 = -3.3452620661800362e-07
        elif i == 10:
            c0 = 4.7824764029482715e-05
            c1 = 1.3210963439975028e-04
            c2 = 1.6594105667277307e-04
            c3 = 1.2466422784726883e-04
            c4 = 6.1740613696036075e-05
            c5 = 2.0784779413288423e-05
            c6 = 4.5914893479576135e-06
            c7 = 5.0875618282444679e-07
            c8 = -1.9684788103161201e-08
        else:  # i == 11
            c0 = 2.4000869177589039e-08
            c1 = 1.3165411022430253e-07
            c2 = 3.2556973918858355e-07
            c3 = 4.8177614434635299e-07
            c4 = 4.7338360924466032e-07
            c5 = 3.2428381026124624e-07
            c6 = 1.6569803565964398e-07
            c7 = 6.7724562122472354e-08
            c8 = 1.6797773443491893e-08

        return c0 + z * (
            c1
            + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * c8))))))
        )

    @pk.function
    def _basic_kaiser_poly_p14(x: pk.double, i: int) -> float:
        """
        Evaluate the basic kaiser polynomial of support 14 and degree 9.

        Parameters
        ----------
        x : pk.double,
            the point to be evaluated
        i : int
            the fourier grid positioning of the
            point where the 0 index is the
            front/top/left and the 9 index
            is the back/bottom/right corner
            of a p-cube

        Returns
        -------
        pk.double
            output of the basic kaiser polynomial
        """
        z: pk.double = 2.0 * x - 1.0
        c0: pk.double = 0.0
        c1: pk.double = 0.0
        c2: pk.double = 0.0
        c3: pk.double = 0.0
        c4: pk.double = 0.0
        c5: pk.double = 0.0
        c6: pk.double = 0.0
        c7: pk.double = 0.0
        c8: pk.double = 0.0
        c9: pk.double = 0.0

        if i == 0:
            c0 = 4.5650884746772444e-10
            c1 = -2.7424632687222392e-09
            c2 = 7.4960616433264958e-09
            c3 = -1.2411738247248583e-08
            c4 = 1.3965713227454019e-08
            c5 = -1.1242600932743078e-08
            c6 = 6.6336143123578231e-09
            c7 = -3.0635359127578443e-09
            c8 = 1.2062670111769458e-09
            c9 = -2.9791182637037096e-10
        elif i == 1:
            c0 = 2.0311290191486935e-06
            c1 = -6.2989468836761287e-06
            c2 = 9.0301373204448727e-06
            c3 = -7.9169076498762400e-06
            c4 = 4.7243540337050546e-06
            c5 = -2.0153132992200313e-06
            c6 = 6.2485224414050194e-07
            c7 = -1.3817718138078912e-07
            c8 = 1.9434612162418479e-08
            c9 = -1.0453625470829540e-09
        elif i == 2:
            c0 = 3.1717334603774096e-04
            c1 = -6.5294130509538275e-04
            c2 = 6.1075189969064532e-04
            c3 = -3.3965256565287444e-04
            c4 = 1.2245431039806500e-04
            c5 = -2.8712312740508972e-05
            c6 = 3.8579045452562357e-06
            c7 = -5.9920790161922475e-08
            c8 = -8.6091446465706539e-08
            c9 = 1.5033495345024980e-08
        elif i == 3:
            c0 = 9.8861552221418767e-03
            c1 = -1.4032042263910159e-02
            c2 = 8.6275966108163293e-03
            c3 = -2.8851176116984384e-03
            c4 = 5.0361548720741060e-04
            c5 = -1.4142554910700822e-05
            c6 = -1.2992824924057952e-05
            c7 = 2.5159820955038229e-06
            c8 = -4.2911627077487240e-08
            c9 = -4.2706285147731341e-08
        elif i == 4:
            c0 = 1.0291157506258378e-01
            c1 = -9.6856629772374594e-02
            c2 = 3.4499622954484006e-02
            c3 = -4.1906595354069699e-03
            c4 = -6.7336595398648561e-04
            c5 = 2.6188447553473246e-04
            c6 = -1.5238556791400218e-05
            c7 = -4.7970037590755593e-06
            c8 = 7.4124699126584075e-07
            c9 = 2.7857495043738248e-08
        elif i == 5:
            c0 = 4.4880124345572281e-01
            c1 = -2.4251983649949740e-01
            c2 = 2.3189519838824005e-02
            c3 = 1.0405381647915258e-02
            c4 = -2.2972967661092744e-03
            c5 = -1.3819187748221083e-04
            c6 = 7.5581499455361028e-05
            c7 = -1.5626534159094266e-06
            c8 = -1.4222147024791690e-06
            c9 = 9.2239575719794799e-08
        elif i == 6:
            c0 = 9.1565859602900845e-01
            c1 = -1.6156402467968725e-01
            c2 = -6.6936528937115325e-02
            c3 = 1.3077096441992437e-02
            c4 = 2.3398546024140485e-03
            c5 = -5.1512344124803478e-04
            c6 = -5.1839507638156238e-05
            c7 = 1.3160158102549420e-05
            c8 = 7.8932965274175755e-07
            c9 = -2.3698744514140577e-07
        elif i == 7:
            c0 = 9.1565859602900845e-01
            c1 = 1.6156402467968778e-01
            c2 = -6.6936528937117601e-02
            c3 = -1.3077096441998188e-02
            c4 = 2.3398546024267198e-03
            c5 = 5.1512344126816419e-04
            c6 = -5.1839507659917003e-05
            c7 = -1.3160158129298492e-05
            c8 = 7.8932966416400849e-07
            c9 = 2.3698745706004939e-07
        elif i == 8:
            c0 = 4.4880124345571842e-01
            c1 = 2.4251983649949432e-01
            c2 = 2.3189519838867478e-02
            c3 = -1.0405381647884269e-02
            c4 = -2.2972967662552557e-03
            c5 = 1.3819187737476170e-04
            c6 = 7.5581499655938335e-05
            c7 = 1.5626535707743192e-06
            c8 = -1.4222147994293300e-06
            c9 = -9.2239654291454217e-08
        elif i == 9:
            c0 = 1.0291157506258380e-01
            c1 = 9.6856629772374844e-02
            c2 = 3.4499622954481550e-02
            c3 = 4.1906595353943697e-03
            c4 = -6.7336595399194847e-04
            c5 = -2.6188447550185413e-04
            c6 = -1.5238556769727995e-05
            c7 = 4.7970037326992634e-06
            c8 = 7.4124697734072349e-07
            c9 = -2.7857489157315068e-08
        elif i == 10:
            c0 = 9.8861552221418698e-03
            c1 = 1.4032042263910163e-02
            c2 = 8.6275966108163137e-03
            c3 = 2.8851176116984787e-03
            c4 = 5.0361548720771602e-04
            c5 = 1.4142554910226985e-05
            c6 = -1.2992824924598051e-05
            c7 = -2.5159820947249160e-06
            c8 = -4.2911626817648475e-08
            c9 = 4.2706284798982148e-08
        elif i == 11:
            c0 = 3.1717334603773939e-04
            c1 = 6.5294130509538036e-04
            c2 = 6.1075189969064878e-04
            c3 = 3.3965256565287970e-04
            c4 = 1.2245431039810365e-04
            c5 = 2.8712312740551970e-05
            c6 = 3.8579045451470134e-06
            c7 = 5.9920790035837690e-08
            c8 = -8.6091446400365954e-08
            c9 = -1.5033495268409152e-08
        elif i == 12:
            c0 = 2.0311290191486833e-06
            c1 = 6.2989468836761160e-06
            c2 = 9.0301373204448219e-06
            c3 = 7.9169076498759012e-06
            c4 = 4.7243540337045472e-06
            c5 = 2.0153132992204971e-06
            c6 = 6.2485224414217377e-07
            c7 = 1.3817718138138786e-07
            c8 = 1.9434612161236931e-08
            c9 = 1.0453625462927024e-09
        else:  # i == 13
            c0 = 4.5650884746770252e-10
            c1 = 2.7424632687221924e-09
            c2 = 7.4960616433270996e-09
            c3 = 1.2411738247249071e-08
            c4 = 1.3965713227448958e-08
            c5 = 1.1242600932737978e-08
            c6 = 6.6336143123681165e-09
            c7 = 3.0635359127688218e-09
            c8 = 1.2062670111706536e-09
            c9 = 2.9791182636356358e-10

        return c0 + z * (
            c1
            + z
            * (
                c2
                + z
                * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9))))))
            )
        )

        # @pk.function
        # def kaiserBesselFunction(
        #     x: pk.double, window_shape: pk.double, over_p_half_squared: pk.double
        # ) -> pk.double:
        #     t: pk.double = pk.sqrt(1 - x * x * over_p_half_squared)
        #     return pk.cyl_bessel_i0(window_shape * t) / pk.cyl_bessel_i0(window_shape)


# END TEMPLATE G2PWORKLOAD

# START APPLICATION G2PWORKLOAD single_force_cuda
@pk.workload(
    potentials = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    targets = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.CudaSpace
    ),
    t_l2g = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    H=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight, space=pk.MemorySpace.CudaSpace),
)
# END APPLICATION G2PWORKLOAD single_force_cuda
# START APPLICATION G2PWORKLOAD single_force_hip
@pk.workload(
    potentials = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    targets = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HIPSpace
    ),
    t_l2g = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    H=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HIPSpace),
)
# END APPLICATION G2PWORKLOAD single_force_hip
# START APPLICATION G2PWORKLOAD single_force_host
@pk.workload(
    potentials = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HostSpace
    ),
    targets = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    t_l2g = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    H=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace),
)
# END APPLICATION G2PWORKLOAD single_force_host
# START APPLICATION G2PWORKLOAD multi_forces_cuda
@pk.workload(
    potentials = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.CudaSpace
    ),
    targets = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.CudaSpace
    ),
    t_l2g = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.CudaSpace
    ),
    H=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight, space=pk.MemorySpace.CudaSpace),
)
# END APPLICATION G2PWORKLOAD multi_forces_cuda
# START APPLICATION G2PWORKLOAD multi_forces_hip
@pk.workload(
    potentials = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HIPSpace
    ),
    targets = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HIPSpace
    ),
    t_l2g = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutLeft, space=pk.MemorySpace.HIPSpace
    ),
    H=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HIPSpace),
)
# END APPLICATION G2PWORKLOAD multi_forces_hip
# START APPLICATION G2PWORKLOAD multi_forces_host
@pk.workload(
    potentials = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    targets = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    t_l2g = pk.ViewTypeInfo(
        layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace
    ),
    H=pk.ViewTypeInfo(layout=pk.Layout.LayoutRight, space=pk.MemorySpace.HostSpace),
)
# END APPLICATION G2PWORKLOAD multi_forces_host
