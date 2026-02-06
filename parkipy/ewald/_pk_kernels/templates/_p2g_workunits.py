import pykokkos as pk


# START TEMPLATE P2G
@pk.classtype
class Cell_P2G:
    def __init__(self):
        self.x: pk.double = 0.0
        self.y: pk.double = 0.0
        self.z: pk.double = 0.0

        self.x_shift: pk.double = 0.0
        self.y_shift: pk.double = 0.0
        self.z_shift: pk.double = 0.0

        self.inbounds: bool = True


@pk.workunit
def p2g_base_P2G(
    team_member: pk.TeamMember,
    yj,
    qj,
    nj,
    H,
    H_shape,
    ny,
    window_P,
    periodicity,
    threads,
):
    """
    args:
        yj: (3, ny) list of source particles
            scaled to the Fourier grid
        qj: (dx, ny) list of source densities
        nj: (3, ny) list of source normal vectors
        H: (gx, gy, gz, dh) Fourier grid
        H_shape: shape of the Fourier grid
        ny: int, number of source particles
        window_P: int, window function support in grid points
        periodicity: int, periodicity of the problem
        threads: int, threads per team
    """
    pp: int = window_P * window_P
    po2: int = window_P / 2
    s_off: int = team_member.league_rank() * threads

    def thread_loop(tid: int):
        s: int = s_off + tid
        if s >= ny:
            return
        # get source position
        s_x: pk.double = yj[0][s]
        s_y: pk.double = yj[1][s]
        s_z: pk.double = yj[2][s]
        # get the nearest grid point for binning
        g_x: int = int(s_x)
        g_y: int = int(s_y)
        g_z: int = int(s_z)
        # get the distances for binning
        d_x: pk.double = s_x - g_x
        d_y: pk.double = s_y - g_y
        d_z: pk.double = s_z - g_z
        # get the window anchor
        a_x: int = g_x - (po2 - 1)
        a_y: int = g_y - (po2 - 1)
        a_z: int = g_z - (po2 - 1)
        # allocate registers for window function
        w0: List[pk.double] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        w1: List[pk.double] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        w2: List[pk.double] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for r in range(window_P):
            if window_P == 14:
                w0[r] = _basic_kaiser_poly_p14_fp64(d_x, r)
                w1[r] = _basic_kaiser_poly_p14_fp64(d_y, r)
                w2[r] = _basic_kaiser_poly_p14_fp64(d_z, r)
            elif window_P == 12:
                w0[r] = _basic_kaiser_poly_p12_fp64(d_x, r)
                w1[r] = _basic_kaiser_poly_p12_fp64(d_y, r)
                w2[r] = _basic_kaiser_poly_p12_fp64(d_z, r)
            elif window_P == 10:
                w0[r] = _basic_kaiser_poly_p10_fp64(d_x, r)
                w1[r] = _basic_kaiser_poly_p10_fp64(d_y, r)
                w2[r] = _basic_kaiser_poly_p10_fp64(d_z, r)
            if window_P == 8:
                w0[r] = _basic_kaiser_poly_p8_fp64(d_x, r)
                w1[r] = _basic_kaiser_poly_p8_fp64(d_y, r)
                w2[r] = _basic_kaiser_poly_p8_fp64(d_z, r)
            elif window_P == 6:
                w0[r] = _basic_kaiser_poly_p6_fp64(d_x, r)
                w1[r] = _basic_kaiser_poly_p6_fp64(d_y, r)
                w2[r] = _basic_kaiser_poly_p6_fp64(d_z, r)
            elif window_P == 4:
                w0[r] = _basic_kaiser_poly_p4_fp64(d_x, r)
                w1[r] = _basic_kaiser_poly_p4_fp64(d_y, r)
                w2[r] = _basic_kaiser_poly_p4_fp64(d_z, r)
            elif window_P == 2:
                w0[r] = _basic_kaiser_poly_p2_fp64(d_x, r)
                w1[r] = _basic_kaiser_poly_p2_fp64(d_y, r)
                w2[r] = _basic_kaiser_poly_p2_fp64(d_z, r)
        # loop over window cell
        wx: pk.double = 0.0
        wx_wy: pk.double = 0.0
        w: pk.double = 0.0
        t_x: int = 0
        t_y: int = 0
        t_z: int = 0
        for i in range(window_P):
            wx = w0[i]
            t_x = a_x + i
            if periodicity >= 1:
                t_x -= (t_x >= H_shape[0]) * H_shape[0]
                t_x += (t_x < 0) * H_shape[0]
            else:
                if t_x >= H_shape[0] or t_x < 0:
                    continue
            for j in range(window_P):
                wx_wy = wx * w1[j]
                t_y = a_y + j
                if periodicity >= 2:
                    t_y -= (t_y >= H_shape[1]) * H_shape[1]
                    t_y += (t_y < 0) * H_shape[1]
                else:
                    if t_y >= H_shape[1] or t_y < 0:
                        continue
                for k in range(window_P):
                    t_z = a_z + k
                    if periodicity >= 3:
                        t_z -= (t_z >= H_shape[2]) * H_shape[2]
                        t_z += (t_z < 0) * H_shape[2]
                    else:
                        if t_z >= H_shape[2] or t_z < 0:
                            continue
                    w = wx_wy * w2[k]
                    # interpolate densities with windows
                    for d in range(3):
                        pk.atomic_add(H, [t_x, t_y, t_z, d], w * qj[d][s])
                    dsl: List[pk.double] = [0, 0, 0]
                    norms: List[pk.double] = [0, 0, 0]
                    for d in range(3):
                        dsl[d] = qj[d + 3][s]
                        norms[d] = nj[d][s]
                    for d in range(9):
                        pk.atomic_add(
                            H,
                            [t_x, t_y, t_z, d + 3],
                            w * dsl[d % 3] * norms[d // 3],
                        )

    pk.parallel_for(pk.TeamThreadRange(team_member, threads), thread_loop)


@pk.workunit
def p2g_source_P2G(
    team_member: pk.TeamMember,
    yj,
    qj,
    nj,
    H,
    H_shape,
    ny,
    window_P,
    periodicity,
    cell_size,
    cell_teams,
    threads,
):
    """
    args:
        yj: (3, ny) list of source particles
            scaled to the Fourier grid
        qj: (dx, ny) list of source densities
        nj: (3, ny) list of source normal vectors
        H: (gx, gy, gz, dh) Fourier grid
        H_shape: shape of the Fourier grid
        ny: int, number of source particles
        window_P: int, window function support in grid points
        periodicity: int, periodicity of the problem
        cell_size: int, number of source particles in a cell
        cell_teams: int, number of teams assigned to a cell
        threads: int, threads per team
    """
    po2: int = window_P / 2
    nz_cell: int = team_member.league_rank() // cell_teams
    cell_team: int = team_member.league_rank() % cell_teams
    cell_off: int = cell_team * threads
    s_off: int = nz_cell * cell_size + cell_off

    def thread_loop(tid: int):
        if cell_off + tid >= cell_size:
            return
        s: int = s_off + tid
        # get source position
        s_x: pk.double = yj[0][s]
        s_y: pk.double = yj[1][s]
        s_z: pk.double = yj[2][s]
        # get the nearest grid point for binning
        g_x: int = int(s_x)
        g_y: int = int(s_y)
        g_z: int = int(s_z)
        # get the distances for binning
        d_x: pk.double = s_x - g_x
        d_y: pk.double = s_y - g_y
        d_z: pk.double = s_z - g_z
        # get the window anchor
        a_x: int = g_x - (po2 - 1)
        a_y: int = g_y - (po2 - 1)
        a_z: int = g_z - (po2 - 1)
        # allocate registers for window function
        w0: List[pk.double] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        w1: List[pk.double] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        w2: List[pk.double] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for r in range(window_P):
            if window_P == 14:
                w0[r] = _basic_kaiser_poly_p14_fp64(d_x, r)
                w1[r] = _basic_kaiser_poly_p14_fp64(d_y, r)
                w2[r] = _basic_kaiser_poly_p14_fp64(d_z, r)
            elif window_P == 12:
                w0[r] = _basic_kaiser_poly_p12_fp64(d_x, r)
                w1[r] = _basic_kaiser_poly_p12_fp64(d_y, r)
                w2[r] = _basic_kaiser_poly_p12_fp64(d_z, r)
            elif window_P == 10:
                w0[r] = _basic_kaiser_poly_p10_fp64(d_x, r)
                w1[r] = _basic_kaiser_poly_p10_fp64(d_y, r)
                w2[r] = _basic_kaiser_poly_p10_fp64(d_z, r)
            if window_P == 8:
                w0[r] = _basic_kaiser_poly_p8_fp64(d_x, r)
                w1[r] = _basic_kaiser_poly_p8_fp64(d_y, r)
                w2[r] = _basic_kaiser_poly_p8_fp64(d_z, r)
            elif window_P == 6:
                w0[r] = _basic_kaiser_poly_p6_fp64(d_x, r)
                w1[r] = _basic_kaiser_poly_p6_fp64(d_y, r)
                w2[r] = _basic_kaiser_poly_p6_fp64(d_z, r)
            elif window_P == 4:
                w0[r] = _basic_kaiser_poly_p4_fp64(d_x, r)
                w1[r] = _basic_kaiser_poly_p4_fp64(d_y, r)
                w2[r] = _basic_kaiser_poly_p4_fp64(d_z, r)
            elif window_P == 2:
                w0[r] = _basic_kaiser_poly_p2_fp64(d_x, r)
                w1[r] = _basic_kaiser_poly_p2_fp64(d_y, r)
                w2[r] = _basic_kaiser_poly_p2_fp64(d_z, r)
        # loop over window cell
        wx: pk.double = 0.0
        wx_wy: pk.double = 0.0
        w: pk.double = 0.0
        t_x: int = 0
        t_y: int = 0
        t_z: int = 0
        for i in range(window_P):
            wx = w0[i]
            t_x = a_x + i
            if periodicity >= 1:
                t_x -= (t_x >= H_shape[0]) * H_shape[0]
                t_x += (t_x < 0) * H_shape[0]
            else:
                if t_x >= H_shape[0] or t_x < 0:
                    continue
            for j in range(window_P):
                wx_wy = wx * w1[j]
                t_y = a_y + j
                if periodicity >= 2:
                    t_y -= (t_y >= H_shape[1]) * H_shape[1]
                    t_y += (t_y < 0) * H_shape[1]
                else:
                    if t_y >= H_shape[1] or t_y < 0:
                        continue
                for k in range(window_P):
                    t_z = a_z + k
                    if periodicity >= 3:
                        t_z -= (t_z >= H_shape[2]) * H_shape[2]
                        t_z += (t_z < 0) * H_shape[2]
                    else:
                        if t_z >= H_shape[2] or t_z < 0:
                            continue
                    w = wx_wy * w2[k]
                    # interpolate densities with windows
                    for d in range(3):
                        pk.atomic_add(H, [d, t_x, t_y, t_z], w * qj[d][s])
                    dsl: List[pk.double] = [0, 0, 0]
                    norms: List[pk.double] = [0, 0, 0]
                    for d in range(3):
                        dsl[d] = qj[d + 3][s]
                        norms[d] = nj[d][s]
                    for d in range(9):
                        pk.atomic_add(
                            H,
                            [d + 3, t_x, t_y, t_z],
                            w * dsl[d % 3] * norms[d // 3],
                        )

    pk.parallel_for(pk.TeamThreadRange(team_member, threads), thread_loop)


@pk.workunit
def p2g_grid_P2G(
    team_member: pk.TeamMember,
    yj,
    qj,
    nj,
    H,
    H_shape,
    window_P,
    periodicity,
    cell_size,
    num_cells,
    nonempty_cell_index,
    threads,
):
    """
    args:
        yj: (3, ny) list of source particles
            scaled to the Fourier grid
        qj: (dx, ny) list of source densities
        nj: (3, ny) list of source normal vectors
        H: (gx, gy, gz, dh) Fourier grid
        H_shape: shape of the Fourier grid
        window_P: int, window function support in grid points
        periodicity: int, periodicity of the problem
        cell_size: int, number of source particles in a cell
        num_cells: (3), number of cells in each spatial direction
        nonempty_cell_index: cell index -> nonempty cell index map
        threads: int, threads per team
    """
    pp: int = window_P * window_P
    po2: int = window_P / 2
    num_cells12: int = num_cells[1] * num_cells[2]
    H_shape12: int = H_shape[1] * H_shape[2]
    ng: int = H_shape[0] * H_shape12
    t_off: int = team_member.league_rank() * threads

    def thread_loop(tid: int):
        t: int = t_off + tid
        if t >= ng:
            return
        t_x: int = t // H_shape12
        t_y: int = t % H_shape12 // H_shape[2]
        t_z: int = t % H_shape12 % H_shape[2]
        # get target cell
        t_cell_x: int = t_x // po2
        t_cell_y: int = t_y // po2
        t_cell_z: int = t_z // po2
        t_cell: int = t_cell_x * num_cells12 + t_cell_y * num_cells[2] + t_cell_z
        pot: List[pk.double] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # loop over sources
        for k in range(27):
            source_cell: Cell_fp64 = get_source_cell_fp64(
                k, t_cell_x, t_cell_y, t_cell_z, periodicity
            )
            if source_cell.inbounds == False:
                continue
            s_cell: int = (
                source_cell.x * num_cells12
                + source_cell.y * num_cells[2]
                + source_cell.z
            )
            nz_cell: int = nonempty_cell_index[s_cell]
            if nz_cell < 0:
                continue
            s_off: int = nz_cell * cell_size
            for s in range(s_off, s_off + cell_size):
                w: pk.double = naive_window_kernel_fp64(
                    s, t_x, t_y, t_z, periodicity, pp, po2, yj, H_shape
                )
                # interpolate densities with windows
                for d in range(3):
                    pot[d] += w * qj[d][s]
                dsl: List[pk.double] = [0, 0, 0]
                norms: List[pk.double] = [0, 0, 0]
                for d in range(3):
                    dsl[d] = qj[d + 3][s]
                    norms[d] = nj[d][s]
                for d in range(9):
                    pot[d + 3] += w * dsl[d % 3] * norms[d // 3]
        # write interpolated densities to grid
        for d in range(12):
            H[t_x][t_y][t_z][d] += pot[d]

    pk.parallel_for(pk.TeamThreadRange(team_member, threads), thread_loop)


@pk.workunit
def p2g_hybrid(team_member: pk.TeamMember):
    nz_cell: int = team_member.league_rank() // chunks_per_cell
    cell_chunk: int = team_member.league_rank() % chunks_per_cell
    s_cell: int = nz2s_cell_map[nz_cell]
    s_cell_x: int = s_cell // num_cells12
    s_cell_y: int = (s_cell % num_cells12) // num_cells[2]
    s_cell_z: int = (s_cell % num_cells12) % num_cells[2]
    cell_off: int = cell_chunk * cell_chunk_size
    s_off: int = nz_cell * cell_size + cell_off
    # declare shared memory array
    shmem_w: pk.ScratchView3D[pk.double] = pk.ScratchView3D(
        team_member.team_scratch(0), cell_chunk_size, window_P, 3
    )
    shmem_a: pk.ScratchView2D[int] = pk.ScratchView2D(
        team_member.team_scratch(0), cell_chunk_size, 3
    )
    shmem_f: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
        team_member.team_scratch(0), cell_chunk_size, (dim_f + has_normals * dim_n)
    )

    def source_loop(ii: int):
        s: int = s_off + ii
        # get source position
        s_x: pk.double = yj[0][s]
        s_y: pk.double = yj[1][s]
        s_z: pk.double = yj[2][s]
        # get the nearest grid point for binning
        g_x: int = int(s_x)
        g_y: int = int(s_y)
        g_z: int = int(s_z)
        # get the distances for binning
        d_x: pk.double = s_x - g_x
        d_y: pk.double = s_y - g_y
        d_z: pk.double = s_z - g_z
        # load the window anchor into shared memory
        anchor_x: pk.double = g_x - (po2 - 1)
        if periodicity >= 1:
            anchor_x += (anchor_x < 0) * H_shape[0]
            anchor_x -= (anchor_x >= H_shape[0]) * H_shape[0]
        anchor_y: pk.double = g_y - (po2 - 1)
        if periodicity >= 2:
            anchor_y += (anchor_y < 0) * H_shape[1]
            anchor_y -= (anchor_y >= H_shape[1]) * H_shape[1]
        anchor_z: pk.double = g_z - (po2 - 1)
        if periodicity >= 3:
            anchor_z += (anchor_z < 0) * H_shape[2]
            anchor_z -= (anchor_z >= H_shape[2]) * H_shape[2]
        shmem_a[ii][0] = anchor_x
        shmem_a[ii][1] = anchor_y
        shmem_a[ii][2] = anchor_z
        # load densities and normals into shared memory
        for iii in range(dim_f):
            shmem_f[ii][iii] = qj[iii][s]
        for iii in range(dim_n):
            shmem_f[ii][iii + dim_f] = nj[iii][s]
        for r in range(window_P):
            if window_P == 14:
                shmem_w[ii][r][0] = _basic_kaiser_poly_p14_fp64(d_x, r)
                shmem_w[ii][r][1] = _basic_kaiser_poly_p14_fp64(d_y, r)
                shmem_w[ii][r][2] = _basic_kaiser_poly_p14_fp64(d_z, r)
            elif window_P == 12:
                shmem_w[ii][r][0] = _basic_kaiser_poly_p12_fp64(d_x, r)
                shmem_w[ii][r][1] = _basic_kaiser_poly_p12_fp64(d_y, r)
                shmem_w[ii][r][2] = _basic_kaiser_poly_p12_fp64(d_z, r)
            elif window_P == 10:
                shmem_w[ii][r][0] = _basic_kaiser_poly_p10_fp64(d_x, r)
                shmem_w[ii][r][1] = _basic_kaiser_poly_p10_fp64(d_y, r)
                shmem_w[ii][r][2] = _basic_kaiser_poly_p10_fp64(d_z, r)
            if window_P == 8:
                shmem_w[ii][r][0] = _basic_kaiser_poly_p8_fp64(d_x, r)
                shmem_w[ii][r][1] = _basic_kaiser_poly_p8_fp64(d_y, r)
                shmem_w[ii][r][2] = _basic_kaiser_poly_p8_fp64(d_z, r)
            elif window_P == 6:
                shmem_w[ii][r][0] = _basic_kaiser_poly_p6_fp64(d_x, r)
                shmem_w[ii][r][1] = _basic_kaiser_poly_p6_fp64(d_y, r)
                shmem_w[ii][r][2] = _basic_kaiser_poly_p6_fp64(d_z, r)
            elif window_P == 4:
                shmem_w[ii][r][0] = _basic_kaiser_poly_p4_fp64(d_x, r)
                shmem_w[ii][r][1] = _basic_kaiser_poly_p4_fp64(d_y, r)
                shmem_w[ii][r][2] = _basic_kaiser_poly_p4_fp64(d_z, r)
            elif window_P == 2:
                shmem_w[ii][r][0] = _basic_kaiser_poly_p2_fp64(d_x, r)
                shmem_w[ii][r][1] = _basic_kaiser_poly_p2_fp64(d_y, r)
                shmem_w[ii][r][2] = _basic_kaiser_poly_p2_fp64(d_z, r)

    pk.parallel_for(pk.TeamThreadRange(team_member, cell_chunk_size), source_loop)
    team_member.team_barrier()

    def target_loop(ll: int):
        xx: int = ll // (po2_sqr)
        yy: int = (ll % po2_sqr) // po2
        zz: int = (ll % po2_sqr) % po2
        t_x: int = t_off_x + xx
        t_y: int = t_off_y + yy
        t_z: int = t_off_z + zz
        # periodic corrections
        if periodicity >= 1:
            if t_x >= H_shape[0]:
                if s_cell_x == num_cells[0] - 2:
                    t_x -= H_shape[0]
                elif s_cell_x == 0:
                    t_x -= po2
        if periodicity >= 2:
            if t_y >= H_shape[1]:
                if s_cell_y == num_cells[1] - 2:
                    t_y -= H_shape[1]
                elif s_cell_y == 0:
                    t_y -= po2
        if periodicity >= 3:
            if t_z >= H_shape[2]:
                if s_cell_z == num_cells[2] - 2:
                    t_z -= H_shape[2]
                elif s_cell_z == 0:
                    t_z -= po2

        # check that the target node is in bounds
        if t_x >= H_shape[0] or t_y >= H_shape[1] or t_z >= H_shape[2]:
            return
        pot: List[pk.double] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        _range_bound: int = (cell_chunk_size + cell_off) < cell_size
        _range: int = (_range_bound) * cell_chunk_size + (not _range_bound) * (
            cell_size - cell_off
        )
        for ii in range(_range):
            # TODO: simplify this integer arithmatic
            bin_x: int = t_x - shmem_a[ii][0]
            bin_x += (bin_x < 0) * H_shape[0]
            bin_x -= (bin_x >= H_shape[0]) * H_shape[0]

            bin_y: int = t_y - shmem_a[ii][1]
            bin_y += (bin_y < 0) * H_shape[1]
            bin_y -= (bin_y >= H_shape[1]) * H_shape[1]

            bin_z: int = t_z - shmem_a[ii][2]
            bin_z += (bin_z < 0) * H_shape[2]
            bin_z += (bin_z >= H_shape[2]) * H_shape[2]

            depth_indicator: int = bin_x < window_P
            width_indicator: int = bin_y < window_P
            height_indicator: int = bin_z < window_P
            bin_in_cube: int = depth_indicator and width_indicator and height_indicator

            if not bin_in_cube:
                continue

            wx: pk.double = shmem_w[ii][bin_x][0]
            wy: pk.double = shmem_w[ii][bin_y][1]
            wz: pk.double = shmem_w[ii][bin_z][2]
            w: pk.double = wx * wy * wz
            # interpolate densities with windows
            dlx: pk.double = shmem_f[ii][3]
            dly: pk.double = shmem_f[ii][4]
            dlz: pk.double = shmem_f[ii][5]
            nx: pk.double = shmem_f[ii][6]
            ny: pk.double = shmem_f[ii][7]
            nz: pk.double = shmem_f[ii][8]
            w_nx: pk.double = w * nx
            w_ny: pk.double = w * ny
            w_nz: pk.double = w * nz

            pot[0] += w * shmem_f[ii][0]
            pot[1] += w * shmem_f[ii][1]
            pot[2] += w * shmem_f[ii][2]
            pot[3] += w_nx * dlx
            pot[4] += w_nx * dly
            pot[5] += w_nx * dlz

            pot[6] += w_ny * dlx
            pot[7] += w_ny * dly
            pot[8] += w_ny * dlz

            pot[9] += w_nz * dlx
            pot[10] += w_nz * dly
            pot[11] += w_nz * dlz

        for d in range(dim_H):
            pk.atomic_add(H, [d, t_x, t_y, t_z], pot[d])

    # call the target loop for neighboring target cells
    for dx in range(-1, 2):
        t_cell_x: int = s_cell_x + dx
        if periodicity >= 1:
            if t_cell_x < 0:
                t_cell_x += num_cells[0]
            elif t_cell_x >= num_cells[0]:
                t_cell_x -= num_cells[0]
        if t_cell_x < 0 or t_cell_x >= num_cells[0]:
            continue
        t_off_x: int = t_cell_x * po2
        for dy in range(-1, 2):
            t_cell_y: int = s_cell_y + dy
            if periodicity >= 2:
                if t_cell_y < 0:
                    t_cell_y += num_cells[1]
                elif t_cell_y >= num_cells[1]:
                    t_cell_y -= num_cells[1]
            if t_cell_y < 0 or t_cell_y >= num_cells[1]:
                continue
            t_off_y: int = t_cell_y * po2
            for dz in range(-1, 2):
                t_cell_z: int = s_cell_z + dz
                if periodicity >= 3:
                    if t_cell_z < 0:
                        t_cell_z += num_cells[2]
                    elif t_cell_z >= num_cells[2]:
                        t_cell_z -= num_cells[2]
                if t_cell_z < 0 or t_cell_z >= num_cells[2]:
                    continue
                t_off_z: int = t_cell_z * po2
                pk.parallel_for(pk.TeamThreadRange(team_member, po2_cubed), target_loop)


@pk.workunit
def p2g_hybrid_wo_normals(team_member: pk.TeamMember):
    nz_cell: int = team_member.league_rank() // chunks_per_cell
    cell_chunk: int = team_member.league_rank() % chunks_per_cell
    s_cell: int = nz2s_cell_map[nz_cell]
    s_cell_x: int = s_cell // num_cells12
    s_cell_y: int = (s_cell % num_cells12) // num_cells[2]
    s_cell_z: int = (s_cell % num_cells12) % num_cells[2]
    cell_off: int = cell_chunk * cell_chunk_size
    s_off: int = nz_cell * cell_size + cell_off
    # declare shared memory array
    shmem_w: pk.ScratchView3D[pk.double] = pk.ScratchView3D(
        team_member.team_scratch(0), cell_chunk_size, window_P, 3
    )
    shmem_a: pk.ScratchView2D[int] = pk.ScratchView2D(
        team_member.team_scratch(0), cell_chunk_size, 3
    )
    shmem_f: pk.ScratchView2D[pk.double] = pk.ScratchView2D(
        team_member.team_scratch(0), cell_chunk_size, (dim_f + has_normals * dim_n)
    )

    def source_loop(ii: int):
        s: int = s_off + ii
        # get source position
        s_x: pk.double = yj[0][s]
        s_y: pk.double = yj[1][s]
        s_z: pk.double = yj[2][s]
        # get the nearest grid point for binning
        g_x: int = int(s_x)
        g_y: int = int(s_y)
        g_z: int = int(s_z)
        # get the distances for binning
        d_x: pk.double = s_x - g_x
        d_y: pk.double = s_y - g_y
        d_z: pk.double = s_z - g_z
        # load the window anchor into shared memory
        anchor_x: pk.double = g_x - (po2 - 1)
        if periodicity >= 1:
            anchor_x += (anchor_x < 0) * H_shape[0]
            anchor_x -= (anchor_x >= H_shape[0]) * H_shape[0]
        anchor_y: pk.double = g_y - (po2 - 1)
        if periodicity >= 2:
            anchor_y += (anchor_y < 0) * H_shape[1]
            anchor_y -= (anchor_y >= H_shape[1]) * H_shape[1]
        anchor_z: pk.double = g_z - (po2 - 1)
        if periodicity >= 3:
            anchor_z += (anchor_z < 0) * H_shape[2]
            anchor_z -= (anchor_z >= H_shape[2]) * H_shape[2]
        shmem_a[ii][0] = anchor_x
        shmem_a[ii][1] = anchor_y
        shmem_a[ii][2] = anchor_z
        # load densities and normals into shared memory
        for iii in range(dim_f):
            shmem_f[ii][iii] = qj[iii][s]
        for iii in range(dim_n):
            shmem_f[ii][iii + dim_f] = nj[iii][s]
        for r in range(window_P):
            if window_P == 14:
                shmem_w[ii][r][0] = _basic_kaiser_poly_p14_fp64(d_x, r)
                shmem_w[ii][r][1] = _basic_kaiser_poly_p14_fp64(d_y, r)
                shmem_w[ii][r][2] = _basic_kaiser_poly_p14_fp64(d_z, r)
            elif window_P == 12:
                shmem_w[ii][r][0] = _basic_kaiser_poly_p12_fp64(d_x, r)
                shmem_w[ii][r][1] = _basic_kaiser_poly_p12_fp64(d_y, r)
                shmem_w[ii][r][2] = _basic_kaiser_poly_p12_fp64(d_z, r)
            elif window_P == 10:
                shmem_w[ii][r][0] = _basic_kaiser_poly_p10_fp64(d_x, r)
                shmem_w[ii][r][1] = _basic_kaiser_poly_p10_fp64(d_y, r)
                shmem_w[ii][r][2] = _basic_kaiser_poly_p10_fp64(d_z, r)
            if window_P == 8:
                shmem_w[ii][r][0] = _basic_kaiser_poly_p8_fp64(d_x, r)
                shmem_w[ii][r][1] = _basic_kaiser_poly_p8_fp64(d_y, r)
                shmem_w[ii][r][2] = _basic_kaiser_poly_p8_fp64(d_z, r)
            elif window_P == 6:
                shmem_w[ii][r][0] = _basic_kaiser_poly_p6_fp64(d_x, r)
                shmem_w[ii][r][1] = _basic_kaiser_poly_p6_fp64(d_y, r)
                shmem_w[ii][r][2] = _basic_kaiser_poly_p6_fp64(d_z, r)
            elif window_P == 4:
                shmem_w[ii][r][0] = _basic_kaiser_poly_p4_fp64(d_x, r)
                shmem_w[ii][r][1] = _basic_kaiser_poly_p4_fp64(d_y, r)
                shmem_w[ii][r][2] = _basic_kaiser_poly_p4_fp64(d_z, r)
            elif window_P == 2:
                shmem_w[ii][r][0] = _basic_kaiser_poly_p2_fp64(d_x, r)
                shmem_w[ii][r][1] = _basic_kaiser_poly_p2_fp64(d_y, r)
                shmem_w[ii][r][2] = _basic_kaiser_poly_p2_fp64(d_z, r)

    pk.parallel_for(pk.TeamThreadRange(team_member, cell_chunk_size), source_loop)
    team_member.team_barrier()

    def target_loop(ll: int):
        xx: int = ll // (po2_sqr)
        yy: int = (ll % po2_sqr) // po2
        zz: int = (ll % po2_sqr) % po2
        t_x: int = t_off_x + xx
        t_y: int = t_off_y + yy
        t_z: int = t_off_z + zz
        # periodic corrections
        if periodicity >= 1:
            if t_x >= H_shape[0]:
                if s_cell_x == num_cells[0] - 2:
                    t_x -= H_shape[0]
                elif s_cell_x == 0:
                    t_x -= po2
        if periodicity >= 2:
            if t_y >= H_shape[1]:
                if s_cell_y == num_cells[1] - 2:
                    t_y -= H_shape[1]
                elif s_cell_y == 0:
                    t_y -= po2
        if periodicity >= 3:
            if t_z >= H_shape[2]:
                if s_cell_z == num_cells[2] - 2:
                    t_z -= H_shape[2]
                elif s_cell_z == 0:
                    t_z -= po2

        # check that the target node is in bounds
        if t_x >= H_shape[0] or t_y >= H_shape[1] or t_z >= H_shape[2]:
            return
        pot: List[pk.double] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        _range_bound: int = (cell_chunk_size + cell_off) < cell_size
        _range: int = (_range_bound) * cell_chunk_size + (not _range_bound) * (
            cell_size - cell_off
        )
        for ii in range(_range):
            # TODO: simplify this integer arithmatic
            bin_x: int = t_x - shmem_a[ii][0]
            bin_x += (bin_x < 0) * H_shape[0]
            bin_x -= (bin_x >= H_shape[0]) * H_shape[0]

            bin_y: int = t_y - shmem_a[ii][1]
            bin_y += (bin_y < 0) * H_shape[1]
            bin_y -= (bin_y >= H_shape[1]) * H_shape[1]

            bin_z: int = t_z - shmem_a[ii][2]
            bin_z += (bin_z < 0) * H_shape[2]
            bin_z += (bin_z >= H_shape[2]) * H_shape[2]

            depth_indicator: int = bin_x < window_P
            width_indicator: int = bin_y < window_P
            height_indicator: int = bin_z < window_P
            bin_in_cube: int = depth_indicator and width_indicator and height_indicator

            if not bin_in_cube:
                continue

            wx: pk.double = shmem_w[ii][bin_x][0]
            wy: pk.double = shmem_w[ii][bin_y][1]
            wz: pk.double = shmem_w[ii][bin_z][2]
            w: pk.double = wx * wy * wz
            # interpolate densities with windows
            for d in range(dim_f1):
                pot[d] += w * shmem_f[ii][d]
        for d in range(dim_H):
            pk.atomic_add(H, [d, t_x, t_y, t_z], pot[d])

    # call the target loop for neighboring target cells
    for dx in range(-1, 2):
        t_cell_x: int = s_cell_x + dx
        if periodicity >= 1:
            if t_cell_x < 0:
                t_cell_x += num_cells[0]
            elif t_cell_x >= num_cells[0]:
                t_cell_x -= num_cells[0]
        if t_cell_x < 0 or t_cell_x >= num_cells[0]:
            continue
        t_off_x: int = t_cell_x * po2
        for dy in range(-1, 2):
            t_cell_y: int = s_cell_y + dy
            if periodicity >= 2:
                if t_cell_y < 0:
                    t_cell_y += num_cells[1]
                elif t_cell_y >= num_cells[1]:
                    t_cell_y -= num_cells[1]
            if t_cell_y < 0 or t_cell_y >= num_cells[1]:
                continue
            t_off_y: int = t_cell_y * po2
            for dz in range(-1, 2):
                t_cell_z: int = s_cell_z + dz
                if periodicity >= 3:
                    if t_cell_z < 0:
                        t_cell_z += num_cells[2]
                    elif t_cell_z >= num_cells[2]:
                        t_cell_z -= num_cells[2]
                if t_cell_z < 0 or t_cell_z >= num_cells[2]:
                    continue
                t_off_z: int = t_cell_z * po2
                pk.parallel_for(pk.TeamThreadRange(team_member, po2_cubed), target_loop)


# helpers


@pk.function
def get_source_cell_P2G(
    self,
    k: int,
    t_cell_x: int,
    t_cell_y: int,
    t_cell_z: int,
    periodicity: int,
) -> Cell_fp64:
    # NOTE: black formats it like this, ARGH!
    offsets: List[int] = [
        -1,
        -1,
        -1,
        -1,
        -1,
        0,
        -1,
        -1,
        1,
        -1,
        0,
        -1,
        -1,
        0,
        0,
        -1,
        0,
        1,
        -1,
        1,
        -1,
        -1,
        1,
        0,
        -1,
        1,
        1,
        0,
        -1,
        -1,
        0,
        -1,
        0,
        0,
        -1,
        1,
        0,
        0,
        -1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        -1,
        0,
        1,
        0,
        0,
        1,
        1,
        1,
        -1,
        -1,
        1,
        -1,
        0,
        1,
        -1,
        1,
        1,
        0,
        -1,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        -1,
        1,
        1,
        0,
        1,
        1,
        1,
    ]
    dx: pk.double = offsets[k * 3]
    dy: pk.double = offsets[k * 3 + 1]
    dz: pk.double = offsets[k * 3 + 2]

    source_cell: Cell_fp64 = Cell_fp64()

    # x coord
    source_cell.x = t_cell_x + dx
    if source_cell.x < 0:
        if periodicity >= 1:
            source_cell.x += num_cells[0]
            source_cell.x_shift = -H_shape[0]
        else:
            source_cell.inbounds = False
    if source_cell.x >= num_cells[0]:
        if periodicity >= 1:
            source_cell.x -= num_cells[0]
            source_cell.x_shift = H_shape[0]
        else:
            source_cell.inbounds = False

    # y coord
    source_cell.y = t_cell_y + dy
    if source_cell.y < 0:
        if periodicity >= 2:
            source_cell.y += num_cells[1]
            source_cell.y_shift = -H_shape[1]
        else:
            source_cell.inbounds = False
    if source_cell.y >= num_cells[1]:
        if periodicity >= 2:
            source_cell.y -= num_cells[1]
            source_cell.y_shift = H_shape[1]
        else:
            source_cell.inbounds = False

    # z coord
    source_cell.z = t_cell_z + dz
    if source_cell.z < 0:
        if periodicity >= 3:
            source_cell.z += num_cells[2]
            source_cell.z_shift = -H_shape[2]
        else:
            source_cell.inbounds = False
    if source_cell.z >= num_cells[2]:
        if periodicity >= 3:
            source_cell.z -= num_cells[2]
            source_cell.z_shift = H_shape[2]
        else:
            source_cell.inbounds = False

    return source_cell


@pk.function
def naive_window_kernel_P2G(
    s: int,
    t_x: int,
    t_y: int,
    t_z: int,
    periodicity: int,
    pp: int,
    po2: int,
    yj: pk.View3D[pk.double],
    H_shape: pk.View1D[int],
) -> pk.double:
    # get source position
    s_x: pk.double = yj[0][s]
    s_y: pk.double = yj[1][s]
    s_z: pk.double = yj[2][s]
    # get the nearest grid point for binning
    g_x: int = int(s_x)
    g_y: int = int(s_y)
    g_z: int = int(s_z)
    # get the distances for binning
    d_x: pk.double = s_x - g_x
    d_y: pk.double = s_y - g_y
    d_z: pk.double = s_z - g_z
    # get the window anchor
    a_x: int = g_x - (po2 - 1)
    a_y: int = g_y - (po2 - 1)
    a_z: int = g_z - (po2 - 1)
    # get the bin idx of the t given s
    bin: int = _grid2bin(a_x, a_y, a_z, t_x, t_y, t_z, pp, H_shape, periodicity)
    if bin < 0:
        return 0
    # calculate the window function if in bounds
    bin_x: int = bin // pp
    bin_y: int = (bin % pp) // window_P
    bin_z: int = (bin % pp) % window_P
    wx: pk.double = 0
    wy: pk.double = 0
    wz: pk.double = 0
    if window_P == 14:
        wx = _basic_kaiser_poly_p14_fp64(d_x, bin_x)
        wy = _basic_kaiser_poly_p14_fp64(d_y, bin_y)
        wz = _basic_kaiser_poly_p14_fp64(d_z, bin_z)
    elif window_P == 12:
        wx = _basic_kaiser_poly_p12_fp64(d_x, bin_x)
        wy = _basic_kaiser_poly_p12_fp64(d_y, bin_y)
        wz = _basic_kaiser_poly_p12_fp64(d_z, bin_z)
    elif window_P == 10:
        wx = _basic_kaiser_poly_p10_fp64(d_x, bin_x)
        wy = _basic_kaiser_poly_p10_fp64(d_y, bin_y)
        wz = _basic_kaiser_poly_p10_fp64(d_z, bin_z)
    if window_P == 8:
        wx = _basic_kaiser_poly_p8_fp64(d_x, bin_x)
        wy = _basic_kaiser_poly_p8_fp64(d_y, bin_y)
        wz = _basic_kaiser_poly_p8_fp64(d_z, bin_z)
    elif window_P == 6:
        wx = _basic_kaiser_poly_p6_fp64(d_x, bin_x)
        wy = _basic_kaiser_poly_p6_fp64(d_y, bin_y)
        wz = _basic_kaiser_poly_p6_fp64(d_z, bin_z)
    elif window_P == 4:
        wx = _basic_kaiser_poly_p4_fp64(d_x, bin_x)
        wy = _basic_kaiser_poly_p4_fp64(d_y, bin_y)
        wz = _basic_kaiser_poly_p4_fp64(d_z, bin_z)
    elif window_P == 2:
        wx = _basic_kaiser_poly_p2_fp64(d_x, bin_x)
        wy = _basic_kaiser_poly_p2_fp64(d_y, bin_y)
        wz = _basic_kaiser_poly_p2_fp64(d_z, bin_z)
    w: pk.double = wx * wy * wz
    return w


@pk.function
def _grid2bin(
    anchor_x: int,
    anchor_y: int,
    anchor_z: int,
    grid_x: int,
    grid_y: int,
    grid_z: int,
    pp: int,
    H_shape: pk.View1D[int],
    periodicity: int,
) -> int:
    # periodic wrap of anchor
    if periodicity >= 1:
        anchor_x += (anchor_x < 0) * H_shape[0]
        anchor_x -= (anchor_x >= H_shape[0]) * H_shape[0]
    if periodicity >= 2:
        anchor_y += (anchor_y < 0) * H_shape[1]
        anchor_y -= (anchor_y >= H_shape[1]) * H_shape[1]
    if periodicity >= 3:
        anchor_z += (anchor_z < 0) * H_shape[2]
        anchor_z -= (anchor_z >= H_shape[2]) * H_shape[2]

    # compute distance from target to anchor
    bin_x: int = grid_x - anchor_x
    bin_x -= (bin_x >= H_shape[0]) * H_shape[0]
    bin_x += (bin_x < 0) * H_shape[0]

    bin_y: int = grid_y - anchor_y
    bin_y -= (bin_y >= H_shape[1]) * H_shape[1]
    bin_y += (bin_y < 0) * H_shape[1]

    bin_z: int = grid_z - anchor_z
    bin_z -= (bin_z >= H_shape[2]) * H_shape[2]
    bin_z += (bin_z < 0) * H_shape[2]

    # in bounds indicators
    in_cube_indicator: int = (
        0 <= bin_x
        and bin_x < window_P
        and 0 <= bin_y
        and bin_y < window_P
        and 0 <= bin_z
        and bin_z < window_P
    )

    # convert to linear bin id
    bin_id: int = (in_cube_indicator) * (bin_x * pp + bin_y * window_P + bin_z) + (
        not in_cube_indicator
    ) * -1
    return bin_id


# NOTE: These functions are generated by ``generate_basic_kaiser_poly_functions.py``.
# They should not be modified by hand.
@pk.function
def _basic_kaiser_poly_p2_P2G(x: pk.double, i: int) -> float:
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
def _basic_kaiser_poly_p4_P2G(x: pk.double, i: int) -> float:
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
def _basic_kaiser_poly_p6_P2G(x: pk.double, i: int) -> float:
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
def _basic_kaiser_poly_p8_P2G(x: pk.double, i: int) -> float:
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
def _basic_kaiser_poly_p10_P2G(x: pk.double, i: int) -> float:
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
def _basic_kaiser_poly_p12_P2G(x: pk.double, i: int) -> float:
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
        c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * c8))))))
    )


@pk.function
def _basic_kaiser_poly_p14_P2G(x: pk.double, i: int) -> float:
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
            + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9))))))
        )
    )


# END TEMPLATE P2G

# START APPLICATION P2G _
# END APPLICATION P2G _
