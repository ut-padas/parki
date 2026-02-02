import pykokkos as pk


# START TEMPLATE KVEC
@pk.function
def _k_vector_KVEC(M: int, L: pk.double, ups: pk.double, i: int) -> pk.double:
    TWO_PI: pk.double = 6.28318530717958647692529
    Mu: int = int(round(M * ups))
    a: int = -1
    if (Mu % 2) == 0:
        a = Mu / 2 - 1
    else:
        a = (Mu - 1) / 2

    k_int: int = i
    if i > a:
        k_int -= Mu

    h: pk.double = L / M
    Lu: pk.double = Mu * h
    k: pk.double = (TWO_PI) * k_int / Lu
    return k


# END TEMPLATE KVEC


# START TEMPLATE KBEXACT
@pk.function
def _kaiser_exact_ft_KBEXACT(
    k1: pk.double, b2: pk.double, w: pk.double, scale: pk.double
) -> pk.double:
    t: pk.double = pk.sqrt(b2 - k1 * w * w)
    F: pk.double = 2 * w * pk.sinh(t) / t * scale
    return F


# END TEMPLATE KBEXACT


# START TEMPLATE STOKESKERN
@pk.function
def stokes_kernel_STOKESKERN(
    kk: pk.double,
    k0: pk.double,
    k1: pk.double,
    k2: pk.double,
    xi: pk.double,
    wsh: pk.double,
    whw: pk.double,
    ksc: pk.double,
    pw: int,
) -> pk.double:
    EIGHT_PI: pk.double = 25.1327412287183459077011
    # See the MATLAB code for details on the method
    # scaling for the global Fourier domain or local pad
    biharm: pk.double = (EIGHT_PI) / (kk * kk)
    C: pk.double = kk / (4.0 * xi * xi)
    screen: pk.double = (1.0 + C) * pk.exp(-C)
    # compute window_m2 on the fly
    b2: pk.double = wsh * wsh
    f1: pk.double = _kaiser_exact_ft_fp64(k0 * k0, b2, whw, ksc)
    f2: pk.double = _kaiser_exact_ft_fp64(k1 * k1, b2, whw, ksc)
    f3: pk.double = _kaiser_exact_ft_fp64(k2 * k2, b2, whw, ksc)
    F: pk.double = f1 * f2 * f3  # tensor product of spatial directions
    if pw == 2:
        F *= F
    window_m2: pk.double = 1.0 / F
    scaling: pk.double = biharm * screen * window_m2

    # now the element corresponding to the zero mode is Inf

    # this is fine since we overwrite it later on

    return scaling


# END TEMPLATE STOKESKERN


# START TEMPLATE LAPLACECNV
@pk.workunit
def laplace_convolution_kernel_LAPLACECNV(
    team_member: pk.TeamMember,
    H,
    grid_size_1: int,
    grid_size_2: int,
    box,
    k1_off: int,
    xi: pk.double,
    pw: int,
    wsh: pk.double,
    ksc: pk.double,
    grid: pk.View1D[int],
    whw: pk.double,
    ups,
    locals: pk.View1D[int],
    num_locals: int,
    freq_range: int,
    freq_offset: pk.View1D[int],
    threads: int,
):
    wid_off: int = team_member.league_rank() * threads

    def thread_loop(tid: int):
        wid: int = wid_off + tid
        if wid >= freq_range:
            return
        # global index for H1, H2, H3
        i: int = (wid // (grid_size_1 * grid_size_2)) + freq_offset[0]
        j: int = (wid % (grid_size_1 * grid_size_2)) // grid_size_2 + freq_offset[1]
        k: int = wid % grid_size_2 + freq_offset[2]

        # compute k-vector
        k0: pk.double = 0.0
        if num_locals == 0:
            k0 = _k_vector_fp64(grid[0], box[0], ups[0], i)
        else:
            k0 = _k_vector_fp64(grid[0], box[0], ups[0], locals[i])
        k1: pk.double = _k_vector_fp64(
            grid[1], box[1], ups[1], j + k1_off
        )  # free direction
        k2: pk.double = _k_vector_fp64(grid[2], box[2], ups[2], k)  # free direction
        kk: pk.double = k0 * k0 + k1 * k1 + k2 * k2

        # kernel dependent computations
        FOUR_PI: pk.double = 12.566370614359172954
        C: pk.double = kk / (4.0 * xi * xi)
        screen: pk.double = pk.exp(-C)
        # compute window_m2 on the fly
        b2: pk.double = wsh * wsh
        f1: pk.double = _kaiser_exact_ft_fp64(k0 * k0, b2, whw, ksc)
        f2: pk.double = _kaiser_exact_ft_fp64(k1 * k1, b2, whw, ksc)
        f3: pk.double = _kaiser_exact_ft_fp64(k2 * k2, b2, whw, ksc)
        F: pk.double = f1 * f2 * f3  # tensor product of spatial directions
        if pw == 2:
            F *= F
        window_m2: pk.double = 1.0 / F
        scaling: pk.double = FOUR_PI * window_m2 * screen / kk
        H[i][j][k][0] *= scaling
        H[i][j][k][1] *= scaling

    pk.parallel_for(pk.TeamThreadRange(team_member, threads), thread_loop)


# END TEMPLATE LAPLACECNV


# START TEMPLATE STOKESCNV
@pk.workunit
def stokeslet_convolution_kernel_STOKESCNV(
    team_member: pk.TeamMember,
    H1,
    H2,
    H3,
    grid_size_1: int,
    grid_size_2: int,
    box,
    k1_off: int,
    xi: pk.double,
    pw: int,
    wsh: pk.double,
    ksc: pk.double,
    grid: pk.View1D[int],
    whw: pk.double,
    ups,
    locals: pk.View1D[int],
    num_locals: int,
    freq_range: int,
    freq_offset: pk.View1D[int],
    threads: int,
):
    wid_off: int = team_member.league_rank() * threads

    def thread_loop(tid: int):
        wid: int = wid_off + tid
        if wid >= freq_range:
            return
        # global index for H1, H2, H3
        i: int = (wid // (grid_size_1 * grid_size_2)) + freq_offset[0]
        j: int = (wid % (grid_size_1 * grid_size_2)) // grid_size_2 + freq_offset[1]
        k: int = wid % grid_size_2 + freq_offset[2]

        # compute k-vector
        k0: pk.double = 0.0
        if num_locals == 0:
            k0 = _k_vector_fp64(grid[0], box[0], ups[0], i)
        else:
            k0 = _k_vector_fp64(grid[0], box[0], ups[0], locals[i])
        k1: pk.double = _k_vector_fp64(
            grid[1], box[1], ups[1], j + k1_off
        )  # free direction
        k2: pk.double = _k_vector_fp64(grid[2], box[2], ups[2], k)  # free direction
        kk: pk.double = k0 * k0 + k1 * k1 + k2 * k2

        scaling: pk.double = stokes_kernel_fp64(kk, k0, k1, k2, xi, wsh, whw, ksc, pw)

        # use relation between stokeslet and biharmonic Green's function
        k_dot_H_re: pk.double = (
            k0 * H1[i][j][k][0] + k1 * H2[i][j][k][0] + k2 * H3[i][j][k][0]
        )
        k_dot_H_im: pk.double = (
            k0 * H1[i][j][k][1] + k1 * H2[i][j][k][1] + k2 * H3[i][j][k][1]
        )
        H1[i][j][k][0] = scaling * (kk * H1[i][j][k][0] - k_dot_H_re * k0)
        H1[i][j][k][1] = scaling * (kk * H1[i][j][k][1] - k_dot_H_im * k0)
        H2[i][j][k][0] = scaling * (kk * H2[i][j][k][0] - k_dot_H_re * k1)
        H2[i][j][k][1] = scaling * (kk * H2[i][j][k][1] - k_dot_H_im * k1)
        H3[i][j][k][0] = scaling * (kk * H3[i][j][k][0] - k_dot_H_re * k2)
        H3[i][j][k][1] = scaling * (kk * H3[i][j][k][1] - k_dot_H_im * k2)

    pk.parallel_for(pk.TeamThreadRange(team_member, threads), thread_loop)


# END TEMPLATE STOKESCNV


@pk.workunit
def stokeslet_convolution_kernel_range(
    wid: int,
    H1,
    H2,
    H3,
    grid_size_1: int,
    grid_size_2: int,
    box,
    k1_off: int,
    xi,
    pw: int,
    wsh,
    ksc,
    grid: pk.View1D[int],
    whw,
    ups,
    locals: pk.View1D[int],
    num_locals: int,
):
    EIGHT_PI: pk.double = 25.1327412287183459077011
    # global index for H1, H2, H3
    i: int = (wid // (grid_size_1 * grid_size_2)) + freq_offset[0]
    j: int = (wid % (grid_size_1 * grid_size_2)) // grid_size_2 + freq_offset[1]
    k: int = wid % grid_size_2 + freq_offset[2]

    # compute k-vector
    k0: pk.double = 0.0
    if num_locals == 0:
        k0 = _k_vector_fp64(grid[0], box[0], ups[0], i)
    else:
        k0 = _k_vector_fp64(grid[0], box[0], ups[0], locals[i])
    k1: pk.double = _k_vector_fp64(
        grid[1], box[1], ups[1], j + k1_off
    )  # free direction
    k2: pk.double = _k_vector_fp64(grid[2], box[2], ups[2], k)  # free direction
    kk: pk.double = k0 * k0 + k1 * k1 + k2 * k2

    # See the MATLAB code for details on the method

    # scaling for the global Fourier domain or local pad
    biharm: pk.double = (EIGHT_PI) / (kk * kk)
    C: pk.double = kk / (4.0 * xi * xi)
    screen: pk.double = (1.0 + C) * pk.exp(-C)
    # compute window_m2 on the fly
    b2: pk.double = wsh * wsh
    f1: pk.double = _kaiser_exact_ft_fp64(k0 * k0, b2, whw, ksc)
    f2: pk.double = _kaiser_exact_ft_fp64(k1 * k1, b2, whw, ksc)
    f3: pk.double = _kaiser_exact_ft_fp64(k2 * k2, b2, whw, ksc)
    F: pk.double = f1 * f2 * f3  # tensor product of spatial directions
    if pw == 2:
        F *= F
    window_m2: pk.double = 1.0 / F
    scaling: pk.double = biharm * screen * window_m2

    # now the element corresponding to the zero mode is Inf

    # this is fine since we overwrite it later on

    # use relation between stokeslet and biharmonic Green's function
    k_dot_H_re: pk.double = (
        k0 * H1[i][j][k][0] + k1 * H2[i][j][k][0] + k2 * H3[i][j][k][0]
    )
    k_dot_H_im: pk.double = (
        k0 * H1[i][j][k][1] + k1 * H2[i][j][k][1] + k2 * H3[i][j][k][1]
    )
    H1[i][j][k][0] = scaling * (kk * H1[i][j][k][0] - k_dot_H_re * k0)
    H1[i][j][k][1] = scaling * (kk * H1[i][j][k][1] - k_dot_H_im * k0)
    H2[i][j][k][0] = scaling * (kk * H2[i][j][k][0] - k_dot_H_re * k1)
    H2[i][j][k][1] = scaling * (kk * H2[i][j][k][1] - k_dot_H_im * k1)
    H3[i][j][k][0] = scaling * (kk * H3[i][j][k][0] - k_dot_H_re * k2)
    H3[i][j][k][1] = scaling * (kk * H3[i][j][k][1] - k_dot_H_im * k2)


@pk.workunit
def convolution_sum_sl_dl(
    team_member: pk.TeamMember,
    H1,
    H2,
    H3,
    H11,
    H21,
    H31,
    D1,
    D2,
    D3,
    grid_size_1: int,
    grid_size_2: int,
    freq_range: int,
    freq_offset: pk.View1D[int],
    threads: int,
):
    wid_off: int = team_member.league_rank() * threads

    def thread_loop(tid: int):
        wid: int = wid_off + tid
        if wid >= freq_range * 2:
            return
        # global index for H1, H2, H3
        i: int = (wid // (grid_size_1 * grid_size_2 * 2)) + freq_offset[0]
        j: int = (wid % (grid_size_1 * grid_size_2 * 2)) // (
            grid_size_2 * 2
        ) + freq_offset[1]
        k: int = wid % (grid_size_2 * 2) // 2 + freq_offset[2]
        l: int = wid % 2

        D1[i][j][k][l] = H1[i][j][k][l] + H11[i][j][k][l]
        D2[i][j][k][l] = H2[i][j][k][l] + H21[i][j][k][l]
        D3[i][j][k][l] = H3[i][j][k][l] + H31[i][j][k][l]

    pk.parallel_for(pk.TeamThreadRange(team_member, threads), thread_loop)


@pk.workunit
def convolution_sum_sl(
    team_member: pk.TeamMember,
    H1,
    H2,
    H3,
    D1,
    D2,
    D3,
    grid_size_1: int,
    grid_size_2: int,
    freq_range: int,
    freq_offset: pk.View1D[int],
    threads: int,
):
    wid_off: int = team_member.league_rank() * threads

    def thread_loop(tid: int):
        wid: int = wid_off + tid
        if wid >= freq_range * 2:
            return
        # global index for H1, H2, H3
        i: int = (wid // (grid_size_1 * grid_size_2 * 2)) + freq_offset[0]
        j: int = (wid % (grid_size_1 * grid_size_2 * 2)) // (
            grid_size_2 * 2
        ) + freq_offset[1]
        k: int = wid % (grid_size_2 * 2) // 2 + freq_offset[2]
        l: int = wid % 2

        D1[i][j][k][l] = H1[i][j][k][l]
        D2[i][j][k][l] = H2[i][j][k][l]
        D3[i][j][k][l] = H3[i][j][k][l]

    pk.parallel_for(pk.TeamThreadRange(team_member, threads), thread_loop)


@pk.workunit
def convolution_sum_sl_dl_range(
    wid: int,
    H1,
    H2,
    H3,
    H11,
    H21,
    H31,
    D1,
    D2,
    D3,
    grid_size_1: int,
    grid_size_2: int,
):

    # global index for H1, H2, H3
    i: int = (wid // (grid_size_1 * grid_size_2 * 2)) + freq_offset[0]
    j: int = (wid % (grid_size_1 * grid_size_2 * 2)) // (grid_size_2 * 2) + freq_offset[
        1
    ]
    k: int = wid % (grid_size_2 * 2) // 2 + freq_offset[2]
    l: int = wid % 2

    D1[i][j][k][l] = H1[i][j][k][l] + H11[i][j][k][l]
    D2[i][j][k][l] = H2[i][j][k][l] + H21[i][j][k][l]
    D3[i][j][k][l] = H3[i][j][k][l] + H31[i][j][k][l]


# START TEMPLATE STOKESCNV0
@pk.workunit
def stokeslet_convolution_zero_kernel_STOKESCNV0(
    team_member: pk.TeamMember,
    H1,
    H2,
    H3,
    grid_size_1: int,
    box,
    k1_off: int,
    xi,
    pw: int,
    wsh,
    ksc,
    grid: pk.View1D[int],
    whw,
    ups,
    gR,
    vico_var: int,
    freq_range: int,
    periodicity: int,
    threads: int,
):
    PI: pk.double = 3.14159265358979323846264
    FOUR_PI: pk.double = 12.5663706143591729538506
    EIGHT_PI: pk.double = 25.1327412287183459077011
    wid_off: int = team_member.league_rank() * threads

    def thread_loop(tid: int):
        wid: int = wid_off + tid
        if wid >= freq_range:
            return
        # global index for H1, H2, H3
        i: int = wid // grid_size_1
        j: int = wid % grid_size_1

        # compute k-vector
        k0: pk.double = 0.0
        k1: pk.double = _k_vector_fp64(
            grid[1], box[1], ups[1], i + k1_off
        )  # free direction
        k2: pk.double = _k_vector_fp64(grid[2], box[2], ups[2], j)  # free direction
        kk: pk.double = k0 * k0 + k1 * k1 + k2 * k2

        scaling_b: pk.double = 0.0
        if periodicity == 1:
            # See the MATLAB code for details on the method

            # scaling for the zero mode [k0==0] (Vico method)
            C: pk.double = kk / (4 * xi * xi)
            kn: pk.double = pk.sqrt(kk)
            # modified Green's function
            BJ0: pk.double = pk.cyl_bessel_j0(gR * kn)
            BJ1: pk.double = pk.cyl_bessel_j1(gR * kn)
            biharm: pk.double = 0.0
            if (i + k1_off) == 0 and j == 0:
                biharm = (-1.0 / 8.0) * PI * gR * gR * gR * gR
            else:
                biharm = EIGHT_PI * (
                    (BJ0 - 1.0) / (kk * kk) + (gR * BJ1) / (2 * kk * kn)
                )
            screen: pk.double = (1 + C) * pk.exp(-C)
            # compute window_m2 on the fly
            b2: pk.double = wsh * wsh
            f1: pk.double = _kaiser_exact_ft_fp64(k0 * k0, b2, whw, ksc)
            f2: pk.double = _kaiser_exact_ft_fp64(k1 * k1, b2, whw, ksc)
            f3: pk.double = _kaiser_exact_ft_fp64(k2 * k2, b2, whw, ksc)
            F: pk.double = f1 * f2 * f3  # tensor product of spatial directions
            if pw == 2:
                F *= F
            window_m2: pk.double = 1.0 / F
            combo: pk.double = screen * window_m2
            scaling_b = biharm * combo

            # use harmonic or biharmonic for the first component
            if vico_var == 2:
                harm: pk.double = 0.0
                if (i + k1_off) == 0 and j == 0:
                    harm = PI * (gR * gR)  # finite limit at k1==k2==0
                else:
                    harm = FOUR_PI * (1.0 - BJ0) / kk
                scaling_H: pk.double = 2 * harm * combo
                H1[i][j][0] = scaling_H * H1[i][j][0]
                H1[i][j][1] = scaling_H * H1[i][j][1]
        elif periodicity == 3:
            scaling_b = stokes_kernel_fp64(kk, k0, k1, k2, xi, wsh, whw, ksc, pw)

        # relate harmonic/biharmonic to stokeslet
        SKK: pk.double = -kk * scaling_b

        if vico_var != 2:
            H1[i][j][0] = SKK * H1[i][j][0]
            H1[i][j][1] = SKK * H1[i][j][1]

        # always use biharmonic for the other two components
        k_dot_H_0: pk.double = k1 * H2[i][j][0] + k2 * H3[i][j][0]
        k_dot_H_1: pk.double = k1 * H2[i][j][1] + k2 * H3[i][j][1]
        k_dot_H_0 = scaling_b * k_dot_H_0
        k_dot_H_1 = scaling_b * k_dot_H_1
        H2[i][j][0] = SKK * H2[i][j][0] + k_dot_H_0 * k1
        H2[i][j][1] = SKK * H2[i][j][1] + k_dot_H_1 * k1
        H3[i][j][0] = SKK * H3[i][j][0] + k_dot_H_0 * k2
        H3[i][j][1] = SKK * H3[i][j][1] + k_dot_H_1 * k2

    pk.parallel_for(pk.TeamThreadRange(team_member, threads), thread_loop)


# END TEMPLATE STOKESCNV0


@pk.workunit
def stokeslet_convolution_zero_kernel_range(
    wid: int,
    H1,
    H2,
    H3,
    grid_size_1: int,
    box,
    k1_off: int,
    xi,
    pw: int,
    wsh,
    ksc,
    grid: pk.View1D[int],
    whw,
    ups,
    gR,
    vico_var: int,
    periodicity,
):
    PI: pk.double = 3.14159265358979323846264
    FOUR_PI: pk.double = 12.5663706143591729538506
    EIGHT_PI: pk.double = 25.1327412287183459077011
    # global index for H1, H2, H3
    i: int = wid // grid_size_1
    j: int = wid % grid_size_1

    # compute k-vector
    k0: pk.double = 0.0
    k1: pk.double = _k_vector_fp64(
        grid[1], box[1], ups[1], i + k1_off
    )  # free direction
    k2: pk.double = _k_vector_fp64(grid[2], box[2], ups[2], j)  # free direction
    kk: pk.double = k0 * k0 + k1 * k1 + k2 * k2

    scaling_b: pk.double = 0.0
    if periodicity == 1:
        # See the MATLAB code for details on the method

        # scaling for the zero mode [k0==0] (Vico method)
        C: pk.double = kk / (4 * xi * xi)
        kn: pk.double = pk.sqrt(kk)
        # modified Green's function
        BJ0: pk.double = pk.cyl_bessel_j0(gR * kn)
        BJ1: pk.double = pk.cyl_bessel_j1(gR * kn)
        biharm: pk.double = 0.0
        if (i + k1_off) == 0 and j == 0:
            biharm = (-1.0 / 8.0) * PI * gR * gR * gR * gR
        else:
            biharm = EIGHT_PI * ((BJ0 - 1.0) / (kk * kk) + (gR * BJ1) / (2 * kk * kn))
        screen: pk.double = (1 + C) * pk.exp(-C)
        # compute window_m2 on the fly
        b2: pk.double = wsh * wsh
        f1: pk.double = _kaiser_exact_ft_fp64(k0 * k0, b2, whw, ksc)
        f2: pk.double = _kaiser_exact_ft_fp64(k1 * k1, b2, whw, ksc)
        f3: pk.double = _kaiser_exact_ft_fp64(k2 * k2, b2, whw, ksc)
        F: pk.double = f1 * f2 * f3  # tensor product of spatial directions
        if pw == 2:
            F *= F
        window_m2: pk.double = 1.0 / F
        combo: pk.double = screen * window_m2
        scaling_b = biharm * combo

        # use harmonic or biharmonic for the first component
        if vico_var == 2:
            harm: pk.double = 0.0
            if (i + k1_off) == 0 and j == 0:
                harm = PI * (gR * gR)  # finite limit at k1==k2==0
            else:
                harm = FOUR_PI * (1.0 - BJ0) / kk
            scaling_H: pk.double = 2 * harm * combo
            H1[i][j][0] = scaling_H * H1[i][j][0]
            H1[i][j][1] = scaling_H * H1[i][j][1]
    elif periodicity == 3:
        scaling_b = stokes_kernel_fp64(kk, k0, k1, k2, xi, wsh, whw, ksc, pw)

    # relate harmonic/biharmonic to stokeslet
    SKK: pk.double = -kk * scaling_b

    if vico_var != 2:
        H1[i][j][0] = SKK * H1[i][j][0]
        H1[i][j][1] = SKK * H1[i][j][1]

    # always use biharmonic for the other two components
    k_dot_H_0: pk.double = k1 * H2[i][j][0] + k2 * H3[i][j][0]
    k_dot_H_1: pk.double = k1 * H2[i][j][1] + k2 * H3[i][j][1]
    k_dot_H_0 = scaling_b * k_dot_H_0
    k_dot_H_1 = scaling_b * k_dot_H_1
    H2[i][j][0] = SKK * H2[i][j][0] + k_dot_H_0 * k1
    H2[i][j][1] = SKK * H2[i][j][1] + k_dot_H_1 * k1
    H3[i][j][0] = SKK * H3[i][j][0] + k_dot_H_0 * k2
    H3[i][j][1] = SKK * H3[i][j][1] + k_dot_H_1 * k2


# START TEMPLATE STRESSCNV
@pk.workunit
def stresslet_convolution_kernel_STRESSCNV(
    team_member: pk.TeamMember,
    H11,
    H21,
    H31,
    H12,
    H22,
    H32,
    H13,
    H23,
    H33,
    grid_size_1: int,
    grid_size_2: int,
    box,
    k1_off: int,
    xi,
    pw: int,
    wsh,
    ksc,
    grid: pk.View1D[int],
    whw,
    ups,
    locals: pk.View1D[int],
    num_locals: int,
    freq_range: int,
    freq_offset: pk.View1D[int],
    threads: int,
):
    EIGHT_PI: pk.double = 25.1327412287183459077011
    wid_off: int = team_member.league_rank() * threads

    def thread_loop(tid: int):
        wid: int = wid_off + tid
        if wid >= freq_range:
            return
        # global index for H1, H2, H3
        i: int = (wid // (grid_size_1 * grid_size_2)) + freq_offset[0]
        j: int = (wid % (grid_size_1 * grid_size_2)) // grid_size_2 + freq_offset[1]
        k: int = wid % grid_size_2 + freq_offset[2]

        # compute k-vector
        k0: pk.double = 0.0
        if num_locals == 0:
            k0 = _k_vector_fp64(grid[0], box[0], ups[0], i)
        else:
            k0 = _k_vector_fp64(grid[0], box[0], ups[0], locals[i])
        k1: pk.double = _k_vector_fp64(
            grid[1], box[1], ups[1], j + k1_off
        )  # free direction
        k2: pk.double = _k_vector_fp64(grid[2], box[2], ups[2], k)  # free direction
        kk: pk.double = k0 * k0 + k1 * k1 + k2 * k2

        # see the Matlab code for details on the method

        # scaling for the global Fourier domain or local pad
        biharm: pk.double = -(EIGHT_PI) / (kk * kk)
        C: pk.double = kk / (4 * xi * xi)
        screen: pk.double = (1 + C) * pk.exp(-C)
        # compute window_m2 on the fly
        b2: pk.double = wsh * wsh
        f1: pk.double = _kaiser_exact_ft_fp64(k0 * k0, b2, whw, ksc)
        f2: pk.double = _kaiser_exact_ft_fp64(k1 * k1, b2, whw, ksc)
        f3: pk.double = _kaiser_exact_ft_fp64(k2 * k2, b2, whw, ksc)
        F: pk.double = f1 * f2 * f3  # tensor product of spatial directions
        if pw == 2:
            F *= F
        window_m2: pk.double = 1.0 / F
        scaling: pk.double = biharm * screen * window_m2
        # now the element corresponding to the zero mode is Inf.
        # this is fine since we overwrite it later on

        # use relation between stresslet and biharmonic Green's function
        H_dot_k0_0: pk.double = (
            H11[i][j][k][0] * k0 + H12[i][j][k][0] * k1 + H13[i][j][k][0] * k2
        )
        H_dot_k0_1: pk.double = (
            H11[i][j][k][1] * k0 + H12[i][j][k][1] * k1 + H13[i][j][k][1] * k2
        )
        H_dot_k1_0: pk.double = (
            H21[i][j][k][0] * k0 + H22[i][j][k][0] * k1 + H23[i][j][k][0] * k2
        )
        H_dot_k1_1: pk.double = (
            H21[i][j][k][1] * k0 + H22[i][j][k][1] * k1 + H23[i][j][k][1] * k2
        )
        H_dot_k2_0: pk.double = (
            H31[i][j][k][0] * k0 + H32[i][j][k][0] * k1 + H33[i][j][k][0] * k2
        )
        H_dot_k2_1: pk.double = (
            H31[i][j][k][1] * k0 + H32[i][j][k][1] * k1 + H33[i][j][k][1] * k2
        )
        k_dot_H_dot_k_0: pk.double = k0 * H_dot_k0_0 + k1 * H_dot_k1_0 + k2 * H_dot_k2_0
        k_dot_H_dot_k_1: pk.double = k0 * H_dot_k0_1 + k1 * H_dot_k1_1 + k2 * H_dot_k2_1
        k_dot_H1_0: pk.double = (
            k0 * H11[i][j][k][0] + k1 * H21[i][j][k][0] + k2 * H31[i][j][k][0]
        )
        k_dot_H1_1: pk.double = (
            k0 * H11[i][j][k][1] + k1 * H21[i][j][k][1] + k2 * H31[i][j][k][1]
        )
        k_dot_H2_0: pk.double = (
            k0 * H12[i][j][k][0] + k1 * H22[i][j][k][0] + k2 * H32[i][j][k][0]
        )
        k_dot_H2_1: pk.double = (
            k0 * H12[i][j][k][1] + k1 * H22[i][j][k][1] + k2 * H32[i][j][k][1]
        )
        k_dot_H3_0: pk.double = (
            k0 * H13[i][j][k][0] + k1 * H23[i][j][k][0] + k2 * H33[i][j][k][0]
        )
        k_dot_H3_1: pk.double = (
            k0 * H13[i][j][k][1] + k1 * H23[i][j][k][1] + k2 * H33[i][j][k][1]
        )
        trace_H_0: pk.double = H11[i][j][k][0] + H22[i][j][k][0] + H33[i][j][k][0]
        trace_H_1: pk.double = H11[i][j][k][1] + H22[i][j][k][1] + H33[i][j][k][1]
        vp1_0: pk.double = (
            H_dot_k0_1 + k_dot_H1_1 + k0 * trace_H_1
        ) * kk - 2.0 * k0 * k_dot_H_dot_k_1
        vp1_1: pk.double = (
            -(H_dot_k0_0 + k_dot_H1_0 + k0 * trace_H_0) * kk
            + 2.0 * k0 * k_dot_H_dot_k_0
        )
        vp2_0: pk.double = (
            H_dot_k1_1 + k_dot_H2_1 + k1 * trace_H_1
        ) * kk - 2.0 * k1 * k_dot_H_dot_k_1
        vp2_1: pk.double = (
            -(H_dot_k1_0 + k_dot_H2_0 + k1 * trace_H_0) * kk
            + 2.0 * k1 * k_dot_H_dot_k_0
        )
        vp3_0: pk.double = (
            H_dot_k2_1 + k_dot_H3_1 + k2 * trace_H_1
        ) * kk - 2.0 * k2 * k_dot_H_dot_k_1
        vp3_1: pk.double = (
            -(H_dot_k2_0 + k_dot_H3_0 + k2 * trace_H_0) * kk
            + 2.0 * k2 * k_dot_H_dot_k_0
        )
        H11[i][j][k][0] = scaling * vp1_0
        H11[i][j][k][1] = scaling * vp1_1
        H21[i][j][k][0] = scaling * vp2_0
        H21[i][j][k][1] = scaling * vp2_1
        H31[i][j][k][0] = scaling * vp3_0
        H31[i][j][k][1] = scaling * vp3_1

    pk.parallel_for(pk.TeamThreadRange(team_member, threads), thread_loop)


# END TEMPLATE STRESSCNV


@pk.workunit
def stresslet_convolution_kernel_range(
    wid: int,
    H11,
    H21,
    H31,
    H12,
    H22,
    H32,
    H13,
    H23,
    H33,
    grid_size_1: int,
    grid_size_2: int,
    box,
    k1_off: int,
    xi,
    pw: int,
    wsh,
    ksc,
    grid: pk.View1D[int],
    whw,
    ups,
    locals: pk.View1D[int],
    num_locals: int,
):
    EIGHT_PI: pk.double = 25.1327412287183459077011

    # global index for H1, H2, H3
    i: int = (wid // (grid_size_1 * grid_size_2)) + freq_offset[0]
    j: int = (wid % (grid_size_1 * grid_size_2)) // grid_size_2 + freq_offset[1]
    k: int = wid % grid_size_2 + freq_offset[2]

    # compute k-vector
    k0: pk.double = 0.0
    if num_locals == 0:
        k0 = _k_vector_fp64(grid[0], box[0], ups[0], i)
    else:
        k0 = _k_vector_fp64(grid[0], box[0], ups[0], locals[i])
    k1: pk.double = _k_vector_fp64(
        grid[1], box[1], ups[1], j + k1_off
    )  # free direction
    k2: pk.double = _k_vector_fp64(grid[2], box[2], ups[2], k)  # free direction
    kk: pk.double = k0 * k0 + k1 * k1 + k2 * k2

    # see the Matlab code for details on the method

    # scaling for the global Fourier domain or local pad
    biharm: pk.double = -(EIGHT_PI) / (kk * kk)
    C: pk.double = kk / (4 * xi * xi)
    screen: pk.double = (1 + C) * pk.exp(-C)
    # compute window_m2 on the fly
    b2: pk.double = wsh * wsh
    f1: pk.double = _kaiser_exact_ft_fp64(k0 * k0, b2, whw, ksc)
    f2: pk.double = _kaiser_exact_ft_fp64(k1 * k1, b2, whw, ksc)
    f3: pk.double = _kaiser_exact_ft_fp64(k2 * k2, b2, whw, ksc)
    F: pk.double = f1 * f2 * f3  # tensor product of spatial directions
    if pw == 2:
        F *= F
    window_m2: pk.double = 1.0 / F
    scaling: pk.double = biharm * screen * window_m2
    # now the element corresponding to the zero mode is Inf.
    # this is fine since we overwrite it later on

    # use relation between stresslet and biharmonic Green's function
    H_dot_k0_0: pk.double = (
        H11[i][j][k][0] * k0 + H12[i][j][k][0] * k1 + H13[i][j][k][0] * k2
    )
    H_dot_k0_1: pk.double = (
        H11[i][j][k][1] * k0 + H12[i][j][k][1] * k1 + H13[i][j][k][1] * k2
    )
    H_dot_k1_0: pk.double = (
        H21[i][j][k][0] * k0 + H22[i][j][k][0] * k1 + H23[i][j][k][0] * k2
    )
    H_dot_k1_1: pk.double = (
        H21[i][j][k][1] * k0 + H22[i][j][k][1] * k1 + H23[i][j][k][1] * k2
    )
    H_dot_k2_0: pk.double = (
        H31[i][j][k][0] * k0 + H32[i][j][k][0] * k1 + H33[i][j][k][0] * k2
    )
    H_dot_k2_1: pk.double = (
        H31[i][j][k][1] * k0 + H32[i][j][k][1] * k1 + H33[i][j][k][1] * k2
    )
    k_dot_H_dot_k_0: pk.double = k0 * H_dot_k0_0 + k1 * H_dot_k1_0 + k2 * H_dot_k2_0
    k_dot_H_dot_k_1: pk.double = k0 * H_dot_k0_1 + k1 * H_dot_k1_1 + k2 * H_dot_k2_1
    k_dot_H1_0: pk.double = (
        k0 * H11[i][j][k][0] + k1 * H21[i][j][k][0] + k2 * H31[i][j][k][0]
    )
    k_dot_H1_1: pk.double = (
        k0 * H11[i][j][k][1] + k1 * H21[i][j][k][1] + k2 * H31[i][j][k][1]
    )
    k_dot_H2_0: pk.double = (
        k0 * H12[i][j][k][0] + k1 * H22[i][j][k][0] + k2 * H32[i][j][k][0]
    )
    k_dot_H2_1: pk.double = (
        k0 * H12[i][j][k][1] + k1 * H22[i][j][k][1] + k2 * H32[i][j][k][1]
    )
    k_dot_H3_0: pk.double = (
        k0 * H13[i][j][k][0] + k1 * H23[i][j][k][0] + k2 * H33[i][j][k][0]
    )
    k_dot_H3_1: pk.double = (
        k0 * H13[i][j][k][1] + k1 * H23[i][j][k][1] + k2 * H33[i][j][k][1]
    )
    trace_H_0: pk.double = H11[i][j][k][0] + H22[i][j][k][0] + H33[i][j][k][0]
    trace_H_1: pk.double = H11[i][j][k][1] + H22[i][j][k][1] + H33[i][j][k][1]
    vp1_0: pk.double = (
        H_dot_k0_1 + k_dot_H1_1 + k0 * trace_H_1
    ) * kk - 2.0 * k0 * k_dot_H_dot_k_1
    vp1_1: pk.double = (
        -(H_dot_k0_0 + k_dot_H1_0 + k0 * trace_H_0) * kk + 2.0 * k0 * k_dot_H_dot_k_0
    )
    vp2_0: pk.double = (
        H_dot_k1_1 + k_dot_H2_1 + k1 * trace_H_1
    ) * kk - 2.0 * k1 * k_dot_H_dot_k_1
    vp2_1: pk.double = (
        -(H_dot_k1_0 + k_dot_H2_0 + k1 * trace_H_0) * kk + 2.0 * k1 * k_dot_H_dot_k_0
    )
    vp3_0: pk.double = (
        H_dot_k2_1 + k_dot_H3_1 + k2 * trace_H_1
    ) * kk - 2.0 * k2 * k_dot_H_dot_k_1
    vp3_1: pk.double = (
        -(H_dot_k2_0 + k_dot_H3_0 + k2 * trace_H_0) * kk + 2.0 * k2 * k_dot_H_dot_k_0
    )
    H11[i][j][k][0] = scaling * vp1_0
    H11[i][j][k][1] = scaling * vp1_1
    H21[i][j][k][0] = scaling * vp2_0
    H21[i][j][k][1] = scaling * vp2_1
    H31[i][j][k][0] = scaling * vp3_0
    H31[i][j][k][1] = scaling * vp3_1


# START TEMPLATE STRESSCNV0
@pk.workunit
def stresslet_convolution_zero_kernel_STRESSCNV0(
    team_member: pk.TeamMember,
    H11,
    H21,
    H31,
    H12,
    H22,
    H32,
    H13,
    H23,
    H33,
    grid_size_1: int,
    box,
    k1_off: int,
    xi,
    pw: int,
    wsh,
    ksc,
    grid: pk.View1D[int],
    whw,
    ups,
    gR,
    vico_var: int,
    periodicity: int,
    freq_range: int,
    threads: int,
):
    PI: pk.double = 3.14159265358979323846264
    FOUR_PI: pk.double = 12.5663706143591729538506
    EIGHT_PI: pk.double = 25.1327412287183459077011
    wid_off: int = team_member.league_rank() * threads

    def thread_loop(tid: int):
        wid: int = wid_off + tid
        if wid >= freq_range:
            return
        # global index for H1, H2, H3
        i: int = wid // grid_size_1
        j: int = wid % grid_size_1

        # compute k-vector
        k0: pk.double = 0.0
        k1: pk.double = _k_vector_fp64(
            grid[1], box[1], ups[1], i + k1_off
        )  # free direction
        k2: pk.double = _k_vector_fp64(grid[2], box[2], ups[2], j)  # free direction
        kk: pk.double = k0 * k0 + k1 * k1 + k2 * k2

        scaling_b: pk.double = 0.0
        BJ0: pk.double = 0.0
        combo: pk.double = 0.0
        if periodicity == 1:
            # see the matlab code for details on the method

            # scaling for the zero mode [k0 == 0] (Vico method)
            C: pk.double = kk / (4 * xi * xi)
            kn: pk.double = pk.sqrt(kk)
            # modified Green's function
            BJ0 = pk.cyl_bessel_j0(gR * kn)
            BJ1: pk.double = pk.cyl_bessel_j1(gR * kn)
            biharm: pk.double = 0.0
            if (i + k1_off) == 0 and j == 0:
                biharm = (-1.0 / 8.0) * PI * (gR * gR * gR * gR)
            else:
                biharm = EIGHT_PI * ((BJ0 - 1) / (kk * kk) + (gR * BJ1) / (2 * kk * kn))
            screen: pk.double = (1 + C) * pk.exp(-C)
            # compute window_m2 on the fly
            b2: pk.double = wsh * wsh
            f1: pk.double = _kaiser_exact_ft_fp64(k0 * k0, b2, whw, ksc)
            f2: pk.double = _kaiser_exact_ft_fp64(k1 * k1, b2, whw, ksc)
            f3: pk.double = _kaiser_exact_ft_fp64(k2 * k2, b2, whw, ksc)
            F: pk.double = f1 * f2 * f3  # tensor product of spatial directions
            if pw == 2:
                F *= F
            window_m2: pk.double = 1.0 / F
            combo = screen * window_m2
            scaling_b = biharm * combo
        elif periodicity == 3:
            scaling_b = stokes_kernel_fp64(kk, k0, k1, k2, xi, wsh, whw, ksc, pw)

        # relate harmonic/biharmonic to stresslet
        H_dot_k0_0: pk.double = H12[i][j][0] * k1 + H13[i][j][0] * k2
        H_dot_k0_1: pk.double = H12[i][j][1] * k1 + H13[i][j][1] * k2
        H_dot_k1_0: pk.double = H22[i][j][0] * k1 + H23[i][j][0] * k2
        H_dot_k1_1: pk.double = H22[i][j][1] * k1 + H23[i][j][1] * k2
        H_dot_k2_0: pk.double = H32[i][j][0] * k1 + H33[i][j][0] * k2
        H_dot_k2_1: pk.double = H32[i][j][1] * k1 + H33[i][j][1] * k2
        k_dot_H_dot_k_0: pk.double = k1 * H_dot_k1_0 + k2 * H_dot_k2_0
        k_dot_H_dot_k_1: pk.double = k1 * H_dot_k1_1 + k2 * H_dot_k2_1
        scaling_H: pk.double = 0.0
        trace_H_0: pk.double = 0.0
        trace_H_1: pk.double = 0.0
        if vico_var == 2:
            harm: pk.double = 0.0
            if (i + k1_off) == 0 and j == 0:
                harm = PI * (gR * gR)
            else:
                harm = FOUR_PI * (1 - BJ0) / kk
            scaling_H = harm * combo
            trace_H_0 = H22[i][j][0] + H33[i][j][0]
            trace_H_1 = H22[i][j][1] + H33[i][j][1]
        else:
            trace_H_0 = H11[i][j][0] + H22[i][j][0] + H33[i][j][0]
            trace_H_1 = H11[i][j][1] + H22[i][j][1] + H33[i][j][1]
        # relation between stresslet and biharmonic
        k_dot_H1_0: pk.double = k1 * H21[i][j][0] + k2 * H31[i][j][0]
        k_dot_H1_1: pk.double = k1 * H21[i][j][1] + k2 * H31[i][j][1]
        vp1_0: pk.double = 0.0
        vp1_1: pk.double = 0.0
        if vico_var != 2:
            vp1_0 = (
                H_dot_k0_0 + k_dot_H1_0 + k0 * trace_H_0
            ) * kk - 2 * k0 * k_dot_H_dot_k_0
            vp1_1 = (
                H_dot_k0_1 + k_dot_H1_1 + k0 * trace_H_1
            ) * kk - 2 * k0 * k_dot_H_dot_k_1
            H11[i][j][0] = scaling_b * vp1_1
            H11[i][j][1] = -scaling_b * vp1_0
        k_dot_H2_0: pk.double = k1 * H22[i][j][0] + k2 * H32[i][j][0]
        k_dot_H2_1: pk.double = k1 * H22[i][j][1] + k2 * H32[i][j][1]
        vp2_0: pk.double = (
            H_dot_k1_0 + k_dot_H2_0 + k1 * trace_H_0
        ) * kk - 2 * k1 * k_dot_H_dot_k_0
        vp2_1: pk.double = (
            H_dot_k1_1 + k_dot_H2_1 + k1 * trace_H_1
        ) * kk - 2 * k1 * k_dot_H_dot_k_1
        H21[i][j][0] = scaling_b * vp2_1
        H21[i][j][1] = -scaling_b * vp2_0
        k_dot_H3_0: pk.double = k1 * H23[i][j][0] + k2 * H33[i][j][0]
        k_dot_H3_1: pk.double = k1 * H23[i][j][1] + k2 * H33[i][j][1]
        vp3_0: pk.double = (
            H_dot_k2_0 + k_dot_H3_0 + k2 * trace_H_0
        ) * kk - 2 * k2 * k_dot_H_dot_k_0
        vp3_1: pk.double = (
            H_dot_k2_1 + k_dot_H3_1 + k2 * trace_H_1
        ) * kk - 2 * k2 * k_dot_H_dot_k_1
        H31[i][j][0] = scaling_b * vp3_1
        H31[i][j][1] = -scaling_b * vp3_0
        # relation between stresslet and harmonic
        if vico_var == 2:
            H11i_0: pk.double = H11[i][j][0]
            H11i_1: pk.double = H11[i][j][1]
            vp1_0 = k0 * H11i_0 + k_dot_H1_0 + H_dot_k0_0
            vp1_1 = k0 * H11i_1 + k_dot_H1_1 + H_dot_k0_1
            H11[i][j][0] = -2 * scaling_H * vp1_1
            H11[i][j][1] = 2 * scaling_H * vp1_0
            vp2_0 = k1 * H11i_0
            vp2_1 = k1 * H11i_1
            H21[i][j][0] -= 2 * scaling_H * vp2_1
            H21[i][j][1] += 2 * scaling_H * vp2_0
            vp3_0 = k2 * H11i_0
            vp3_1 = k2 * H11i_1
            H31[i][j][0] -= 2 * scaling_H * vp3_1
            H31[i][j][1] += 2 * scaling_H * vp3_0

    pk.parallel_for(pk.TeamThreadRange(team_member, threads), thread_loop)


# END TEMPLATE STRESSCNV0


@pk.workunit
def stresslet_convolution_zero_kernel_range(
    wid: int,
    H11,
    H21,
    H31,
    H12,
    H22,
    H32,
    H13,
    H23,
    H33,
    grid_size_1: int,
    box,
    k1_off: int,
    xi,
    pw: int,
    wsh,
    ksc,
    grid: pk.View1D[int],
    whw,
    ups,
    gR,
    vico_var: int,
    periodicity: int,
):
    PI: pk.double = 3.14159265358979323846264
    FOUR_PI: pk.double = 12.5663706143591729538506
    EIGHT_PI: pk.double = 25.1327412287183459077011

    # global index for H1, H2, H3
    i: int = wid // grid_size_1
    j: int = wid % grid_size_1

    # compute k-vector
    k0: pk.double = 0.0
    k1: pk.double = _k_vector_fp64(
        grid[1], box[1], ups[1], i + k1_off
    )  # free direction
    k2: pk.double = _k_vector_fp64(grid[2], box[2], ups[2], j)  # free direction
    kk: pk.double = k0 * k0 + k1 * k1 + k2 * k2

    scaling_b: pk.double = 0.0
    BJ0: pk.double = 0.0
    combo: pk.double = 0.0
    if periodicity == 1:
        # see the matlab code for details on the method

        # scaling for the zero mode [k0 == 0] (Vico method)
        C: pk.double = kk / (4 * xi * xi)
        kn: pk.double = pk.sqrt(kk)
        # modified Green's function
        BJ0 = pk.cyl_bessel_j0(gR * kn)
        BJ1: pk.double = pk.cyl_bessel_j1(gR * kn)
        biharm: pk.double = 0.0
        if (i + k1_off) == 0 and j == 0:
            biharm = (-1.0 / 8.0) * PI * (gR * gR * gR * gR)
        else:
            biharm = EIGHT_PI * ((BJ0 - 1) / (kk * kk) + (gR * BJ1) / (2 * kk * kn))
        screen: pk.double = (1 + C) * pk.exp(-C)
        # compute window_m2 on the fly
        b2: pk.double = wsh * wsh
        f1: pk.double = _kaiser_exact_ft_fp64(k0 * k0, b2, whw, ksc)
        f2: pk.double = _kaiser_exact_ft_fp64(k1 * k1, b2, whw, ksc)
        f3: pk.double = _kaiser_exact_ft_fp64(k2 * k2, b2, whw, ksc)
        F: pk.double = f1 * f2 * f3  # tensor product of spatial directions
        if pw == 2:
            F *= F
        window_m2: pk.double = 1.0 / F
        combo = screen * window_m2
        scaling_b = biharm * combo
    elif periodicity == 3:
        scaling_b = stokes_kernel_fp64(kk, k0, k1, k2, xi, wsh, whw, ksc, pw)

    # relate harmonic/biharmonic to stresslet
    H_dot_k0_0: pk.double = H12[i][j][0] * k1 + H13[i][j][0] * k2
    H_dot_k0_1: pk.double = H12[i][j][1] * k1 + H13[i][j][1] * k2
    H_dot_k1_0: pk.double = H22[i][j][0] * k1 + H23[i][j][0] * k2
    H_dot_k1_1: pk.double = H22[i][j][1] * k1 + H23[i][j][1] * k2
    H_dot_k2_0: pk.double = H32[i][j][0] * k1 + H33[i][j][0] * k2
    H_dot_k2_1: pk.double = H32[i][j][1] * k1 + H33[i][j][1] * k2
    k_dot_H_dot_k_0: pk.double = k1 * H_dot_k1_0 + k2 * H_dot_k2_0
    k_dot_H_dot_k_1: pk.double = k1 * H_dot_k1_1 + k2 * H_dot_k2_1
    scaling_H: pk.double = 0.0
    trace_H_0: pk.double = 0.0
    trace_H_1: pk.double = 0.0
    if vico_var == 2:
        harm: pk.double = 0.0
        if (i + k1_off) == 0 and j == 0:
            harm = PI * (gR * gR)
        else:
            harm = FOUR_PI * (1 - BJ0) / kk
        scaling_H = harm * combo
        trace_H_0 = H22[i][j][0] + H33[i][j][0]
        trace_H_1 = H22[i][j][1] + H33[i][j][1]
    else:
        trace_H_0 = H11[i][j][0] + H22[i][j][0] + H33[i][j][0]
        trace_H_1 = H11[i][j][1] + H22[i][j][1] + H33[i][j][1]
    # relation between stresslet and biharmonic
    k_dot_H1_0: pk.double = k1 * H21[i][j][0] + k2 * H31[i][j][0]
    k_dot_H1_1: pk.double = k1 * H21[i][j][1] + k2 * H31[i][j][1]
    vp1_0: pk.double = 0.0
    vp1_1: pk.double = 0.0
    if vico_var != 2:
        vp1_0 = (
            H_dot_k0_0 + k_dot_H1_0 + k0 * trace_H_0
        ) * kk - 2 * k0 * k_dot_H_dot_k_0
        vp1_1 = (
            H_dot_k0_1 + k_dot_H1_1 + k0 * trace_H_1
        ) * kk - 2 * k0 * k_dot_H_dot_k_1
        H11[i][j][0] = scaling_b * vp1_1
        H11[i][j][1] = -scaling_b * vp1_0
    k_dot_H2_0: pk.double = k1 * H22[i][j][0] + k2 * H32[i][j][0]
    k_dot_H2_1: pk.double = k1 * H22[i][j][1] + k2 * H32[i][j][1]
    vp2_0: pk.double = (
        H_dot_k1_0 + k_dot_H2_0 + k1 * trace_H_0
    ) * kk - 2 * k1 * k_dot_H_dot_k_0
    vp2_1: pk.double = (
        H_dot_k1_1 + k_dot_H2_1 + k1 * trace_H_1
    ) * kk - 2 * k1 * k_dot_H_dot_k_1
    H21[i][j][0] = scaling_b * vp2_1
    H21[i][j][1] = -scaling_b * vp2_0
    k_dot_H3_0: pk.double = k1 * H23[i][j][0] + k2 * H33[i][j][0]
    k_dot_H3_1: pk.double = k1 * H23[i][j][1] + k2 * H33[i][j][1]
    vp3_0: pk.double = (
        H_dot_k2_0 + k_dot_H3_0 + k2 * trace_H_0
    ) * kk - 2 * k2 * k_dot_H_dot_k_0
    vp3_1: pk.double = (
        H_dot_k2_1 + k_dot_H3_1 + k2 * trace_H_1
    ) * kk - 2 * k2 * k_dot_H_dot_k_1
    H31[i][j][0] = scaling_b * vp3_1
    H31[i][j][1] = -scaling_b * vp3_0
    # relation between stresslet and harmonic
    if vico_var == 2:
        H11i_0: pk.double = H11[i][j][0]
        H11i_1: pk.double = H11[i][j][1]
        vp1_0 = k0 * H11i_0 + k_dot_H1_0 + H_dot_k0_0
        vp1_1 = k0 * H11i_1 + k_dot_H1_1 + H_dot_k0_1
        H11[i][j][0] = -2 * scaling_H * vp1_1
        H11[i][j][1] = 2 * scaling_H * vp1_0
        vp2_0 = k1 * H11i_0
        vp2_1 = k1 * H11i_1
        H21[i][j][0] -= 2 * scaling_H * vp2_1
        H21[i][j][1] += 2 * scaling_H * vp2_0
        vp3_0 = k2 * H11i_0
        vp3_1 = k2 * H11i_1
        H31[i][j][0] -= 2 * scaling_H * vp3_1
        H31[i][j][1] += 2 * scaling_H * vp3_0


# START APPLICATION KVEC _
# END APPLICATION KVEC _

# START APPLICATION KBEXACT _
# END APPLICATION KBEXACT _

# START APPLICATION STOKESKERN _
# END APPLICATION STOKESKERN _

# START APPLICATION LAPLACECNV _
# END APPLICATION LAPLACECNV _

# START APPLICATION STOKESCNV _
# END APPLICATION STOKESCNV _

# START APPLICATION STOKESCNV0 _
# END APPLICATION STOKESCNV0 _

# START APPLICATION STRESSCNV _
# END APPLICATION STRESSCNV _

# START APPLICATION STRESSCNV0 _
# END APPLICATION STRESSCNV0 _
