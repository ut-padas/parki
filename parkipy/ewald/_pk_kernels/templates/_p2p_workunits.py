import pykokkos as pk


# START TEMPLATE STOKESSL
@pk.workunit
def p2p_stokes_sl_STOKESSL(
    member,
    u: pk.View2D[pk.double],
    out_list: pk.View2D[pk.dobule],
    out_ordering: pk.View1D[int],
    out_cell_chunk_size: int,
    out_ne_cells: int,
    out_cell_size: int,
    in_nonempty_neighbors: pk.View2D[int],
    in_cell_size: int,
    periodicity: int,
    periodic_shif: List[int],
    split: pk.double,
    split2: pk.double,
):
    ioff: int = member.league_rank() * out_cell_chunk_size

    def thread_loop(ii: int):
        i: int = ioff + ii
        if i >= out_ne_cells * out_cell_size:
            return
        iglb: int = out_ordering[i]
        if iglb < 0:
            return
        xi: List[pk.double] = [0, 0, 0]
        yj: List[pk.double] = [0, 0, 0]
        qj: List[pk.double] = [0, 0, 0]
        r: List[pk.double] = [0, 0, 0]
        ui: List[pk.double] = [0, 0, 0]
        for l in range(3):
            xi[l] = out_list[l][i]
        for k in range(27):
            incell: int = in_nonempty_neighbors[out_cell][k]
            if incell < 0:
                continue
            inoff: int = incell * in_cell_size
            for j in range(inoff, inoff + in_cell_size):
                if in_list[0][j] < 0:
                    continue
                qr: pk.double = 0.0
                d2: pk.double = 0.0
                for l in range(d_in):
                    yj[l] = y_list[l][j]
                    qj[l] = y_list_forces[l][j]
                    r[l] = xi[l] - yj[l] - periodic_shift[l]
                    d2 += r[l] * r[l]
                    qr += qj[l] * r[l]
                # kernel dependent code
                d: pk.double = pk.sqrt(d2)
                od: pk.double = 1.0 / d
                od = od + (od - od)  # od -> NaN if od = inf
                od = pk.fmax(od, 0.0)  # max(NaN, 0.0) = 0.0
                # Ewald correction
                splitd: pk.double = split * d
                splitd2: pk.double = split2 * d2
                A: pk.double = (splitd < 1e-14) * (self.xi_two_inv_sqrt_pi) + (
                    splitd >= 1e-14
                ) * (pk.erf(splitd) * od)
                B: pk.double = self.xi * pk.exp(-splitd2) * self.two_inv_sqrt_pi
                # Replace C by its Taylor expansion close to zero
                # (relative error at most around 3e-13 for four terms)
                # NOTE: This might be a bit unnecessary, as C goes into terms
                # that will anyway go to zero as r -> 0. But we do it
                # anyway for good measure.
                # Reformulation of if/else statement
                term2: pk.double = splitd2 * self.C_term1
                term3: pk.double = splitd2 * term2
                C: pk.double = (splitd < 4.75e-2) * (
                    self.m_c1_C_term1
                    - self.c2 * term2
                    + self.c3 * term3
                    - self.c4 * (splitd2 * term3)
                ) + (splitd >= 4.75e-2) * ((A - B) * od2)
                tmp1: pk.double = od - B - A
                tmp2: pk.double = qr * (od * od * od - C)
                for l in range(3):
                    u[l] += tmp1 * qj[l] + tmp2 * r[l]
            for l in range(3):
                u[iglb][l] += u[l]

        pk.parallel_for(pk.TeamThreadRange(member, out_cell_chunk_size), thread_loop)


# END TEMPLATE STOKESSL

# START APPLICATION STOKESSL _
# END APPLICATION STOKESSL _
