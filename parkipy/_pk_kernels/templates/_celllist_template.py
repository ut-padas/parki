"""
PyKokkos kernels for the parkipy.CellList class
"""

from typing import List
import pykokkos as pk


@pk.workunit
def get_nonempty_neighbors(
    cell: int,
    grid_area,
    cell_grid_shape_0,
    cell_grid_shape_1,
    cell_grid_shape_2,
    nonempty_cell_index,
    nonempty_neighbors,
):
    cell_x: int = cell // grid_area
    cell_y: int = (cell % grid_area) // cell_grid_shape_2
    cell_z: int = (cell % grid_area) % cell_grid_shape_2
    for dx in range(-1, 2):
        neighbor_x: int = cell_x + dx
        if neighbor_x < 0 or neighbor_x >= cell_grid_shape_0:
            continue
        for dy in range(-1, 2):
            neighbor_y: int = cell_y + dy
            if neighbor_y < 0 or neighbor_y >= cell_grid_shape_1:
                continue
            for dz in range(-1, 2):
                neighbor_z: int = cell_z + dz
                if neighbor_z < 0 or neighbor_z >= cell_grid_shape_2:
                    continue
                neighbor_empty: int = (
                    neighbor_x * grid_area + neighbor_y * cell_grid_shape_2 + neighbor_z
                )

                neighbor: int = nonempty_cell_index[neighbor_empty]
                nonempty_neighbors[cell][dx + 1][dy + 1][dz + 1] = neighbor


# START TEMPLATE COUNT
@pk.workunit
def count_particles_COUNT(i: int, counter, p, rc, box):
    cell: int = _get_cell_fp64(p, i, rc, box[0], box[1], box[2])
    pk.atomic_add(counter, [cell], 1)


# END TEMPLATE COUNT


# START TEMPLATE GETCELL
@pk.function
def _get_cell_GETCELL(
    p: pk.View2D[pk.double],
    i: int,
    rc: pk.double,
    box_length_x: pk.double,
    box_length_y: pk.double,
    box_length_z: pk.double,
) -> int:
    num_cells: List[int] = [
        int(box_length_x / rc),
        int(box_length_y / rc),
        int(box_length_z / rc),
    ]
    cell_size: List[pk.double] = [
        box_length_x / num_cells[0],
        box_length_y / num_cells[1],
        box_length_z / num_cells[2],
    ]
    cell_xyz: List[int] = [
        p[0][i] / cell_size[0],
        p[1][i] / cell_size[1],
        p[2][i] / cell_size[2],
    ]
    cell: int = (
        cell_xyz[0] * num_cells[2] * num_cells[1]
        + cell_xyz[1] * num_cells[2]
        + cell_xyz[2]
    )
    return cell


# END TEMPLATE GETCELL


# START TEMPLATE RP
@pk.workunit
def reshuffle_particles_RP(
    i: int,
    p_list,
    counter,
    l2g,
    p,
    cell2nz,
    rc,
    box,
    cell_size,
    dp,
):
    cell: int = _get_cell_fp64(p, i, rc, box[0], box[1], box[2])  # read
    cell_off: int = pk.atomic_fetch_add(counter, [cell], 1)  # write
    nz: int = cell2nz[cell]  # read
    l_idx: int = nz * cell_size + cell_off
    l2g[l_idx] = i  # write
    for k in range(dp):
        p_list[k][l_idx] = p[k][i]  # read + write


# END TEMPLATE RP


# START TEMPLATE RF
@pk.workunit
def reshuffle_forces_RF(
    i: int,
    q_list,
    l2g,
    q,
    dq,
):
    glb: int = l2g[i]
    if glb < 0:
        return
    for k in range(dq):
        q_list[k][i] = q[k][glb]


# END TEMPLATE RF


# START APPLICATION COUNT _
# END APPLICATION COUNT _

# START APPLICATION GETCELL _
# END APPLICATION GETCELL _

# START APPLICATION RP _
# END APPLICATION RP _

# START APPLICATION RF _
# END APPLICATION RF _
