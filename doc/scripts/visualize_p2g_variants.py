"""
Visualize P2G variants on a three-level parallel machine
(league -> team -> thread).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path

MARKERS = ["*", "p", "P", "o", "D"]
MARKER_SIZE = 8
ARROW_COLOR = "black"

# Macro for team hatch patterns
HATCHES = ["//", "..", "xx"]

# Macros for footprint overlap shading
SAME_TEAM_OVERLAP_COLOR = "blue"
DIFF_TEAM_OVERLAP_COLOR = "red"
OVERLAP_ALPHA = 0.4


def make_wavy_path(start, end, num_waves=3.5, amplitude=0.18):
    """
    Generates a parametric sinusoidal wavy path between start and end points.
    """
    x1, y1 = start
    x2, y2 = end
    t = np.linspace(0, 1, 100)
    x_line = x1 + t * (x2 - x1)
    y_line = y1 + t * (y2 - y1)
    dx = x2 - x1
    dy = y2 - y1
    length = np.hypot(dx, dy)
    nx = -dy / length
    ny = dx / length

    # Envelope sin(pi * t) keeps endpoints anchored nicely
    wave = amplitude * np.sin(2 * np.pi * num_waves * t) * (np.sin(np.pi * t) ** 2)
    x_wavy = x_line + wave * nx
    y_wavy = y_line + wave * ny

    verts = list(zip(x_wavy, y_wavy))
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
    return Path(verts, codes)


def wavy_arrow(ax, start, end):
    wavy_p = make_wavy_path(start, end, num_waves=2.5, amplitude=0.18)

    arrow = FancyArrowPatch(
        path=wavy_p,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.8,
        color="black",
        zorder=5,
        shrinkA=6,
        shrinkB=6,
    )
    ax.add_patch(arrow)


def box(ax, x0, y0, ncols, nrows, grid=True):
    """
    Plot grid lines originating from (x0, y0).
    Each grid cell has width 1.
    If grid=False, only plot the outline of the computational box.
    """
    xs = np.linspace(x0, x0 + ncols, ncols + 1)
    ys = np.linspace(y0, y0 + nrows, nrows + 1)

    if grid:
        for x in xs:
            ax.plot([x, x], [y0, y0 + nrows], color="black", linewidth=1)
        for y in ys:
            ax.plot([x0, x0 + ncols], [y, y], color="black", linewidth=1)
    else:
        x = [x0, x0 + ncols, x0 + ncols, x0, x0]
        y = [y0, y0, y0 + nrows, y0 + nrows, y0]
        ax.plot(x, y, color="black", linewidth=1)


def setup_axes():
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_aspect("equal")
    return fig, ax


def add_particles(ax, x0, y0, ncols, nrows, subset=None):
    """
    Plot particle markers. If subset is given (an iterable of marker
    indices), only those particles are plotted; positions for all
    particles are still returned.
    """
    positions = {
        0: (x0 + 1.1, y0 + nrows - 1.2),
        1: (x0 + 1.8, y0 + nrows - 1.66),
        2: (x0 + 2.6, y0 + nrows - 3.6),
        3: (x0 + 3.75, y0 + nrows - 2.2),
        4: (x0 + 3.2, y0 + nrows - 2.8),
    }
    for ii, (px, py) in positions.items():
        if subset is not None and ii not in subset:
            continue
        ax.plot(
            px, py, marker=MARKERS[ii], markersize=MARKER_SIZE, color="black", zorder=4
        )
    return positions


def add_grid(ax, x0, y0, nrows, ncols, positions, P, order=None):
    """
    Draws each particle's local footprint box, fills it with its team's hatch pattern,
    and shades overlapping footprint regions in red (different team) or green (same team).
    """
    h = 1.0 / P
    radius = P * h  # Support radius

    # 1. Map particle index -> team ID & hatch pattern using the HATCHES macro
    if order is not None:
        particle_teams = {p_id: i // 2 for i, p_id in enumerate(order)}
        particle_hatches = {
            p_id: HATCHES[(i // 2) % len(HATCHES)] for i, p_id in enumerate(order)
        }
    else:
        particle_teams = {}
        particle_hatches = {}

    # 2. Draw footprint boxes & hatch fills
    for p_id, (px, py) in positions.items():
        hatch_pattern = particle_hatches.get(p_id, None)
        if hatch_pattern:
            ax.fill_between(
                [px - radius, px + radius],
                py - radius,
                py + radius,
                facecolor="none",
                edgecolor="gray",
                hatch=hatch_pattern,
                linewidth=1,
                zorder=0,
            )

        # Local footprint bounding box outline
        bx = [px - radius, px + radius, px + radius, px - radius, px - radius]
        by = [py - radius, py - radius, py + radius, py + radius, py - radius]
        ax.plot(bx, by, color="black", zorder=2)

    # 3. Detect overlaps between particle footprints and shade accordingly
    p_ids = list(positions.keys())
    for i in range(len(p_ids)):
        for j in range(i + 1, len(p_ids)):
            p1_id, p2_id = p_ids[i], p_ids[j]
            px1, py1 = positions[p1_id]
            px2, py2 = positions[p2_id]

            # Calculate rectangular intersection bounds
            x_min = max(px1 - radius, px2 - radius)
            x_max = min(px1 + radius, px2 + radius)
            y_min = max(py1 - radius, py2 - radius)
            y_max = min(py1 + radius, py2 + radius)

            # Check if there is a spatial overlap
            if x_min < x_max and y_min < y_max:
                team1 = particle_teams.get(p1_id)
                team2 = particle_teams.get(p2_id)

                if team1 is not None and team2 is not None:
                    color = (
                        SAME_TEAM_OVERLAP_COLOR
                        if team1 == team2
                        else DIFF_TEAM_OVERLAP_COLOR
                    )

                    ax.fill_between(
                        [x_min, x_max],
                        y_min,
                        y_max,
                        facecolor=color,
                        alpha=OVERLAP_ALPHA,
                        edgecolor="none",
                        zorder=1,
                    )


def add_grid_nodes(ax, x0, y0, nrows, ncols, positions, P=4):
    """
    Draws active grid nodes for output-driven visualization.
    Only nodes within particle footprints are drawn.
    - Overlapping nodes (>1 footprint) are highlighted in BLUE.
    - Non-overlapping active nodes (1 footprint) are GRAY.
    - Active footprint boxes are shaded with a soft background fill.
    """
    h = 1.0 / P
    radius = P * h  # Support radius

    # Grid node coordinates across the domain
    grid_x = np.linspace(x0, x0 + ncols, ncols * P, endpoint=False)
    grid_y = np.linspace(y0, y0 + nrows, nrows * P, endpoint=False)

    if positions is not None:
        node_counts = {}
        for px, py in positions.values():
            for gx in grid_x:
                if abs(gx - px) <= radius + 1e-9:
                    for gy in grid_y:
                        if abs(gy - py) <= radius + 1e-9:
                            pt = (round(gx, 4), round(gy, 4))
                            node_counts[pt] = node_counts.get(pt, 0) + 1

        # Render only active nodes (count >= 1)
        for (nx, ny), count in node_counts.items():
            if count > 1:
                # Overlapping active node -> Blue
                ax.plot(nx, ny, marker=".", color="blue", markersize=3, zorder=3)
            else:
                # Non-overlapping active node -> Gray
                ax.plot(nx, ny, marker=".", color="black", markersize=3, zorder=3)
    else:
        # plot every node
        for nx in grid_x:
            for ny in grid_y:
                ax.plot(nx, ny, marker=".", color="black", markersize=3, zorder=3)


def set_team(ax, x0, y0, ncols, order, positions):
    """
    Plots team elements and highlights pairs of adjacent boxes
    using alternating hatch patterns configured via HATCHES macro.
    """
    for i in range(0, ncols, 2):
        x_start = x0 + i
        x_end = x0 + min(i + 2, ncols)

        hatch_pattern = HATCHES[(i // 2) % len(HATCHES)]

        ax.fill_between(
            [x_start, x_end],
            y0,
            y0 + 1,
            facecolor="none",
            edgecolor="gray",
            hatch=hatch_pattern,
            linewidth=0,
            zorder=0,
        )

    for i, ii in enumerate(order):
        target = (x0 + i + 0.5, y0 + 0.5)
        arrow = FancyArrowPatch(
            positions[ii],
            target,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=2,
            color=ARROW_COLOR,
            shrinkA=6,
            shrinkB=6,
            zorder=1,
        )
        ax.add_patch(arrow)
        ax.plot(
            *target, marker=MARKERS[ii], markersize=MARKER_SIZE, color="black", zorder=2
        )


def add_team_connections(ax, x0, y0_team_bottom, order, positions_l3, P=4):
    ncols = len(order)
    h = 1.0 / P
    radius = P * h

    for t in range(0, (ncols + 1) // 2):
        idx1 = 2 * t
        idx2 = 2 * t + 1

        p1_id = order[idx1] if idx1 < ncols else None
        p2_id = order[idx2] if idx2 < ncols else None

        x1_start = x0 + idx1 + 0.5
        y_start = y0_team_bottom

        if p2_id is not None:
            x2_start = x0 + idx2 + 0.5
            x_mid_start = (x1_start + x2_start) / 2.0
            y_merge = y_start - 0.35

            t1_pos = positions_l3[p1_id]
            t2_pos = positions_l3[p2_id]

            targets = [(t1_pos, p1_id), (t2_pos, p2_id)]
            targets.sort(key=lambda item: item[0][0])

            t_left_pos, _ = targets[0]
            t_right_pos, _ = targets[1]

            t_left_target = (t_left_pos[0], t_left_pos[1] + radius)
            t_right_target = (t_right_pos[0], t_right_pos[1] + radius)

            x_mid_end = (t_left_target[0] + t_right_target[0]) / 2.0
            y_split = max(t_left_target[1], t_right_target[1]) + 0.5

            path_m1 = Path(
                [(x1_start, y_start), (x1_start, y_merge), (x_mid_start, y_merge)],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3],
            )

            path_m2 = Path(
                [(x2_start, y_start), (x2_start, y_merge), (x_mid_start, y_merge)],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3],
            )

            path_trunk = Path(
                [
                    (x_mid_start, y_merge),
                    (x_mid_start, (y_merge + y_split) / 2),
                    (x_mid_end, (y_merge + y_split) / 2),
                    (x_mid_end, y_split),
                ],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
            )

            path_s1 = Path(
                [
                    (x_mid_end, y_split),
                    (x_mid_end, t_left_target[1] + 0.3),
                    (t_left_target[0], t_left_target[1] + 0.3),
                    t_left_target,
                ],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
            )

            path_s2 = Path(
                [
                    (x_mid_end, y_split),
                    (x_mid_end, t_right_target[1] + 0.3),
                    (t_right_target[0], t_right_target[1] + 0.3),
                    t_right_target,
                ],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
            )

            for p in [path_m1, path_m2, path_trunk]:
                patch = PathPatch(
                    p, facecolor="none", edgecolor="black", lw=1.8, zorder=3
                )
                ax.add_patch(patch)

            for p in [path_s1, path_s2]:
                arrow = FancyArrowPatch(
                    path=p,
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.8,
                    color="black",
                    shrinkA=0,
                    shrinkB=2,
                    zorder=3,
                )
                ax.add_patch(arrow)

        else:
            t1_pos = positions_l3[p1_id]
            t1_target = (t1_pos[0] + radius, t1_pos[1])

            path = Path(
                [
                    (x1_start, y_start),
                    (x1_start + 0.2, y_start - 3),
                    (t1_target[0] + 2, t1_target[1] - 0.1),
                    t1_target,
                ],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
            )

            arrow = FancyArrowPatch(
                path=path,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.8,
                color="black",
                shrinkA=0,
                shrinkB=2,
                zorder=3,
            )
            ax.add_patch(arrow)


def input_driven(order, cell_grid, grid_nodes):
    fig, ax = setup_axes()
    x0, y0 = (0, 0)
    ncols, nrows = (5, 5)

    # Level 1: League parallelism
    box(ax, x0, y0, ncols, nrows, grid=cell_grid)
    positions = add_particles(ax, x0, y0, ncols, nrows)

    # Level 2: Team parallelism
    box(ax, x0, y0 - 2, ncols, 1)
    set_team(ax, x0, y0 - 2, ncols, order, positions)

    # Level 3: Thread parallelism
    box(ax, x0, y0 - 3 - nrows, ncols, nrows, grid=grid_nodes)
    positions_l3 = add_particles(ax, x0, y0 - 3 - nrows, ncols, nrows)
    if grid_nodes:
        add_grid_nodes(ax, x0, y0 - 3 - nrows, nrows, ncols, positions_l3, P=4)
        wavy_arrow(
            ax, (x0 + ncols / 2.0, y0 - 2 - 0.1), (x0 + ncols / 2.0, y0 - 3 + 0.1)
        )
    else:
        add_grid(ax, x0, y0 - 3 - nrows, nrows, ncols, positions_l3, P=4, order=order)

        # Level 2 -> Level 3 team connections
        add_team_connections(ax, x0, y0 - 2, order, positions_l3, P=4)
    return fig, ax


def output_driven():
    fig, ax = setup_axes()
    x0, y0 = (0, 0)
    ncols, nrows = (5, 5)

    # Level 1: League parallelism
    box(ax, x0, y0, ncols, nrows, grid=True)
    add_grid_nodes(ax, x0, y0, nrows, ncols, None, P=4)

    # Level 2 / 3: Output driven discrete grid nodes
    y3 = y0 - 3 - nrows
    box(ax, x0, y3, ncols, nrows, grid=True)
    positions = add_particles(ax, x0, y3, ncols, nrows)
    add_grid_nodes(ax, x0, y3, nrows, ncols, positions, P=4)

    # Wavy arrow connecting the bottom of Level 1 to the top of Level 3
    wavy_arrow(ax, (x0 + ncols / 2.0, y0 - 0.1), (x0 + ncols / 2.0, y3 + nrows + 0.1))

    return fig, ax


def hybrid(order):
    fig, ax = setup_axes()
    x0, y0 = (0, 0)
    ncols, nrows = (5, 5)

    # Level 1: League parallelism
    box(ax, x0, y0, ncols, nrows, grid=True)
    positions = add_particles(ax, x0, y0, ncols, nrows)

    # Level 2: Team parallelism
    box(ax, x0, y0 - 2, ncols, 1)
    set_team(ax, x0, y0 - 2, ncols, order, positions)

    # Level 3: Thread parallelism
    box(ax, x0, y0 - 3 - nrows, ncols, nrows, grid=False)
    positions_l3 = add_particles(ax, x0, y0 - 3 - nrows, ncols, nrows)
    add_grid(ax, x0, y0 - 3 - nrows, nrows, ncols, positions_l3, P=4, order=order)

    # Level 2 -> Level 3 team connections
    add_team_connections(ax, x0, y0 - 2, order, positions_l3, P=4)
    return fig, ax


def main(args):
    # P2G-BASE
    fig_base, ax = input_driven([0, 4, 1, 3, 2], False, False)
    # P2G-SOURCE
    fig_source, ax = input_driven([0, 1, 4, 3, 2], True, False)
    # P2G-GRID
    fig_grid, ax = output_driven()
    # P2G-HYBRID
    fig_hybrid, ax = input_driven([0, 1, 4, 3, 2], True, True)

    if args.save:
        fig_base.savefig("p2g_base_viz.pdf", bbox_inches="tight")
        fig_source.savefig("p2g_source_viz.pdf", bbox_inches="tight")
        fig_grid.savefig("p2g_grid_viz.pdf", bbox_inches="tight")
        fig_hybrid.savefig("p2g_hybrid_viz.pdf", bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize P2G variants and optionally export to PGF."
    )
    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="Save output figures as .pgf files instead of showing them interactively.",
    )
    args = parser.parse_args()
    exit(main(args))
