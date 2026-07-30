"""
Visualize P2G variants on a three-level parallel machine
(league -> team -> thread).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

MARKERS = ["*", "p", "P", "o", "D"]
MARKER_SIZE = 8
ARROW_COLOR = "black"  # "#3f8f4f"


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
            ax.plot([x, x], [y0, y0 + nrows], color="black")
        for y in ys:
            ax.plot([x0, x0 + ncols], [y, y], color="black")
    else:
        x = [x0, x0 + ncols, x0 + nrows, x0, x0]
        y = [y0, y0, y0 + nrows, y0 + nrows, y0]
        ax.plot(x, y, color="black")


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
    # cell 1
    positions = {
        0: (x0 + 1.1, y0 + nrows - 1.5),
        1: (x0 + 1.8, y0 + nrows - 1.66),
        # cell 2
        2: (x0 + 2.6, y0 + nrows - 3.5),
        # cell 3
        3: (x0 + 3.75, y0 + nrows - 2.2),
        4: (x0 + 3.5, y0 + nrows - 2.8),
    }
    for ii, (px, py) in positions.items():
        if subset is not None and ii not in subset:
            continue
        ax.plot(px, py, marker=MARKERS[ii], markersize=MARKER_SIZE, color="black")
    return positions


def add_grid(ax, x0, y0, nrows, ncols, positions, P, order=None):
    """
    Subdivide each unit cell of the (ncols x nrows) box rooted at (x0, y0)
    into a PxP grid of nodes (spacing h = 1/P). Plot nodes in green within radius.

    Fills each particle's local footprint with the hatch pattern corresponding
    to its team assignment in `order`.
    """
    h = 1.0 / P
    radius = P * h / 2  # support radius

    # 1. Map particle index -> team hatch pattern matching set_team
    hatches = ["//", "..", "xx"]
    if order is not None:
        particle_hatches = {
            p_id: hatches[(i // 2) % len(hatches)] for i, p_id in enumerate(order)
        }
    else:
        particle_hatches = {}

    # 2. Fill local grid area (square of radius around particle position)
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


def set_team(ax, x0, y0, ncols, order, positions):
    """
    Plots team elements and highlights pairs of adjacent boxes
    using alternating hatch patterns instead of background colors.
    """
    # Define alternating hatch patterns (e.g., forward slash and backslash)
    hatches = ["//", "..", "xx"]

    # 1. Fill pairs of 2 boxes with hatches
    for i in range(0, ncols, 2):
        x_start = x0 + i
        x_end = x0 + min(i + 2, ncols)  # Keeps within ncols bound if odd

        # Pick hatch style for the current 2-box block
        hatch_pattern = hatches[(i // 2) % len(hatches)]

        ax.fill_between(
            [x_start, x_end],
            y0,
            y0 + 1,
            facecolor="none",  # Removes background fill color
            edgecolor="gray",  # Line color for the hatch pattern
            hatch=hatch_pattern,
            linewidth=0,  # Prevents drawing extra border edges over the grid
            zorder=0,  # Keeps hatch pattern behind arrows and markers
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


def main():
    # P2G-BASE
    fig, ax = setup_axes()
    x0, y0 = (0, 0)
    ncols, nrows = (5, 5)
    order = [0, 4, 1, 3, 2]

    # level 1: league parallelism
    box(ax, x0, y0, ncols, nrows, grid=True)
    positions = add_particles(ax, x0, y0, ncols, nrows)

    # level 2: team parallelism
    box(ax, x0, y0 - 2, ncols, 1)
    set_team(ax, x0, y0 - 2, ncols, order, positions)

    # level 3: thread parallelism
    box(ax, x0, y0 - 3 - nrows, ncols, nrows, grid=False)
    # Include all particles (no subset)
    positions_l3 = add_particles(ax, x0, y0 - 3 - nrows, ncols, nrows)
    add_grid(ax, x0, y0 - 3 - nrows, nrows, ncols, positions_l3, P=4, order=order)
    plt.show()
    # P2G-SOURCE
    # P2G-GRID
    # P2G-HYBRID


if __name__ == "__main__":
    exit(main())
