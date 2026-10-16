"""Generates visualizations of cells from the cell database"""

import utils.database as db
import utils.cellmath as cellmath
from utils.color_formatter import color_formatter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

UNASSIGNED_CELL_ID = 10000


def draw_cell_boundaries(
    ax,
    fig,
    color_bar_label,
    frame: int,
    cell_id_list,
    hue_vals={},
    omega_max=1,
    omega_min=0,
    border_width=0.8,
    do_normalization=True,
):
    cell_data = db.get_frame(frame, db.get_database_path())
    all_dbonds = cell_data["directed_bonds"]
    all_vertices = cell_data["vertices"]
    cache = cellmath.create_dbond_vertex_cache(all_dbonds, all_vertices)

    colors = color_formatter(omega_max, omega_min)

    for cell_id in cell_id_list:

        if cell_id == UNASSIGNED_CELL_ID:
            continue

        cell_verticies = cellmath.get_cell_vertices(cell_id, cache)
        cell_points = [(v["x_pos"], v["y_pos"]) for v in cell_verticies]

        if len(cell_points) < 3:
            print(f"Cell with id: {cell_id} has less than 3 vertices")
            continue

        omega = hue_vals.get(cell_id, np.nan)

        if np.isnan(omega):
            facecolor = "lightgray"
        else:
            facecolor = colors.getColor(omega)

        polygon = Polygon(
            cell_points,
            closed=True,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=border_width,
        )

        ax.add_patch(polygon)
        ax.autoscale()
        ax.set_aspect("equal")

    colors.plotColorbar(ax, fig, color_bar_label)


if __name__ == "__main__":
    data = db.get_frame(1, db.get_database_path())

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    cellsListAx1 = cellmath.get_neighbors_vertex(2050, data["directed_bonds"])
    ax1.set_title("Cells at Vertex 2050")
    draw_cell_boundaries(ax1, fig, "No colors needed", 1, cellsListAx1)

    ax2.set_title("Cells bordering cell 10203")
    cellsListAx2 = cellmath.get_neighbors_cell(
        10203, data["directed_bonds"], data["vertices"]
    )
    draw_cell_boundaries(ax2, fig, "No colors needed", 1, cellsListAx2)

    plt.show()
