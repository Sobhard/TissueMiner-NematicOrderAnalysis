"""Generates visualizations of cells from the cell database"""

from matplotlib.cm import ScalarMappable

import utils.database as db
import utils.cellmath as cellmath
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

UNASSIGNED_CELL_ID = 10000


def draw_cell_boundaries(ax, frame: int, cell_id_list, hue_vals={}, border_width=0.8):
    cell_data = db.get_frame(frame, db.get_database_path())
    all_dbonds = cell_data["directed_bonds"]
    all_vertices = cell_data["vertices"]

    if hue_vals:
        omega_max = calcMax(hue_vals)
    else:
        omega_max = 1

    for cell_id in cell_id_list:

        if cell_id == UNASSIGNED_CELL_ID:
            continue

        cell_verticies = cellmath.get_cell_verticies(cell_id, all_dbonds, all_vertices)
        cell_points = [(v["x_pos"], v["y_pos"]) for v in cell_verticies]

        if len(cell_points) < 3:
            print(f"Cell with id: {cell_id} has less than 3 vertices")
            continue

        omega = hue_vals.get(cell_id, np.nan)

        if np.isnan(omega):
            facecolor = "lightgray"
        else:
            facecolor = getColor(omega, omega_max)

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


def calcMax(vals: dict):
    return np.nanmax(np.abs(np.array(list(vals.values()), dtype=np.float32)))


def getColor(val, max_val, cmap_color="coolwarm"):
    norm = Normalize(-max_val, max_val)
    cmap = plt.get_cmap(cmap_color)

    return cmap(norm(val))


def plotColorBar(
    ax, fig, hue_vals: dict, normalize: bool, label, cmap_color="coolwarm"
):
    max_val = calcMax(hue_vals)
    cmap = plt.get_cmap(cmap_color)

    if normalize:
        norm = Normalize(-max_val, max_val)
        scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    else:
        scalar_mappable = ScalarMappable(cmap=cmap)

    scalar_mappable.set_array(list(hue_vals.values()))
    colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(label)


if __name__ == "__main__":
    data = db.get_frame(1, db.get_database_path())

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    cellsListAx1 = cellmath.get_neighbors_vertex(2050, data["directed_bonds"])
    ax1.set_title("Cells at Vertex 2050")
    draw_cell_boundaries(ax1, 1, cellsListAx1)

    ax2.set_title("Cells bordering cell 10203")
    cellsListAx2 = cellmath.get_neighbors_cell(
        10203, data["directed_bonds"], data["vertices"]
    )
    draw_cell_boundaries(ax2, 1, cellsListAx2)

    plt.show()
