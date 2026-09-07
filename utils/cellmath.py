"""Does mathematical operations and miscellaneous utils on cells"""

import pandas as pd
import numpy as np


def get_cell_verticies(cell_id, dbonds_data, vertex_data):
    cell_dbonds_data = dbonds_data[dbonds_data["cell_id"] == cell_id]
    verticies = []
    seen = set()
    curr_id = cell_dbonds_data.iloc[0]["dbond_id"]

    while curr_id not in seen:
        seen.add(curr_id)
        dbond = cell_dbonds_data[cell_dbonds_data["dbond_id"] == curr_id].iloc[0]
        vertex = vertex_data[vertex_data["vertex_id"] == dbond["vertex_id"]].iloc[0]

        verticies.append(vertex)

        curr_id = dbond["left_dbond_id"]

    return verticies


def get_neighbors_cell(cell_id, dbonds_data, vertex_data):
    verticies = get_cell_verticies(cell_id, dbonds_data, vertex_data)
    vertex_ids = pd.Series([v["vertex_id"] for v in verticies])

    neighbor_bonds = dbonds_data[dbonds_data["vertex_id"].isin(vertex_ids)]
    return neighbor_bonds["cell_id"].tolist()


def get_neighbors_vertex(vertex_id, dbonds_data):
    neighbor_bonds = dbonds_data[dbonds_data["vertex_id"] == vertex_id]
    return neighbor_bonds["cell_id"].tolist()


def shoelace_area(vertices):
    vertices = vertices[::-1]

    xList = [v["x_pos"] for v in vertices]
    yList = [v["y_pos"] for v in vertices]

    area = 0

    for i in range(len(vertices)):
        j = (i + 1) % len(vertices)

        area += xList[i] * yList[j]
        area -= yList[i] * xList[j]

    return abs(area) / 2


def nematic_order(vertices):
    real_sum = 0
    imaginary_sum = 0

    for i in range(len(vertices)):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % len(vertices)]

        dx = v2["x_pos"] - v1["x_pos"]
        dy = v2["y_pos"] - v1["y_pos"]

        theta = np.arctan2(dy, dx)

        real_sum += np.cos(2 * theta)
        imaginary_sum += np.sin(2 * theta)

    return real_sum, imaginary_sum
