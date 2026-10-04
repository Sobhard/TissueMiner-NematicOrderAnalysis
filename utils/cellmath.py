"""Does mathematical operations and miscellaneous utils on cells"""

import pandas as pd
import numpy as np
import time


def create_dbond_vertex_cache(dbonds_data, vertices_data):
    """
    Create fast lookup dictionaries for dbonds and vertices.

    The cache contains:
        dbond_by_id:
            dbond_id -> dbond record

        vertex_by_id:
            vertex_id -> vertex record

        start_dbond_by_cell:
            cell_id -> first dbond_id belonging to that cell

        cells_by_vertex:
            vertex_id -> list of cell_ids touching that vertex

        dbonds_by_vertex:
            vertex_id -> list of dbond_ids touching that vertex

    The cache should be created once per dataset/frame and reused.
    """

    cache = {
        # Direct ID -> row lookups
        "dbond_by_id": (dbonds_data.set_index("dbond_id").to_dict("index")),
        "vertex_by_id": (vertices_data.set_index("vertex_id").to_dict("index")),
        # One starting dbond for each cell
        "start_dbond_by_cell": (
            dbonds_data.groupby("cell_id")["dbond_id"].first().to_dict()
        ),
        # All cells touching each vertex
        "cells_by_vertex": (
            dbonds_data.groupby("vertex_id")["cell_id"].apply(list).to_dict()
        ),
        # All dbonds touching each vertex
        "dbonds_by_vertex": (
            dbonds_data.groupby("vertex_id")["dbond_id"].apply(list).to_dict()
        ),
    }

    return cache


def get_cell_vertices(cell_id, cache):

    # start_time = time.perf_counter()

    dbond_by_id = cache["dbond_by_id"]
    vertex_by_id = cache["vertex_by_id"]
    start_dbond_by_cell = cache["start_dbond_by_cell"]

    curr_id = start_dbond_by_cell[cell_id]

    vertices = []
    seen = set()

    while curr_id not in seen:
        seen.add(curr_id)

        dbond = dbond_by_id[curr_id]
        vertex = vertex_by_id[dbond["vertex_id"]]

        vertices.append(vertex)

        curr_id = dbond["dbond_left_id"]

    # end_time = time.perf_counter()

    # print(f"Getting cell vertices took {(end_time - start_time):.6f}s", end="\r")

    return vertices


def get_neighbors_cell(cell_id, cache, dbonds_data):
    verticies = get_cell_vertices(cell_id, cache)
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


def find_center(vertices):
    # Add formula for this function (make documentation)
    area = 0
    x_center = 0
    y_center = 0

    for i in range(len(vertices)):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % len(vertices)]

        x1 = v1["x_pos"]
        y1 = v1["y_pos"]
        x2 = v2["x_pos"]
        y2 = v2["y_pos"]

        cross = x1 * y2 - x2 * y1

        area += cross
        x_center += (x1 + x2) * cross
        y_center += (y1 + y2) * cross

    if area == 0:
        return None

    area /= 2

    x_center /= 6 * area
    y_center /= 6 * area

    return (x_center, y_center)


# def calc_order(vertices, n):
#     real_sum = 0.0
#     imaginary_sum = 0.0
#     center = find_center(vertices)

#     for i in range(len(vertices)):
#         x1 = vertices[i]["x_pos"] - center[0]
#         y1 = vertices[i]["y_pos"] - center[1]

#         theta = np.arctan2(x1, y1)

#         real_sum += np.cos(n * theta)
#         imaginary_sum += np.sin(n * theta)

#     real_sum *= 1.0 / float(len(vertices))

#     return real_sum, imaginary_sum


def calc_order(vertices, n, use_reference=False):
    """DO NOT USE THIS FOR NEMATIC ORDER!!!!"""
    real_sum = 0.0
    imaginary_sum = 0.0
    center = find_center(vertices)
    reference_angle = 0

    if not center:
        return 0, 0

    # Center the vertices
    points = np.array(
        [[v["x_pos"] - center[0], v["y_pos"] - center[1]] for v in vertices],
        dtype=np.float32,
    )

    if use_reference:
        # Covariance matrix
        covariance = np.cov(points, rowvar=False)

        # Find principal axes
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Eigenvector corresponding to largest eigenvalue
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

        # Angle of the principal axis relative to x-axis
        reference_angle = np.arctan2(principal_axis[1], principal_axis[0])

    # Calculate order relative to principal axis
    for x, y in points:
        theta = np.arctan2(y, x) - reference_angle

        real_sum += np.cos(n * theta)
        imaginary_sum += np.sin(n * theta)

    real_sum /= len(vertices)
    imaginary_sum /= len(vertices)

    return real_sum, imaginary_sum


def calc_nematic(vertices):
    cos_part = 0.0
    sin_part = 0.0
    center = find_center(vertices)

    if not center:
        return 0, 0

    points = np.array(
        [[v["x_pos"] - center[0], v["y_pos"] - center[1]] for v in vertices],
        dtype=np.float32,
    )

    covariance = np.cov(points, rowvar=False)
    # Find principal axes
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # Eigenvector corresponding to largest eigenvalue
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

    phi = np.arctan2(principal_axis[1], principal_axis[0])

    return np.cos(2 * phi), np.sin(2 * phi)


def calc_aspect_ratio(vertices):

    center = find_center(vertices)
    if not center:
        return 1

    # Center the vertices
    points = np.array(
        [[v["x_pos"] - center[0], v["y_pos"] - center[1]] for v in vertices],
        dtype=np.float32,
    )

    # Covariance matrix
    covariance = np.cov(points, rowvar=False)

    # Find principal axes
    eigenvalues, _ = np.linalg.eigh(covariance)

    a = max(eigenvalues)
    b = min(eigenvalues)

    return np.sqrt(a / b)


# def ellipse_aspect_ratio(vertices):
#     points = np.array([[v["x_pos"], v["y_pos"]] for v in vertices], dtype=np.float32)
#     print("Points: ")
#     print(points)

#     if len(points) < 5:
#         return 0

#     ellipse = EllipseModel.from_estimate(points)

#     if not ellipse:
#         print("ellipse approx failed")
#         return 0

#     xc, yc, a, b, theta = ellipse.params

#     # Make sure a is the major axis
#     major = max(a, b)
#     minor = min(a, b)

#     return major / minor


def order_dict(cell_data, cache, n, use_reference=True):
    orders = {}

    for cell_id in cell_data["cell_id"].unique():
        vertices = get_cell_vertices(cell_id, cache)
        if len(vertices) < 3:
            # print("ERROR NO VERTICES IN CELL", cell_id)
            orders[cell_id] = 1
            continue
        order = calc_order(vertices, n, use_reference)

        orders[cell_id] = order[0]  # real part

    return orders


def nematic_order_dict(cell_data, cache, use_reference=True):
    nematic_orders = {}

    for cell_id in cell_data["cell_id"].unique():
        vertices = get_cell_vertices(cell_id, cache)
        if len(vertices) < 3:
            # print("ERROR NO VERTICES IN CELL", cell_id)
            nematic_orders[cell_id] = 0
            continue

        n_order = calc_nematic(vertices)

        nematic_orders[cell_id] = n_order[0]  # real part

    return nematic_orders


# def aspect_ratio_dict(cell_data, dbonds_data, vertices_data):
#     ellipse_ratios = {}

#     for cell_id in cell_data["cell_id"].unique():
#         vertices = get_cell_verticies(cell_id, dbonds_data, vertices_data)
#         aspect_ratio = ellipse_aspect_ratio(vertices)

#         ellipse_ratios[cell_id] = aspect_ratio

#     return ellipse_ratios


def aspect_ratio_dict(cell_data, cache):

    ellipse_ratios = {}

    for cell_id in cell_data["cell_id"].unique():

        vertices = get_cell_vertices(cell_id, cache)
        if len(vertices) < 3:
            # print("ERROR NO VERTICES IN CELL", cell_id)
            ellipse_ratios[cell_id] = 1
            continue
        aspect_ratio = calc_aspect_ratio(vertices)

        ellipse_ratios[cell_id] = aspect_ratio

    return ellipse_ratios


def calc_local_hexatic_order(cell_id, cache, dbonds_data):
    # Get this cell's vertices and center
    vertices = get_cell_vertices(cell_id, cache)
    center = find_center(vertices)

    if not center:
        return 0

    # Get neighboring cells
    neighbors = get_neighbors_cell(cell_id, cache, dbonds_data)

    # Remove this cell itself and duplicate neighbors
    neighbors = set(neighbors)
    neighbors.discard(cell_id)

    real_sum = 0.0
    imaginary_sum = 0.0
    valid_neighbors = 0

    for neighbor_id in neighbors:
        # Get neighbor's vertices and center
        neighbor_vertices = get_cell_vertices(neighbor_id, cache)
        neighbor_center = find_center(neighbor_vertices)

        if not neighbor_center:
            continue

        # Vector from this cell to its neighbor
        dx = neighbor_center[0] - center[0]
        dy = neighbor_center[1] - center[1]

        # Ignore zero-length vectors
        if dx == 0 and dy == 0:
            continue

        # Angle of the neighbor relative to this cell
        theta = np.arctan2(dy, dx)

        # Add e^(i * 6 * theta)
        real_sum += np.cos(6 * theta)
        imaginary_sum += np.sin(6 * theta)

        valid_neighbors += 1

    if valid_neighbors == 0:
        return np.nan

    # Calculate complex local hexatic order parameter
    real_sum /= valid_neighbors
    imaginary_sum /= valid_neighbors

    # Return magnitude
    return np.sqrt(real_sum**2 + imaginary_sum**2)


def weighted_order_dict(nematic_order_dict, aspect_ratio_dict):
    avg_aspect_ratio = np.mean(list(aspect_ratio_dict.values()))
    weighted_order_dict = dict()

    for cell_id, n_order in nematic_order_dict.items():
        weighted_order_dict[cell_id] = (
            aspect_ratio_dict[cell_id] * n_order / avg_aspect_ratio
        )

    return weighted_order_dict


def local_hexatic_order_dict(cell_data, cache, dbonds_data):
    hexatic_orders = {}

    for cell_id in cell_data["cell_id"].unique():
        hexatic_order = calc_local_hexatic_order(cell_id, cache, dbonds_data)

        hexatic_orders[cell_id] = hexatic_order

    return hexatic_orders
