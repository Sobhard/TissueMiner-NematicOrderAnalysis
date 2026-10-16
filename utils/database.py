"""Contains functions to manage database queries"""

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from utils.cellmath import create_dbond_vertex_cache
import time
from enum import StrEnum


class Tables(StrEnum):
    CELLS = "cells"
    VERTICES = "vertices"
    BONDS = "bonds"
    DBONDS = "directed_bonds"


class SeriesKeys(StrEnum):
    CELL_ID = "cell_id"
    CENTER_X = "center_x"
    CENTER_Y = "center_y"
    AREA = "area"
    ELONG_XX = "elong_xx"
    ELONG_XY = "elong_xy"
    VERTEX_ID = "vertex_id"
    X = "x_pos"
    Y = "y_pos"
    BOND_ID = "bond_id"
    BOND_LENGTH = "bond_length"
    DBOND_ID = "dbond_id"
    ASPECT_RATIO = "aspect_ratio"
    NEMATIC_ORDER = "nematic_order"
    HEXATIC_ORDER = "hexatic_order"
    OMEGA = "weighted_nematic_order"


def get_database_path():
    return Path("data/WT_25deg_111102.sqlite")


def get_frame_count(database_path: Path):
    print(database_path)

    if str(database_path) == "data/demo.sqlite":
        return 70

    conn = sqlite3.connect(database_path)

    query = """
    SELECT frame, MAX(time_sec) FROM timepoints;
    """

    max_frame = pd.read_sql_query(query, conn)

    return max(max_frame["frame"])


def get_frame(frame, database_path):
    """
    Load all data necessary to visualize one frame.

    if the we are using WT:
        ubond_id instead of bond_id
        dbond_left_id instead of left_dbond_id
        ubonds instead of bonds
        dbonds instead of directed_bonds

    """

    conn = sqlite3.connect(database_path)

    cells_query = """
        SELECT
            frame,
            cell_id,
            center_x,
            center_y,
            area,
            elong_xx,
            elong_xy
        FROM cells
        WHERE frame = ?
    """

    cells = pd.read_sql_query(cells_query, conn, params=(frame,))

    vertices_query = """
        SELECT
            frame,
            vertex_id,
            x_pos,
            y_pos
        FROM vertices
        WHERE frame = ?
    """

    vertices = pd.read_sql_query(vertices_query, conn, params=(frame,))

    bonds_query = """
        SELECT
            frame,
            ubond_id,
            bond_length
        FROM ubonds
        WHERE frame = ?
    """

    bonds = pd.read_sql_query(bonds_query, conn, params=(frame,))

    directed_bonds_query = """
        SELECT
            frame,
            cell_id,
            dbond_id,
            conj_dbond_id,
            ubond_id,
            vertex_id,
            dbond_left_id
        FROM dbonds
        WHERE frame = ?
    """

    directed_bonds = pd.read_sql_query(directed_bonds_query, conn, params=(frame,))

    conn.close()

    return {
        "cells": cells,
        "vertices": vertices,
        "bonds": bonds,
        "directed_bonds": directed_bonds,
    }


def get_timestamps(database_path):

    conn = sqlite3.connect(database_path)
    timestamp_query = """
    SELECT
        frame,
        time_sec
    FROM timepoints
    """

    timestamps = pd.read_sql_query(timestamp_query, conn)

    conn.close()

    return {"timestamps": timestamps}


def create_database_cache(database_path: Path):
    """
    Create a global cache for the entire database.

    Each frame gets its own cellmath cache containing:
        - dbond_id -> dbond
        - vertex_id -> vertex
        - cell_id -> starting dbond
        - vertex_id -> cells touching that vertex
        - vertex_id -> dbonds touching that vertex

    The resulting cache can be reused throughout the program so that
    cellmath functions do not repeatedly perform pandas lookups.
    """

    start_time = time.perf_counter()

    conn = sqlite3.connect(database_path)

    # --------------------------------------------------------
    # Load all vertices
    # --------------------------------------------------------

    vertices_query = """
        SELECT
            frame,
            vertex_id,
            x_pos,
            y_pos
        FROM vertices
    """

    vertices = pd.read_sql_query(vertices_query, conn)

    # --------------------------------------------------------
    # Load all directed bonds
    # --------------------------------------------------------

    dbonds_query = """
        SELECT
            frame,
            cell_id,
            dbond_id,
            conj_dbond_id,
            ubond_id,
            vertex_id,
            dbond_left_id
        FROM dbonds
    """

    dbonds = pd.read_sql_query(dbonds_query, conn)

    conn.close()

    # --------------------------------------------------------
    # Create a cellmath cache for each frame
    # --------------------------------------------------------

    database_cache = {}

    for frame in sorted(dbonds["frame"].unique()):
        print(f"Creating cache for frame {frame}/278...", end="\r")

        frame_dbonds = dbonds[dbonds["frame"] == frame]

        frame_vertices = vertices[vertices["frame"] == frame]

        database_cache[frame] = create_dbond_vertex_cache(
            frame_dbonds,
            frame_vertices,
        )

    end_time = time.perf_counter()

    print(f"Created databse cache in f{(end_time - start_time):.6f}s")

    return database_cache
