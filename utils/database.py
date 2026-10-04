"""Contains functions to manage database queries"""

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np


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
    FROM frames
    """

    timestamps = pd.read_sql_query(timestamp_query, conn)

    conn.close()

    return {"timestamps": timestamps}
