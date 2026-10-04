"""This code will evaluate metrics over all the frames of the database"""

import utils.database as db
from utils import cellmath, cellplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from enum import StrEnum
from pathlib import Path


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


# Agregate frame data into a .csv file for easy analytics
index_path = Path(f"indices/{str(db.get_database_path())[5:-7]}.pkl")
index_path.parent.mkdir(parents=True, exist_ok=True)

if not index_path.exists():

    tmp_index = dict()
    frame_count = db.get_frame_count(db.get_database_path())

    for i in range(1, frame_count + 1):
        frame_data = db.get_frame(i, db.get_database_path())

        # Creating series in the CELLS dataframe for nematic roder, hexatic order, shoelace_area and aspect ratio
        nematic_orders = cellmath.order_dict(
            frame_data[Tables.CELLS],
            frame_data[Tables.DBONDS],
            frame_data[Tables.VERTICES],
            2,
        )

        hexatic_orders = cellmath.order_dict(
            frame_data[Tables.CELLS],
            frame_data[Tables.DBONDS],
            frame_data[Tables.VERTICES],
            6,
        )

        aspect_ratios = cellmath.aspect_ratio_dict(
            frame_data[Tables.CELLS],
            frame_data[Tables.DBONDS],
            frame_data[Tables.VERTICES],
        )

        frame_data[Tables.CELLS][SeriesKeys.ASPECT_RATIO] = frame_data[Tables.CELLS][
            SeriesKeys.CELL_ID
        ].map(aspect_ratios)

        frame_data[Tables.CELLS][SeriesKeys.NEMATIC_ORDER] = frame_data[Tables.CELLS][
            SeriesKeys.CELL_ID
        ].map(nematic_orders)

        frame_data[Tables.CELLS][SeriesKeys.HEXATIC_ORDER] = frame_data[Tables.CELLS][
            SeriesKeys.CELL_ID
        ].map(hexatic_orders)

        tmp_index[i] = frame_data

    with open(index_path, "wb") as f:
        pickle.dump(tmp_index, f)

index: dict[int, pd.DataFrame] = pd.read_pickle(index_path)
timestamps: dict[str, pd.DataFrame] = db.get_timestamps(db.get_database_path())

# CREATING THE ORDER VS TIME PLOTS
