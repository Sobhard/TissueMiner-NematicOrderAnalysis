"""This code will evaluate metrics over all the frames of the database"""

import utils.database as db
from utils.database import Tables, SeriesKeys
from utils import cellmath
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import time

# Agregate frame data into a .pkl file for easy analytics
index_path = Path(f"indices/{str(db.get_database_path())[5:-7]}.pkl")
index_path.parent.mkdir(parents=True, exist_ok=True)

if not index_path.exists():

    tmp_index = dict()
    frame_count = db.get_frame_count(db.get_database_path())

    for i in range(1, frame_count + 1):
        start_time = time.perf_counter()

        frame_data = db.get_frame(i, db.get_database_path())
        frame_cache = cellmath.create_dbond_vertex_cache(
            frame_data[Tables.DBONDS], frame_data[Tables.VERTICES]
        )

        # Creating series in the CELLS dataframe for nematic roder, hexatic order, shoelace_area and aspect ratio
        nematic_orders = cellmath.nematic_order_dict(
            frame_data[Tables.CELLS], frame_cache, False
        )

        hexatic_orders = cellmath.order_dict(
            frame_data[Tables.CELLS],
            frame_cache,
            6,
            use_reference=False,
        )

        aspect_ratios = cellmath.aspect_ratio_dict(
            frame_data[Tables.CELLS], frame_cache
        )

        omegas = cellmath.weighted_order_dict(nematic_orders, aspect_ratios)

        frame_data[Tables.CELLS][SeriesKeys.ASPECT_RATIO] = frame_data[Tables.CELLS][
            SeriesKeys.CELL_ID
        ].map(aspect_ratios)

        frame_data[Tables.CELLS][SeriesKeys.NEMATIC_ORDER] = frame_data[Tables.CELLS][
            SeriesKeys.CELL_ID
        ].map(nematic_orders)

        frame_data[Tables.CELLS][SeriesKeys.HEXATIC_ORDER] = frame_data[Tables.CELLS][
            SeriesKeys.CELL_ID
        ].map(hexatic_orders)

        frame_data[Tables.CELLS][SeriesKeys.OMEGA] = frame_data[Tables.CELLS][
            SeriesKeys.CELL_ID
        ].map(omegas)

        tmp_index[i] = frame_data

        end_time = time.perf_counter()
        print(
            f"done with frame {i}/{frame_count} in {(end_time - start_time):.6f}s",
            end="\r",
        )

    with open(index_path, "wb") as f:
        pickle.dump(tmp_index, f)


index: dict[int, pd.DataFrame] = pd.read_pickle(index_path)
timestamps: pd.Series = db.get_timestamps(db.get_database_path())["timestamps"][
    "time_sec"
][1:]

# CREATING THE ORDER VS TIME PLOTS
average_nematic_orders = []

for frame_data in index.values():
    mean_nematic = np.mean(frame_data[Tables.CELLS][SeriesKeys.NEMATIC_ORDER][1:])
    average_nematic_orders.append(mean_nematic)


average_aspect_ratios = []

for frame_data in index.values():

    # Clean the aspect ratio data
    aspect_ratios = np.asarray(
        frame_data[Tables.CELLS][SeriesKeys.ASPECT_RATIO], dtype=float
    )

    # Remove NaN values
    aspect_ratios = aspect_ratios[~np.isnan(aspect_ratios)]

    if len(aspect_ratios) == 0:
        average_aspect_ratios.append(np.nan)
        continue

    q1 = np.percentile(aspect_ratios, 25)
    q3 = np.percentile(aspect_ratios, 75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    eliminated_vals = aspect_ratios[(aspect_ratios < lower) | (aspect_ratios > upper)]

    aspect_ratios = aspect_ratios[(aspect_ratios >= lower) & (aspect_ratios <= upper)]

    if len(eliminated_vals) > 0:
        print(f"Eliminated aspect ratio values: {eliminated_vals}")

    mean_aspect = np.mean(aspect_ratios)
    average_aspect_ratios.append(mean_aspect)

average_hexatic_orders = []

for frame_data in index.values():
    mean_hexatic = np.mean(frame_data[Tables.CELLS][SeriesKeys.HEXATIC_ORDER][1:])
    average_hexatic_orders.append(mean_hexatic)

average_omegas = []

for frame_data in index.values():

    # Clean the omega data
    omegas = np.asarray(frame_data[Tables.CELLS][SeriesKeys.OMEGA], dtype=float)

    # Remove NaN values
    omegas = omegas[~np.isnan(omegas)]

    # if len(omegas) == 0:
    #     average_omegas.append(average_omegas[-1] if average_omegas else np.nan)
    #     continue

    # q1 = np.percentile(omegas, 25)
    # q3 = np.percentile(omegas, 75)
    # iqr = q3 - q1

    # lower = q1 - 1.5 * iqr
    # upper = q3 + 1.5 * iqr

    # eliminated_vals = omegas[(omegas < lower) | (omegas > upper)]

    # omegas = omegas[(omegas >= lower) & (omegas <= upper)]

    # if len(eliminated_vals) > 0:
    #     print(f"Eliminated omega values: {eliminated_vals}")

    mean_omega = np.mean(omegas)
    if average_omegas:
        if abs(average_omegas[-1] - mean_omega) > 0.1:
            average_omegas.append(average_omegas[-1])
            continue

    average_omegas.append(mean_omega)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

frame_counts = range(len(index.values()))

# --- Nematic Order ---
ax1.plot(timestamps, average_nematic_orders)
ax1.set_title("Average Nematic Order vs. Time")
ax1.set_ylabel("Nematic Order")
ax1.grid(True)

# --- Aspect Ratio ---
ax2.plot(timestamps, average_aspect_ratios)
ax2.set_title("Average Aspect Ratio vs. Time")
ax2.set_ylabel("Aspect Ratio")
ax2.grid(True)

# --- Hexatic Order ---
ax3.plot(timestamps, average_hexatic_orders)
ax3.set_title("Average Hexatic Order vs. Time")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Hexatic Order")
ax3.grid(True)

# ---  OMEGAS ---
ax4.plot(timestamps, average_omegas)
ax4.set_title("Average Omega vs. Time")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Weighted Nematic Order")
ax4.grid(True)


plt.tight_layout()
plt.show()
