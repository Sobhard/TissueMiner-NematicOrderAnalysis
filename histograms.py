"""Generate per-frame histograms of cell metrics from the cached frame index.

Requires the index .pkl produced by the frame-metrics script. Histograms are
computed every HISTOGRAM_INTERVAL frames and saved to ./histograms/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # saving only, no windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils.database as db
from utils.database import Tables, SeriesKeys

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
HISTOGRAM_INTERVAL = 1  # make a histogram every N frames
N_BINS = 30
OUTPUT_DIR = Path("histograms")

# Each metric: (series key, title, x-axis label, skip first row, remove IQR outliers)
# "skip first row" mirrors the [1:] slicing used for nematic order in the main script.
METRICS = [
    (SeriesKeys.NEMATIC_ORDER, "Nematic Order", "Nematic Order", True, False),
    (SeriesKeys.OMEGA, "Omega", "Weighted Nematic Order", False, True),
    (SeriesKeys.ASPECT_RATIO, "Aspect Ratio", "Aspect Ratio", False, True),
]

# Area: picked up automatically once SeriesKeys.AREA exists and the column is in the cache.
AREA_KEY = getattr(SeriesKeys, "AREA", None)
if AREA_KEY is not None:
    METRICS.append((AREA_KEY, "Area", "Area", False, True))


# ---------------------------------------------------------------------------
# Load index
# ---------------------------------------------------------------------------
index_path = Path(f"indices/{str(db.get_database_path())[5:-7]}.pkl")

if not index_path.exists():
    raise FileNotFoundError(
        f"Index file not found at '{index_path}'. The index must exist before "
        "generating histograms; run the frame-metrics script first to create it."
    )

index: dict[int, dict] = pd.read_pickle(index_path)

timestamps = list(
    db.get_timestamps(db.get_database_path())["timestamps"]["time_sec"][1:]
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_values(cells: pd.DataFrame, key, skip_first: bool, remove_outliers: bool):
    """Return a clean float array for one metric, or None if the column is missing."""
    if key not in cells.columns:
        return None

    series = cells[key]
    if skip_first:
        series = series.iloc[1:]

    values = np.asarray(series, dtype=float)
    values = values[~np.isnan(values)]

    if remove_outliers and len(values) > 0:
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        values = values[(values >= lower) & (values <= upper)]

    return values


# Only the frames we actually histogram
frame_ids = sorted(index.keys())
sampled = [(pos, fid) for pos, fid in enumerate(frame_ids)][::HISTOGRAM_INTERVAL]

# Only keep metrics that are actually present in the cache
first_cells = index[frame_ids[0]][Tables.CELLS]
active_metrics = [m for m in METRICS if m[0] in first_cells.columns]
skipped = [m[1] for m in METRICS if m[0] not in first_cells.columns]
if skipped:
    print(f"Skipping metrics not found in the index: {', '.join(skipped)}")
if not active_metrics:
    raise ValueError("None of the requested metrics are present in the index.")

# Compute values once and find global x-ranges so axes match across frames
frame_values: dict[int, list] = {}
global_ranges = {m[0]: [np.inf, -np.inf] for m in active_metrics}

for pos, fid in sampled:
    cells = index[fid][Tables.CELLS]
    values_per_metric = []
    for key, _, _, skip_first, remove_outliers in active_metrics:
        vals = clean_values(cells, key, skip_first, remove_outliers)
        values_per_metric.append(vals)
        if vals is not None and len(vals) > 0:
            global_ranges[key][0] = min(global_ranges[key][0], vals.min())
            global_ranges[key][1] = max(global_ranges[key][1], vals.max())
    frame_values[fid] = values_per_metric


# ---------------------------------------------------------------------------
# Plot and save
# ---------------------------------------------------------------------------
n = len(active_metrics)

for pos, fid in sampled:
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    axes = np.atleast_1d(axes)

    for ax, (key, title, xlabel, _, _), vals in zip(
        axes, active_metrics, frame_values[fid]
    ):
        lo, hi = global_ranges[key]
        if vals is None or len(vals) == 0 or not np.isfinite(lo):
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
        else:
            bins = np.linspace(lo, hi, N_BINS + 1) if hi > lo else N_BINS
            ax.hist(vals, bins=bins, edgecolor="black", alpha=0.8)
            ax.axvline(
                np.mean(vals),
                color="red",
                linestyle="--",
                label=f"mean = {np.mean(vals):.3f}",
            )
            ax.legend(fontsize=8)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cell count")
        ax.grid(True, alpha=0.3)

    time_str = f" (t = {timestamps[pos]:.1f} s)" if pos < len(timestamps) else ""
    fig.suptitle(f"Frame {fid}{time_str}")
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"frame_{fid:05d}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}", end="\r")

print(f"\nDone. Saved {len(sampled)} histogram figures to '{OUTPUT_DIR}/'.")
