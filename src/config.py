"""Single source of configuration for the pipeline: paths, seeds, cleaning thresholds, model settings."""
from __future__ import annotations
from pathlib import Path

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

# Monthly TLC Yellow Taxi files to use, relative to RAW_DIR. Every listed file is loaded;
# the file name must contain the month as YYYY-MM (used to drop out-of-month pickups).
INPUT_FILES = [
    "yellow_tripdata_2025-01.parquet",
]

# ---------------- Reproducibility ----------------
RANDOM_STATE = 42
SAMPLE_SIZE = 300_000      # rows sampled across all INPUT_FILES (proportional to file size)
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---------------- Cleaning thresholds ----------------
# Percentiles quoted are from the 300k-row sample of yellow_tripdata_2025-01 (after dropping
# non-positive fare/distance); see README "Cleaning" for the reasoning behind each bound.
MAX_FARE = 500.0           # USD; p99.99 = 279.66, sample max = 500.00; full file contains 863,372
MAX_DISTANCE_MI = 100.0    # p99.99 = 54.2 mi; values above are odometer errors (up to 92,186 mi)
MAX_DURATION_MIN = 180.0   # p99.9 = 99.8 min, p99.99 = 1,428.6 min (meter left running)
MAX_SPEED_MPH = 80.0
MAX_PASSENGERS = 6

RUSH_HOURS = (7, 8, 9, 16, 17, 18, 19)

# ---------------- Models ----------------
RIDGE_ALPHA = 1.0

RANDOM_FOREST_PARAMS = dict(
    n_estimators=150, max_features="sqrt", min_samples_leaf=2, n_jobs=-1,
)

HIST_GB_PARAMS = dict(max_iter=150, learning_rate=0.08, max_leaf_nodes=31)

HIST_GB_GRID = {
    "model__learning_rate": [0.05, 0.08],
    "model__max_iter": [100, 150],
    "model__max_leaf_nodes": [15, 31],
    "model__l2_regularization": [0.0, 0.5],
}

# Surcharge and tolls models
AUX_MODEL_PARAMS = dict(max_iter=100, learning_rate=0.08, max_leaf_nodes=31)

PERMUTATION_REPEATS = 5
