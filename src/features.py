from __future__ import annotations
import numpy as np
import pandas as pd

from src.config import RUSH_HOURS

# Regulated per-trip surcharges. tolls_amount is deliberately excluded: tolls depend on the
# route (bridges/tunnels), not on fare rules, so they are modeled separately (TOLLS_TARGET).
SURCHARGE_COLUMNS = [
    "extra", "mta_tax", "improvement_surcharge",
    "congestion_surcharge", "Airport_fee", "cbd_congestion_fee",
]
TOLLS_TARGET = "tolls_amount"

# In TLC data these columns are null together (payment_type 0 records); see README.
METADATA_COLUMNS = [
    "passenger_count", "RatecodeID", "store_and_fwd_flag",
    "congestion_surcharge", "Airport_fee",
]

FARE_FEATURES = [
    "trip_distance", "trip_duration_min", "passenger_count",
    "avg_speed_mph", "pickup_hour", "day_of_week",
    "is_weekend", "is_rush_hour", "log_trip_distance",
    "metadata_missing",
]

# Surcharge model is trained only on rows with complete metadata, so no missing indicator.
SURCHARGE_FEATURES = [
    "trip_distance", "trip_duration_min", "passenger_count",
    "avg_speed_mph", "pickup_hour", "day_of_week",
    "is_weekend", "is_rush_hour",
]

TOLLS_FEATURES = SURCHARGE_FEATURES + ["metadata_missing"]


def add_trip_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trip_duration_min"] = (
        df["dropoff_datetime"] - df["pickup_datetime"]
    ).dt.total_seconds() / 60.0
    df["avg_speed_mph"] = df["trip_distance"] / (df["trip_duration_min"] / 60.0)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    if "trip_duration_min" not in df.columns:
        df = add_trip_metrics(df)
    else:
        df = df.copy()
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["pickup_hour"].isin(RUSH_HOURS).astype(int)
    df["log_trip_distance"] = np.log1p(df["trip_distance"])
    df["metadata_missing"] = df[METADATA_COLUMNS].isna().any(axis=1).astype(int)
    # NaN when any component is unknown: a missing surcharge is not a zero surcharge.
    df["total_surcharge"] = df[SURCHARGE_COLUMNS].sum(axis=1, skipna=False)
    return df
