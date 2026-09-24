from __future__ import annotations
import numpy as np
import pandas as pd

from src.config import (
    MAX_FARE, MAX_DISTANCE_MI, MAX_DURATION_MIN, MAX_SPEED_MPH, MAX_PASSENGERS,
)
from src.features import add_trip_metrics


def _drop(df: pd.DataFrame, bad: pd.Series, key: str, log: dict) -> pd.DataFrame:
    log[key] = int(bad.sum())
    return df.loc[~bad]


def clean_data(df: pd.DataFrame):
    """Apply the cleaning rules in order; returns the cleaned frame and per-rule removal counts."""
    log = {"initial_rows": int(len(df))}
    df = df.copy()

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")
    bad = df["pickup_datetime"].isna() | df["dropoff_datetime"].isna()
    df = _drop(df, bad, "invalid_timestamps_removed", log)

    pickup_month = df["pickup_datetime"].dt.strftime("%Y-%m")
    df = _drop(df, pickup_month != df["source_month"], "pickup_outside_source_month_removed", log)

    df = add_trip_metrics(df)

    duration = df["trip_duration_min"]
    df = _drop(df, ~np.isfinite(duration) | (duration <= 0), "nonpositive_duration_removed", log)
    df = _drop(df, df["trip_duration_min"] > MAX_DURATION_MIN,
               f"duration_above_{MAX_DURATION_MIN:g}_min_removed", log)

    distance = df["trip_distance"]
    df = _drop(df, ~np.isfinite(distance) | (distance <= 0), "nonpositive_distance_removed", log)
    df = _drop(df, df["trip_distance"] > MAX_DISTANCE_MI,
               f"distance_above_{MAX_DISTANCE_MI:g}_mi_removed", log)

    fare = df["fare_amount"]
    df = _drop(df, ~np.isfinite(fare) | (fare <= 0), "nonpositive_fare_removed", log)
    df = _drop(df, df["fare_amount"] > MAX_FARE, f"fare_above_{MAX_FARE:g}_removed", log)

    # Missing passenger_count is kept (handled by imputation / missing indicator downstream).
    pax = df["passenger_count"]
    df = _drop(df, pax.notna() & (pax <= 0), "passenger_count_nonpositive_removed", log)
    df = _drop(df, df["passenger_count"].notna() & (df["passenger_count"] > MAX_PASSENGERS),
               f"passenger_count_above_{MAX_PASSENGERS}_removed", log)

    speed = df["avg_speed_mph"]
    df = _drop(df, ~np.isfinite(speed) | (speed > MAX_SPEED_MPH),
               f"speed_above_{MAX_SPEED_MPH:g}_mph_removed", log)

    df = df.reset_index(drop=True)
    log["final_rows"] = int(len(df))
    return df, log
