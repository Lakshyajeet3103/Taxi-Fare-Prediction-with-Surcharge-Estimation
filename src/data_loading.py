from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import pyarrow.parquet as pq

# Only the columns the pipeline uses are read from Parquet.
LOAD_COLUMNS = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
    "trip_distance", "RatecodeID", "store_and_fwd_flag", "payment_type",
    "fare_amount", "extra", "mta_tax", "improvement_surcharge",
    "congestion_surcharge", "Airport_fee", "cbd_congestion_fee", "tolls_amount",
]

# Fees introduced after some TLC files were published (Airport_fee in 2021,
# cbd_congestion_fee in January 2025). If a file has no such column, the fee
# was not charged that month, so it is filled with 0.
ZERO_IF_ABSENT = ("Airport_fee", "cbd_congestion_fee")

TIMESTAMP_RENAMES = {
    "tpep_pickup_datetime": "pickup_datetime",
    "tpep_dropoff_datetime": "dropoff_datetime",
}


def file_month(path: Path) -> str:
    match = re.search(r"(\d{4})-(\d{2})", path.name)
    if not match:
        raise ValueError(f"Cannot infer YYYY-MM month from file name: {path.name}")
    return f"{match.group(1)}-{match.group(2)}"


def _read_columns(path: Path) -> pd.DataFrame:
    # Column capitalisation varies between TLC releases (e.g. airport_fee vs Airport_fee).
    lookup = {name.lower(): name for name in pq.ParquetFile(path).schema_arrow.names}
    selected, renames = [], {}
    for col in LOAD_COLUMNS:
        actual = lookup.get(col.lower())
        if actual is None:
            if col in ZERO_IF_ABSENT:
                continue
            raise KeyError(f"{path.name} is missing required column '{col}'")
        selected.append(actual)
        if actual != col:
            renames[actual] = col
    df = pd.read_parquet(path, columns=selected).rename(columns=renames)
    for col in ZERO_IF_ABSENT:
        if col not in df.columns:
            df[col] = 0.0
    return df[LOAD_COLUMNS]


def _allocate_sample(row_counts: list[int], sample_size: int) -> list[int]:
    """Split sample_size across files in proportion to their row counts (largest remainder)."""
    total = sum(row_counts)
    if total <= sample_size:
        return list(row_counts)
    exact = [n * sample_size / total for n in row_counts]
    alloc = [int(x) for x in exact]
    by_remainder = sorted(range(len(exact)), key=lambda i: exact[i] - alloc[i], reverse=True)
    for i in by_remainder[: sample_size - sum(alloc)]:
        alloc[i] += 1
    return alloc


def load_tlc_data(raw_dir: Path, file_names: list[str], sample_size: int, random_state: int):
    paths = [raw_dir / name for name in file_names]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing input file(s) in {raw_dir}: {missing}. Download them from "
            "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page "
            "or edit INPUT_FILES in src/config.py."
        )

    row_counts = [pq.ParquetFile(p).metadata.num_rows for p in paths]
    allocation = _allocate_sample(row_counts, sample_size)

    frames = []
    for path, n_rows, n_take in zip(paths, row_counts, allocation):
        df = _read_columns(path)
        if n_take < n_rows:
            df = df.sample(n=n_take, random_state=random_state)
        df["source_month"] = file_month(path)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True).rename(columns=TIMESTAMP_RENAMES)
    info = {
        "files": [p.name for p in paths],
        "source_rows": int(sum(row_counts)),
        "source_rows_per_file": {p.name: int(n) for p, n in zip(paths, row_counts)},
        "sampled_rows_per_file": {p.name: int(n) for p, n in zip(paths, allocation)},
    }
    return df, info
