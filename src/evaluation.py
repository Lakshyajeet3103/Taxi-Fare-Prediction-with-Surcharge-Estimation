from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import RANDOM_STATE, PERMUTATION_REPEATS


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_eda_plots(feat: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(feat["fare_amount"], bins=80)
    ax.set(xlabel="Fare amount", ylabel="Count", title="Fare Distribution")
    _save(fig, out_dir / "fare_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(feat["trip_distance"], feat["fare_amount"], s=5, alpha=.2)
    ax.set(xlabel="Trip distance (miles)", ylabel="Fare", title="Fare vs Distance")
    _save(fig, out_dir / "fare_vs_distance.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(feat["trip_duration_min"], feat["fare_amount"], s=5, alpha=.2)
    ax.set(xlabel="Duration (min)", ylabel="Fare", title="Fare vs Duration")
    _save(fig, out_dir / "fare_vs_duration.png")

    byhour = feat.groupby("pickup_hour")["fare_amount"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(byhour.index, byhour.values, marker="o")
    ax.set(xlabel="Pickup hour", ylabel="Mean fare", title="Mean Fare by Hour")
    _save(fig, out_dir / "fare_by_hour.png")

    corr_cols = [
        "trip_distance", "trip_duration_min", "passenger_count", "fare_amount",
        "extra", "mta_tax", "improvement_surcharge", "congestion_surcharge",
        "Airport_fee", "cbd_congestion_fee", "tolls_amount", "total_surcharge",
        "pickup_hour", "avg_speed_mph", "metadata_missing",
    ]
    corr = feat[corr_cols].corr()  # pairwise: null-metadata rows drop out where needed
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="viridis", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)), corr.columns)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation Heatmap")
    _save(fig, out_dir / "correlation_heatmap.png")

    known = feat.dropna(subset=["total_surcharge"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(known["total_surcharge"], known["fare_amount"], s=5, alpha=.2)
    ax.set(xlabel="Total surcharge (excl. tolls)", ylabel="Fare",
           title="Fare vs Total Surcharge (rows with complete metadata)")
    _save(fig, out_dir / "surcharge_effect.png")


def save_regression_plots(y_true, y_pred, out_dir: Path, prefix: str, label: str = "fare"):
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_true, y_pred, s=5, alpha=0.2)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi])
    ax.set(xlabel=f"Actual {label}", ylabel=f"Predicted {label}", title="Actual vs Predicted (test set)")
    _save(fig, out_dir / f"{prefix}_actual_vs_predicted.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, s=5, alpha=0.2)
    ax.axhline(0)
    ax.set(xlabel=f"Predicted {label}", ylabel="Residual", title="Residual Plot (test set)")
    _save(fig, out_dir / f"{prefix}_residuals.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=60)
    ax.set(xlabel="Residual", ylabel="Count", title="Residual Distribution (test set)")
    _save(fig, out_dir / f"{prefix}_residual_histogram.png")


def compute_permutation_importance(model, X, y) -> pd.DataFrame:
    result = permutation_importance(
        model, X, y, n_repeats=PERMUTATION_REPEATS, random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error", n_jobs=1,
    )
    return pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def plot_importance(imp: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    top = imp.head(10).sort_values("importance_mean")
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set(xlabel="Permutation importance (increase in RMSE)",
           title="Permutation Feature Importance (test set)")
    _save(fig, path)


def plot_model_comparison(results: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(results["Model"], results["CV_RMSE"], yerr=results["CV_RMSE_std"])
    ax.set(ylabel="Cross-validated RMSE (training split)", title="Model Comparison")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, path)


def variance_inflation_factors(X: pd.DataFrame) -> pd.Series:
    """VIF_j = [R^-1]_jj, where R is the feature correlation matrix."""
    inv = np.linalg.inv(X.corr().to_numpy())
    return pd.Series(np.diag(inv), index=X.columns)
