"""End-to-end pipeline: load → clean → features → EDA → model selection (CV) → final test evaluation."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Make `src` importable no matter which directory the script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.data_loading import load_tlc_data
from src.data_cleaning import clean_data
from src.features import (
    engineer_features, FARE_FEATURES, SURCHARGE_FEATURES, TOLLS_FEATURES,
    TOLLS_TARGET, METADATA_COLUMNS, SURCHARGE_COLUMNS,
)
from src.training import compare_fare_models, fit_final, fit_aux_model, metrics
from src.evaluation import (
    save_eda_plots, save_regression_plots, compute_permutation_importance,
    plot_importance, plot_model_comparison, variance_inflation_factors,
)

_START = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - _START:7.1f}s] {msg}", flush=True)


def describe_null_subset(raw: pd.DataFrame) -> dict:
    null = raw[METADATA_COLUMNS].isna()
    all_null = null.all(axis=1)
    return {
        "columns": METADATA_COLUMNS,
        "rows_any_null": int(null.any(axis=1).sum()),
        "rows_all_null": int(all_null.sum()),
        "share_all_null": float(all_null.mean()),
        "payment_type_counts_in_subset": {
            str(k): int(v) for k, v in raw.loc[all_null, "payment_type"].value_counts(dropna=False).items()
        },
    }


def main():
    config.FIGURES_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)

    # ---------------- Load ----------------
    raw, source = load_tlc_data(
        config.RAW_DIR, config.INPUT_FILES, config.SAMPLE_SIZE, config.RANDOM_STATE
    )
    log(f"Files: {source['files']}")
    print(f"Original rows in file(s): {source['source_rows']:,}")
    print(f"Rows sampled for project: {len(raw):,}")
    print("\nDtypes:")
    print(raw.dtypes)
    print("\nMissing values:")
    print(raw.isna().sum().sort_values(ascending=False))
    null_subset = describe_null_subset(raw)
    print("\nRows with null metadata columns:")
    print(json.dumps(null_subset, indent=2))

    # ---------------- Clean + features ----------------
    cleaned, clean_log = clean_data(raw)
    feat = engineer_features(cleaned)
    print("\nCleaning log:")
    print(json.dumps(clean_log, indent=2))

    save_eda_plots(feat, config.FIGURES_DIR)
    log("EDA figures saved")

    # ---------------- Single train/test split ----------------
    train_idx, test_idx = train_test_split(
        feat.index, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    train, test = feat.loc[train_idx], feat.loc[test_idx]

    # ---------------- Fare model: select by CV on train ----------------
    Xtr, ytr = train[FARE_FEATURES], train["fare_amount"]
    Xte, yte = test[FARE_FEATURES], test["fare_amount"]

    log("Cross-validating fare models")
    cv_table, models, search = compare_fare_models(Xtr, ytr, log=log)
    best_name = cv_table.loc[cv_table["CV_RMSE"].idxmin(), "Model"]
    log(f"Selected by CV RMSE: {best_name}")
    best_model = fit_final(best_name, models, Xtr, ytr)

    # Test set: evaluated once, for the selected model only.
    best_pred = best_model.predict(Xte)
    fare_test = metrics(yte, best_pred)

    save_regression_plots(yte, best_pred, config.FIGURES_DIR, prefix="best_model")
    plot_model_comparison(cv_table, config.FIGURES_DIR / "model_comparison.png")
    log("Computing permutation importance")
    importance_df = compute_permutation_importance(best_model, Xte, yte)
    plot_importance(importance_df, config.FIGURES_DIR / "feature_importance.png")

    ridge = fit_final("Ridge", models, Xtr, ytr)
    Xtr_imputed = pd.DataFrame(ridge.named_steps["imputer"].transform(Xtr), columns=FARE_FEATURES)
    coef_df = pd.DataFrame({
        "feature": FARE_FEATURES,
        "coefficient": ridge.named_steps["model"].coef_,
        "vif": variance_inflation_factors(Xtr_imputed).values,
    })
    coef_df["absolute_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("absolute_coefficient", ascending=False)

    # ---------------- Surcharge model (complete-metadata rows only) ----------------
    s_train = train.dropna(subset=["total_surcharge"])
    s_test = test.dropna(subset=["total_surcharge"])
    log(f"Surcharge model: {len(s_train):,} train / {len(s_test):,} test rows")
    s_model, s_cv = fit_aux_model(s_train[SURCHARGE_FEATURES], s_train["total_surcharge"])
    s_test_metrics = metrics(s_test["total_surcharge"], s_model.predict(s_test[SURCHARGE_FEATURES]))

    # ---------------- Tolls model (all rows; tolls_amount is never null) ----------------
    log("Tolls model")
    t_model, t_cv = fit_aux_model(train[TOLLS_FEATURES], train[TOLLS_TARGET])
    t_test_metrics = metrics(test[TOLLS_TARGET], t_model.predict(test[TOLLS_FEATURES]))

    # ---------------- Save ----------------
    cv_table.to_csv(config.RESULTS_DIR / "model_comparison.csv", index=False)
    importance_df.to_csv(config.RESULTS_DIR / "permutation_importance.csv", index=False)
    coef_df.to_csv(config.RESULTS_DIR / "ridge_coefficients.csv", index=False)

    baseline_cv_rmse = float(cv_table.loc[cv_table["Model"] == "Mean Baseline", "CV_RMSE"].iloc[0])
    best_cv_rmse = float(cv_table["CV_RMSE"].min())

    summary = {
        "data_source": "Official NYC Taxi & Limousine Commission Yellow Taxi Trip Record Data",
        **source,
        "sampled_rows": int(len(raw)),
        "final_rows": int(len(cleaned)),
        "config": {
            "random_state": config.RANDOM_STATE, "sample_size": config.SAMPLE_SIZE,
            "test_size": config.TEST_SIZE, "cv_folds": config.CV_FOLDS,
            "max_fare": config.MAX_FARE, "max_distance_mi": config.MAX_DISTANCE_MI,
            "max_duration_min": config.MAX_DURATION_MIN, "max_speed_mph": config.MAX_SPEED_MPH,
            "max_passengers": config.MAX_PASSENGERS,
        },
        "null_metadata_subset_in_sample": null_subset,
        "null_metadata_rows_after_cleaning": int(feat["metadata_missing"].sum()),
        "cleaning_log": clean_log,
        "split": {"train_rows": int(len(train)), "test_rows": int(len(test))},
        "fare_model": {
            "features": FARE_FEATURES,
            "cv_results": cv_table.round(6).to_dict(orient="records"),
            "selected_model": best_name,
            "selection_criterion": f"lowest mean {config.CV_FOLDS}-fold CV RMSE on training split",
            "baseline_cv_rmse": baseline_cv_rmse,
            "selected_cv_rmse": best_cv_rmse,
            "cv_improvement_over_baseline_percent": 100 * (baseline_cv_rmse - best_cv_rmse) / baseline_cv_rmse,
            "tuned_hist_params": search.best_params_,
            "test_metrics": fare_test,
        },
        "surcharge_model": {
            "target_components": SURCHARGE_COLUMNS,
            "features": SURCHARGE_FEATURES,
            "train_rows": int(len(s_train)), "test_rows": int(len(s_test)),
            "excluded_null_metadata_rows": {
                "train": int(len(train) - len(s_train)), "test": int(len(test) - len(s_test)),
            },
            "cv_metrics": s_cv,
            "test_metrics": s_test_metrics,
        },
        "tolls_model": {
            "target": TOLLS_TARGET,
            "features": TOLLS_FEATURES,
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "cv_metrics": t_cv,
            "test_metrics": t_test_metrics,
        },
        "top_permutation_features": importance_df.head(3).to_dict(orient="records"),
        "top_ridge_coefficients": coef_df.head(5).round(6).to_dict(orient="records"),
    }

    (config.RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2, default=str))
    log("Done")


if __name__ == "__main__":
    main()
