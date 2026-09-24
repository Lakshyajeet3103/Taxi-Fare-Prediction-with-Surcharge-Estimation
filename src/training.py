from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    RANDOM_STATE, CV_FOLDS, RIDGE_ALPHA, RANDOM_FOREST_PARAMS,
    HIST_GB_PARAMS, HIST_GB_GRID, AUX_MODEL_PARAMS,
)

SCORING = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}
TUNED_NAME = "Tuned HistGradientBoosting"


def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def kfold():
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def _cv_row(mae, rmse, rmse_std, r2):
    return {
        "CV_MAE": float(mae), "CV_RMSE": float(rmse),
        "CV_RMSE_std": float(rmse_std), "CV_R2": float(r2),
    }


def cross_validate_model(model, X, y):
    # Fits run sequentially; the estimators themselves are multi-threaded.
    scores = cross_validate(model, X, y, cv=kfold(), scoring=SCORING, n_jobs=1)
    return _cv_row(
        -scores["test_MAE"].mean(), -scores["test_RMSE"].mean(),
        scores["test_RMSE"].std(), scores["test_R2"].mean(),
    )


def _pipe(model, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def fare_candidates():
    return {
        "Mean Baseline": _pipe(DummyRegressor(strategy="mean")),
        "Linear Regression": _pipe(LinearRegression(), scale=True),
        "Ridge": _pipe(Ridge(alpha=RIDGE_ALPHA), scale=True),
        "Random Forest": _pipe(RandomForestRegressor(
            **RANDOM_FOREST_PARAMS, random_state=RANDOM_STATE
        )),
        "HistGradientBoosting": _pipe(HistGradientBoostingRegressor(
            **HIST_GB_PARAMS, random_state=RANDOM_STATE
        )),
    }


def tune_hist_gradient_boosting(Xtr, ytr):
    pipe = _pipe(HistGradientBoostingRegressor(random_state=RANDOM_STATE))
    search = GridSearchCV(
        pipe, HIST_GB_GRID, cv=kfold(), scoring=SCORING, refit="RMSE", n_jobs=1
    )
    search.fit(Xtr, ytr)
    return search


def compare_fare_models(Xtr, ytr, log=print):
    """Cross-validate every candidate on the training split only. Returns the CV table,
    unfitted candidate pipelines (plus the refit tuned model) and the grid search."""
    rows, models = [], fare_candidates()
    for name, model in models.items():
        log(f"  CV: {name}")
        rows.append({"Model": name, **cross_validate_model(model, Xtr, ytr)})

    log(f"  Grid search: {TUNED_NAME}")
    search = tune_hist_gradient_boosting(Xtr, ytr)
    i, cv = search.best_index_, search.cv_results_
    rows.append({"Model": TUNED_NAME, **_cv_row(
        -cv["mean_test_MAE"][i], -cv["mean_test_RMSE"][i],
        cv["std_test_RMSE"][i], cv["mean_test_R2"][i],
    )})
    models[TUNED_NAME] = search.best_estimator_  # already refit on all of Xtr
    return pd.DataFrame(rows), models, search


def fit_final(name, models, Xtr, ytr):
    if name == TUNED_NAME:
        return models[name]
    return clone(models[name]).fit(Xtr, ytr)


def fit_aux_model(Xtr, ytr):
    """HistGradientBoosting for the surcharge / tolls targets: CV on train, then fit on all of train."""
    model = HistGradientBoostingRegressor(**AUX_MODEL_PARAMS, random_state=RANDOM_STATE)
    cv = cross_validate_model(model, Xtr, ytr)
    return model.fit(Xtr, ytr), cv
