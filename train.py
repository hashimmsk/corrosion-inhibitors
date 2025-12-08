"""
Model selection and training workflow for corrosion inhibitor IE prediction.

Loads canonical train/val/test splits, runs compact hyperparameter searches,
selects the best model on validation R^2, refits on train+val, evaluates on test,
and saves metrics plus the fitted model artifact under data/models/.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR
from joblib import parallel_backend

import features


MetricDict = Dict[str, float]
ResultDict = Dict[str, object]


def _project_root() -> Path:
    current = Path(__file__).resolve().parent
    if (current / "data").exists():
        return current
    return current.parent


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> MetricDict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _candidates() -> Dict[str, Tuple[object, Dict[str, List[object]]]]:
    return {
        "linear": (LinearRegression(), {}),
        "ridge": (
            Ridge(),
            {"alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        "lasso": (
            Lasso(max_iter=5000),
            {"alpha": [0.001, 0.01, 0.1, 1.0]},
        ),
        "elasticnet": (
            ElasticNet(max_iter=5000),
            {"alpha": [0.001, 0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
        ),
        "random_forest": (
            RandomForestRegressor(random_state=0, n_jobs=-1),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 5, 10],
                "min_samples_leaf": [1, 2],
            },
        ),
        "gbr": (
            GradientBoostingRegressor(random_state=0),
            {
                "n_estimators": [200, 400],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
            },
        ),
        "svr": (
            SVR(),
            {
                "C": [1.0, 10.0, 50.0],
                "gamma": ["scale", "auto"],
                "epsilon": [0.01, 0.1],
            },
        ),
    }


def _serialize(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    return value


def main() -> None:
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = features.load_splits()

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    candidates = _candidates()
    results: List[ResultDict] = []
    best_estimators: Dict[str, object] = {}

    with parallel_backend("threading"):
        for name, (estimator, param_grid) in candidates.items():
            search = GridSearchCV(
                estimator,
                param_grid if param_grid else {},
                scoring="r2",
                cv=cv,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            best_estimators[name] = best

            val_pred = best.predict(X_val)
            val_metrics = _metrics(y_val, val_pred)

            results.append(
                {
                    "model": name,
                    "cv_best_score": float(search.best_score_),
                    "best_params": _serialize(search.best_params_),
                    "val_metrics": _serialize(val_metrics),
                }
            )

    results_sorted = sorted(results, key=lambda r: r["val_metrics"]["r2"], reverse=True)
    best_name = results_sorted[0]["model"]
    best_model = best_estimators[best_name]

    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    best_model.fit(X_trainval, y_trainval)

    test_pred = best_model.predict(X_test)
    test_metrics = _metrics(y_test, test_pred)

    root = _project_root()
    models_dir = root / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    results_path = models_dir / "model_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(_serialize(results_sorted), f, indent=2)

    best_metrics_path = models_dir / "best_model_test_metrics.json"
    with best_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_name,
                "test_metrics": _serialize(test_metrics),
                "best_params": _serialize(results_sorted[0]["best_params"]),
            },
            f,
            indent=2,
        )

    artifact_path = models_dir / f"{best_name}_model.pkl"
    with artifact_path.open("wb") as f:
        pickle.dump(best_model, f)

    print("Model selection complete.")
    print(f"Results written to: {results_path}")
    print(f"Best model test metrics: {best_metrics_path}")
    print(f"Saved best model artifact: {artifact_path}")


if __name__ == "__main__":
    main()

