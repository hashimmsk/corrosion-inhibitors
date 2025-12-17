"""
1) Load train/val/test splits
2) Tune each model on TRAIN using CV (RandomizedSearchCV)
3) Evaluate best tuned model on VAL and pick the winner
4) Refit winner on TRAIN+VAL
5) Evaluate once on TEST
6) Save best pipeline + metrics to data/models/
"""

import json
import numpy as np
import pandas as pd
import features

from pathlib import Path
from joblib import parallel_backend
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.svm import SVR

SEED = 0
CV_SPLITS = 3
N_ITER = 18
OUT_DIR = Path(__file__).parent / "data" / "models"

def get_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

def build_models():
    return {
        "random_forest": (
            RandomForestRegressor(random_state=SEED, n_jobs=1),
            {"n_estimators": [300, 600], "max_depth": [None, 6, 10], "min_samples_leaf": [1, 2, 4]},
        ),
        "svr": (
            SVR(kernel="rbf", cache_size=2000),
            {"C": [10.0, 50.0, 100.0], "gamma": ["scale", 0.1, 0.01], "epsilon": [0.01, 0.05, 0.1]},
        ),
    }

def main():
    # 1) Load splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = features.load_splits()

    # 2) CV setup
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)

    models = build_models()
    results = []

    # 3) Tune each model on TRAIN, evaluate on VAL
    with parallel_backend("threading"):
        for model_name, (estimator, param_space) in models.items():
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_space,
                n_iter=N_ITER,
                scoring="r2",
                cv=cv,
                random_state=SEED,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            val_pred = best_model.predict(X_val)
            val_metrics = get_metrics(y_val, val_pred)

            results.append(
                {
                    "model": model_name,
                    "val_metrics": val_metrics,
                    "best_params": search.best_params_,
                    "best_estimator": best_model,
                }
            )

            print(f"{model_name}: VAL R² = {val_metrics['r2']:.4f} | VAL RMSE = {val_metrics['rmse']:.3f}")

    # 4) Pick best model by validation R²
    best = max(results, key=lambda r: r["val_metrics"]["r2"])
    best_name = best["model"]
    best_model = best["best_estimator"]

    # 5) Refit on TRAIN+VAL
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    best_model.fit(X_trainval, y_trainval)

    # 6) Test evaluation
    test_pred = best_model.predict(X_test)
    test_metrics = get_metrics(y_test, test_pred)

    # 7) Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save a clean JSON report
    report = {
        "best_model": best_name,
        "best_params": best["best_params"],
        "val_metrics_best": best["val_metrics"],
        "test_metrics": test_metrics,
        "all_models": [
            {"model": r["model"], "val_metrics": r["val_metrics"], "best_params": r["best_params"]}
            for r in results
        ],
        "config": {"seed": SEED, "cv_splits": CV_SPLITS, "n_iter": N_ITER},
    }
    (OUT_DIR / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Save test predictions for inspection (include pH and medium for analysis)
    pred_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": test_pred,
        "residual": y_test.values - test_pred,
    })
    
    # Add original pH and medium from test metadata
    test_meta = features.load_test_metadata()
    for col in test_meta.columns:
        pred_df[col] = test_meta[col].values
    
    pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    print("\nBest model:", best_name)
    print("Test metrics:", test_metrics)

if __name__ == "__main__":
    main()
