"""
Combined approach: Extended features (6 + 40 RDKit) + VSG augmentation.

1) Load extended train/val/test (46 features)
2) Apply VSG to extended TRAIN only (rows augmented)
3) Train RF and SVR on augmented extended data
4) Evaluate on original val and test

Requires: pip install rdkit
Run preprocess_extended.py first.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.svm import SVR

from virtual_sample_generation import generate_virtual_samples

SEED = 0
CV_SPLITS = 5
N_ITER = 50
N_VIRTUAL_SAMPLES = 500
OUT_DIR = Path(__file__).resolve().parent / "data" / "models" / "extended_vsg"
EXTENDED_DIR = Path(__file__).resolve().parent / "data" / "processed" / "extended"
LABEL_COLUMN = "IE"


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
            {
                "n_estimators": [300, 600, 900],
                "max_depth": [None, 6, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "svr": (
            SVR(kernel="rbf", cache_size=2000),
            {
                "C": [10.0, 50.0, 100.0],
                "gamma": ["scale", 0.1, 0.01],
                "epsilon": [0.01, 0.05, 0.1],
            },
        ),
    }


def load_extended_splits():
    """Load extended train/val/test and metadata."""
    train_df = pd.read_csv(EXTENDED_DIR / "train.csv")
    val_df = pd.read_csv(EXTENDED_DIR / "val.csv")
    test_df = pd.read_csv(EXTENDED_DIR / "test.csv")

    with open(EXTENDED_DIR / "rdkit_columns.json") as f:
        meta = json.load(f)
    extended_cols = meta["extended_columns"]

    X_train = train_df[extended_cols]
    y_train = train_df[LABEL_COLUMN]
    X_val = val_df[extended_cols]
    y_val = val_df[LABEL_COLUMN]
    X_test = test_df[extended_cols]
    y_test = test_df[LABEL_COLUMN]

    medium_train = train_df["medium"] if "medium" in train_df.columns else None

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "medium_train": medium_train,
        "extended_cols": extended_cols,
    }


def main():
    if not (EXTENDED_DIR / "train.csv").exists():
        raise FileNotFoundError(
            f"Extended data not found. Run: python3 preprocess_extended.py"
        )

    data = load_extended_splits()
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    extended_cols = data["extended_cols"]
    medium_train = data["medium_train"]

    # Apply VSG to extended training only
    print("Applying VSG to extended training set (46 features)...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    synthetic_path = OUT_DIR / "synthetic_samples.csv"

    X_train_aug, y_train_aug = generate_virtual_samples(
        X_train,
        y_train,
        n_samples=N_VIRTUAL_SAMPLES,
        medium=medium_train,
        medium_balance=True,
        random_state=SEED,
        save_path=synthetic_path,
        feature_columns=extended_cols,
    )

    print(f"  Original train: {len(X_train)} samples")
    print(f"  After VSG:      {len(X_train_aug)} samples (+{len(X_train_aug) - len(X_train)} virtual)")
    print(f"  Features:       {len(extended_cols)} (6 original + 40 RDKit)")

    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    models = build_models()
    results = []

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
            search.fit(X_train_aug, y_train_aug)

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
            print(
                f"{model_name}: VAL R² = {val_metrics['r2']:.4f} | "
                f"VAL RMSE = {val_metrics['rmse']:.3f}"
            )

    # Refit on augmented train+val
    X_trainval = pd.concat([X_train_aug, X_val], axis=0, ignore_index=True)
    y_trainval = pd.concat([y_train_aug, y_val], axis=0, ignore_index=True)

    print("\nRefitting on AUGMENTED TRAIN+VAL, evaluating on TEST...")
    for r in results:
        model = r["best_estimator"]
        model.fit(X_trainval, y_trainval)
        test_pred = model.predict(X_test)
        r["test_metrics"] = get_metrics(y_test, test_pred)
        r["test_predictions"] = test_pred
        print(
            f"  {r['model']}: TEST R² = {r['test_metrics']['r2']:.4f} | "
            f"TEST RMSE = {r['test_metrics']['rmse']:.3f}"
        )

    best = max(results, key=lambda r: r["val_metrics"]["r2"])

    # Save
    report = {
        "best_model": best["model"],
        "best_params": best["best_params"],
        "val_metrics_best": best["val_metrics"],
        "test_metrics": best["test_metrics"],
        "config": {
            "n_features": len(extended_cols),
            "n_virtual_samples": N_VIRTUAL_SAMPLES,
            "seed": SEED,
            "cv_splits": CV_SPLITS,
        },
        "all_models": [
            {
                "model": r["model"],
                "val_metrics": r["val_metrics"],
                "test_metrics": r["test_metrics"],
                "best_params": r["best_params"],
            }
            for r in results
        ],
    }
    (OUT_DIR / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    pred_df = pd.DataFrame({"y_true": y_test.values})
    for r in results:
        pred_df[f"y_pred_{r['model']}"] = r["test_predictions"]
    pred_df["y_pred"] = best["test_predictions"]
    if "medium" in pd.read_csv(EXTENDED_DIR / "test.csv").columns:
        pred_df["medium"] = pd.read_csv(EXTENDED_DIR / "test.csv")["medium"].values
    pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    print("\n" + "=" * 60)
    print("Extended + VSG Training Complete")
    print("=" * 60)
    print(f"Best model: {best['model']}")
    print(f"Val R²:  {best['val_metrics']['r2']:.4f}")
    print(f"Test R²: {best['test_metrics']['r2']:.4f}")
    print(f"Results: {OUT_DIR}")


if __name__ == "__main__":
    main()
