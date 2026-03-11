"""
Train models with KDE-based Virtual Sample Generation (Pathway A).

1) Load train/val/test splits
2) Apply VSG to TRAIN only (val and test remain unchanged)
3) Tune models on augmented TRAIN using CV
4) Evaluate on VAL, pick winner
5) Refit winner on augmented TRAIN+VAL
6) Evaluate on TEST (original held-out set - no augmentation)
7) Save results to data/models/vsg/

Critical: Test set is NEVER augmented. All evaluation uses original data.
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

from virtual_sample_generation import generate_virtual_samples, load_train_with_medium

SEED = 0
CV_SPLITS = 5  # Upgraded from 3 to 5 per Pathway G
N_ITER = 50   # Increased from 18 for better hyperparameter search
N_VIRTUAL_SAMPLES = 500
OUT_DIR = Path(__file__).parent / "data" / "models" / "vsg"


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


def main():
    # 1) Load splits (original, unaugmented)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = features.load_splits()

    # 2) Load medium for stratified VSG
    X_train_full, _, medium = load_train_with_medium()

    # 3) Apply VSG to TRAIN only
    print("Applying KDE-based Virtual Sample Generation to training set...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    synthetic_path = OUT_DIR / "synthetic_samples.csv"
    X_train_aug, y_train_aug = generate_virtual_samples(
        X_train,
        y_train,
        n_samples=N_VIRTUAL_SAMPLES,
        medium=medium,
        medium_balance=True,
        random_state=SEED,
        save_path=synthetic_path,
    )
    print(f"  Original train: {len(X_train)} samples")
    print(f"  After VSG:      {len(X_train_aug)} samples (+{len(X_train_aug) - len(X_train)} virtual)")

    # 4) CV setup (stratified by medium if possible - KFold for now)
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)

    models = build_models()
    results = []

    # 5) Tune each model on AUGMENTED TRAIN, evaluate on VAL (original)
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

    # 6) Refit on AUGMENTED TRAIN+VAL and evaluate on TEST
    # For refit, we augment train+val together (still no test leakage)
    X_val_df = pd.DataFrame(X_val.values, columns=X_val.columns)
    X_trainval = pd.concat([X_train_aug, X_val_df], axis=0, ignore_index=True)
    y_trainval = pd.concat([y_train_aug, y_val], axis=0, ignore_index=True)

    print("\nRefitting all models on AUGMENTED TRAIN+VAL and evaluating on TEST...")
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

    # 7) Pick best by validation R²
    best = max(results, key=lambda r: r["val_metrics"]["r2"])
    best_name = best["model"]

    # 8) Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "best_model": best_name,
        "best_params": best["best_params"],
        "val_metrics_best": best["val_metrics"],
        "test_metrics": best["test_metrics"],
        "vsg_config": {
            "n_virtual_samples": N_VIRTUAL_SAMPLES,
            "medium_balance": True,
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
        "config": {
            "seed": SEED,
            "cv_splits": CV_SPLITS,
            "n_iter": N_ITER,
        },
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # Save test predictions
    pred_df = pd.DataFrame({"y_true": y_test.values})
    for r in results:
        pred_df[f"y_pred_{r['model']}"] = r["test_predictions"]
        pred_df[f"residual_{r['model']}"] = y_test.values - r["test_predictions"]
    pred_df["y_pred"] = best["test_predictions"]
    pred_df["residual"] = y_test.values - best["test_predictions"]

    test_meta = features.load_test_metadata()
    for col in test_meta.columns:
        pred_df[col] = test_meta[col].values

    pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    # Save report
    save_report(OUT_DIR / "report.txt", results, best, X_train_aug, X_val, X_test)

    print("\n" + "=" * 60)
    print("Pathway A (VSG) Training Complete")
    print("=" * 60)
    print(f"Best model: {best_name}")
    print(f"Val R²:  {best['val_metrics']['r2']:.4f}")
    print(f"Test R²: {best['test_metrics']['r2']:.4f}")
    print(f"Results saved to: {OUT_DIR}")


def save_report(report_path, results, best, X_train_aug, X_val, X_test):
    """Generate human-readable report."""
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PATHWAY A: TRAINING WITH KDE-BASED VIRTUAL SAMPLE GENERATION\n")
        f.write("=" * 70 + "\n\n")

        f.write("VSG CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Virtual samples generated: {N_VIRTUAL_SAMPLES}\n")
        f.write("Medium-aware balancing: Yes\n")
        f.write("Applied to: TRAIN only (val/test unchanged)\n\n")

        f.write("DATASET INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Augmented train samples: {len(X_train_aug)}\n")
        f.write(f"Val samples: {len(X_val)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Features: C#, Mw, HLB, EO, Conc, pH\n")
        f.write(f"Target: IE (Inhibition Efficiency %)\n\n")

        f.write("TRAINING APPROACH\n")
        f.write("-" * 70 + "\n")
        f.write("1. Apply KDE-based VSG to training set only\n")
        f.write("2. Tune RF and SVR on augmented TRAIN\n")
        f.write(f"   ({CV_SPLITS}-fold CV, {N_ITER} iterations)\n")
        f.write("3. Evaluate on original VAL, pick winner\n")
        f.write("4. Refit on augmented TRAIN+VAL\n")
        f.write("5. Final evaluation on original TEST set\n\n")

        f.write("MODEL COMPARISON (VALIDATION SET)\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"{'Model':<20} | {'Val R²':<10} | {'Val MAE':<10} | {'Val RMSE':<10}\n"
        )
        f.write("-" * 70 + "\n")
        for r in sorted(
            results, key=lambda x: x["val_metrics"]["r2"], reverse=True
        ):
            m = r["val_metrics"]
            f.write(
                f"{r['model']:<20} | {m['r2']:<10.3f} | {m['mae']:<10.2f} | {m['rmse']:<10.2f}\n"
            )

        f.write("\nMODEL COMPARISON (TEST SET)\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"{'Model':<20} | {'Test R²':<10} | {'Test MAE':<10} | {'Test RMSE':<10}\n"
        )
        f.write("-" * 70 + "\n")
        for r in sorted(
            results, key=lambda x: x["test_metrics"]["r2"], reverse=True
        ):
            m = r["test_metrics"]
            f.write(
                f"{r['model']:<20} | {m['r2']:<10.3f} | {m['mae']:<10.2f} | {m['rmse']:<10.2f}\n"
            )

        f.write("\nBEST HYPERPARAMETERS\n")
        f.write("-" * 70 + "\n")
        for param, value in best["best_params"].items():
            f.write(f"  {param}: {value}\n")


if __name__ == "__main__":
    main()
