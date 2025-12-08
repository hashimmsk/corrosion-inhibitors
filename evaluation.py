"""
Step 8: Model evaluation & interpretation.

Generates residual plots, learning curves, permutation importance, and partial
dependence plots for the best-trained model artifact. Outputs are written to
data/evaluation/.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve

import features


def _project_root() -> Path:
    current = Path(__file__).resolve().parent
    if (current / "data").exists():
        return current
    return current.parent


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _ensure_dirs(root: Path) -> Path:
    out_dir = root / "data" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.7, edgecolor="k")
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Predicted IE")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_learning_curve(model, X, y, out_path: Path, cv: int = 5) -> None:
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring="r2",
        n_jobs=1,
        train_sizes=np.linspace(0.2, 1.0, 5),
        shuffle=True,
        random_state=0,
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_mean, "o-", label="Train R²")
    plt.plot(train_sizes, val_mean, "o-", label="CV R²")
    plt.xlabel("Training examples")
    plt.ylabel("R²")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_permutation_importance(model, X: pd.DataFrame, y: pd.Series, out_path: Path, n_repeats: int = 20) -> None:
    perm = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=0,
        n_jobs=1,
        scoring="r2",
    )
    importances = pd.Series(perm.importances_mean, index=X.columns)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(6, 4))
    plt.barh(importances.index, importances.values, color="tab:blue")
    plt.xlabel("Permutation importance (mean ΔR²)")
    plt.title("Permutation Importance (validation)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_partial_dependence(model, X: pd.DataFrame, features_to_plot: List[str], out_dir: Path) -> None:
    for feat in features_to_plot:
        if feat not in X.columns:
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        PartialDependenceDisplay.from_estimator(
            model,
            X,
            [feat],
            ax=ax,
            grid_resolution=30,
        )
        ax.set_title(f"Partial Dependence: {feat}")
        plt.tight_layout()
        fig.savefig(out_dir / f"pdp_{feat}.png", dpi=300)
        plt.close(fig)


def main() -> None:
    root = _project_root()
    out_dir = _ensure_dirs(root)

    # Load data splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = features.load_splits()

    # Load best model artifact
    model_path = root / "data" / "models" / "random_forest_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Best model artifact not found at {model_path}")
    with model_path.open("rb") as f:
        model = pickle.load(f)

    # Fit learning curve uses a fresh clone; we can reuse the loaded model type
    # but to preserve the fitted model for eval metrics, keep separate.
    # We assume the artifact is the best model already fitted on train+val.

    # Metrics on val/test
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    val_metrics = _metrics(y_val, val_pred)
    test_metrics = _metrics(y_test, test_pred)

    metrics_out = {
        "val": val_metrics,
        "test": test_metrics,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    # Residual plots
    plot_residuals(y_val, val_pred, "Residuals (Validation)", out_dir / "residuals_val.png")
    plot_residuals(y_test, test_pred, "Residuals (Test)", out_dir / "residuals_test.png")

    # Learning curve on train split
    plot_learning_curve(model, X_train, y_train, out_dir / "learning_curve.png", cv=5)

    # Permutation importance on validation
    plot_permutation_importance(model, X_val, y_val, out_dir / "permutation_importance_val.png", n_repeats=20)

    # Partial dependence on top drivers
    pd_features = [feat for feat in ["pH", "Conc", "Mw", "EO"] if feat in X_train.columns]
    plot_partial_dependence(model, X_train, pd_features, out_dir)

    print("Evaluation artifacts saved to:", out_dir)


if __name__ == "__main__":
    main()

