"""
Baseline feature importance analysis for corrosion inhibitor modeling.

This script demonstrates how to:
1. Load the canonical train/validation/test splits via ``features.load_splits``.
2. Fit simple baseline models (linear regression, random forest).
3. Compute feature importance scores (coefficients, tree-based importances,
   permutation importance).
4. Report the results to the console and optionally save them as CSV files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

import features
from preprocessing import FEATURE_COLUMNS


def _project_root() -> Path:
  current = Path(__file__).resolve().parent
  if (current / "data").exists():
    return current
  return current.parent


def fit_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
  model = LinearRegression()
  model.fit(X_train, y_train)
  return model


def fit_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
  model = RandomForestRegressor(
      n_estimators=300,
      random_state=0,
      n_jobs=-1,
  )
  model.fit(X_train, y_train)
  return model


def normalize_importance(values: np.ndarray) -> np.ndarray:
  values = np.abs(values)
  max_value = values.max(initial=0.0)
  if max_value == 0:
    return np.zeros_like(values)
  return values / max_value


def compute_linear_importance(model: LinearRegression) -> pd.Series:
  return pd.Series(normalize_importance(model.coef_), index=FEATURE_COLUMNS, name="LinearRegression")


def compute_rf_importance(model: RandomForestRegressor) -> pd.Series:
  return pd.Series(
      normalize_importance(model.feature_importances_),
      index=FEATURE_COLUMNS,
      name="RandomForest",
  )


def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    scoring="r2",
    n_repeats: int = 20,
) -> pd.Series:
  perm = permutation_importance(
      model,
      X,
      y,
      n_repeats=n_repeats,
      random_state=0,
      n_jobs=-1,
      scoring=scoring,
  )
  return pd.Series(
      normalize_importance(perm.importances_mean),
      index=FEATURE_COLUMNS,
      name="PermutationImportance",
  )


def summarize_importances(importances: Tuple[pd.Series, ...]) -> pd.DataFrame:
  combined = pd.concat(importances, axis=1)
  combined["Mean"] = combined.mean(axis=1)
  combined.sort_values("Mean", ascending=False, inplace=True)
  return combined


def save_results(df: pd.DataFrame, root: Path) -> Path:
  out_dir = root / "data" / "feature_importance"
  out_dir.mkdir(parents=True, exist_ok=True)
  output_path = out_dir / "baseline_importance.csv"
  df.to_csv(output_path)
  return output_path


def main() -> None:
  (X_train, y_train), (X_val, y_val), (X_test, y_test) = features.load_splits()

  linear_model = fit_linear_regression(X_train, y_train)
  rf_model = fit_random_forest(X_train, y_train)

  lin_importance = compute_linear_importance(linear_model)
  rf_importance = compute_rf_importance(rf_model)
  perm_importance = compute_permutation_importance(rf_model, X_val, y_val)

  summary = summarize_importances((lin_importance, rf_importance, perm_importance))

  root = _project_root()
  out_path = save_results(summary, root)

  print("Baseline feature importance summary:")
  print(summary.to_string(float_format=lambda v: f"{v:0.3f}"))
  print(f"\nSaved detailed results to {out_path}")

  for name, model in [("LinearRegression", linear_model), ("RandomForest", rf_model)]:
    r2_val = r2_score(y_val, model.predict(X_val))
    r2_test = r2_score(y_test, model.predict(X_test))
    print(f"{name} R^2 -> validation: {r2_val:0.3f}, test: {r2_test:0.3f}")


if __name__ == "__main__":
  main()

