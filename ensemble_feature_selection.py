"""
Pathway D: Ensemble Feature Selection using 5 algorithms.

Ranks features using 5 independent methods, normalizes each to [0, 1],
and aggregates into a composite score. Selection uses TRAIN data only.

Methods:
  1. Pearson Correlation (absolute value with target)
  2. Mutual Information (nonlinear dependency)
  3. F-test (statistical significance)
  4. Recursive Feature Elimination (RFE with Gradient Boosting)
  5. Permutation Feature Importance (PFI with Gradient Boosting)

Reference: Tale Masoule et al. (2025) used 5-algorithm composite scoring.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import (
    f_regression,
    mutual_info_regression,
    RFE,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler


def _clean_and_scale(X: pd.DataFrame, y: pd.Series):
    """Handle inf/nan, impute, clip, scale. Returns (X_scaled, valid_cols)."""
    X_clean = X.replace([np.inf, -np.inf], np.nan)
    valid_cols = X_clean.columns[X_clean.notna().any()]
    X_clean = X_clean[valid_cols]

    imputer = SimpleImputer(strategy="median")
    X_imp = np.clip(imputer.fit_transform(X_clean), -1e10, 1e10)

    scaler = StandardScaler()
    X_sc = np.nan_to_num(scaler.fit_transform(X_imp), nan=0.0, posinf=0.0, neginf=0.0)

    return X_sc, list(valid_cols)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1]. Handles edge cases."""
    scores = np.nan_to_num(scores, nan=0.0)
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def rank_pearson(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlation with target."""
    scores = np.array([abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])])
    return np.nan_to_num(scores, nan=0.0)


def rank_mutual_info(X: np.ndarray, y: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Mutual information regression scores."""
    return mutual_info_regression(X, y, random_state=random_state)


def rank_ftest(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """F-test regression scores."""
    f_scores, _ = f_regression(X, y)
    return np.nan_to_num(f_scores, nan=0.0)


def rank_rfe(X: np.ndarray, y: np.ndarray, random_state: int = 0) -> np.ndarray:
    """RFE ranking (lower rank = more important). Converted to importance scores."""
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=random_state,
    )
    rfe = RFE(model, n_features_to_select=1, step=1)
    rfe.fit(X, y)
    # Convert ranking (1=best) to scores (higher=better)
    return 1.0 / rfe.ranking_.astype(float)


def rank_pfi(X: np.ndarray, y: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Permutation Feature Importance with Gradient Boosting."""
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=5, random_state=random_state,
    )
    model.fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=10, random_state=random_state)
    scores = result.importances_mean
    return np.clip(scores, 0, None)  # Negative importances → 0


def ensemble_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_top: int = 10,
    random_state: int = 0,
    verbose: bool = True,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select top N features using 5-algorithm ensemble.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (RDKit descriptors only, not original 6).
    y_train : pd.Series
        Training target.
    n_top : int
        Number of features to select.
    random_state : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    selected : list of str
        Top N feature names.
    ranking_df : pd.DataFrame
        Full ranking with all scores.
    """
    X_sc, valid_cols = _clean_and_scale(X_train, y_train)
    y = y_train.values
    n_features = len(valid_cols)

    if verbose:
        print(f"    Ensemble feature selection on {n_features} features...")

    # 1. Pearson
    if verbose:
        print(f"      [1/5] Pearson correlation...")
    pearson = _normalize_scores(rank_pearson(X_sc, y))

    # 2. Mutual Information
    if verbose:
        print(f"      [2/5] Mutual information...")
    mi = _normalize_scores(rank_mutual_info(X_sc, y, random_state))

    # 3. F-test
    if verbose:
        print(f"      [3/5] F-test...")
    ftest = _normalize_scores(rank_ftest(X_sc, y))

    # 4. RFE
    if verbose:
        print(f"      [4/5] Recursive Feature Elimination...")
    rfe = _normalize_scores(rank_rfe(X_sc, y, random_state))

    # 5. PFI
    if verbose:
        print(f"      [5/5] Permutation Feature Importance...")
    pfi = _normalize_scores(rank_pfi(X_sc, y, random_state))

    # Composite score (equal weights)
    composite = (pearson + mi + ftest + rfe + pfi) / 5.0

    ranking_df = pd.DataFrame({
        "feature": valid_cols,
        "pearson": pearson,
        "mutual_info": mi,
        "f_test": ftest,
        "rfe": rfe,
        "pfi": pfi,
        "composite": composite,
    }).sort_values("composite", ascending=False).reset_index(drop=True)

    selected = ranking_df["feature"].head(n_top).tolist()

    if verbose:
        print(f"    Top {n_top} by composite score:")
        for i, row in ranking_df.head(n_top).iterrows():
            print(f"      {i+1:>2}. {row['feature']:<25} composite={row['composite']:.4f}")

    return selected, ranking_df


def select_top_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    n_top: int = 10,
    random_state: int = 0,
) -> List[str]:
    """Convenience wrapper returning just the feature names."""
    selected, _ = ensemble_feature_selection(X, y, n_top, random_state)
    return selected


if __name__ == "__main__":
    """Demo on original dataset RDKit descriptors."""
    import json
    from pathlib import Path
    from rdkit_descriptors import RDKIT_AVAILABLE

    ROOT = Path(__file__).resolve().parent
    cache_dir = ROOT / "data" / "processed" / "rdkit_cache"

    if not (cache_dir / "all_rdkit_descriptors.csv").exists():
        print("RDKit cache not found. Run feature_selection_rdkit.py first.")
        exit(1)

    df = pd.read_csv(cache_dir / "all_rdkit_descriptors.csv")
    with open(cache_dir / "descriptor_names.json") as f:
        desc_names = json.load(f)

    from preprocessing import (
        PreprocessingConfig, load_raw_dataset, clean_dataset,
        infer_feature_columns, split_dataset,
    )
    config = PreprocessingConfig(dataset_path=str(ROOT / "dataset.csv"), random_state=0)
    raw = load_raw_dataset(config.dataset_path)
    cleaned = clean_dataset(raw, config)
    feat_cols = list(infer_feature_columns(cleaned, config))
    target = cleaned["IE"]
    splits = split_dataset(cleaned[feat_cols], target, config)
    train_idx = splits["X_train"].index

    selected, ranking = ensemble_feature_selection(
        df.loc[train_idx, desc_names], target.loc[train_idx], n_top=20,
    )

    out_dir = ROOT / "data" / "feature_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out_dir / "ensemble_ranking.csv", index=False)
    print(f"\nSaved ranking to {out_dir / 'ensemble_ranking.csv'}")
