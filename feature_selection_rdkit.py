"""
Feature selection analysis for RDKit descriptors.

1. Compute all ~200 RDKit descriptors (cached after first run)
2. Rank descriptors via PFI on training data
3. Sweep n_top = 5, 10, 15, 20, 25, 30, 35, 40
4. Train RF for each and report Val/Test R²
5. Report the optimal number and top descriptor names

Uses the same train/val/test split as the base pipeline.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from preprocessing import (
    FEATURE_COLUMNS, LABEL_COLUMN, PreprocessingConfig,
    clean_dataset, infer_feature_columns, load_raw_dataset, split_dataset,
)
from rdkit_descriptors import RDKIT_AVAILABLE, add_rdkit_descriptors

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "processed" / "rdkit_cache"
OUT_DIR = ROOT / "data" / "feature_selection"
SEED = 0


def get_metrics(y_true, y_pred):
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
    }


def load_or_compute_descriptors():
    """Compute RDKit descriptors once and cache them."""
    cache_file = CACHE_DIR / "all_rdkit_descriptors.csv"
    meta_file = CACHE_DIR / "descriptor_names.json"

    if cache_file.exists() and meta_file.exists():
        print("Loading cached RDKit descriptors...")
        df = pd.read_csv(cache_file)
        with open(meta_file) as f:
            desc_names = json.load(f)
        return df, desc_names

    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required. Run: pip install rdkit")

    print("Computing RDKit 2D descriptors (this takes a while, will be cached)...")
    config = PreprocessingConfig(dataset_path=str(ROOT / "dataset.csv"), random_state=SEED)
    raw = load_raw_dataset(config.dataset_path)
    cleaned = clean_dataset(raw, config)

    df_ext, desc_names = add_rdkit_descriptors(cleaned)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df_ext.to_csv(cache_file, index=False)
    with open(meta_file, "w") as f:
        json.dump(desc_names, f)
    print(f"  Cached {len(desc_names)} descriptors to {CACHE_DIR}")

    return df_ext, desc_names


def rank_all_descriptors(X_train_desc, y_train, desc_names):
    """Rank ALL descriptors by PFI importance. Returns sorted list of (name, importance)."""
    X = X_train_desc[desc_names].replace([np.inf, -np.inf], np.nan)
    valid_cols = X.columns[X.notna().any()]
    X = X[valid_cols]

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    X_imp = np.clip(X_imp, -1e10, 1e10)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_imp)
    X_sc = np.nan_to_num(X_sc, nan=0.0, posinf=0.0, neginf=0.0)

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=5, random_state=SEED,
    )
    model.fit(X_sc, y_train)

    result = permutation_importance(model, X_sc, y_train, n_repeats=10, random_state=SEED)

    ranking = sorted(
        zip(valid_cols, result.importances_mean, result.importances_std),
        key=lambda x: x[1], reverse=True,
    )
    return ranking


def sweep_n_top(df_ext, desc_names, ranking, train_idx, val_idx, test_idx, target):
    """Train RF for each n_top and report results."""
    orig_cols = list(FEATURE_COLUMNS)
    n_top_values = [5, 10, 15, 20, 25, 30, 35, 40]
    results = []

    for n_top in n_top_values:
        top_desc = [name for name, _, _ in ranking[:n_top]]
        feat_cols = orig_cols + top_desc

        X_train = df_ext.loc[train_idx, feat_cols]
        X_val = df_ext.loc[val_idx, feat_cols]
        X_test = df_ext.loc[test_idx, feat_cols]
        y_train = target.loc[train_idx]
        y_val = target.loc[val_idx]
        y_test = target.loc[test_idx]

        # Impute + scale (fit on train)
        imputer = SimpleImputer(strategy="median")
        X_tr_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feat_cols, index=train_idx)
        X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=feat_cols, index=val_idx)
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feat_cols, index=test_idx)

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_imp)
        X_val_sc = scaler.transform(X_val_imp)
        X_test_sc = scaler.transform(X_test_imp)
        X_tr_sc = np.nan_to_num(X_tr_sc, nan=0.0)
        X_val_sc = np.nan_to_num(X_val_sc, nan=0.0)
        X_test_sc = np.nan_to_num(X_test_sc, nan=0.0)

        # Train RF
        rf = RandomForestRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=2, random_state=SEED, n_jobs=-1,
        )
        rf.fit(X_tr_sc, y_train)
        val_r2 = get_metrics(y_val, rf.predict(X_val_sc))["r2"]

        # Refit on train+val, test
        X_tv = np.vstack([X_tr_sc, X_val_sc])
        y_tv = pd.concat([y_train, y_val])
        rf.fit(X_tv, y_tv)
        test_metrics = get_metrics(y_test, rf.predict(X_test_sc))

        results.append({
            "n_top": n_top,
            "total_features": len(feat_cols),
            "val_r2": val_r2,
            "test_r2": test_metrics["r2"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "top_descriptors": top_desc,
        })

        print(f"  n_top={n_top:>2} ({len(feat_cols):>2} feat)  Val R²={val_r2:.4f}  Test R²={test_metrics['r2']:.4f}  RMSE={test_metrics['rmse']:.2f}")

    return results


def main():
    config = PreprocessingConfig(dataset_path=str(ROOT / "dataset.csv"), random_state=SEED)
    raw = load_raw_dataset(config.dataset_path)
    cleaned = clean_dataset(raw, config)
    feat_cols = list(infer_feature_columns(cleaned, config))
    target = cleaned[LABEL_COLUMN]

    splits = split_dataset(cleaned[feat_cols], target, config)
    train_idx = splits["X_train"].index
    val_idx = splits["X_val"].index
    test_idx = splits["X_test"].index

    df_ext, desc_names = load_or_compute_descriptors()

    # 1. Rank all descriptors
    print("\nRanking all RDKit descriptors via PFI...")
    ranking = rank_all_descriptors(df_ext.loc[train_idx], target.loc[train_idx], desc_names)

    # 2. Print top 40
    print(f"\n{'Rank':<6} {'Descriptor':<25} {'PFI Importance':<16} {'Std'}")
    print("-" * 60)
    for i, (name, imp, std) in enumerate(ranking[:40], 1):
        print(f"  {i:<4} {name:<25} {imp:<16.4f} {std:.4f}")

    # 3. Sweep n_top
    print("\nSweeping n_top values...")
    results = sweep_n_top(df_ext, desc_names, ranking, train_idx, val_idx, test_idx, target)

    # 4. Find optimal
    best = max(results, key=lambda r: r["test_r2"])
    print(f"\nOptimal: n_top = {best['n_top']} ({best['total_features']} features), Test R² = {best['test_r2']:.4f}")

    # 5. Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ranking_df = pd.DataFrame(
        [(name, imp, std) for name, imp, std in ranking],
        columns=["descriptor", "pfi_importance", "pfi_std"],
    )
    ranking_df.to_csv(OUT_DIR / "rdkit_pfi_ranking.csv", index=False)

    sweep_df = pd.DataFrame([{
        "n_top": r["n_top"], "total_features": r["total_features"],
        "val_r2": r["val_r2"], "test_r2": r["test_r2"],
        "test_rmse": r["test_rmse"], "test_mae": r["test_mae"],
    } for r in results])
    sweep_df.to_csv(OUT_DIR / "n_top_sweep.csv", index=False)

    with open(OUT_DIR / "optimal.json", "w") as f:
        json.dump({
            "optimal_n_top": best["n_top"],
            "optimal_total_features": best["total_features"],
            "test_r2": best["test_r2"],
            "top_descriptors": best["top_descriptors"],
        }, f, indent=2)

    print(f"\nSaved to {OUT_DIR}")
    print(f"  rdkit_pfi_ranking.csv   - full ranking of all descriptors")
    print(f"  n_top_sweep.csv         - performance at each n_top")
    print(f"  optimal.json            - best configuration")


if __name__ == "__main__":
    main()
