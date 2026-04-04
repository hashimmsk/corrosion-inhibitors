"""
Benchmark with Ensemble Feature Selection (Pathway D).

Same structure as train_all_models.py but uses 5-algorithm ensemble
for RDKit descriptor selection instead of PFI-only.

Runs all 10 models x 6 configs x 2 datasets.
Outputs: data/models/benchmark_ensemble/
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb

from ensemble_feature_selection import select_top_ensemble

warnings.filterwarnings("ignore")

SEED = 0
CV_SPLITS = 5
N_ITER = 50
ROOT = Path(__file__).resolve().parent

FEATURE_COLUMNS = ["C#", "Mw", "HLB", "EO", "Conc", "pH"]
LABEL_COLUMN = "IE"


def build_models():
    return {
        "random_forest": (
            RandomForestRegressor(random_state=SEED, n_jobs=1),
            {"n_estimators": [200, 400, 600], "max_depth": [4, 6, 8, None], "min_samples_leaf": [1, 2, 5]},
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(random_state=SEED),
            {"n_estimators": [100, 200, 400], "max_depth": [3, 4, 5], "learning_rate": [0.01, 0.05, 0.1], "subsample": [0.8, 1.0]},
        ),
        "xgboost": (
            xgb.XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0),
            {"n_estimators": [100, 200, 400], "max_depth": [3, 4, 6], "learning_rate": [0.01, 0.05, 0.1], "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]},
        ),
        "lightgbm": (
            lgb.LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1),
            {"n_estimators": [100, 200, 400], "max_depth": [3, 5, 7, -1], "learning_rate": [0.01, 0.05, 0.1], "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]},
        ),
        "extra_trees": (
            ExtraTreesRegressor(random_state=SEED, n_jobs=1),
            {"n_estimators": [200, 400, 600], "max_depth": [4, 6, 8, None], "min_samples_leaf": [1, 2, 5]},
        ),
        "adaboost": (
            AdaBoostRegressor(random_state=SEED),
            {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.5, 1.0], "loss": ["linear", "square", "exponential"]},
        ),
        "bagging": (
            BaggingRegressor(random_state=SEED, n_jobs=1),
            {"n_estimators": [50, 100, 200], "max_samples": [0.5, 0.7, 1.0], "max_features": [0.5, 0.7, 1.0]},
        ),
        "svr": (
            SVR(kernel="rbf", cache_size=2000),
            {"C": [1.0, 10.0, 50.0, 100.0], "gamma": ["scale", "auto", 0.01, 0.1], "epsilon": [0.01, 0.05, 0.1]},
        ),
        "elasticnet": (
            ElasticNet(random_state=SEED, max_iter=5000),
            {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        ),
        "ridge": (
            Ridge(random_state=SEED),
            {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
    }


def get_metrics(y_true, y_pred):
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
    }


def run_experiment(name, X_train, y_train, X_val, y_val, X_test, y_test, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    models = build_models()
    results = []

    print(f"\n  {name} | Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)} Feat:{X_train.shape[1]}")
    print(f"  {'Model':<20} {'Val R²':>8} {'Test R²':>8} {'RMSE':>8}")
    print(f"  {'-'*48}")

    with parallel_backend("threading"):
        for model_name, (estimator, param_space) in models.items():
            n_combos = np.prod([len(v) for v in param_space.values()])
            search = RandomizedSearchCV(
                estimator=estimator, param_distributions=param_space,
                n_iter=min(N_ITER, int(n_combos)), scoring="r2",
                cv=cv, random_state=SEED, n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            val_m = get_metrics(y_val, best.predict(X_val))

            X_tv = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_val)], ignore_index=True)
            y_tv = pd.concat([pd.Series(y_train.values), pd.Series(y_val.values)], ignore_index=True)
            best.fit(X_tv, y_tv)
            test_m = get_metrics(y_test, best.predict(X_test))

            results.append({
                "model": model_name, "val_metrics": val_m, "test_metrics": test_m,
                "best_params": {k: (str(v) if not isinstance(v, (int, float, bool)) else v)
                                for k, v in search.best_params_.items()},
            })
            print(f"  {model_name:<20} {val_m['r2']:>8.4f} {test_m['r2']:>8.4f} {test_m['rmse']:>8.2f}")

    (out_dir / "results.json").write_text(json.dumps({"config": name, "all_models": results}, indent=2))
    return results


def generate_vsg(X_train, y_train, medium, n_samples=200, feature_cols=None):
    if feature_cols is None:
        feature_cols = list(X_train.columns)
    X = X_train[feature_cols].values
    y = y_train.values

    if medium is not None:
        mediums = medium.unique()
        n_per = max(1, n_samples // len(mediums))
        X_list, y_list = [], []
        for m in mediums:
            mask = medium.values == m
            X_m, y_m = X[mask], y[mask]
            if len(X_m) < 2:
                continue
            data = np.hstack([X_m, y_m.reshape(-1, 1)])
            kde = KernelDensity(kernel="gaussian", bandwidth="scott")
            kde.fit(data)
            samples = kde.sample(int(n_per * 1.5), random_state=SEED)
            X_list.append(np.clip(samples[:n_per, :-1], X_m.min(0), X_m.max(0)))
            y_list.append(np.clip(samples[:n_per, -1], y_m.min(), y_m.max()))
        X_virt, y_virt = np.vstack(X_list), np.concatenate(y_list)
    else:
        data = np.hstack([X, y.reshape(-1, 1)])
        kde = KernelDensity(kernel="gaussian", bandwidth="scott")
        kde.fit(data)
        samples = kde.sample(int(n_samples * 1.5), random_state=SEED)
        X_virt = np.clip(samples[:n_samples, :-1], X.min(0), X.max(0))
        y_virt = np.clip(samples[:n_samples, -1], y.min(), y.max())

    return (pd.DataFrame(np.vstack([X, X_virt]), columns=feature_cols),
            pd.Series(np.concatenate([y, y_virt])))


def build_extended_ensemble(df, train_idx, val_idx, test_idx, n_top, cache_label):
    """Build extended features using ENSEMBLE selection (not PFI-only)."""
    from rdkit_descriptors import add_rdkit_descriptors

    cache_dir = ROOT / "data" / "processed" / f"rdkit_cache_{cache_label}"
    cache_file = cache_dir / "all_descriptors.csv"
    meta_file = cache_dir / "desc_names.json"

    if cache_file.exists() and meta_file.exists():
        df_ext = pd.read_csv(cache_file)
        with open(meta_file) as f:
            desc_names = json.load(f)
    else:
        print(f"    Computing RDKit descriptors for {cache_label}...")
        df_ext, desc_names = add_rdkit_descriptors(df)
        cache_dir.mkdir(parents=True, exist_ok=True)
        df_ext.to_csv(cache_file, index=False)
        with open(meta_file, "w") as f:
            json.dump(desc_names, f)

    target = df[LABEL_COLUMN]

    # ENSEMBLE selection instead of PFI-only
    top_desc = select_top_ensemble(
        df_ext.loc[train_idx, desc_names], target.loc[train_idx],
        n_top=n_top, random_state=SEED,
    )
    ext_cols = FEATURE_COLUMNS + top_desc

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr = np.clip(imputer.fit_transform(df_ext.loc[train_idx, ext_cols]), -1e10, 1e10)
    X_va = np.clip(imputer.transform(df_ext.loc[val_idx, ext_cols]), -1e10, 1e10)
    X_te = np.clip(imputer.transform(df_ext.loc[test_idx, ext_cols]), -1e10, 1e10)

    X_tr_sc = pd.DataFrame(np.nan_to_num(scaler.fit_transform(X_tr)), columns=ext_cols, index=train_idx)
    X_va_sc = pd.DataFrame(np.nan_to_num(scaler.transform(X_va)), columns=ext_cols, index=val_idx)
    X_te_sc = pd.DataFrame(np.nan_to_num(scaler.transform(X_te)), columns=ext_cols, index=test_idx)

    medium = df["medium"].loc[train_idx] if "medium" in df.columns else None
    return X_tr_sc, target.loc[train_idx], X_va_sc, target.loc[val_idx], X_te_sc, target.loc[test_idx], medium, ext_cols, top_desc


def load_original():
    from preprocessing import (
        PreprocessingConfig, load_raw_dataset, clean_dataset,
        infer_feature_columns, split_dataset,
    )
    config = PreprocessingConfig(dataset_path=str(ROOT / "dataset.csv"), random_state=SEED)
    raw = load_raw_dataset(config.dataset_path)
    cleaned = clean_dataset(raw, config)
    feat_cols = list(infer_feature_columns(cleaned, config))
    target = cleaned[LABEL_COLUMN]
    splits = split_dataset(cleaned[feat_cols], target, config)
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(splits["X_train"])),
                        columns=feat_cols, index=splits["X_train"].index)
    X_va = pd.DataFrame(scaler.transform(imputer.transform(splits["X_val"])),
                        columns=feat_cols, index=splits["X_val"].index)
    X_te = pd.DataFrame(scaler.transform(imputer.transform(splits["X_test"])),
                        columns=feat_cols, index=splits["X_test"].index)
    medium = cleaned["medium"].loc[splits["X_train"].index] if "medium" in cleaned.columns else None
    return cleaned, X_tr, splits["y_train"], X_va, splits["y_val"], X_te, splits["y_test"], medium


def load_imputed():
    df = pd.read_excel(ROOT / "data_imputed.xlsx")
    df = df.rename(columns={"Mw (g/mol)": "Mw", "ph": "pH", "S%": "Conc", "liquid": "medium", "AA (%)": "AA"})
    df_clean = df[FEATURE_COLUMNS + [LABEL_COLUMN] + (["medium"] if "medium" in df.columns else [])].copy()
    df_clean = df_clean.dropna(subset=[LABEL_COLUMN]).reset_index(drop=True)
    X = df_clean[FEATURE_COLUMNS]
    y = df_clean[LABEL_COLUMN]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(X_tr)), columns=FEATURE_COLUMNS, index=X_tr.index)
    X_va_sc = pd.DataFrame(scaler.transform(imputer.transform(X_va)), columns=FEATURE_COLUMNS, index=X_va.index)
    X_te_sc = pd.DataFrame(scaler.transform(imputer.transform(X_te)), columns=FEATURE_COLUMNS, index=X_te.index)
    medium = df_clean["medium"].loc[X_tr.index] if "medium" in df_clean.columns else None
    return df_clean, X_tr_sc, y_tr, X_va_sc, y_va, X_te_sc, y_te, medium


def run_all_configs(dataset_name, df_clean, X_tr, y_tr, X_va, y_va, X_te, y_te, medium, out_base):
    all_results = {}

    print(f"\n{'#'*60}")
    print(f"  DATASET: {dataset_name} (Ensemble Feature Selection)")
    print(f"{'#'*60}")

    train_idx, val_idx, test_idx = X_tr.index, X_va.index, X_te.index
    cache_label = dataset_name.lower()

    # 1. Baseline (no feature selection needed)
    res = run_experiment("Baseline (6 feat)", X_tr, y_tr, X_va, y_va, X_te, y_te, out_base / "baseline")
    all_results["Baseline"] = res

    # 2. VSG
    X_aug, y_aug = generate_vsg(X_tr, y_tr, medium, n_samples=200)
    res = run_experiment("VSG (6 feat, 200)", X_aug, y_aug, X_va, y_va, X_te, y_te, out_base / "vsg")
    all_results["VSG"] = res

    # 3. Extended top 10 (Ensemble)
    print("\n  Building extended top 10 (Ensemble FS)...")
    X_e10_tr, y_e10_tr, X_e10_va, y_e10_va, X_e10_te, y_e10_te, med_e10, cols_10, top10 = \
        build_extended_ensemble(df_clean, train_idx, val_idx, test_idx, n_top=10, cache_label=cache_label)
    res = run_experiment("EnsFS top10 (16 feat)", X_e10_tr, y_e10_tr, X_e10_va, y_e10_va, X_e10_te, y_e10_te,
                         out_base / "ensfs_top10")
    all_results["EnsFS top10"] = res

    # 4. Extended top 10 + VSG
    X_e10_aug, y_e10_aug = generate_vsg(X_e10_tr, y_e10_tr, med_e10, n_samples=200, feature_cols=cols_10)
    res = run_experiment("EnsFS top10+VSG", X_e10_aug, y_e10_aug, X_e10_va, y_e10_va, X_e10_te, y_e10_te,
                         out_base / "ensfs_top10_vsg")
    all_results["EnsFS top10+VSG"] = res

    # 5. Extended top 20 (Ensemble)
    print("\n  Building extended top 20 (Ensemble FS)...")
    X_e20_tr, y_e20_tr, X_e20_va, y_e20_va, X_e20_te, y_e20_te, med_e20, cols_20, top20 = \
        build_extended_ensemble(df_clean, train_idx, val_idx, test_idx, n_top=20, cache_label=cache_label)
    res = run_experiment("EnsFS top20 (26 feat)", X_e20_tr, y_e20_tr, X_e20_va, y_e20_va, X_e20_te, y_e20_te,
                         out_base / "ensfs_top20")
    all_results["EnsFS top20"] = res

    # 6. Extended top 20 + VSG
    X_e20_aug, y_e20_aug = generate_vsg(X_e20_tr, y_e20_tr, med_e20, n_samples=200, feature_cols=cols_20)
    res = run_experiment("EnsFS top20+VSG", X_e20_aug, y_e20_aug, X_e20_va, y_e20_va, X_e20_te, y_e20_te,
                         out_base / "ensfs_top20_vsg")
    all_results["EnsFS top20+VSG"] = res

    # Save selected features
    (out_base / "selected_features.json").write_text(json.dumps(
        {"ensemble_top10": top10, "ensemble_top20": top20}, indent=2))

    return all_results


def print_summary_table(dataset_name, all_results):
    print(f"\n{'='*80}")
    print(f"  {dataset_name} - ENSEMBLE FS RESULTS")
    print(f"{'='*80}")
    print(f"  {'Configuration':<20} {'Model':<20} {'Val R²':>8} {'Test R²':>8} {'RMSE':>8}")
    print(f"  {'-'*68}")

    best_r2 = -999
    best_label = ""
    for config_name, results in all_results.items():
        for r in sorted(results, key=lambda x: x["test_metrics"]["r2"], reverse=True):
            vm, tm = r["val_metrics"], r["test_metrics"]
            if tm["r2"] > best_r2:
                best_r2 = tm["r2"]
                best_label = f"{config_name} / {r['model']}"
            print(f"  {config_name:<20} {r['model']:<20} {vm['r2']:>8.4f} {tm['r2']:>8.4f} {tm['rmse']:>8.2f}")
        config_name = ""

    print(f"  {'-'*68}")
    print(f"  BEST: {best_label} → Test R² = {best_r2:.4f}")
    print(f"{'='*80}")
    return best_r2, best_label


def main():
    out_root = ROOT / "data" / "models" / "benchmark_ensemble"

    # Original dataset
    df_orig, X_tr_o, y_tr_o, X_va_o, y_va_o, X_te_o, y_te_o, med_o = load_original()
    res_orig = run_all_configs("ORIGINAL", df_orig, X_tr_o, y_tr_o, X_va_o, y_va_o, X_te_o, y_te_o, med_o,
                               out_root / "original")
    best_orig_r2, best_orig = print_summary_table("ORIGINAL DATASET", res_orig)

    # Imputed dataset
    df_imp, X_tr_i, y_tr_i, X_va_i, y_va_i, X_te_i, y_te_i, med_i = load_imputed()
    res_imp = run_all_configs("IMPUTED", df_imp, X_tr_i, y_tr_i, X_va_i, y_va_i, X_te_i, y_te_i, med_i,
                              out_root / "imputed")
    best_imp_r2, best_imp = print_summary_table("IMPUTED DATASET", res_imp)

    # Save CSV
    def flatten(ds, results):
        rows = []
        for cfg, res_list in results.items():
            for r in res_list:
                rows.append({"dataset": ds, "config": cfg, "model": r["model"],
                             "val_r2": r["val_metrics"]["r2"], "test_r2": r["test_metrics"]["r2"],
                             "test_rmse": r["test_metrics"]["rmse"], "test_mae": r["test_metrics"]["mae"]})
        return rows

    all_rows = flatten("original", res_orig) + flatten("imputed", res_imp)
    pd.DataFrame(all_rows).to_csv(out_root / "full_benchmark_ensemble.csv", index=False)

    # Compare with PFI-only
    print(f"\n{'='*60}")
    print("  COMPARISON: PFI-only vs Ensemble FS")
    print(f"{'='*60}")
    print(f"  Original: PFI best = 0.5321 (LightGBM)  |  Ensemble best = {best_orig_r2:.4f} ({best_orig})")
    print(f"  Imputed:  PFI best = 0.4829 (LightGBM)  |  Ensemble best = {best_imp_r2:.4f} ({best_imp})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
