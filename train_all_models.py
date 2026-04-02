"""
Expanded model benchmark on BOTH datasets (original + imputed).

Models (10):
  Ensemble:    Random Forest, Gradient Boosting, XGBoost, LightGBM,
               AdaBoost, Extra Trees, Bagging
  Kernel:      SVR (RBF)
  Linear:      ElasticNet, Ridge

Configurations per dataset:
  1. Baseline (6 features)
  2. VSG (6 feat, 200 samples)
  3. Extended top 10 (6 + 10 RDKit)
  4. Extended top 10 + VSG
  5. Extended top 20 (6 + 20 RDKit)
  6. Extended top 20 + VSG

Outputs: data/models/benchmark/{original,imputed}/
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

warnings.filterwarnings("ignore")

SEED = 0
CV_SPLITS = 5
N_ITER = 50
ROOT = Path(__file__).resolve().parent

FEATURE_COLUMNS = ["C#", "Mw", "HLB", "EO", "Conc", "pH"]
LABEL_COLUMN = "IE"


# ── Models ───────────────────────────────────────────────────

def build_models():
    return {
        "random_forest": (
            RandomForestRegressor(random_state=SEED, n_jobs=1),
            {
                "n_estimators": [200, 400, 600],
                "max_depth": [4, 6, 8, None],
                "min_samples_leaf": [1, 2, 5],
            },
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(random_state=SEED),
            {
                "n_estimators": [100, 200, 400],
                "max_depth": [3, 4, 5],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0],
            },
        ),
        "xgboost": (
            xgb.XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0),
            {
                "n_estimators": [100, 200, 400],
                "max_depth": [3, 4, 6],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        ),
        "lightgbm": (
            lgb.LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1),
            {
                "n_estimators": [100, 200, 400],
                "max_depth": [3, 5, 7, -1],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        ),
        "extra_trees": (
            ExtraTreesRegressor(random_state=SEED, n_jobs=1),
            {
                "n_estimators": [200, 400, 600],
                "max_depth": [4, 6, 8, None],
                "min_samples_leaf": [1, 2, 5],
            },
        ),
        "adaboost": (
            AdaBoostRegressor(random_state=SEED),
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.5, 1.0],
                "loss": ["linear", "square", "exponential"],
            },
        ),
        "bagging": (
            BaggingRegressor(random_state=SEED, n_jobs=1),
            {
                "n_estimators": [50, 100, 200],
                "max_samples": [0.5, 0.7, 1.0],
                "max_features": [0.5, 0.7, 1.0],
            },
        ),
        "svr": (
            SVR(kernel="rbf", cache_size=2000),
            {
                "C": [1.0, 10.0, 50.0, 100.0],
                "gamma": ["scale", "auto", 0.01, 0.1],
                "epsilon": [0.01, 0.05, 0.1],
            },
        ),
        "elasticnet": (
            ElasticNet(random_state=SEED, max_iter=5000),
            {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        ),
        "ridge": (
            Ridge(random_state=SEED),
            {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        ),
    }


# ── Metrics ──────────────────────────────────────────────────

def get_metrics(y_true, y_pred):
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
    }


# ── Experiment Runner ────────────────────────────────────────

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
                estimator=estimator,
                param_distributions=param_space,
                n_iter=min(N_ITER, int(n_combos)),
                scoring="r2",
                cv=cv,
                random_state=SEED,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            val_m = get_metrics(y_val, best.predict(X_val))

            # Refit on train+val
            X_tv = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_val)], ignore_index=True)
            y_tv = pd.concat([pd.Series(y_train.values), pd.Series(y_val.values)], ignore_index=True)
            best.fit(X_tv, y_tv)
            test_pred = best.predict(X_test)
            test_m = get_metrics(y_test, test_pred)

            results.append({
                "model": model_name,
                "val_metrics": val_m,
                "test_metrics": test_m,
                "best_params": {k: (str(v) if not isinstance(v, (int, float, bool)) else v)
                                for k, v in search.best_params_.items()},
            })
            print(f"  {model_name:<20} {val_m['r2']:>8.4f} {test_m['r2']:>8.4f} {test_m['rmse']:>8.2f}")

    # Save
    report = {
        "config": name,
        "all_models": results,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    return results


# ── VSG ──────────────────────────────────────────────────────

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
            Xs = np.clip(samples[:, :-1], X_m.min(0), X_m.max(0))[:n_per]
            ys = np.clip(samples[:, -1], y_m.min(), y_m.max())[:n_per]
            X_list.append(Xs)
            y_list.append(ys)
        X_virt, y_virt = np.vstack(X_list), np.concatenate(y_list)
    else:
        data = np.hstack([X, y.reshape(-1, 1)])
        kde = KernelDensity(kernel="gaussian", bandwidth="scott")
        kde.fit(data)
        samples = kde.sample(int(n_samples * 1.5), random_state=SEED)
        X_virt = np.clip(samples[:n_samples, :-1], X.min(0), X.max(0))
        y_virt = np.clip(samples[:n_samples, -1], y.min(), y.max())

    X_aug = pd.DataFrame(np.vstack([X, X_virt]), columns=feature_cols)
    y_aug = pd.Series(np.concatenate([y, y_virt]))
    return X_aug, y_aug


# ── RDKit Extended ───────────────────────────────────────────

def build_extended(df, train_idx, val_idx, test_idx, n_top, cache_label):
    from rdkit_descriptors import add_rdkit_descriptors, select_top_descriptors_pfi
    cache_dir = ROOT / "data" / "processed" / f"rdkit_cache_{cache_label}"
    cache_file = cache_dir / "all_descriptors.csv"
    meta_file = cache_dir / "desc_names.json"

    if cache_file.exists() and meta_file.exists():
        df_ext = pd.read_csv(cache_file)
        with open(meta_file) as f:
            desc_names = json.load(f)
    else:
        df_ext, desc_names = add_rdkit_descriptors(df)
        cache_dir.mkdir(parents=True, exist_ok=True)
        df_ext.to_csv(cache_file, index=False)
        with open(meta_file, "w") as f:
            json.dump(desc_names, f)

    target = df[LABEL_COLUMN]
    top_desc = select_top_descriptors_pfi(
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
    return X_tr_sc, target.loc[train_idx], X_va_sc, target.loc[val_idx], X_te_sc, target.loc[test_idx], medium, ext_cols


# ── Dataset Loaders ──────────────────────────────────────────

def load_original():
    """Load original dataset.csv, clean, split, scale."""
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
    """Load data_imputed.xlsx, standardize columns, split, scale."""
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


# ── Run All Configs for One Dataset ──────────────────────────

def run_all_configs(dataset_name, df_clean, X_tr, y_tr, X_va, y_va, X_te, y_te, medium, out_base):
    all_results = {}

    print(f"\n{'#'*60}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'#'*60}")

    train_idx = X_tr.index
    val_idx = X_va.index
    test_idx = X_te.index
    cache_label = dataset_name.lower()

    # 1. Baseline
    res = run_experiment("Baseline (6 feat)", X_tr, y_tr, X_va, y_va, X_te, y_te,
                         out_base / "baseline")
    all_results["Baseline"] = res

    # 2. VSG
    X_aug, y_aug = generate_vsg(X_tr, y_tr, medium, n_samples=200)
    res = run_experiment("VSG (6 feat, 200)", X_aug, y_aug, X_va, y_va, X_te, y_te,
                         out_base / "vsg")
    all_results["VSG"] = res

    # 3. Extended top 10
    print("\n  Building extended top 10...")
    X_e10_tr, y_e10_tr, X_e10_va, y_e10_va, X_e10_te, y_e10_te, med_e10, cols_10 = \
        build_extended(df_clean, train_idx, val_idx, test_idx, n_top=10, cache_label=cache_label)
    res = run_experiment("Extended top10 (16 feat)", X_e10_tr, y_e10_tr, X_e10_va, y_e10_va, X_e10_te, y_e10_te,
                         out_base / "extended_top10")
    all_results["Ext top10"] = res

    # 4. Extended top 10 + VSG
    X_e10_aug, y_e10_aug = generate_vsg(X_e10_tr, y_e10_tr, med_e10, n_samples=200, feature_cols=cols_10)
    res = run_experiment("Ext top10 + VSG", X_e10_aug, y_e10_aug, X_e10_va, y_e10_va, X_e10_te, y_e10_te,
                         out_base / "extended_top10_vsg")
    all_results["Ext top10+VSG"] = res

    # 5. Extended top 20
    print("\n  Building extended top 20...")
    X_e20_tr, y_e20_tr, X_e20_va, y_e20_va, X_e20_te, y_e20_te, med_e20, cols_20 = \
        build_extended(df_clean, train_idx, val_idx, test_idx, n_top=20, cache_label=cache_label)
    res = run_experiment("Extended top20 (26 feat)", X_e20_tr, y_e20_tr, X_e20_va, y_e20_va, X_e20_te, y_e20_te,
                         out_base / "extended_top20")
    all_results["Ext top20"] = res

    # 6. Extended top 20 + VSG
    X_e20_aug, y_e20_aug = generate_vsg(X_e20_tr, y_e20_tr, med_e20, n_samples=200, feature_cols=cols_20)
    res = run_experiment("Ext top20 + VSG", X_e20_aug, y_e20_aug, X_e20_va, y_e20_va, X_e20_te, y_e20_te,
                         out_base / "extended_top20_vsg")
    all_results["Ext top20+VSG"] = res

    return all_results


def print_summary_table(dataset_name, all_results):
    """Print a formatted table showing all configs x all models."""
    print(f"\n{'='*80}")
    print(f"  {dataset_name} - FULL RESULTS TABLE")
    print(f"{'='*80}")
    print(f"  {'Configuration':<20} {'Model':<20} {'Val R²':>8} {'Test R²':>8} {'RMSE':>8}")
    print(f"  {'-'*68}")

    best_r2 = -999
    best_label = ""

    for config_name, results in all_results.items():
        for r in sorted(results, key=lambda x: x["test_metrics"]["r2"], reverse=True):
            vm = r["val_metrics"]
            tm = r["test_metrics"]
            marker = ""
            if tm["r2"] > best_r2:
                best_r2 = tm["r2"]
                best_label = f"{config_name} / {r['model']}"
            print(f"  {config_name:<20} {r['model']:<20} {vm['r2']:>8.4f} {tm['r2']:>8.4f} {tm['rmse']:>8.2f}")
        config_name = ""  # Only print config name on first row of group

    print(f"  {'-'*68}")
    print(f"  BEST: {best_label} → Test R² = {best_r2:.4f}")
    print(f"{'='*80}")


# ── Main ─────────────────────────────────────────────────────

def main():
    out_root = ROOT / "data" / "models" / "benchmark"

    # Original dataset
    df_orig, X_tr_o, y_tr_o, X_va_o, y_va_o, X_te_o, y_te_o, med_o = load_original()
    results_orig = run_all_configs(
        "ORIGINAL", df_orig, X_tr_o, y_tr_o, X_va_o, y_va_o, X_te_o, y_te_o, med_o,
        out_root / "original",
    )
    print_summary_table("ORIGINAL DATASET", results_orig)

    # Imputed dataset
    df_imp, X_tr_i, y_tr_i, X_va_i, y_va_i, X_te_i, y_te_i, med_i = load_imputed()
    results_imp = run_all_configs(
        "IMPUTED", df_imp, X_tr_i, y_tr_i, X_va_i, y_va_i, X_te_i, y_te_i, med_i,
        out_root / "imputed",
    )
    print_summary_table("IMPUTED DATASET", results_imp)

    # Save combined summary
    def flatten(dataset_name, all_results):
        rows = []
        for config, res_list in all_results.items():
            for r in res_list:
                rows.append({
                    "dataset": dataset_name,
                    "config": config,
                    "model": r["model"],
                    "val_r2": r["val_metrics"]["r2"],
                    "test_r2": r["test_metrics"]["r2"],
                    "test_rmse": r["test_metrics"]["rmse"],
                    "test_mae": r["test_metrics"]["mae"],
                })
        return rows

    all_rows = flatten("original", results_orig) + flatten("imputed", results_imp)
    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(out_root / "full_benchmark.csv", index=False)
    print(f"\nSaved full benchmark to {out_root / 'full_benchmark.csv'}")


if __name__ == "__main__":
    main()
