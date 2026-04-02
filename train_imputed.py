"""
Full pipeline on data_imputed.xlsx.

Runs all configurations that were run on the original dataset:
1. Baseline (6 features)
2. VSG (6 feat, 200 samples)
3. Extended top 10 (6 + 10 RDKit)
4. Extended top 10 + VSG
5. Extended top 20 (6 + 20 RDKit)
6. Extended top 20 + VSG

The imputed dataset has different column names and additional features
(Density, CP). IE is already corrected (no AA rescaling needed).

Column mapping:
  C# -> C#, Mw (g/mol) -> Mw, HLB -> HLB, EO -> EO,
  S% -> Conc, ph -> pH, liquid -> medium
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KernelDensity

SEED = 0
CV_SPLITS = 5
N_ITER = 60
ROOT = Path(__file__).resolve().parent
OUT_BASE = ROOT / "data" / "models" / "imputed"
PROCESSED_BASE = ROOT / "data" / "processed" / "imputed"

FEATURE_COLUMNS = ["C#", "Mw", "HLB", "EO", "Conc", "pH"]
LABEL_COLUMN = "IE"


# ── Helpers ──────────────────────────────────────────────────

def get_metrics(y_true, y_pred):
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
    }


def build_models():
    return {
        "random_forest": (
            RandomForestRegressor(random_state=SEED, n_jobs=1),
            {
                "n_estimators": [200, 400, 600, 800],
                "max_depth": [4, 6, 8, 10, None],
                "min_samples_leaf": [1, 2, 3, 5],
                "min_samples_split": [2, 3, 5],
            },
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(random_state=SEED),
            {
                "n_estimators": [100, 200, 400, 600],
                "max_depth": [3, 4, 5, 6],
                "min_samples_leaf": [2, 5, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0],
            },
        ),
        "svr": (
            SVR(kernel="rbf", cache_size=2000),
            {
                "C": [1.0, 10.0, 50.0, 100.0, 500.0],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                "epsilon": [0.01, 0.05, 0.1, 0.2],
            },
        ),
    }


def run_experiment(name, X_train, y_train, X_val, y_val, X_test, y_test, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    models = build_models()
    results = []

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}, Features: {X_train.shape[1]}")
    print(f"{'='*60}")

    with parallel_backend("threading"):
        for model_name, (estimator, param_space) in models.items():
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_space,
                n_iter=min(N_ITER, np.prod([len(v) for v in param_space.values()])),
                scoring="r2",
                cv=cv,
                random_state=SEED,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            val_pred = best.predict(X_val)
            val_m = get_metrics(y_val, val_pred)
            results.append({
                "model": model_name, "val_metrics": val_m,
                "best_params": search.best_params_, "best_estimator": best,
            })
            print(f"  {model_name:<22} VAL R²={val_m['r2']:.4f}  RMSE={val_m['rmse']:.2f}")

    X_tv = pd.concat([X_train, X_val], ignore_index=True)
    y_tv = pd.concat([y_train, y_val], ignore_index=True)

    print(f"  --- Refit on train+val, evaluate on test ---")
    for r in results:
        r["best_estimator"].fit(X_tv, y_tv)
        test_pred = r["best_estimator"].predict(X_test)
        r["test_metrics"] = get_metrics(y_test, test_pred)
        r["test_predictions"] = test_pred
        tm = r["test_metrics"]
        print(f"  {r['model']:<22} TEST R²={tm['r2']:.4f}  RMSE={tm['rmse']:.2f}")

    winner = max(results, key=lambda r: r["val_metrics"]["r2"])

    report = {
        "best_model": winner["model"],
        "best_params": winner["best_params"],
        "val_metrics": winner["val_metrics"],
        "test_metrics": winner["test_metrics"],
        "all_models": [
            {"model": r["model"], "val_metrics": r["val_metrics"],
             "test_metrics": r["test_metrics"], "best_params": r["best_params"]}
            for r in results
        ],
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))

    pred_df = pd.DataFrame({"y_true": y_test.values})
    for r in results:
        pred_df[f"y_pred_{r['model']}"] = r["test_predictions"]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    return winner["test_metrics"]["r2"], winner["model"], results


# ── Data Loading ─────────────────────────────────────────────

def load_imputed_data():
    """Load and standardize column names from data_imputed.xlsx."""
    df = pd.read_excel(ROOT / "data_imputed.xlsx")

    # Rename columns to match our canonical names
    col_map = {
        "Mw (g/mol)": "Mw",
        "ph": "pH",
        "S%": "Conc",
        "liquid": "medium",
        "AA (%)": "AA",
    }
    df = df.rename(columns=col_map)

    # Keep only what we need
    keep = FEATURE_COLUMNS + [LABEL_COLUMN]
    if "medium" in df.columns:
        keep.append("medium")

    df_clean = df[keep].copy()
    df_clean = df_clean.dropna(subset=[LABEL_COLUMN]).reset_index(drop=True)

    return df_clean


def split_and_scale(df, feature_cols):
    """Split 70/15/15, impute, scale. Returns dict of splits + medium."""
    X = df[feature_cols]
    y = df[LABEL_COLUMN]

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

    train_idx = X_train.index
    val_idx = X_val.index
    test_idx = X_test.index

    # Impute + scale
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_tr = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(X_train)), columns=feature_cols, index=train_idx)
    X_va = pd.DataFrame(scaler.transform(imputer.transform(X_val)), columns=feature_cols, index=val_idx)
    X_te = pd.DataFrame(scaler.transform(imputer.transform(X_test)), columns=feature_cols, index=test_idx)

    medium = df["medium"].loc[train_idx] if "medium" in df.columns else None

    return X_tr, y_train, X_va, y_val, X_te, y_test, medium, train_idx, val_idx, test_idx


# ── VSG ──────────────────────────────────────────────────────

def generate_vsg(X_train, y_train, medium, n_samples=200, feature_cols=None):
    """KDE-based VSG, medium-aware."""
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
            Xs = np.clip(samples[:, :-1], X_m.min(0), X_m.max(0))
            ys = np.clip(samples[:, -1], y_m.min(), y_m.max())
            X_list.append(Xs[:n_per])
            y_list.append(ys[:n_per])
        X_virt = np.vstack(X_list)
        y_virt = np.concatenate(y_list)
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

def build_extended(df, train_idx, val_idx, test_idx, n_top=10):
    """Add RDKit descriptors, PFI select, impute, scale."""
    from rdkit_descriptors import add_rdkit_descriptors, select_top_descriptors_pfi, RDKIT_AVAILABLE
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required.")

    cache_dir = PROCESSED_BASE / f"rdkit_cache"
    cache_file = cache_dir / "all_descriptors.csv"
    meta_file = cache_dir / "desc_names.json"

    if cache_file.exists() and meta_file.exists():
        df_ext = pd.read_csv(cache_file)
        with open(meta_file) as f:
            desc_names = json.load(f)
    else:
        print("  Computing RDKit descriptors for imputed data...")
        df_ext, desc_names = add_rdkit_descriptors(df)
        cache_dir.mkdir(parents=True, exist_ok=True)
        df_ext.to_csv(cache_file, index=False)
        with open(meta_file, "w") as f:
            json.dump(desc_names, f)

    target = df[LABEL_COLUMN]

    print(f"  Selecting top {n_top} RDKit descriptors via PFI...")
    top_desc = select_top_descriptors_pfi(
        df_ext.loc[train_idx, desc_names], target.loc[train_idx],
        n_top=n_top, random_state=SEED,
    )
    ext_cols = FEATURE_COLUMNS + top_desc

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_tr = df_ext.loc[train_idx, ext_cols]
    X_va = df_ext.loc[val_idx, ext_cols]
    X_te = df_ext.loc[test_idx, ext_cols]

    X_tr_imp = imputer.fit_transform(X_tr)
    X_va_imp = imputer.transform(X_va)
    X_te_imp = imputer.transform(X_te)

    X_tr_imp = np.clip(X_tr_imp, -1e10, 1e10)
    X_va_imp = np.clip(X_va_imp, -1e10, 1e10)
    X_te_imp = np.clip(X_te_imp, -1e10, 1e10)

    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr_imp), columns=ext_cols, index=train_idx)
    X_va_sc = pd.DataFrame(scaler.transform(X_va_imp), columns=ext_cols, index=val_idx)
    X_te_sc = pd.DataFrame(scaler.transform(X_te_imp), columns=ext_cols, index=test_idx)

    X_tr_sc = X_tr_sc.apply(lambda c: np.nan_to_num(c, nan=0.0))
    X_va_sc = X_va_sc.apply(lambda c: np.nan_to_num(c, nan=0.0))
    X_te_sc = X_te_sc.apply(lambda c: np.nan_to_num(c, nan=0.0))

    medium = df["medium"].loc[train_idx] if "medium" in df.columns else None

    return X_tr_sc, target.loc[train_idx], X_va_sc, target.loc[val_idx], X_te_sc, target.loc[test_idx], medium, ext_cols, top_desc


# ── Main ─────────────────────────────────────────────────────

def main():
    print("Loading data_imputed.xlsx...")
    df = load_imputed_data()
    print(f"  Rows: {len(df)}, Features: {FEATURE_COLUMNS}")
    if "medium" in df.columns:
        print(f"  Medium: {df['medium'].value_counts().to_dict()}")

    summary = []

    # ── 1. Baseline ──
    X_tr, y_tr, X_va, y_va, X_te, y_te, medium, train_idx, val_idx, test_idx = split_and_scale(df, FEATURE_COLUMNS)

    # Save processed splits
    PROCESSED_BASE.mkdir(parents=True, exist_ok=True)
    for label, Xdf, yser, idx in [("train", X_tr, y_tr, train_idx), ("val", X_va, y_va, val_idx), ("test", X_te, y_te, test_idx)]:
        out = Xdf.copy()
        out[LABEL_COLUMN] = yser.values
        if "medium" in df.columns:
            out["medium"] = df.loc[idx, "medium"].values
        out.to_csv(PROCESSED_BASE / f"{label}.csv", index=False)

    r2, model, _ = run_experiment("Baseline (6 feat)", X_tr, y_tr, X_va, y_va, X_te, y_te, OUT_BASE / "baseline")
    summary.append(("Baseline", model, r2))

    # ── 2. VSG ──
    X_aug, y_aug = generate_vsg(X_tr, y_tr, medium, n_samples=200)
    r2, model, _ = run_experiment("VSG (6 feat, 200 samples)", X_aug, y_aug, X_va, y_va, X_te, y_te, OUT_BASE / "vsg")
    summary.append(("VSG", model, r2))

    # ── 3. Extended top 10 ──
    X_tr_e10, y_tr_e10, X_va_e10, y_va_e10, X_te_e10, y_te_e10, med_e10, ext_cols_10, top10 = build_extended(df, train_idx, val_idx, test_idx, n_top=10)
    r2, model, _ = run_experiment("Extended top10 (6+10 RDKit)", X_tr_e10, y_tr_e10, X_va_e10, y_va_e10, X_te_e10, y_te_e10, OUT_BASE / "extended_top10")
    summary.append(("Extended top10", model, r2))

    # ── 4. Extended top 10 + VSG ──
    X_aug10, y_aug10 = generate_vsg(X_tr_e10, y_tr_e10, med_e10, n_samples=200, feature_cols=ext_cols_10)
    r2, model, _ = run_experiment("Extended top10 + VSG", X_aug10, y_aug10, X_va_e10, y_va_e10, X_te_e10, y_te_e10, OUT_BASE / "extended_top10_vsg")
    summary.append(("Extended top10+VSG", model, r2))

    # ── 5. Extended top 20 ──
    X_tr_e20, y_tr_e20, X_va_e20, y_va_e20, X_te_e20, y_te_e20, med_e20, ext_cols_20, top20 = build_extended(df, train_idx, val_idx, test_idx, n_top=20)
    r2, model, _ = run_experiment("Extended top20 (6+20 RDKit)", X_tr_e20, y_tr_e20, X_va_e20, y_va_e20, X_te_e20, y_te_e20, OUT_BASE / "extended_top20")
    summary.append(("Extended top20", model, r2))

    # ── 6. Extended top 20 + VSG ──
    X_aug20, y_aug20 = generate_vsg(X_tr_e20, y_tr_e20, med_e20, n_samples=200, feature_cols=ext_cols_20)
    r2, model, _ = run_experiment("Extended top20 + VSG", X_aug20, y_aug20, X_va_e20, y_va_e20, X_te_e20, y_te_e20, OUT_BASE / "extended_top20_vsg")
    summary.append(("Extended top20+VSG", model, r2))

    # ── Summary ──
    print("\n" + "=" * 60)
    print("IMPUTED DATASET - FULL RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Config':<30} {'Best Model':<22} {'Test R²'}")
    print("-" * 60)
    for name, m, r2 in summary:
        print(f"  {name:<28} {m:<22} {r2:.4f}")
    print("=" * 60)

    # Save top descriptors
    with open(OUT_BASE / "rdkit_top_descriptors.json", "w") as f:
        json.dump({"top10": top10, "top20": top20}, f, indent=2)

    # Save full summary
    with open(OUT_BASE / "summary.json", "w") as f:
        json.dump([{"config": name, "best_model": m, "test_r2": r2} for name, m, r2 in summary], f, indent=2)


if __name__ == "__main__":
    main()
