"""
Improved training run for VSG, Extended, and Extended+VSG.

Changes from previous runs:
- Added GradientBoostingRegressor (strong on tabular data per QSPR paper)
- Expanded hyperparameter grids
- VSG: reduced to 200 synthetic samples (less noise)
- Extended: top 20 RDKit descriptors instead of 40 (less overfitting risk)
- SVR: wider C/gamma range for high-dimensional data
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from preprocessing import FEATURE_COLUMNS, LABEL_COLUMN
from virtual_sample_generation import generate_virtual_samples
import features

SEED = 0
CV_SPLITS = 5
N_ITER = 60
ROOT = Path(__file__).resolve().parent
OUT_BASE = ROOT / "data" / "models" / "improved"


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
    """Train all models, evaluate, save results. Returns best test R²."""
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
                "model": model_name,
                "val_metrics": val_m,
                "best_params": search.best_params_,
                "best_estimator": best,
            })
            print(f"  {model_name:<22} VAL R²={val_m['r2']:.4f}  RMSE={val_m['rmse']:.2f}")

    # Refit on train+val
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

    return winner["test_metrics"]["r2"], winner["model"]


def load_extended(n_top=20):
    """Load or rebuild extended data with n_top RDKit descriptors."""
    ext_dir = ROOT / "data" / "processed" / f"extended_top{n_top}"

    if not (ext_dir / "train.csv").exists():
        print(f"Building extended features with top {n_top} RDKit descriptors...")
        from rdkit_descriptors import add_rdkit_descriptors, select_top_descriptors_pfi, RDKIT_AVAILABLE
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required.")
        from preprocessing import (
            PreprocessingConfig, load_raw_dataset, clean_dataset,
            infer_feature_columns, split_dataset,
        )

        config = PreprocessingConfig(dataset_path=str(ROOT / "dataset.csv"), random_state=SEED)
        raw = load_raw_dataset(config.dataset_path)
        cleaned = clean_dataset(raw, config)
        feat_cols = infer_feature_columns(cleaned, config)
        feats = cleaned[feat_cols]
        target = cleaned[LABEL_COLUMN]
        splits = split_dataset(feats, target, config)

        train_idx = splits["X_train"].index
        val_idx = splits["X_val"].index
        test_idx = splits["X_test"].index

        df_ext, desc_names = add_rdkit_descriptors(cleaned)

        top_desc = select_top_descriptors_pfi(
            df_ext.loc[train_idx, desc_names],
            target.loc[train_idx],
            n_top=n_top,
            random_state=SEED,
        )
        ext_cols = list(FEATURE_COLUMNS) + top_desc

        ext_dir.mkdir(parents=True, exist_ok=True)

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        train_X = df_ext.loc[train_idx, ext_cols]
        val_X = df_ext.loc[val_idx, ext_cols]
        test_X = df_ext.loc[test_idx, ext_cols]

        train_imp = pd.DataFrame(imputer.fit_transform(train_X), columns=ext_cols, index=train_idx)
        val_imp = pd.DataFrame(imputer.transform(val_X), columns=ext_cols, index=val_idx)
        test_imp = pd.DataFrame(imputer.transform(test_X), columns=ext_cols, index=test_idx)

        train_sc = pd.DataFrame(scaler.fit_transform(train_imp), columns=ext_cols, index=train_idx)
        val_sc = pd.DataFrame(scaler.transform(val_imp), columns=ext_cols, index=val_idx)
        test_sc = pd.DataFrame(scaler.transform(test_imp), columns=ext_cols, index=test_idx)

        for label, Xdf, idx in [("train", train_sc, train_idx), ("val", val_sc, val_idx), ("test", test_sc, test_idx)]:
            out = Xdf.copy()
            out[LABEL_COLUMN] = target.loc[idx].values
            if "medium" in cleaned.columns:
                out["medium"] = cleaned.loc[idx, "medium"].values
            out.to_csv(ext_dir / f"{label}.csv", index=False)

        with open(ext_dir / "rdkit_columns.json", "w") as f:
            json.dump({"rdkit_columns": top_desc, "extended_columns": ext_cols}, f, indent=2)

        print(f"  Saved {len(ext_cols)} features to {ext_dir}")

    with open(ext_dir / "rdkit_columns.json") as f:
        meta = json.load(f)
    ext_cols = meta["extended_columns"]

    train_df = pd.read_csv(ext_dir / "train.csv")
    val_df = pd.read_csv(ext_dir / "val.csv")
    test_df = pd.read_csv(ext_dir / "test.csv")

    return (
        train_df[ext_cols], train_df[LABEL_COLUMN],
        val_df[ext_cols], val_df[LABEL_COLUMN],
        test_df[ext_cols], test_df[LABEL_COLUMN],
        train_df.get("medium"), ext_cols,
    )


def main():
    summary = []

    # --- 1. VSG (6 features, 200 synthetic samples) ---
    (X_tr, y_tr), (X_val, y_val), (X_test, y_test) = features.load_splits()
    train_csv = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    medium = train_csv["medium"] if "medium" in train_csv.columns else None

    X_tr_aug, y_tr_aug = generate_virtual_samples(
        X_tr, y_tr, n_samples=200, medium=medium,
        medium_balance=True, random_state=SEED,
    )
    r2, model = run_experiment(
        "VSG (6 feat, 200 samples, +GB)",
        X_tr_aug, y_tr_aug, X_val, y_val, X_test, y_test,
        OUT_BASE / "vsg",
    )
    summary.append(("VSG (improved)", model, r2))

    # --- 2. Extended top 10 (optimal from feature selection) ---
    X_tr_e10, y_tr_e10, X_val_e10, y_val_e10, X_test_e10, y_test_e10, med_e10, ext_cols_10 = load_extended(n_top=10)
    r2, model = run_experiment(
        "Extended (6+10 RDKit, no VSG, +GB)",
        X_tr_e10, y_tr_e10, X_val_e10, y_val_e10, X_test_e10, y_test_e10,
        OUT_BASE / "extended_top10",
    )
    summary.append(("Extended top10", model, r2))

    # --- 3. Extended top 10 + VSG ---
    X_tr_e10_aug, y_tr_e10_aug = generate_virtual_samples(
        X_tr_e10, y_tr_e10, n_samples=200, medium=med_e10,
        medium_balance=True, random_state=SEED, feature_columns=ext_cols_10,
    )
    r2, model = run_experiment(
        "Extended+VSG (6+10 RDKit, 200 samples, +GB)",
        X_tr_e10_aug, y_tr_e10_aug, X_val_e10, y_val_e10, X_test_e10, y_test_e10,
        OUT_BASE / "extended_top10_vsg",
    )
    summary.append(("Extended top10+VSG", model, r2))

    # --- 4. Extended top 20 (previous best) ---
    X_tr_e20, y_tr_e20, X_val_e20, y_val_e20, X_test_e20, y_test_e20, med_e20, ext_cols_20 = load_extended(n_top=20)
    r2, model = run_experiment(
        "Extended (6+20 RDKit, no VSG, +GB)",
        X_tr_e20, y_tr_e20, X_val_e20, y_val_e20, X_test_e20, y_test_e20,
        OUT_BASE / "extended_top20",
    )
    summary.append(("Extended top20", model, r2))

    # --- 5. Extended top 20 + VSG ---
    X_tr_e20_aug, y_tr_e20_aug = generate_virtual_samples(
        X_tr_e20, y_tr_e20, n_samples=200, medium=med_e20,
        medium_balance=True, random_state=SEED, feature_columns=ext_cols_20,
    )
    r2, model = run_experiment(
        "Extended+VSG (6+20 RDKit, 200 samples, +GB)",
        X_tr_e20_aug, y_tr_e20_aug, X_val_e20, y_val_e20, X_test_e20, y_test_e20,
        OUT_BASE / "extended_top20_vsg",
    )
    summary.append(("Extended top20+VSG", model, r2))

    # --- Summary ---
    print("\n" + "=" * 60)
    print("IMPROVEMENT SUMMARY")
    print("=" * 60)
    print(f"{'Config':<30} {'Best Model':<22} {'Test R²'}")
    print("-" * 60)

    baselines = [
        ("Baseline (original)", "random_forest", 0.417),
    ]
    for name, m, r2 in baselines:
        print(f"  {name:<28} {m:<22} {r2:.4f}")
    print("-" * 60)
    for name, m, r2 in summary:
        print(f"  {name:<28} {m:<22} {r2:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
