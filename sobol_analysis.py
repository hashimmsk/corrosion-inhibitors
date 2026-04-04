"""
Pathway E: Sobol Global Sensitivity Analysis.

Runs Sobol analysis on the best model for both datasets to decompose
output variance into individual feature contributions and interactions.

Analysis:
  1. Sobol on original 6 features (baseline model)
  2. Sobol with pH removed (to see how importance redistributes)
  3. Sobol per medium (HCl, NaCl, CPS separately)
  4. Compare S1 (individual) vs ST (total incl. interactions)

Uses SALib for Sobol sampling and analysis.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "sobol_analysis"
FEATURE_COLUMNS = ["C#", "Mw", "HLB", "EO", "Conc", "pH"]
LABEL_COLUMN = "IE"
SEED = 0
N_SOBOL = 4096  # Sobol samples (must be power of 2)


def train_model(X, y, model_type="gradient_boosting"):
    """Train a model on the full dataset for Sobol analysis."""
    if model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=SEED,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=2, random_state=SEED,
        )
    model.fit(X, y)
    return model


def run_sobol(model, feature_names, feature_bounds, n_samples=N_SOBOL):
    """
    Run Sobol analysis on a trained model.

    Parameters
    ----------
    model : trained sklearn model
    feature_names : list of str
    feature_bounds : list of [min, max] for each feature
    n_samples : int (power of 2)

    Returns
    -------
    dict with S1, ST, S2 indices and confidence intervals
    """
    problem = {
        "num_vars": len(feature_names),
        "names": feature_names,
        "bounds": feature_bounds,
    }

    # Generate Sobol sample points
    X_sobol = sobol_sample.sample(problem, n_samples, calc_second_order=True)

    # Evaluate model on all Sobol points
    Y = model.predict(X_sobol)

    # Analyze
    Si = sobol_analyze.analyze(problem, Y, calc_second_order=True)

    return Si, problem


def format_sobol_results(Si, feature_names):
    """Create a readable DataFrame from Sobol results."""
    df = pd.DataFrame({
        "Feature": feature_names,
        "S1": Si["S1"],
        "S1_conf": Si["S1_conf"],
        "ST": Si["ST"],
        "ST_conf": Si["ST_conf"],
        "Interaction": Si["ST"] - Si["S1"],  # ST - S1 = interaction contribution
    }).sort_values("ST", ascending=False).reset_index(drop=True)
    return df


def print_sobol_table(title, df):
    """Print formatted Sobol results."""
    print(f"\n  {title}")
    print(f"  {'Feature':<20} {'S1':>8} {'ST':>8} {'Interact':>10} {'% Interact':>12}")
    print(f"  {'-'*60}")
    for _, row in df.iterrows():
        pct = (row["Interaction"] / row["ST"] * 100) if row["ST"] > 0.01 else 0
        print(f"  {row['Feature']:<20} {row['S1']:>8.4f} {row['ST']:>8.4f} {row['Interaction']:>10.4f} {pct:>11.1f}%")


def load_and_prepare_original():
    """Load original data, impute, return raw (unscaled) for Sobol bounds."""
    from preprocessing import (
        PreprocessingConfig, load_raw_dataset, clean_dataset,
        infer_feature_columns,
    )
    config = PreprocessingConfig(dataset_path=str(ROOT / "dataset.csv"), random_state=SEED)
    raw = load_raw_dataset(config.dataset_path)
    cleaned = clean_dataset(raw, config)
    feat_cols = list(infer_feature_columns(cleaned, config))
    X = cleaned[feat_cols].copy()
    y = cleaned[LABEL_COLUMN].copy()

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feat_cols, index=X.index)

    return X_imp, y, cleaned


def load_and_prepare_imputed():
    """Load imputed data."""
    df = pd.read_excel(ROOT / "data_imputed.xlsx")
    df = df.rename(columns={"Mw (g/mol)": "Mw", "ph": "pH", "S%": "Conc", "liquid": "medium"})
    X = df[FEATURE_COLUMNS].copy()
    y = df[LABEL_COLUMN].copy()
    return X, y, df


def analyze_dataset(name, X, y, df_full, out_dir):
    """Run full Sobol analysis for one dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    print(f"\n{'#'*60}")
    print(f"  SOBOL ANALYSIS: {name}")
    print(f"  Samples: {len(X)}, Features: {list(X.columns)}")
    print(f"{'#'*60}")

    feat_cols = list(X.columns)
    bounds = [[X[c].min(), X[c].max()] for c in feat_cols]

    # ── 1. Full model (all 6 features) ──
    print("\n  [1] Training model on all 6 features...")
    model = train_model(X.values, y.values)
    Si, _ = run_sobol(model, feat_cols, bounds)
    df_full_sobol = format_sobol_results(Si, feat_cols)
    print_sobol_table("All 6 Features", df_full_sobol)
    df_full_sobol.to_csv(out_dir / "sobol_all_features.csv", index=False)
    all_results["all_features"] = df_full_sobol.to_dict("records")

    # ── 2. Without pH ──
    print("\n  [2] Training model WITHOUT pH...")
    no_ph_cols = [c for c in feat_cols if c != "pH"]
    bounds_no_ph = [[X[c].min(), X[c].max()] for c in no_ph_cols]
    model_no_ph = train_model(X[no_ph_cols].values, y.values)
    Si_no_ph, _ = run_sobol(model_no_ph, no_ph_cols, bounds_no_ph)
    df_no_ph = format_sobol_results(Si_no_ph, no_ph_cols)
    print_sobol_table("Without pH", df_no_ph)
    df_no_ph.to_csv(out_dir / "sobol_without_ph.csv", index=False)
    all_results["without_ph"] = df_no_ph.to_dict("records")

    # ── 3. Per medium ──
    if "medium" in df_full.columns:
        print("\n  [3] Per-medium Sobol analysis...")
        mediums = df_full["medium"].unique()
        no_ph_cols_med = [c for c in feat_cols if c != "pH"]

        for medium in mediums:
            mask = df_full["medium"] == medium
            X_m = X.loc[mask, no_ph_cols_med]
            y_m = y.loc[mask]

            if len(X_m) < 20:
                print(f"    Skipping {medium} (only {len(X_m)} samples)")
                continue

            bounds_m = [[X_m[c].min(), X_m[c].max()] for c in no_ph_cols_med]

            # Ensure bounds have non-zero range
            for i, (lo, hi) in enumerate(bounds_m):
                if lo == hi:
                    bounds_m[i] = [lo - 0.01, hi + 0.01]

            model_m = train_model(X_m.values, y_m.values)
            Si_m, _ = run_sobol(model_m, no_ph_cols_med, bounds_m)
            df_m = format_sobol_results(Si_m, no_ph_cols_med)
            print_sobol_table(f"Medium: {medium} (n={len(X_m)}, no pH)", df_m)
            df_m.to_csv(out_dir / f"sobol_{medium}.csv", index=False)
            all_results[f"medium_{medium}"] = df_m.to_dict("records")

    # ── 4. Interaction matrix (S2) ──
    print("\n  [4] Second-order interaction indices...")
    s2_matrix = np.array(Si["S2"])
    s2_df = pd.DataFrame(s2_matrix, index=feat_cols, columns=feat_cols)
    s2_df.to_csv(out_dir / "sobol_s2_interactions.csv")

    # Print top interactions
    interactions = []
    for i in range(len(feat_cols)):
        for j in range(i + 1, len(feat_cols)):
            interactions.append((feat_cols[i], feat_cols[j], s2_matrix[i, j]))
    interactions.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\n  Top feature interactions (S2):")
    print(f"  {'Feature 1':<12} {'Feature 2':<12} {'S2':>10}")
    print(f"  {'-'*36}")
    for f1, f2, s2 in interactions[:10]:
        print(f"  {f1:<12} {f2:<12} {s2:>10.4f}")

    # Save summary
    with open(out_dir / "sobol_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Original dataset
    X_orig, y_orig, df_orig = load_and_prepare_original()
    results_orig = analyze_dataset("ORIGINAL", X_orig, y_orig, df_orig, OUT_DIR / "original")

    # Imputed dataset
    X_imp, y_imp, df_imp = load_and_prepare_imputed()
    results_imp = analyze_dataset("IMPUTED", X_imp, y_imp, df_imp, OUT_DIR / "imputed")

    # ── Cross-dataset comparison ──
    print(f"\n{'='*60}")
    print("  CROSS-DATASET COMPARISON (All 6 features, ST index)")
    print(f"{'='*60}")
    print(f"  {'Feature':<12} {'Original ST':>14} {'Imputed ST':>14}")
    print(f"  {'-'*42}")

    orig_st = {r["Feature"]: r["ST"] for r in results_orig["all_features"]}
    imp_st = {r["Feature"]: r["ST"] for r in results_imp["all_features"]}

    for feat in FEATURE_COLUMNS:
        o = orig_st.get(feat, 0)
        i = imp_st.get(feat, 0)
        print(f"  {feat:<12} {o:>14.4f} {i:>14.4f}")

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
