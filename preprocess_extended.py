"""
Extended preprocessing: original 6 features + top 40 RDKit descriptors.

Produces:
- data/processed/extended/train.csv, val.csv, test.csv (46 features, no VSG)
- data/processed/extended/cleaned_full.csv
- data/processed/extended/rdkit_columns.json (selected descriptor names)

Uses same train/val/test split as base preprocessing (random_state=0).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from preprocessing import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    PreprocessingConfig,
    clean_dataset,
    infer_feature_columns,
    load_raw_dataset,
    split_dataset,
)
from rdkit_descriptors import (
    RDKIT_AVAILABLE,
    add_rdkit_descriptors,
    select_top_descriptors_pfi,
)

OUT_DIR = Path(__file__).resolve().parent / "data" / "processed" / "extended"
N_PFI_DESCRIPTORS = 40
SEED = 0


def main():
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required. Run: pip install rdkit")

    root = Path(__file__).resolve().parent
    dataset_path = str(root / "dataset.csv")
    config = PreprocessingConfig(
        dataset_path=dataset_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=SEED,
    )

    # 1) Load and clean (same as base preprocessing)
    raw = load_raw_dataset(config.dataset_path)
    cleaned = clean_dataset(raw, config)
    feature_cols = infer_feature_columns(cleaned, config)
    features = cleaned[feature_cols]
    target = cleaned[config.label_column]

    # 2) Split (same logic as preprocessing)
    splits = split_dataset(features, target, config)
    train_idx = splits["X_train"].index
    val_idx = splits["X_val"].index
    test_idx = splits["X_test"].index

    # 3) Add RDKit descriptors to full cleaned df
    print("Computing RDKit 2D descriptors (surrogate SMILES from C#, EO)...")
    df_ext, desc_names = add_rdkit_descriptors(cleaned)

    # 4) PFI selection on TRAIN only
    print("Running PFI to select top 40 descriptors...")
    X_train_desc = df_ext.loc[train_idx, desc_names]
    y_train = target.loc[train_idx]
    top_desc = select_top_descriptors_pfi(
        X_train_desc, y_train, n_top=N_PFI_DESCRIPTORS, random_state=SEED
    )

    extended_cols = list(FEATURE_COLUMNS) + top_desc

    # 5) Extract splits with extended features
    train_ext = df_ext.loc[train_idx, extended_cols].copy()
    val_ext = df_ext.loc[val_idx, extended_cols].copy()
    test_ext = df_ext.loc[test_idx, extended_cols].copy()
    train_y = target.loc[train_idx]
    val_y = target.loc[val_idx]
    test_y = target.loc[test_idx]

    # 6) Impute and scale (fit on train only)
    imputer = SimpleImputer(strategy="median")
    train_imp = pd.DataFrame(
        imputer.fit_transform(train_ext),
        columns=extended_cols,
        index=train_ext.index,
    )
    val_imp = pd.DataFrame(
        imputer.transform(val_ext),
        columns=extended_cols,
        index=val_ext.index,
    )
    test_imp = pd.DataFrame(
        imputer.transform(test_ext),
        columns=extended_cols,
        index=test_ext.index,
    )

    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_imp),
        columns=extended_cols,
        index=train_imp.index,
    )
    val_scaled = pd.DataFrame(
        scaler.transform(val_imp),
        columns=extended_cols,
        index=val_imp.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_imp),
        columns=extended_cols,
        index=test_imp.index,
    )

    # 7) Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, X_df, y_ser in [
        ("train", train_scaled, train_y),
        ("val", val_scaled, val_y),
        ("test", test_scaled, test_y),
    ]:
        out = X_df.copy()
        out[LABEL_COLUMN] = y_ser.values
        if "medium" in cleaned.columns:
            out["medium"] = cleaned.loc[X_df.index, "medium"].values
        out["pH_original"] = cleaned.loc[X_df.index, "pH"].values
        out.to_csv(OUT_DIR / f"{name}.csv", index=False)

    # Save full cleaned extended (for reference)
    full_ext = df_ext[extended_cols + [LABEL_COLUMN]].copy()
    if "medium" in cleaned.columns:
        full_ext.loc[:, "medium"] = cleaned["medium"].values
    full_imp = pd.DataFrame(imputer.transform(df_ext[extended_cols]), columns=extended_cols)
    full_scaled = pd.DataFrame(scaler.transform(full_imp), columns=extended_cols)
    full_scaled[LABEL_COLUMN] = cleaned[LABEL_COLUMN].values
    if "medium" in cleaned.columns:
        full_scaled["medium"] = cleaned["medium"].values
    full_scaled.to_csv(OUT_DIR / "cleaned_full.csv", index=False)

    # Save metadata
    with open(OUT_DIR / "rdkit_columns.json", "w") as f:
        json.dump({"rdkit_columns": top_desc, "extended_columns": extended_cols}, f, indent=2)

    print(f"Saved extended data to {OUT_DIR}")
    print(f"  Features: {len(extended_cols)} (6 original + {len(top_desc)} RDKit)")
    print(f"  Train: {len(train_scaled)}, Val: {len(val_scaled)}, Test: {len(test_scaled)}")


if __name__ == "__main__":
    main()
