"""
Data loading and preprocessing utilities.

Responsibilities:
- Load raw dataset from CSV / Excel.
- Apply row-wise cleaning that does NOT learn from data distribution:
  * drop 'No'
  * compute IE = IE * AA / 100
  * drop 'AA'
  * remove duplicates / empty columns.
- Split into train / val / test.
- Fit SimpleImputer & StandardScaler on TRAIN ONLY.
- Apply fitted transforms to train, val, test.
- Return a structured dictionary with all splits and transformers.

Anything related to:
- Outlier detection plots
- Histograms / correlation heatmaps
- Feature selection experiments (RFE, SHAP, etc.)
- Clustering visualizations

will live in eda.py or other analysis scripts, NOT in this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingConfig:
    """
    Configuration for dataset preprocessing.
    """

    dataset_path: str
    label_column: str = "IE"
    feature_columns: Optional[Sequence[str]] = None  # if None, infer as all non-label
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 0
    do_impute: bool = True
    do_scale: bool = True


# ---------------------------------------------------------------------------
# File / dataset loading
# ---------------------------------------------------------------------------


def locate_dataset_file(base_dir: str) -> str:
    """
    Return the first dataset file found in common locations.

    You can call this from config.py or a script, e.g.:

        BASE = os.path.dirname(os.path.dirname(__file__))  # project root
        DATA_PATH = locate_dataset_file(os.path.join(BASE, "data", "raw"))
    """

    candidates = ("Dataset.xlsx", "dataset.xlsx", "dataset.csv", "Dataset.csv")
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not locate a dataset file in {base_dir!r}. "
        f"Expected one of: {', '.join(candidates)}"
    )


def load_raw_dataset(path: str) -> pd.DataFrame:
    """
    Load the raw dataset from CSV or Excel.
    """

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported dataset extension: {ext}")


# ---------------------------------------------------------------------------
# Cleaning and feature/label setup (row-wise, no learning)
# ---------------------------------------------------------------------------


def clean_dataset(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """
    Apply row-wise cleaning that does NOT learn from the data distribution.

    - Drop 'No' if present.
    - If both 'IE' and 'AA' exist, compute IE = IE * AA / 100 and drop 'AA'.
    - Drop duplicated rows.
    - Drop completely empty columns.
    - Drop 'Unnamed:' columns from Excel.
    - Drop rows with missing label.
    - Move the label column to the end.
    """

    cleaned = df.copy()

    # Drop index-like column
    if "No" in cleaned.columns:
        cleaned.drop(columns=["No"], inplace=True)

    # IE correction using AA, then drop AA
    if {"IE", "AA"}.issubset(cleaned.columns):
        cleaned["IE"] = (cleaned["IE"] * cleaned["AA"]) / 100.0
        cleaned.drop(columns=["AA"], inplace=True)

    # Remove duplicate rows
    cleaned.drop_duplicates(inplace=True)

    # Drop columns that are entirely NaN
    cleaned.dropna(axis=1, how="all", inplace=True)

    # Drop Excel "Unnamed" columns
    cleaned = cleaned.loc[:, [
        col for col in cleaned.columns
        if not str(col).lower().startswith("unnamed:")
    ]]

    cleaned.reset_index(drop=True, inplace=True)

    # Ensure label exists
    if config.label_column not in cleaned.columns:
        raise KeyError(
            f"Label column '{config.label_column}' not found in dataset "
            f"columns: {list(cleaned.columns)}"
        )

    # Drop rows where label is NaN (we cannot train on those)
    cleaned = cleaned.dropna(subset=[config.label_column]).reset_index(drop=True)

    # Put label column at the end
    ordered_cols = [c for c in cleaned.columns if c != config.label_column] + [
        config.label_column
    ]
    return cleaned.loc[:, ordered_cols]


def infer_feature_columns(df: pd.DataFrame,
                          config: PreprocessingConfig) -> Sequence[str]:
    """
    Decide which columns are features.

    Priority:
    1. If config.feature_columns is provided, use that (and validate).
    2. Otherwise, use all columns except the label.
    """

    if config.feature_columns is not None:
        missing = [c for c in config.feature_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Missing expected feature columns: {missing}")
        return list(config.feature_columns)

    return [c for c in df.columns if c != config.label_column]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def validate_split_ratios(config: PreprocessingConfig) -> None:
    total = config.train_ratio + config.val_ratio + config.test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Train/val/test ratios must sum to 1.0 (got {total:.3f})."
        )
    if any(r < 0 for r in (config.train_ratio,
                           config.val_ratio,
                           config.test_ratio)):
        raise ValueError("Train/val/test ratios must be non-negative.")


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    config: PreprocessingConfig,
) -> Dict[str, pd.DataFrame]:
    """
    Split features/target into train, val, test according to config ratios.
    """

    validate_split_ratios(config)

    n_samples = len(features)
    if n_samples == 0:
        raise ValueError("Dataset is empty; cannot create splits.")

    feature_cols = list(features.columns)
    empty_X = pd.DataFrame(columns=feature_cols)
    empty_y = target.iloc[0:0]

    # Special case: all train
    if config.train_ratio == 1.0 or (
        config.val_ratio == 0.0 and config.test_ratio == 0.0
    ):
        return {
            "X_train": features.copy(),
            "y_train": target.copy(),
            "X_val": empty_X.copy(),
            "y_val": empty_y.copy(),
            "X_test": empty_X.copy(),
            "y_test": empty_y.copy(),
        }

    # First split: train vs (val+test)
    val_test_ratio = config.val_ratio + config.test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        features,
        target,
        test_size=val_test_ratio,
        shuffle=True,
        random_state=config.random_state,
    )

    # Only test, no val
    if config.val_ratio == 0.0:
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": empty_X.copy(),
            "y_val": empty_y.copy(),
            "X_test": X_temp,
            "y_test": y_temp,
        }

    # Only val, no test
    if config.test_ratio == 0.0:
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_temp,
            "y_val": y_temp,
            "X_test": empty_X.copy(),
            "y_test": empty_y.copy(),
        }

    # Split the temp set into val and test
    test_fraction = config.test_ratio / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_fraction,
        shuffle=True,
        random_state=config.random_state,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_dataset(config: PreprocessingConfig) -> Dict[str, object]:
    """
    Full preprocessing pipeline:

    1. Load raw data from config.dataset_path.
    2. Apply row-wise cleaning (drop 'No', correct IE with AA, etc.).
    3. Infer feature columns.
    4. Split into train / val / test.
    5. Fit imputer and scaler on TRAIN only, then transform all splits.
    6. Build a fully imputed dataset for EDA.

    Returns a dictionary with:
        - 'config'
        - 'cleaned_df'
        - 'dataset_for_eda'
        - 'feature_columns'
        - 'label_column'
        - 'splits': {
              'train': {'X_raw', 'X_imputed', 'X_scaled', 'y'},
              'val':   {...},
              'test':  {...},
          }
        - 'imputer'
        - 'scaler'
    """

    # Load & clean
    raw_df = load_raw_dataset(config.dataset_path)
    cleaned_df = clean_dataset(raw_df, config)

    # Features and label
    feature_columns = infer_feature_columns(cleaned_df, config)
    features = cleaned_df[feature_columns]
    target = cleaned_df[config.label_column]

    # Train/val/test split
    splits = split_dataset(features, target, config)

    # Prepare transformers
    imputer: Optional[SimpleImputer] = None
    scaler: Optional[StandardScaler] = None

    # ---- Imputation ----
    if config.do_impute:
        imputer = SimpleImputer(strategy="mean")
        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(splits["X_train"]),
            columns=feature_columns,
            index=splits["X_train"].index,
        )
        X_val_imputed = pd.DataFrame(
            imputer.transform(splits["X_val"]),
            columns=feature_columns,
            index=splits["X_val"].index,
        )
        X_test_imputed = pd.DataFrame(
            imputer.transform(splits["X_test"]),
            columns=feature_columns,
            index=splits["X_test"].index,
        )

        # For EDA: full dataset imputed (no split)
        full_features_imputed = pd.DataFrame(
            imputer.transform(features),
            columns=feature_columns,
            index=features.index,
        )
    else:
        # No imputation; pass-through
        X_train_imputed = splits["X_train"].copy()
        X_val_imputed = splits["X_val"].copy()
        X_test_imputed = splits["X_test"].copy()
        full_features_imputed = features.copy()

    # ---- Scaling ----
    if config.do_scale:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_imputed),
            columns=feature_columns,
            index=X_train_imputed.index,
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val_imputed),
            columns=feature_columns,
            index=X_val_imputed.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_imputed),
            columns=feature_columns,
            index=X_test_imputed.index,
        )
    else:
        # No scaling; pass-through imputed
        X_train_scaled = X_train_imputed.copy()
        X_val_scaled = X_val_imputed.copy()
        X_test_scaled = X_test_imputed.copy()

    # ---- Dataset for EDA ----
    dataset_for_eda = full_features_imputed.copy()
    dataset_for_eda[config.label_column] = target

    # Build structured output
    processed = {
        "config": config,
        "cleaned_df": cleaned_df,
        "dataset_for_eda": dataset_for_eda,
        "feature_columns": feature_columns,
        "label_column": config.label_column,
        "splits": {
            "train": {
                "X_raw": splits["X_train"],
                "X_imputed": X_train_imputed,
                "X_scaled": X_train_scaled,
                "y": splits["y_train"],
            },
            "val": {
                "X_raw": splits["X_val"],
                "X_imputed": X_val_imputed,
                "X_scaled": X_val_scaled,
                "y": splits["y_val"],
            },
            "test": {
                "X_raw": splits["X_test"],
                "X_imputed": X_test_imputed,
                "X_scaled": X_test_scaled,
                "y": splits["y_test"],
            },
        },
        "imputer": imputer,
        "scaler": scaler,
    }
    return processed


# ---------------------------------------------------------------------------
# Optional: quick CLI-style usage
# ---------------------------------------------------------------------------


def _find_default_dataset_path(this_dir: str) -> str:
    """
    Locate a dataset for CLI usage, preferring data/raw but falling back
    to the current directory if needed.
    """

    project_root = os.path.dirname(this_dir)
    search_dirs = [
        os.path.join(project_root, "data", "raw"),
        os.path.join(project_root, "data"),
        this_dir,
    ]

    for directory in search_dirs:
        if os.path.isdir(directory):
            try:
                return locate_dataset_file(directory)
            except FileNotFoundError:
                continue

    raise FileNotFoundError(
        "Could not locate a dataset file in default search paths. "
        "Please supply PreprocessingConfig.dataset_path explicitly."
    )


if __name__ == "__main__":
    """
    Example usage when running this file directly:

        python -m preprocessing

    This will:
    - Locate the dataset under ../data/raw (relative to this file), or fall back
      to the current directory if that folder does not exist.
    - Run preprocessing with default ratios.
    - Save processed splits to ../data/processed as CSVs.
    """

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(THIS_DIR, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    dataset_path = _find_default_dataset_path(THIS_DIR)

    cfg = PreprocessingConfig(
        dataset_path=dataset_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=0,
    )
    pre = preprocess_dataset(cfg)

    # Save cleaned full dataset and splits (scaled) for convenience
    pre["cleaned_df"].to_csv(
        os.path.join(processed_dir, "cleaned_full.csv"), index=False
    )

    train = pre["splits"]["train"]
    val = pre["splits"]["val"]
    test = pre["splits"]["test"]

    train_out = train["X_scaled"].copy()
    train_out[cfg.label_column] = train["y"]
    train_out.to_csv(os.path.join(processed_dir, "train.csv"), index=False)

    val_out = val["X_scaled"].copy()
    val_out[cfg.label_column] = val["y"]
    val_out.to_csv(os.path.join(processed_dir, "val.csv"), index=False)

    test_out = test["X_scaled"].copy()
    test_out[cfg.label_column] = test["y"]
    test_out.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    print("Preprocessing completed. Saved processed splits to:", processed_dir)

