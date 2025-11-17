"""
Feature loading utilities for the corrosion inhibitor project.

This module centralizes how train/validation/test splits are loaded and how
the canonical feature set is exposed to downstream modeling code. It relies on
the preprocessing pipeline to produce ``data/processed/{train,val,test}.csv``;
if those files are missing it will invoke the preprocessing step automatically.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from preprocessing import FEATURE_COLUMNS, LABEL_COLUMN, PreprocessingConfig, locate_dataset_file, preprocess_dataset


Split = Tuple[pd.DataFrame, pd.Series]


def _project_root() -> Path:
  current = Path(__file__).resolve().parent
  if (current / "data").exists():
    return current
  return current.parent


def _processed_dir(root: Path) -> Path:
  directory = root / "data" / "processed"
  directory.mkdir(parents=True, exist_ok=True)
  return directory


def ensure_processed_splits(root: Path) -> Dict[str, pd.DataFrame]:
  processed_dir = _processed_dir(root)
  required_files = {
      "train": processed_dir / "train.csv",
      "val": processed_dir / "val.csv",
      "test": processed_dir / "test.csv",
  }

  if all(path.exists() for path in required_files.values()):
    return {name: pd.read_csv(path) for name, path in required_files.items()}

  dataset_path = locate_dataset_file(str(root))
  config = PreprocessingConfig(dataset_path=dataset_path)
  processed = preprocess_dataset(config)

  processed_dir.mkdir(parents=True, exist_ok=True)

  for split_name, split_data in processed["splits"].items():
    df = split_data["X_scaled"].copy()
    df[LABEL_COLUMN] = split_data["y"]
    output_path = processed_dir / f"{split_name}.csv"
    df.to_csv(output_path, index=False)

  cleaned_path = processed_dir / "cleaned_full.csv"
  processed["cleaned_df"].to_csv(cleaned_path, index=False)

  return ensure_processed_splits(root)


def load_splits() -> Tuple[Split, Split, Split]:
  root = _project_root()
  splits = ensure_processed_splits(root)

  def extract(split_df: pd.DataFrame) -> Split:
    missing = [col for col in FEATURE_COLUMNS if col not in split_df.columns]
    if missing:
      raise KeyError(f"Processed split is missing expected feature columns: {missing}")
    if LABEL_COLUMN not in split_df.columns:
      raise KeyError(f"Processed split is missing label column '{LABEL_COLUMN}'")
    X = split_df[list(FEATURE_COLUMNS)]
    y = split_df[LABEL_COLUMN]
    return X, y

  return extract(splits["train"]), extract(splits["val"]), extract(splits["test"])


def load_full_dataset() -> pd.DataFrame:
  root = _project_root()
  processed_dir = _processed_dir(root)
  cleaned_path = processed_dir / "cleaned_full.csv"

  if cleaned_path.exists():
    return pd.read_csv(cleaned_path)

  dataset_path = locate_dataset_file(str(root))
  config = PreprocessingConfig(dataset_path=dataset_path)
  processed = preprocess_dataset(config)

  processed["cleaned_df"].to_csv(cleaned_path, index=False)
  return processed["cleaned_df"]

