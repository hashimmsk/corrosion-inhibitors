"""
Exploratory data analysis utilities for the corrosion inhibitor dataset.

This module loads the cleaned dataset produced by the preprocessing pipeline
and generates common exploratory plots:

- Histograms for key features and the inhibition efficiency label.
- Correlation heatmap across features (including IE).
- Simple bivariate scatter plots (IE vs Concentration / pH).

Figures are written to ``data/eda`` relative to the project root so that they
can be versioned or referenced in reports.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from preprocessing import PreprocessingConfig, locate_dataset_file, preprocess_dataset


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
  """
  Return the project root directory.

  If this file lives inside a subdirectory (e.g. ``src/``), the parent folder
  containing the ``data`` directory is treated as the project root. Otherwise
  the current directory is returned.
  """

  current = Path(__file__).resolve().parent
  if (current / "data").exists():
    return current
  return current.parent


def load_dataset_for_eda(
    base_dir: Path,
) -> Tuple[pd.DataFrame, Optional[Path]]:
  """
  Load the dataset for EDA.

  Priority:
  1. Use ``data/processed/cleaned_full.csv`` if it exists.
  2. Otherwise run the preprocessing pipeline to obtain ``dataset_for_eda``.

  Returns:
      dataframe, source_path (if loaded from disk; None if generated in-memory)
  """

  processed_dir = base_dir / "data" / "processed"
  cleaned_path = processed_dir / "cleaned_full.csv"

  if cleaned_path.exists():
    return pd.read_csv(cleaned_path), cleaned_path

  dataset_path = locate_dataset_file(str(base_dir))
  config = PreprocessingConfig(dataset_path=dataset_path)
  processed = preprocess_dataset(config)
  return processed["dataset_for_eda"], None


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def _ensure_output_dir(base_dir: Path) -> Path:
  out_dir = base_dir / "data" / "eda"
  out_dir.mkdir(parents=True, exist_ok=True)
  return out_dir


def plot_histograms(df: pd.DataFrame, columns: Iterable[str], out_dir: Path) -> Path:
  columns = [col for col in columns if col in df.columns]
  if not columns:
    raise ValueError("No valid columns supplied for histogram plotting.")

  n_cols = 3
  n_rows = int(np.ceil(len(columns) / n_cols))

  sns.set_style("whitegrid")
  fig, axes = plt.subplots(
      n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), constrained_layout=True
  )
  axes = np.atleast_1d(axes).flatten()

  for ax, col in zip(axes, columns):
    sns.histplot(df[col].dropna(), kde=False, ax=ax, color="tab:blue", edgecolor="black")
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
  for ax in axes[len(columns):]:
    ax.set_visible(False)

  fig.suptitle("Feature Distributions", fontsize=16)
  output_path = out_dir / "histograms.png"
  fig.savefig(output_path, dpi=300, bbox_inches="tight")
  plt.close(fig)
  return output_path


def plot_correlation_heatmap(df: pd.DataFrame, columns: Iterable[str], out_dir: Path) -> Path:
  columns = [col for col in columns if col in df.columns]
  if not columns:
    raise ValueError("No valid columns supplied for correlation heatmap.")

  corr = df[columns].corr(method="pearson")

  sns.set_style("white")
  fig, ax = plt.subplots(figsize=(10, 8))
  sns.heatmap(
      corr,
      annot=True,
      fmt=".2f",
      cmap="vlag",
      center=0.0,
      linewidths=0.5,
      square=True,
      cbar_kws={"shrink": 0.8, "label": "Pearson correlation"},
      ax=ax,
  )
  ax.set_title("Feature Correlation Heatmap", pad=12)

  output_path = out_dir / "correlation_heatmap.png"
  fig.savefig(output_path, dpi=300, bbox_inches="tight")
  plt.close(fig)
  return output_path


def plot_scatter_pairs(
    df: pd.DataFrame,
    pairs: Iterable[Tuple[str, str]],
    out_dir: Path,
) -> Path:
  valid_pairs = [
      (x, y) for x, y in pairs if x in df.columns and y in df.columns
  ]
  if not valid_pairs:
    raise ValueError("No valid feature pairs provided for scatter plotting.")

  n_cols = 2
  n_rows = int(np.ceil(len(valid_pairs) / n_cols))

  sns.set_style("ticks")
  fig, axes = plt.subplots(
      n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), constrained_layout=True
  )
  axes = np.atleast_1d(axes).flatten()

  for ax, (x_col, y_col) in zip(axes, valid_pairs):
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, color="tab:green", edgecolor="black")
    ax.set_title(f"{y_col} vs {x_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
  for ax in axes[len(valid_pairs):]:
    ax.set_visible(False)

  output_path = out_dir / "scatter_plots.png"
  fig.savefig(output_path, dpi=300, bbox_inches="tight")
  plt.close(fig)
  return output_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_eda() -> None:
  base_dir = _project_root()
  output_dir = _ensure_output_dir(base_dir)

  df, source = load_dataset_for_eda(base_dir)
  if source:
    print(f"Loaded dataset for EDA from {source}")
  else:
    print("Generated dataset for EDA via preprocessing pipeline.")

  feature_columns = ["C#", "Mw", "HLB", "EO", "Conc", "pH", "IE"]

  hist_path = plot_histograms(df, feature_columns, output_dir)
  heatmap_path = plot_correlation_heatmap(df, feature_columns, output_dir)
  scatter_pairs = [("Conc", "IE"), ("HLB", "IE"), ("EO", "IE")]
  if "medium" not in df.columns:
    scatter_pairs.insert(1, ("pH", "IE"))

  scatter_path = plot_scatter_pairs(df, pairs=scatter_pairs, out_dir=output_dir)

  print("EDA artifacts saved:")
  print(f" - Histograms: {hist_path}")
  print(f" - Correlation heatmap: {heatmap_path}")
  print(f" - Scatter plots: {scatter_path}")


if __name__ == "__main__":
  run_eda()

