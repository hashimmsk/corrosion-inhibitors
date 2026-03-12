"""
RDKit 2D molecular descriptors for corrosion inhibitor formulations.

Computes ~200 RDKit descriptors from SMILES. If SMILES are not in the dataset,
generates surrogate SMILES from C# and EO for linear alcohol ethoxylates
(a common surfactant type).

Uses Permutation Feature Importance (PFI) to select top ~40 descriptors.
Reference: Pham et al., RSC Adv. 2024 (QSPR model for organic inhibitors).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# RDKit imports (optional - fail gracefully if not installed)
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def surfactant_to_smiles(C: float, EO: float) -> str:
    """
    Generate surrogate SMILES for linear alcohol ethoxylate from C# and EO.

    Formula: R-O-(CH2CH2O)n-H where R = alkyl chain (C# carbons).
    Used when dataset has no SMILES column.
    """
    C = max(1, int(np.nan_to_num(C, nan=1)))  # Avoid C=0, NaN->1
    EO = max(0, int(np.nan_to_num(EO, nan=0)))
    alkyl = "C" * C  # Linear chain
    if EO == 0:
        return alkyl + "O"
    eo_chain = "CCO" * EO  # Each EO unit is -O-CH2-CH2-
    return alkyl + "O" + eo_chain


def get_rdkit_descriptor_names() -> List[str]:
    """Return list of RDKit 2D descriptor names (~200)."""
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required. Install with: pip install rdkit")
    return [x[0] for x in Descriptors._descList]


def compute_descriptors_for_smiles(smiles: str) -> Optional[dict]:
    """Compute all RDKit descriptors for a single SMILES. Returns None if invalid."""
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required. Install with: pip install rdkit")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(get_rdkit_descriptor_names())
    values = calc.CalcDescriptors(mol)
    return dict(zip(calc.GetDescriptorNames(), values))


def add_rdkit_descriptors(
    df: pd.DataFrame,
    smiles_col: Optional[str] = None,
    C_col: str = "C#",
    EO_col: str = "EO",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add RDKit 2D descriptors to dataframe.

    If smiles_col exists, use it. Otherwise generate surrogate SMILES from C# and EO.
    Returns (df_with_descriptors, list_of_descriptor_column_names).
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required. Install with: pip install rdkit")

    descriptor_names = get_rdkit_descriptor_names()
    desc_values = {name: [] for name in descriptor_names}

    for idx, row in df.iterrows():
        if smiles_col and smiles_col in df.columns and pd.notna(row.get(smiles_col)):
            smiles = str(row[smiles_col]).strip()
        else:
            C = row.get(C_col, 1)
            EO = row.get(EO_col, 0)
            if pd.isna(C):
                C = 1
            if pd.isna(EO):
                EO = 0
            smiles = surfactant_to_smiles(float(C), float(EO))

        desc = compute_descriptors_for_smiles(smiles)
        if desc is None:
            # Fallback: use NaN for failed parses
            for name in descriptor_names:
                desc_values[name].append(np.nan)
        else:
            for name in descriptor_names:
                desc_values[name].append(desc.get(name, np.nan))

    desc_df = pd.DataFrame(desc_values, index=df.index)
    result = pd.concat([df, desc_df], axis=1)
    return result, descriptor_names


def select_top_descriptors_pfi(
    X: pd.DataFrame,
    y: pd.Series,
    n_top: int = 40,
    n_repeats: int = 5,
    random_state: int = 0,
) -> List[str]:
    """
    Use Permutation Feature Importance (PFI) with Gradient Boosting to select
    top N descriptors. Returns list of descriptor column names.
    """
    # Handle inf/nan - replace inf with nan, then impute
    X_clean = X.replace([np.inf, -np.inf], np.nan)
    # Drop columns that are all NaN (imputer would fail or produce bad values)
    valid_cols = X_clean.columns[X_clean.notna().any()]
    X_clean = X_clean[valid_cols]
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_clean)
    # Clip extreme values that can cause overflow in StandardScaler
    X_imputed = np.clip(X_imputed, -1e10, 1e10)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=2,
        random_state=random_state,
    )
    model.fit(X_scaled, y)

    result = permutation_importance(
        model, X_scaled, y, n_repeats=n_repeats, random_state=random_state
    )
    importances = result.importances_mean

    indices = np.argsort(importances)[::-1]
    top_indices = indices[: min(n_top, len(valid_cols))]
    col_names = list(valid_cols)
    return [col_names[i] for i in top_indices]


def build_extended_dataset(
    cleaned_df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    n_pfi_descriptors: int = 40,
    random_state: int = 0,
) -> dict:
    """
    Build extended dataset with original 6 features + top 40 RDKit descriptors.

    Returns dict with:
      - extended_feature_columns: list of 6 + 40 column names
      - train_X, train_y, val_X, val_y, test_X, test_y (DataFrames/Series)
      - imputer, scaler (fitted on train)
      - rdkit_columns: list of selected RDKit descriptor names
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required. Install with: pip install rdkit")

    from preprocessing import FEATURE_COLUMNS, LABEL_COLUMN

    # Add RDKit descriptors to full cleaned df
    df_ext, desc_names = add_rdkit_descriptors(cleaned_df)

    # Split by indices (we need to preserve the split from preprocessing)
    train_df = df_ext.iloc[train_indices]
    val_df = df_ext.iloc[val_indices]
    test_df = df_ext.iloc[test_indices]

    # PFI selection on TRAIN only
    X_train_desc = train_df[desc_names]
    y_train = train_df[LABEL_COLUMN]
    top_desc = select_top_descriptors_pfi(
        X_train_desc, y_train, n_top=n_pfi_descriptors, random_state=random_state
    )

    extended_cols = list(FEATURE_COLUMNS) + top_desc

    return {
        "extended_feature_columns": extended_cols,
        "rdkit_columns": top_desc,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "desc_names": desc_names,
    }


if __name__ == "__main__":
    """Demo: run preprocess_extended (full pipeline)."""
    if not RDKIT_AVAILABLE:
        print("RDKit not installed. Run: pip install rdkit")
        exit(1)
    import subprocess
    import sys
    root = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(root / "preprocess_extended.py")], cwd=root, check=True)
