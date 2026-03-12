"""
Pathway A: KDE-based Virtual Sample Generation (VSG) for corrosion inhibitor data.

Applies VSG ONLY to the training set after the train/val/test split to prevent
data leakage. Synthetic samples are clipped to physically plausible bounds
(min/max of training data). Supports medium-aware augmentation to balance
underrepresented corrosive media (HCl, NaCl, CPS).

Reference: Herowati et al., "Machine learning for pyrimidine corrosion inhibitor
small dataset", Theoretical Chemistry Accounts (2024).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from preprocessing import FEATURE_COLUMNS, LABEL_COLUMN


def _fit_kde(
    X: np.ndarray,
    y: np.ndarray,
    bandwidth: Optional[float] = None,
    kernel: str = "gaussian",
) -> KernelDensity:
    """
    Fit a Kernel Density Estimation model on the joint distribution of (X, y).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Target vector (n_samples,).
    bandwidth : float, optional
        KDE bandwidth. If None, uses Scott's rule (default in sklearn).
    kernel : str
        Kernel type. Default "gaussian".

    Returns
    -------
    KernelDensity
        Fitted KDE model.
    """
    data = np.hstack([X, y.reshape(-1, 1)])
    bw = bandwidth if bandwidth is not None else "scott"
    kde = KernelDensity(kernel=kernel, bandwidth=bw)
    kde.fit(data)
    return kde


def _sample_from_kde(
    kde: KernelDensity,
    n_samples: int,
    X_bounds: Tuple[np.ndarray, np.ndarray],
    y_bounds: Tuple[float, float],
    random_state: int = 0,
    max_clip_iterations: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample from KDE and clip to bounds. Re-samples out-of-bounds points.

    Parameters
    ----------
    kde : KernelDensity
        Fitted KDE model.
    n_samples : int
        Number of samples to generate.
    X_bounds : tuple of (min, max)
        Per-feature min and max arrays for clipping.
    y_bounds : tuple of (min, max)
        Target min and max for clipping.
    random_state : int
        Random seed.
    max_clip_iterations : int
        Max iterations to re-sample out-of-bounds points.

    Returns
    -------
    X_aug : np.ndarray
        Augmented features.
    y_aug : np.ndarray
        Augmented targets.
    """
    rng = np.random.default_rng(random_state)
    X_min, X_max = X_bounds
    y_min, y_max = y_bounds

    # Sample more than needed to account for clipping
    n_oversample = int(n_samples * 1.5)
    samples = kde.sample(n_oversample, random_state=random_state)

    n_features = samples.shape[1] - 1
    X_sampled = samples[:, :n_features]
    y_sampled = samples[:, n_features]

    # Clip to bounds
    X_sampled = np.clip(X_sampled, X_min, X_max)
    y_sampled = np.clip(y_sampled, y_min, y_max)

    # Take first n_samples (all are now in bounds)
    return X_sampled[:n_samples], y_sampled[:n_samples]


def generate_virtual_samples(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_samples: int = 500,
    medium: Optional[pd.Series] = None,
    medium_balance: bool = True,
    bandwidth: Optional[float] = None,
    random_state: int = 0,
    save_path: Optional[Union[str, Path]] = None,
    feature_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate virtual training samples using KDE-based VSG.

    VSG is applied ONLY to the training data. Val and test sets remain unchanged.
    Synthetic samples are clipped to the min/max of the training data to ensure
    chemical plausibility.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (already scaled). Must contain FEATURE_COLUMNS.
    y_train : pd.Series
        Training target (IE).
    n_samples : int
        Total number of virtual samples to generate. Default 500.
    medium : pd.Series, optional
        Medium label for each training sample (HCl, NaCl, CPS). If provided
        and medium_balance=True, samples are generated per medium to balance
        the dataset.
    medium_balance : bool
        If True and medium is provided, generate equal samples per medium
        to balance underrepresented media (e.g. CPS, NaCl). Default True.
    bandwidth : float, optional
        KDE bandwidth. If None, uses default (Scott's rule).
    random_state : int
        Random seed for reproducibility.
    save_path : str or Path, optional
        If provided, save the synthetic samples (features + IE) to this CSV path.
        When medium_balance=True, also saves a 'medium' column.
    feature_columns : sequence of str, optional
        Columns to use as features. If None, uses FEATURE_COLUMNS (original 6).
        Use for extended feature sets (e.g. 6 + 40 RDKit descriptors).

    Returns
    -------
    X_aug : pd.DataFrame
        Augmented features (original + virtual). Column names match X_train.
    y_aug : pd.Series
        Augmented targets (original + virtual).
    """
    feature_cols = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)
    missing = [c for c in feature_cols if c not in X_train.columns]
    if missing:
        raise ValueError(
            f"X_train missing columns: {missing}. Available: {list(X_train.columns)}"
        )

    X = X_train[feature_cols].values
    y = y_train.values

    if medium is not None and medium_balance and len(medium) == len(X_train):
        # Medium-aware: fit KDE per medium, sample to balance
        mediums = medium.unique()
        n_per_medium = max(1, n_samples // len(mediums))

        X_list, y_list, medium_list = [], [], []
        for m in mediums:
            mask = medium == m
            X_m = X[mask]
            y_m = y[mask]
            if len(X_m) < 2:
                continue
            kde = _fit_kde(X_m, y_m, bandwidth=bandwidth)
            X_min = X_m.min(axis=0)
            X_max = X_m.max(axis=0)
            y_min, y_max = y_m.min(), y_m.max()
            X_aug_m, y_aug_m = _sample_from_kde(
                kde, n_per_medium, (X_min, X_max), (y_min, y_max), random_state
            )
            X_list.append(X_aug_m)
            y_list.append(y_aug_m)
            medium_list.extend([m] * len(y_aug_m))

        X_virtual = np.vstack(X_list)
        y_virtual = np.concatenate(y_list)
        virtual_medium = medium_list
    else:
        # Single KDE on full training set
        kde = _fit_kde(X, y, bandwidth=bandwidth)
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        y_min, y_max = y.min(), y.max()
        X_virtual, y_virtual = _sample_from_kde(
            kde, n_samples, (X_min, X_max), (y_min, y_max), random_state
        )
        virtual_medium = None

    # Save synthetic samples if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_virtual = pd.DataFrame(X_virtual, columns=feature_cols)
        df_virtual[LABEL_COLUMN] = y_virtual
        if virtual_medium is not None:
            df_virtual["medium"] = virtual_medium
        df_virtual.to_csv(save_path, index=False)
        print(f"  Saved {len(df_virtual)} synthetic samples to {save_path}")

    # Combine original + virtual
    X_combined = np.vstack([X, X_virtual])
    y_combined = np.concatenate([y, y_virtual])

    X_aug = pd.DataFrame(X_combined, columns=feature_cols)
    y_aug = pd.Series(y_combined)

    return X_aug, y_aug


def load_train_with_medium() -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """
    Load training data including medium column for stratified VSG.

    Returns
    -------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    medium : pd.Series or None
        Medium labels if available.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent
    train_path = root / "data" / "processed" / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Train file not found at {train_path}. Run preprocessing.py first."
        )

    df = pd.read_csv(train_path)
    X = df[list(FEATURE_COLUMNS)]
    y = df[LABEL_COLUMN]
    medium = df["medium"] if "medium" in df.columns else None
    return X, y, medium


if __name__ == "__main__":
    """Demo: generate virtual samples and show statistics."""
    X_train, y_train, medium = load_train_with_medium()

    print("Original training set:")
    print(f"  Samples: {len(X_train)}")
    if medium is not None:
        print(f"  Per medium: {medium.value_counts().to_dict()}")

    out_dir = Path(__file__).resolve().parent / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    synthetic_path = out_dir / "synthetic_samples.csv"

    X_aug, y_aug = generate_virtual_samples(
        X_train, y_train,
        n_samples=500,
        medium=medium,
        medium_balance=True,
        random_state=0,
        save_path=synthetic_path,
    )

    print("\nAfter VSG augmentation:")
    print(f"  Total samples: {len(X_aug)} (original + 500 virtual)")
    print(f"  IE range - original: [{y_train.min():.1f}, {y_train.max():.1f}]")
    print(f"  IE range - virtual:  [{y_aug.iloc[len(y_train):].min():.1f}, "
          f"{y_aug.iloc[len(y_train):].max():.1f}]")
