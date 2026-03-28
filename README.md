# Corrosion Inhibitors – ML Pipeline

Predict corrosion inhibitor performance (inhibition efficiency, IE) from molecular formulation features and test conditions.

## Pipeline Overview

| Step | Script | Description | Status |
|------|--------|-------------|--------|
| 1-4 | `preprocessing.py` | Data cleaning, IE correction, train/val/test split | Done |
| 5 | `eda.py` | Histograms, correlation heatmap, scatter plots | Done |
| 6 | `features.py`, `feature_importance.py` | Feature loading & baseline importance | Done |
| 7 | `train.py` | Model selection & training (RF & SVR) | Done |
| 7a | `medium_specific_preprocessing.py` | Split by medium & preprocess separately | Done |
| 7b | `train_medium_specific.py` | Train models for each medium (HCl, NaCl, CPS) | Done |
| 7c | `analyze_feature_importance.py` | Compare feature importance across mediums | Done |
| 8 | `model_interpretation.py` | SHAP analysis, partial dependence, error analysis | Done |
| 9 | `optimize_inhibitor.py` | Dosage optimization, formulation recommendation | Done |
| A | `virtual_sample_generation.py` | KDE-based Virtual Sample Generation (Pathway A) | Done |
| B | `rdkit_descriptors.py` | RDKit 2D descriptors + PFI selection | Done |
| B | `train_improved.py` | Extended (6+20 RDKit), VSG, Extended+VSG with RF/GB/SVR | Done |

## Key Results

### General Model (All Mediums Combined)
| Model | Val R² | Test R² |
|-------|--------|---------|
| Random Forest | 0.693 | 0.417 |
| SVR (RBF) | 0.555 | 0.370 |

### Pathway A: With Virtual Sample Generation (VSG)
| Model | Val R² | Test R² |
|-------|--------|---------|
| Random Forest | 0.657 | 0.433 |
| Gradient Boosting | 0.633 | 0.458 |
| SVR | 0.597 | 0.419 |
(VSG: 200 virtual samples, medium-aware balancing; run `python train_improved.py`)

### Extended (6 + 20 RDKit descriptors)
| Model | Val R² | Test R² |
|-------|--------|---------|
| Random Forest | 0.653 | **0.489** |
| Gradient Boosting | 0.641 | 0.476 |
| SVR | 0.050 | 0.003 |
(RDKit descriptors from surrogate SMILES; PFI-selected top 20; run `python train_improved.py`)

### Extended + VSG (26 features, 200 samples)
| Model | Val R² | Test R² |
|-------|--------|---------|
| Random Forest | 0.601 | 0.448 |
| Gradient Boosting | 0.604 | 0.473 |
(run `python train_improved.py`)

### Medium-Specific Models
| Medium | Best Model | Val R² | Test R² |
|--------|------------|--------|---------|
| HCl | SVR | 0.141 | 0.036 |
| NaCl | Random Forest | 0.025 | 0.204 |
| CPS | Random Forest | 0.355 | -0.448 |

Key Finding: Medium-specific models underperformed the general model due to very small dataset sizes per medium (train: 37-82, test: 8-18 samples). The general model benefits from cross-medium learning.

## Repository Layout

```
├── preprocessing.py                    # Data cleaning & split pipeline
├── medium_specific_preprocessing.py    # Split by medium & preprocess
├── eda.py                              # Exploratory plots
├── features.py                         # Canonical feature loader
├── feature_importance.py               # Baseline importance analysis
├── train.py                            # General model training
├── train_medium_specific.py            # Medium-specific model training
├── analyze_feature_importance.py       # Feature importance comparison
├── generate_viz_figures.py             # Publication-quality figures
├── dataset.csv                         # Raw dataset
├── plan.txt                            # Project roadmap
├── data/
│   ├── processed/
│   │   ├── train.csv, val.csv, test.csv, cleaned_full.csv  # General splits
│   │   ├── extended_top20/             # Extended (6+20 RDKit) train/val/test
│   │   └── medium_specific/
│   │       ├── HCl/                    # HCl-specific data
│   │       ├── NaCl/                   # NaCl-specific data
│   │       └── CPS/                    # CPS-specific data
│   ├── models/
│   │   ├── results.json, test_predictions.csv  # General model
│   │   ├── improved/                            # VSG, Extended, Extended+VSG results
│   │   └── medium_specific/
│   │       ├── HCl/                    # HCl model results
│   │       ├── NaCl/                   # NaCl model results
│   │       ├── CPS/                    # CPS model results
│   │       └── feature_importance/     # Cross-medium analysis
│   ├── eda/                            # EDA plots
│   ├── feature_importance/             # Baseline importance
│   ├── viz_figures/                    # Publication figures
│   └── archive/                        # Original dataset backup
└── contextual papers/                  # Reference literature
```

## Usage

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# General Pipeline
python preprocessing.py                 # 1. Preprocess data
python eda.py                           # 2. Generate EDA figures
python feature_importance.py            # 3. Compute feature importances
python train.py                         # 4. Train general model

# Medium-Specific Pipeline
python medium_specific_preprocessing.py  # 5. Split by medium & preprocess
python train_medium_specific.py          # 6. Train medium-specific models
python analyze_feature_importance.py     # 7. Compare feature importance

# Generate Figures
python generate_viz_figures.py           # 8. Create publication figures

# Pathway A: Virtual Sample Generation
python virtual_sample_generation.py      # Demo VSG

# Extended + RDKit (improved)
pip install rdkit                       # Required for RDKit descriptors
python train_improved.py               # Runs VSG, Extended (6+20 RDKit), Extended+VSG with RF/GB/SVR
```

## Features

| Feature | Description |
|---------|-------------|
| C# | Carbon number |
| Mw | Molecular weight |
| HLB | Hydrophilic-lipophilic balance |
| EO | Ethylene oxide units |
| Conc | Inhibitor concentration |
| pH | Solution pH |
| IE | Inhibition efficiency (target) |

## Probable Future Enhancements

- External validation on new compounds
- Uncertainty quantification for predictions
- Deep learning comparison (if more data collected at the lab)
- Web app deployment for interactive predictions
