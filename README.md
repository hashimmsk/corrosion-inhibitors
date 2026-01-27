# Corrosion Inhibitors – ML Pipeline

Predict corrosion inhibitor performance (inhibition efficiency, **IE**) from molecular formulation features and test conditions.

## Project Status

✅ **Steps 1–7 Complete** | 🔲 Steps 8–9 Pending

**Best Model:** Random Forest  
**Test Performance:** R² = 0.417 | RMSE = 20.1

## Pipeline Overview

| Step | Script | Description | Status |
|------|--------|-------------|--------|
| 1-4 | `preprocessing.py` | Data cleaning, IE correction, train/val/test split | ✅ |
| 5 | `eda.py` | Histograms, correlation heatmap, scatter plots | ✅ |
| 6 | `features.py`, `feature_importance.py` | Feature loading & baseline importance | ✅ |
| 7 | `train.py` | Model selection & training (general model) | ✅ |
| 7a | `medium_specific_preprocessing.py` | Split by medium & preprocess separately | ✅ |
| 7b | `train_medium_specific.py` | Train models for each medium (HCl, NaCl, CPS) | ✅ |
| 7c | `analyze_feature_importance.py` | Compare feature importance across mediums | ✅ |
| 8 | — | Model evaluation & interpretation | 🔲 |
| 9 | — | Optimization & design use-case | 🔲 |

## Key Results

### General Model (All Mediums Combined)
| Model | Val R² | Test R² |
|-------|--------|---------|
| **Random Forest** | **0.693** | **0.417** |
| SVR (RBF) | 0.555 | — |

### Medium-Specific Models
| Medium | Best Model | Val R² | Test R² |
|--------|------------|--------|---------|
| HCl | SVR | 0.141 | 0.036 |
| NaCl | Random Forest | 0.025 | 0.204 |
| CPS | Random Forest | 0.355 | -0.448 |

**Key Finding:** Medium-specific models underperformed the general model due to very small dataset sizes per medium (train: 37-82, test: 8-18 samples). The general model benefits from cross-medium learning.

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
│   │   └── medium_specific/
│   │       ├── HCl/                    # HCl-specific data
│   │       ├── NaCl/                   # NaCl-specific data
│   │       └── CPS/                    # CPS-specific data
│   ├── models/
│   │   ├── results.json, test_predictions.csv  # General model
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
| **IE** | Inhibition efficiency (target) |

## Next Steps

- Model evaluation & interpretation (Step 8)
- Optimization workflow for inhibitor design (Step 9)
