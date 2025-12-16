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
| 7 | `train.py` | Model selection & training | ✅ |
| 8 | — | Model evaluation & interpretation | 🔲 |
| 9 | — | Optimization & design use-case | 🔲 |

## Key Results

| Model | Val R² | Test R² |
|-------|--------|---------|
| **Random Forest** | **0.693** | **0.417** |
| SVR (RBF) | 0.555 | — |

## Repository Layout

```
├── preprocessing.py          # Data cleaning & split pipeline
├── eda.py                    # Exploratory plots
├── features.py               # Canonical feature loader
├── feature_importance.py     # Baseline importance analysis
├── train.py                  # Model selection & training
├── dataset.csv               # Raw dataset
├── plan.txt                  # Project roadmap
├── data/
│   ├── processed/            # train.csv, val.csv, test.csv, cleaned_full.csv
│   ├── eda/                  # histograms.png, correlation_heatmap.png, scatter_plots.png
│   ├── feature_importance/   # baseline_importance.csv
│   ├── models/               # results.json, test_predictions.csv
│   └── archive/              # Original dataset backup
└── contextual papers/        # Reference literature
```

## Usage

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# Run pipeline
python preprocessing.py       # 1. Preprocess data
python eda.py                 # 2. Generate EDA figures
python feature_importance.py  # 3. Compute feature importances
python train.py               # 4. Train and select best model
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
