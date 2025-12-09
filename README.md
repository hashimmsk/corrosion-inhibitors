# Corrosion Inhibitors – ML Pipeline

Data-driven workflow for predicting corrosion inhibitor performance (inhibition efficiency, **IE**) from molecular formulation features and test conditions. The pipeline covers data preprocessing, exploratory analysis, feature engineering, model training, and evaluation.

## Project Status

✅ **Steps 1–8 Complete** | 🔲 Step 9 (Optimization) Pending

**Best Model:** Random Forest Regressor  
**Test Performance:** R² = 0.449 | MAE = 14.56 | RMSE = 19.53

## Pipeline Overview

| Step | Script | Description | Status |
|------|--------|-------------|--------|
| 1-4 | `preprocessing.py` | Data cleaning, IE correction, train/val/test split | ✅ |
| 5 | `eda.py` | Histograms, correlation heatmap, scatter plots | ✅ |
| 6 | `feature_importance.py` | Baseline importance analysis | ✅ |
| 7 | `train.py` | Model selection & hyperparameter tuning | ✅ |
| 8 | `evaluation.py` | Model diagnostics & interpretation | ✅ |
| 9 | — | Optimization & design use-case | 🔲 |

## Key Results

### Model Comparison (7 algorithms tested)

| Model | Validation R² | Test R² |
|-------|---------------|---------|
| **Random Forest** | **0.675** | **0.449** |
| Gradient Boosting | 0.616 | — |
| SVR | 0.522 | — |
| Linear/Ridge/Lasso/ElasticNet | < 0 | — |

Tree-based models significantly outperform linear models, indicating non-linear relationships between features and IE.

### Feature Importance

Operating conditions (pH, Conc) are the most influential predictors, followed by molecular properties (Mw, EO, HLB, C#).

## Repository Layout

```
.
├── preprocessing.py          # Data cleaning & split pipeline
├── eda.py                    # Exploratory plots
├── features.py               # Canonical feature loader
├── feature_importance.py     # Baseline importance analysis
├── train.py                  # Model selection & training
├── evaluation.py             # Model diagnostics & interpretation
├── dataset.csv               # Raw dataset
├── plan.txt                  # Project roadmap & status
├── data/
│   ├── processed/            # Cleaned CSV splits
│   ├── eda/                  # EDA figures
│   ├── feature_importance/   # Importance rankings
│   ├── models/               # Trained model & results
│   │   ├── random_forest_model.pkl
│   │   ├── model_results.json
│   │   ├── best_model_test_metrics.json
│   │   └── report.txt
│   └── evaluation/           # Diagnostic plots & metrics
│       ├── metrics.json
│       ├── residuals_val.png
│       ├── residuals_test.png
│       ├── learning_curve.png
│       ├── permutation_importance_val.png
│       ├── pdp_pH.png
│       ├── pdp_Conc.png
│       ├── pdp_Mw.png
│       ├── pdp_EO.png
│       └── report.txt
└── contextual papers/        # Reference literature
```

## Usage

### Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Run the Full Pipeline

```bash
# 1. Preprocess data (generates data/processed/)
python preprocessing.py

# 2. Generate EDA figures (generates data/eda/)
python eda.py

# 3. Compute baseline feature importances
python feature_importance.py

# 4. Train and select best model (generates data/models/)
python train.py

# 5. Evaluate model and generate diagnostics (generates data/evaluation/)
python evaluation.py
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

- Implement optimization workflow for inhibitor design recommendations (Step 9)
- Explore SHAP values for individual prediction explanations
- Add prediction intervals for uncertainty quantification

For detailed progress notes, see `plan.txt` and reports in `data/models/report.txt` and `data/evaluation/report.txt`.
