# Corrosion Inhibitors – ML Pipeline

Data-driven workflow for predicting corrosion inhibitor performance (inhibition efficiency, **IE**) from molecular formulation features and test conditions. The current codebase delivers the preprocessing, exploratory analysis, and baseline feature selection stages of the overall plan.

## Current Capabilities (Steps 1–6)

1. **Preprocessing (`preprocessing.py`)**
   - Loads raw data from `dataset.csv` or the Excel equivalent.
   - Cleans rows (drops `No`, applies `IE = IE * AA / 100`, removes `AA`, duplicates, empty columns, and rows with missing IE).
   - Splits data into train/validation/test (70/15/15), performs train-only mean imputation and standard scaling.
   - Writes outputs to `data/processed/{cleaned_full,train,val,test}.csv`.

2. **EDA (`eda.py`)**
   - Consumes the cleaned dataset and generates histograms, correlation heatmap, and scatter plots for the canonical feature set.
   - Outputs figures under `data/eda/`.

3. **Feature Loading (`features.py`)**
   - Provides a single entry point to access `(X, y)` splits with the canonical features (`C#`, `Mw`, `HLB`, `EO`, `Conc`, `pH`) and the label (`IE`).
   - Automatically regenerates processed CSVs if they are missing.

4. **Baseline Feature Importance (`feature_importance.py`)**
   - Fits linear regression and random forest baselines.
  - Computes coefficient, tree-based, and permutation importances.
   - Saves the ranked summary to `data/feature_importance/baseline_importance.csv`.

These steps correspond to sections 1–6 in `plan.txt`. Model development (steps 7–9) will follow.

## Repository Layout

```
.
├── preprocessing.py          # Data cleaning & split pipeline
├── eda.py                    # Exploratory plots
├── features.py               # Canonical feature loader
├── feature_importance.py     # Baseline importance analysis
├── data/
│   ├── processed/            # Cleaned CSV splits (generated)
│   ├── eda/                  # EDA figures (generated)
│   └── feature_importance/   # Importance CSVs (generated)
├── dataset.csv               # Raw dataset (local copy)
└── plan.txt                  # Project roadmap & status
```

## Environment & Usage

1. Create a virtual environment (optional but recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt  # if/when requirements file is added
   ```

2. Run preprocessing (generates `data/processed/`):
   ```powershell
   python preprocessing.py
   ```

3. Generate EDA figures:
   ```powershell
   python eda.py
   ```

4. Compute baseline feature importances:
   ```powershell
   python feature_importance.py
   ```

## Next Steps

- Implement model selection & training experiments (`train.py`).
- Evaluate & interpret models against validation/test splits.
- Optimize inhibitor design recommendation workflow.

For detailed milestones, refer to `plan.txt`.
