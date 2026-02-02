"""
Medium-Specific Model Training

Train separate Random Forest and SVR models for each corrosive medium (HCl, NaCl, CPS).
This allows us to compare whether specialized models outperform the general model.

Workflow (for each medium):
1) Load medium-specific train/val/test splits
2) Tune RF and SVR on TRAIN using RandomizedSearchCV
3) Evaluate on VAL and pick winner
4) Refit winner on TRAIN+VAL
5) Evaluate on TEST
6) Save results to data/models/medium_specific/{medium}/

Output Structure:
    data/models/medium_specific/
        ├── HCl/
        │   ├── results.json
        │   ├── test_predictions.csv
        │   └── report.txt
        ├── NaCl/
        │   └── ...
        └── CPS/
            └── ...
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import parallel_backend
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.svm import SVR

# Configuration
SEED = 0
CV_SPLITS = 3
N_ITER = 18
MEDIUMS = ["HCl", "NaCl", "CPS"]

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "medium_specific"
OUTPUT_BASE = PROJECT_ROOT / "data" / "models" / "medium_specific"

FEATURE_COLUMNS = ["C#", "Mw", "HLB", "EO", "Conc", "pH"]
LABEL_COLUMN = "IE"


def get_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def build_models():
    """Define models and hyperparameter search spaces."""
    return {
        "random_forest": (
            RandomForestRegressor(random_state=SEED, n_jobs=1),
            {
                "n_estimators": [300, 600],
                "max_depth": [None, 6, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "svr": (
            SVR(kernel="rbf", cache_size=2000),
            {
                "C": [10.0, 50.0, 100.0],
                "gamma": ["scale", 0.1, 0.01],
                "epsilon": [0.01, 0.05, 0.1],
            },
        ),
    }


def load_medium_splits(medium_name):
    """Load train/val/test splits for a specific medium."""
    medium_dir = DATA_DIR / medium_name
    
    train_df = pd.read_csv(medium_dir / "train.csv")
    val_df = pd.read_csv(medium_dir / "val.csv")
    test_df = pd.read_csv(medium_dir / "test.csv")
    
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[LABEL_COLUMN]
    
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[LABEL_COLUMN]
    
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[LABEL_COLUMN]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_medium_models(medium_name):
    """Train models for a specific medium."""
    print(f"\n{'='*70}")
    print(f"TRAINING MODELS FOR: {medium_name}")
    print(f"{'='*70}\n")
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_medium_splits(medium_name)
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples\n")
    
    # CV setup
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    models = build_models()
    results = []
    
    # Train and tune each model
    with parallel_backend("threading"):
        for model_name, (estimator, param_space) in models.items():
            print(f"Training {model_name}...")
            
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_space,
                n_iter=N_ITER,
                scoring="r2",
                cv=cv,
                random_state=SEED,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            
            best_model = search.best_estimator_
            val_pred = best_model.predict(X_val)
            val_metrics = get_metrics(y_val, val_pred)
            
            results.append({
                "model": model_name,
                "val_metrics": val_metrics,
                "best_params": search.best_params_,
                "best_estimator": best_model,
            })
            
            print(f"  Val R² = {val_metrics['r2']:.4f} | Val RMSE = {val_metrics['rmse']:.3f}")
    
    # Refit ALL models on TRAIN+VAL and evaluate on TEST
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    
    print("\nRefitting all models on TRAIN+VAL and evaluating on TEST...")
    for r in results:
        model = r["best_estimator"]
        model.fit(X_trainval, y_trainval)
        test_pred = model.predict(X_test)
        r["test_metrics"] = get_metrics(y_test, test_pred)
        r["test_predictions"] = test_pred
        print(f"  {r['model']}: TEST R² = {r['test_metrics']['r2']:.4f} | TEST RMSE = {r['test_metrics']['rmse']:.3f}")
    
    # Pick best model by validation R²
    best = max(results, key=lambda r: r["val_metrics"]["r2"])
    best_name = best["model"]
    test_metrics = best["test_metrics"]
    
    print(f"\nBest model: {best_name}")
    
    # Save outputs
    output_dir = OUTPUT_BASE / medium_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save results.json with test metrics for ALL models
    report_json = {
        "medium": medium_name,
        "best_model": best_name,
        "best_params": best["best_params"],
        "val_metrics_best": best["val_metrics"],
        "test_metrics": test_metrics,
        "all_models": [
            {
                "model": r["model"],
                "val_metrics": r["val_metrics"],
                "test_metrics": r["test_metrics"],
                "best_params": r["best_params"],
            }
            for r in results
        ],
        "config": {
            "seed": SEED,
            "cv_splits": CV_SPLITS,
            "n_iter": N_ITER,
        },
        "dataset_sizes": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(report_json, f, indent=2)
    
    # 2. Save test_predictions.csv (for best model)
    pred_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": best["test_predictions"],
        "residual": y_test.values - best["test_predictions"],
    })
    
    # Add pH_original and medium if available
    test_df = pd.read_csv(DATA_DIR / medium_name / "test.csv")
    if "pH_original" in test_df.columns:
        pred_df["pH_original"] = test_df["pH_original"].values
    if "medium" in test_df.columns:
        pred_df["medium"] = test_df["medium"].values
    
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)
    
    # 3. Save report.txt
    save_report(output_dir / "report.txt", medium_name, results, best, test_metrics, X_train, X_val, X_test)
    
    print(f"✓ Results saved to: {output_dir.relative_to(PROJECT_ROOT)}")
    
    return report_json


def save_report(report_path, medium_name, results, best, test_metrics, X_train, X_val, X_test):
    """Generate a human-readable report."""
    with open(report_path, "w") as f:
        f.write("="*70 + "\n")
        f.write(f"MEDIUM-SPECIFIC MODEL TRAINING: {medium_name}\n")
        f.write("="*70 + "\n\n")
        
        # Dataset info
        f.write("DATASET INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Medium: {medium_name}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Val samples: {len(X_val)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Features: {', '.join(FEATURE_COLUMNS)}\n")
        f.write(f"Target: {LABEL_COLUMN} (Inhibition Efficiency %)\n\n")
        
        # Training approach
        f.write("TRAINING APPROACH\n")
        f.write("-"*70 + "\n")
        f.write("1. Load medium-specific train/val/test splits\n")
        f.write("2. Tune Random Forest and SVR on TRAIN using RandomizedSearchCV\n")
        f.write(f"   (3-fold CV, {N_ITER} iterations)\n")
        f.write("3. Evaluate tuned models on VAL, pick winner by R²\n")
        f.write("4. Refit winner on TRAIN+VAL combined\n")
        f.write("5. Final evaluation on held-out TEST set\n\n")
        
        # Hyperparameters
        f.write("HYPERPARAMETER SEARCH SPACES\n")
        f.write("-"*70 + "\n")
        f.write("Random Forest:\n")
        f.write("  - n_estimators: [300, 600]\n")
        f.write("  - max_depth: [None, 6, 10]\n")
        f.write("  - min_samples_leaf: [1, 2, 4]\n\n")
        f.write("SVR (RBF kernel):\n")
        f.write("  - C: [10.0, 50.0, 100.0]\n")
        f.write("  - gamma: ['scale', 0.1, 0.01]\n")
        f.write("  - epsilon: [0.01, 0.05, 0.1]\n\n")
        
        # Results - Validation
        f.write("MODEL COMPARISON (VALIDATION SET)\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Model':<20} | {'Val R²':<10} | {'Val MAE':<10} | {'Val RMSE':<10}\n")
        f.write("-"*70 + "\n")
        
        for r in sorted(results, key=lambda x: x["val_metrics"]["r2"], reverse=True):
            m = r["val_metrics"]
            f.write(f"{r['model']:<20} | {m['r2']:<10.3f} | {m['mae']:<10.2f} | {m['rmse']:<10.2f}\n")
        
        f.write("\n")
        f.write(f"Winner (by Val R²): {best['model']}\n\n")
        
        # Results - Test (ALL models)
        f.write("MODEL COMPARISON (TEST SET - ALL MODELS)\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Model':<20} | {'Test R²':<10} | {'Test MAE':<10} | {'Test RMSE':<10}\n")
        f.write("-"*70 + "\n")
        
        for r in sorted(results, key=lambda x: x["test_metrics"]["r2"], reverse=True):
            m = r["test_metrics"]
            f.write(f"{r['model']:<20} | {m['r2']:<10.3f} | {m['mae']:<10.2f} | {m['rmse']:<10.2f}\n")
        
        f.write("\n")
        
        # Best hyperparameters
        f.write("BEST HYPERPARAMETERS\n")
        f.write("-"*70 + "\n")
        for param, value in best["best_params"].items():
            f.write(f"  {param}: {value}\n")
        
        # Test results
        f.write("\n")
        f.write("TEST SET PERFORMANCE (HELD-OUT)\n")
        f.write("-"*70 + "\n")
        f.write(f"R²:   {test_metrics['r2']:.3f}\n")
        f.write(f"MAE:  {test_metrics['mae']:.2f}\n")
        f.write(f"RMSE: {test_metrics['rmse']:.2f}\n\n")
        
        # Interpretation
        f.write("INTERPRETATION\n")
        f.write("-"*70 + "\n")
        f.write(f"The {best['model']} model explains {test_metrics['r2']*100:.1f}% of the variance\n")
        f.write(f"in IE for {medium_name} medium on unseen test data.\n\n")
        
        if test_metrics['r2'] < best["val_metrics"]["r2"]:
            f.write("Note: Test R² is lower than validation R², which is normal and expected.\n")
            f.write("This indicates some overfitting to the training distribution, but the\n")
            f.write("model still generalizes reasonably to the held-out test set.\n\n")
        
        # Next steps
        f.write("COMPARISON WITH GENERAL MODEL\n")
        f.write("-"*70 + "\n")
        f.write("Compare this result with the general model (data/models/results.json)\n")
        f.write("to determine if medium-specific models provide better performance.\n\n")
        f.write("Key question: Does specializing by medium improve predictions?\n")
        f.write("- If YES → Deploy separate models for each medium\n")
        f.write("- If NO  → Use the general model (simpler deployment)\n")


def create_summary_report(all_results):
    """Create a summary comparing all mediums."""
    summary_path = OUTPUT_BASE / "COMPARISON_summary.txt"
    
    with open(summary_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("MEDIUM-SPECIFIC MODEL TRAINING - SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("BEST MODEL BY MEDIUM (selected by Val R²)\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Medium':<10} | {'Best Model':<15} | {'Val R²':<10} | {'Test R²':<10} | {'Test RMSE':<10}\n")
        f.write("-"*70 + "\n")
        
        for result in all_results:
            medium = result["medium"]
            model = result["best_model"]
            val_r2 = result["val_metrics_best"]["r2"]
            test_r2 = result["test_metrics"]["r2"]
            test_rmse = result["test_metrics"]["rmse"]
            
            f.write(f"{medium:<10} | {model:<15} | {val_r2:<10.3f} | {test_r2:<10.3f} | {test_rmse:<10.2f}\n")
        
        f.write("\n")
        f.write("ALL MODELS TEST PERFORMANCE\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Medium':<10} | {'Model':<15} | {'Test R²':<10} | {'Test MAE':<10} | {'Test RMSE':<10}\n")
        f.write("-"*70 + "\n")
        
        for result in all_results:
            for model_result in result["all_models"]:
                medium = result["medium"]
                model = model_result["model"]
                test_r2 = model_result["test_metrics"]["r2"]
                test_mae = model_result["test_metrics"]["mae"]
                test_rmse = model_result["test_metrics"]["rmse"]
                f.write(f"{medium:<10} | {model:<15} | {test_r2:<10.3f} | {test_mae:<10.2f} | {test_rmse:<10.2f}\n")
        
        f.write("\n")
        f.write("COMPARISON WITH GENERAL MODEL\n")
        f.write("-"*70 + "\n")
        f.write("General Model (trained on all mediums):\n")
        f.write("  Random Forest - Val R²: 0.693, Test R²: 0.417\n")
        f.write("  SVR           - Val R²: 0.556, Test R²: 0.370\n\n")
        
        f.write("Medium-Specific Best Models:\n")
        for result in all_results:
            f.write(f"  {result['medium']:<6} ({result['best_model']}) - Test R²: {result['test_metrics']['r2']:.3f}")
            diff = result['test_metrics']['r2'] - 0.417
            f.write(f"  (Δ vs general RF = {diff:+.3f})\n")
        
        f.write("\n")
        f.write("INTERPRETATION\n")
        f.write("-"*70 + "\n")
        f.write("If medium-specific models outperform the general model:\n")
        f.write("  → Chemistry varies significantly across pH environments\n")
        f.write("  → Recommend deploying separate models per medium\n\n")
        
        f.write("If general model performs similarly or better:\n")
        f.write("  → General model captures medium differences via pH feature\n")
        f.write("  → Simpler to deploy a single model\n")
    
    print(f"\n{'='*70}")
    print(f"✓ Summary report: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"{'='*70}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("MEDIUM-SPECIFIC MODEL TRAINING")
    print("="*70)
    print(f"\nTraining Random Forest and SVR for: {', '.join(MEDIUMS)}")
    print(f"Output directory: {OUTPUT_BASE.relative_to(PROJECT_ROOT)}\n")
    
    all_results = []
    
    for medium in MEDIUMS:
        result = train_medium_models(medium)
        all_results.append(result)
    
    # Create comparison summary
    create_summary_report(all_results)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print("\nResults saved in:")
    for medium in MEDIUMS:
        medium_dir = OUTPUT_BASE / medium
        print(f"  {medium_dir.relative_to(PROJECT_ROOT)}/")
        print(f"    - results.json")
        print(f"    - test_predictions.csv")
        print(f"    - report.txt")
    print(f"\n  {(OUTPUT_BASE / 'COMPARISON_summary.txt').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
