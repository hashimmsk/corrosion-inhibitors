"""
Leave-One-Medium-Out Cross-Validation

Test whether the model can generalize to unseen corrosive environments.

Experiment Design:
1. Train on HCl + NaCl → Test on CPS
2. Train on HCl + CPS → Test on NaCl  
3. Train on NaCl + CPS → Test on HCl

This answers: "Can the model predict IE for a completely unseen medium?"

If the model generalizes well → it learned underlying chemistry
If it fails → it's learning medium-specific patterns (pH as proxy)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "leave_one_medium_out"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = ["C#", "Mw", "HLB", "EO", "Conc", "pH"]
SEED = 0

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150


def load_full_dataset():
    """Load the full cleaned dataset with medium labels."""
    df = pd.read_csv(DATA_DIR / "processed" / "cleaned_full.csv")
    return df


def get_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def train_and_evaluate(X_train, y_train, X_test, y_test, model_type="random_forest"):
    """Train a model and evaluate on test set."""
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=600,
            max_depth=6,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1
        )
    else:  # SVR
        model = SVR(kernel="rbf", C=100.0, gamma="scale", epsilon=0.1)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = get_metrics(y_test, y_pred)
    
    return model, y_pred, metrics


def run_leave_one_medium_out():
    """Run the leave-one-medium-out experiment."""
    print("\n" + "="*70)
    print("LEAVE-ONE-MEDIUM-OUT CROSS-VALIDATION")
    print("="*70)
    print("\nQuestion: Can the model generalize to unseen corrosive environments?")
    
    # Load data
    df = load_full_dataset()
    print(f"\nDataset: {len(df)} samples")
    print(f"Mediums: {df['medium'].value_counts().to_dict()}")
    
    # Define experiments
    experiments = [
        {"train": ["HCl", "NaCl"], "test": "CPS"},
        {"train": ["HCl", "CPS"], "test": "NaCl"},
        {"train": ["NaCl", "CPS"], "test": "HCl"},
    ]
    
    results = []
    all_predictions = []
    
    for exp in experiments:
        train_mediums = exp["train"]
        test_medium = exp["test"]
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: Train on {' + '.join(train_mediums)} → Test on {test_medium}")
        print("="*70)
        
        # Split data
        train_mask = df["medium"].isin(train_mediums)
        test_mask = df["medium"] == test_medium
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        print(f"\n  Train samples: {len(train_df)} ({', '.join([f'{m}: {(train_df.medium==m).sum()}' for m in train_mediums])})")
        print(f"  Test samples: {len(test_df)} ({test_medium})")
        
        # Extract features and target
        X_train_raw = train_df[FEATURE_COLUMNS]
        y_train = train_df["IE"]
        X_test_raw = test_df[FEATURE_COLUMNS]
        y_test = test_df["IE"]
        
        # Preprocessing: Impute and scale (fit on train only)
        imputer = SimpleImputer(strategy="mean")
        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(X_train_raw),
            columns=FEATURE_COLUMNS
        )
        X_test_imputed = pd.DataFrame(
            imputer.transform(X_test_raw),
            columns=FEATURE_COLUMNS
        )
        
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train_imputed),
            columns=FEATURE_COLUMNS
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test_imputed),
            columns=FEATURE_COLUMNS
        )
        
        # Show pH range difference (key insight)
        train_ph = train_df["pH"].unique()
        test_ph = test_df["pH"].unique()
        print(f"\n  pH in training: {sorted(train_ph)}")
        print(f"  pH in test: {sorted(test_ph)}")
        print(f"  → Model must {'EXTRAPOLATE' if max(test_ph) > max(train_ph) or min(test_ph) < min(train_ph) else 'INTERPOLATE'} to unseen pH range")
        
        # Train and evaluate both models
        exp_results = {
            "train_mediums": train_mediums,
            "test_medium": test_medium,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "train_ph_range": [float(min(train_ph)), float(max(train_ph))],
            "test_ph": float(test_ph[0]),
        }
        
        for model_type in ["random_forest", "svr"]:
            model, y_pred, metrics = train_and_evaluate(
                X_train, y_train, X_test, y_test, model_type
            )
            
            exp_results[f"{model_type}_metrics"] = metrics
            
            print(f"\n  {model_type.upper()}:")
            print(f"    R²:   {metrics['r2']:.3f}")
            print(f"    MAE:  {metrics['mae']:.2f}")
            print(f"    RMSE: {metrics['rmse']:.2f}")
            
            # Store predictions for visualization
            if model_type == "random_forest":
                pred_df = pd.DataFrame({
                    "experiment": f"{'+'.join(train_mediums)} → {test_medium}",
                    "test_medium": test_medium,
                    "y_true": y_test.values,
                    "y_pred": y_pred,
                    "residual": y_test.values - y_pred,
                })
                all_predictions.append(pred_df)
        
        results.append(exp_results)
    
    # Save results
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved: results.json")
    
    # Save predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    print(f"✓ Saved: predictions.csv")
    
    return results, predictions_df


def compare_with_general_model(results):
    """Compare leave-one-out results with the general model."""
    print("\n" + "="*70)
    print("COMPARISON WITH GENERAL MODEL")
    print("="*70)
    
    # General model metrics (from previous training)
    general_rf = {"r2": 0.417, "mae": 15.19, "rmse": 20.09}
    general_svr = {"r2": 0.370, "mae": 16.49, "rmse": 20.88}
    
    print("\n  General Model (trained on ALL mediums, tested on held-out 15%):")
    print(f"    Random Forest: R² = {general_rf['r2']:.3f}, RMSE = {general_rf['rmse']:.2f}")
    print(f"    SVR:           R² = {general_svr['r2']:.3f}, RMSE = {general_svr['rmse']:.2f}")
    
    print("\n  Leave-One-Medium-Out (trained on 2 mediums, tested on 3rd):")
    print("-" * 70)
    print(f"  {'Experiment':<25} | {'RF R²':<8} | {'RF RMSE':<8} | {'SVR R²':<8} | {'SVR RMSE':<8}")
    print("-" * 70)
    
    for exp in results:
        exp_name = f"{'+'.join(exp['train_mediums'])} → {exp['test_medium']}"
        rf = exp["random_forest_metrics"]
        svr = exp["svr_metrics"]
        print(f"  {exp_name:<25} | {rf['r2']:<8.3f} | {rf['rmse']:<8.2f} | {svr['r2']:<8.3f} | {svr['rmse']:<8.2f}")
    
    print("-" * 70)
    
    # Average across experiments
    avg_rf_r2 = np.mean([r["random_forest_metrics"]["r2"] for r in results])
    avg_rf_rmse = np.mean([r["random_forest_metrics"]["rmse"] for r in results])
    avg_svr_r2 = np.mean([r["svr_metrics"]["r2"] for r in results])
    avg_svr_rmse = np.mean([r["svr_metrics"]["rmse"] for r in results])
    
    print(f"  {'AVERAGE':<25} | {avg_rf_r2:<8.3f} | {avg_rf_rmse:<8.2f} | {avg_svr_r2:<8.3f} | {avg_svr_rmse:<8.2f}")
    print("-" * 70)
    
    return {
        "general_rf": general_rf,
        "general_svr": general_svr,
        "lomo_avg_rf_r2": avg_rf_r2,
        "lomo_avg_svr_r2": avg_svr_r2,
    }


def create_visualizations(results, predictions_df, comparison):
    """Create visualizations for the experiment."""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Figure 1: Predicted vs Actual for each experiment
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = {'HCl': '#E94F37', 'NaCl': '#2E86AB', 'CPS': '#4DAA57'}
    
    for i, (exp, ax) in enumerate(zip(results, axes)):
        test_medium = exp["test_medium"]
        rf_r2 = exp["random_forest_metrics"]["r2"]
        
        # Get predictions for this experiment
        mask = predictions_df["test_medium"] == test_medium
        pred = predictions_df[mask]
        
        ax.scatter(pred["y_true"], pred["y_pred"], 
                   c=colors[test_medium], alpha=0.7, s=60, 
                   edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        lims = [min(pred["y_true"].min(), pred["y_pred"].min()) - 10,
                max(pred["y_true"].max(), pred["y_pred"].max()) + 10]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2)
        
        train_str = " + ".join(exp["train_mediums"])
        ax.set_xlabel("Experimental IE (%)")
        ax.set_ylabel("Predicted IE (%)")
        ax.set_title(f"Train: {train_str}\nTest: {test_medium} (R² = {rf_r2:.3f})")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
    
    plt.suptitle("Leave-One-Medium-Out: Can the Model Generalize to Unseen Environments?", 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predicted_vs_actual.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ Saved: predicted_vs_actual.png")
    
    # Figure 2: R² Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    experiments = [f"{'+'.join(r['train_mediums'])} → {r['test_medium']}" for r in results]
    rf_r2_values = [r["random_forest_metrics"]["r2"] for r in results]
    svr_r2_values = [r["svr_metrics"]["r2"] for r in results]
    
    x = np.arange(len(experiments))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rf_r2_values, width, label='Random Forest', 
                   color='#2E86AB', edgecolor='black')
    bars2 = ax.bar(x + width/2, svr_r2_values, width, label='SVR', 
                   color='#E94F37', edgecolor='black')
    
    # Add general model reference line
    ax.axhline(y=comparison["general_rf"]["r2"], color='#2E86AB', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'General RF (R²={comparison["general_rf"]["r2"]:.3f})')
    ax.axhline(y=comparison["general_svr"]["r2"], color='#E94F37', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'General SVR (R²={comparison["general_svr"]["r2"]:.3f})')
    
    ax.set_ylabel('R² Score')
    ax.set_title('Leave-One-Medium-Out vs General Model Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(-0.5, 1.0)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02 if height >= 0 else height - 0.08,
                f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02 if height >= 0 else height - 0.08,
                f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "r2_comparison.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ Saved: r2_comparison.png")
    
    # Figure 3: Residual distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (exp, ax) in enumerate(zip(results, axes)):
        test_medium = exp["test_medium"]
        mask = predictions_df["test_medium"] == test_medium
        residuals = predictions_df.loc[mask, "residual"]
        
        ax.hist(residuals, bins=15, color=colors[test_medium], 
                edgecolor='black', alpha=0.8)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=residuals.mean(), color='orange', linestyle='-', linewidth=2,
                   label=f'Mean: {residuals.mean():.1f}')
        
        ax.set_xlabel('Residual (Actual - Predicted)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Test: {test_medium}')
        ax.legend()
    
    plt.suptitle("Residual Distributions by Held-Out Medium", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_distributions.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ Saved: residual_distributions.png")


def generate_report(results, comparison):
    """Generate a text report with conclusions."""
    report_path = OUTPUT_DIR / "report.txt"
    
    with open(report_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("LEAVE-ONE-MEDIUM-OUT CROSS-VALIDATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("RESEARCH QUESTION\n")
        f.write("-"*70 + "\n")
        f.write("Can ML models for corrosion inhibitors generalize across corrosive\n")
        f.write("environments, or are they environment-specific?\n\n")
        
        f.write("EXPERIMENTAL DESIGN\n")
        f.write("-"*70 + "\n")
        f.write("Train on 2 mediums, test on the 3rd (completely unseen medium)\n\n")
        
        f.write("RESULTS SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Experiment':<30} | {'RF R²':<8} | {'SVR R²':<8}\n")
        f.write("-"*70 + "\n")
        
        for exp in results:
            exp_name = f"{'+'.join(exp['train_mediums'])} → {exp['test_medium']}"
            rf_r2 = exp["random_forest_metrics"]["r2"]
            svr_r2 = exp["svr_metrics"]["r2"]
            f.write(f"{exp_name:<30} | {rf_r2:<8.3f} | {svr_r2:<8.3f}\n")
        
        f.write("-"*70 + "\n")
        avg_rf = comparison["lomo_avg_rf_r2"]
        avg_svr = comparison["lomo_avg_svr_r2"]
        f.write(f"{'AVERAGE (Leave-One-Out)':<30} | {avg_rf:<8.3f} | {avg_svr:<8.3f}\n")
        f.write(f"{'General Model (15% test)':<30} | {comparison['general_rf']['r2']:<8.3f} | {comparison['general_svr']['r2']:<8.3f}\n")
        f.write("\n")
        
        f.write("KEY FINDINGS\n")
        f.write("-"*70 + "\n\n")
        
        # Analyze results
        rf_r2_list = [r["random_forest_metrics"]["r2"] for r in results]
        
        if avg_rf < 0:
            f.write("1. MODEL FAILS TO GENERALIZE ACROSS MEDIUMS\n")
            f.write("   Average R² < 0 indicates the model performs worse than\n")
            f.write("   predicting the mean. The model cannot extrapolate to unseen\n")
            f.write("   corrosive environments.\n\n")
        elif avg_rf < comparison["general_rf"]["r2"] * 0.5:
            f.write("1. MODEL SHOWS POOR CROSS-MEDIUM GENERALIZATION\n")
            f.write("   Leave-one-out R² is significantly lower than the general model,\n")
            f.write("   suggesting the model learns medium-specific patterns rather than\n")
            f.write("   transferable chemistry.\n\n")
        else:
            f.write("1. MODEL SHOWS REASONABLE CROSS-MEDIUM GENERALIZATION\n")
            f.write("   The model maintains reasonable performance on unseen mediums.\n\n")
        
        f.write("2. pH AS A PROXY FOR MEDIUM\n")
        f.write("   The three mediums have distinct, non-overlapping pH values:\n")
        f.write("   - HCl:  pH = 0.5  (acidic)\n")
        f.write("   - NaCl: pH = 4.9  (neutral)\n")
        f.write("   - CPS:  pH = 12.5 (alkaline)\n\n")
        f.write("   When testing on an unseen medium, the model must extrapolate\n")
        f.write("   to a pH range it never encountered during training.\n\n")
        
        f.write("3. IMPLICATIONS FOR MODEL DEPLOYMENT\n")
        if avg_rf < 0.2:
            f.write("   - The model should NOT be used to predict IE for mediums\n")
            f.write("     not present in the training data.\n")
            f.write("   - Medium-specific models or additional features may be needed\n")
            f.write("     to capture environment-dependent behavior.\n")
        else:
            f.write("   - The model may be cautiously used for similar environments.\n")
            f.write("   - Validation on any new medium type is recommended.\n")
        
        f.write("\n\n")
        f.write("CONCLUSION\n")
        f.write("-"*70 + "\n")
        if avg_rf < 0.2:
            f.write("The leave-one-medium-out experiment reveals that the current model\n")
            f.write("does NOT generalize well to unseen corrosive environments. This\n")
            f.write("suggests that pH (which encodes medium type) is being used as a\n")
            f.write("proxy rather than the model learning transferable relationships\n")
            f.write("between molecular properties and inhibition efficiency.\n")
        else:
            f.write("The model shows some ability to generalize across mediums, though\n")
            f.write("performance is reduced compared to the general model.\n")
    
    print(f"  ✓ Saved: report.txt")
    return report_path


def main():
    """Main execution."""
    # Run experiment
    results, predictions_df = run_leave_one_medium_out()
    
    # Compare with general model
    comparison = compare_with_general_model(results)
    
    # Create visualizations
    create_visualizations(results, predictions_df, comparison)
    
    # Generate report
    generate_report(results, comparison)
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nOutputs saved in: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print("\nFiles created:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")
    
    # Print key takeaway
    avg_rf_r2 = np.mean([r["random_forest_metrics"]["r2"] for r in results])
    print("\n" + "="*70)
    print("KEY TAKEAWAY")
    print("="*70)
    if avg_rf_r2 < 0:
        print(f"\nAverage R² = {avg_rf_r2:.3f} (NEGATIVE)")
        print("The model CANNOT generalize to unseen corrosive environments.")
        print("It learns medium-specific patterns rather than transferable chemistry.")
    elif avg_rf_r2 < 0.2:
        print(f"\nAverage R² = {avg_rf_r2:.3f} (POOR)")
        print("The model shows limited ability to generalize across environments.")
    else:
        print(f"\nAverage R² = {avg_rf_r2:.3f}")
        print("The model shows some cross-environment generalization ability.")


if __name__ == "__main__":
    main()
