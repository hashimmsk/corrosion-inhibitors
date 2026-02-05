"""
Step 8: Model Evaluation & Interpretation

This script provides interpretability analysis for the trained models:
1. SHAP values - explain individual predictions and global feature importance
2. Partial Dependence Plots - show marginal effect of each feature on IE
3. Error Analysis - identify patterns in prediction errors

Outputs saved to: data/interpretation/
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
import joblib

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "interpretation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = ["C#", "Mw", "HLB", "EO", "Conc", "pH"]
SEED = 0

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150


def load_data():
    """Load train, validation, and test data."""
    train_df = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "processed" / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "processed" / "test.csv")
    
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["IE"]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["IE"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["IE"]
    
    # Combine train + val for final model
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    
    return X_trainval, y_trainval, X_test, y_test, test_df


def load_best_model_params():
    """Load best hyperparameters from results.json."""
    with open(DATA_DIR / "models" / "results.json") as f:
        results = json.load(f)
    return results["best_params"]


def train_final_model(X_trainval, y_trainval, params):
    """Train the final Random Forest model with best params."""
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X_trainval, y_trainval)
    return model


# ============================================================
# 1. SHAP ANALYSIS
# ============================================================
def run_shap_analysis(model, X_trainval, X_test):
    """Compute and visualize SHAP values."""
    print("\n" + "="*60)
    print("SHAP ANALYSIS")
    print("="*60)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values for test set
    shap_values = explainer.shap_values(X_test)
    
    # 1. Summary plot (beeswarm)
    print("\n  Creating SHAP summary plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance\n(Impact on Predicted IE)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ Saved: shap_summary.png")
    
    # 2. Bar plot (mean absolute SHAP values)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Mean |SHAP Value| by Feature")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_importance_bar.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ Saved: shap_importance_bar.png")
    
    # 3. Dependence plots for top features
    print("\n  Creating SHAP dependence plots...")
    top_features = ["pH", "Conc", "HLB", "EO"]
    
    for feature in top_features:
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(feature, shap_values, X_test, show=False, ax=ax)
        plt.title(f"SHAP Dependence: {feature}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"shap_dependence_{feature}.png", dpi=300, 
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    print(f"  ✓ Saved: shap_dependence_*.png for {', '.join(top_features)}")
    
    # 4. Save SHAP values to CSV
    shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLUMNS)
    shap_df.to_csv(OUTPUT_DIR / "shap_values_test.csv", index=False)
    print(f"  ✓ Saved: shap_values_test.csv")
    
    # 5. Compute mean absolute SHAP importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    importance_df.to_csv(OUTPUT_DIR / "shap_importance.csv", index=False)
    
    print("\n  SHAP Feature Importance (Mean |SHAP|):")
    for _, row in importance_df.iterrows():
        print(f"    {row['Feature']:<6}: {row['Mean_Abs_SHAP']:.3f}")
    
    return shap_values


# ============================================================
# 2. PARTIAL DEPENDENCE PLOTS
# ============================================================
def run_partial_dependence(model, X_trainval):
    """Create partial dependence plots for all features."""
    print("\n" + "="*60)
    print("PARTIAL DEPENDENCE PLOTS")
    print("="*60)
    
    # Individual PDP for each feature
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(FEATURE_COLUMNS):
        print(f"  Computing PDP for {feature}...")
        PartialDependenceDisplay.from_estimator(
            model, X_trainval, [feature], ax=axes[i],
            kind='average', line_kw={'color': '#2E86AB', 'linewidth': 2}
        )
        axes[i].set_title(f'Partial Dependence: {feature}')
        axes[i].set_ylabel('Predicted IE')
    
    plt.suptitle('Partial Dependence Plots\n(Marginal Effect of Each Feature on IE)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "partial_dependence_all.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ Saved: partial_dependence_all.png")
    
    # 2D interaction plots for key pairs
    print("\n  Creating 2D interaction plots...")
    interaction_pairs = [("pH", "Conc"), ("HLB", "EO"), ("Conc", "HLB")]
    
    for feat1, feat2 in interaction_pairs:
        fig, ax = plt.subplots(figsize=(8, 6))
        idx1 = FEATURE_COLUMNS.index(feat1)
        idx2 = FEATURE_COLUMNS.index(feat2)
        PartialDependenceDisplay.from_estimator(
            model, X_trainval, [(idx1, idx2)], ax=ax,
            kind='average'
        )
        plt.title(f'2D Partial Dependence: {feat1} × {feat2}')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"pdp_2d_{feat1}_{feat2}.png", dpi=300, 
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    print(f"  ✓ Saved: pdp_2d_*.png for interaction pairs")


# ============================================================
# 3. ERROR ANALYSIS
# ============================================================
def run_error_analysis(model, X_test, y_test, test_df):
    """Analyze prediction errors by medium, feature ranges, etc."""
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    # Get predictions
    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred
    abs_errors = np.abs(residuals)
    
    # Create analysis dataframe
    error_df = pd.DataFrame({
        'y_true': y_test.values,
        'y_pred': y_pred,
        'residual': residuals,
        'abs_error': abs_errors,
        'medium': test_df['medium'].values if 'medium' in test_df.columns else 'Unknown'
    })
    
    # Add features
    for col in FEATURE_COLUMNS:
        error_df[col] = X_test[col].values
    
    error_df.to_csv(OUTPUT_DIR / "error_analysis.csv", index=False)
    print(f"  ✓ Saved: error_analysis.csv")
    
    # 1. Error by Medium
    print("\n  Error by Medium:")
    print("-" * 50)
    medium_stats = error_df.groupby('medium').agg({
        'abs_error': ['mean', 'std', 'max'],
        'residual': 'mean',
        'y_true': 'count'
    }).round(2)
    medium_stats.columns = ['MAE', 'Std', 'Max_Error', 'Mean_Bias', 'N_Samples']
    print(medium_stats.to_string())
    medium_stats.to_csv(OUTPUT_DIR / "error_by_medium.csv")
    
    # Plot error by medium
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Boxplot of absolute errors
    ax = axes[0]
    mediums = error_df['medium'].unique()
    colors = {'HCl': '#E94F37', 'NaCl': '#2E86AB', 'CPS': '#4DAA57'}
    for i, medium in enumerate(mediums):
        data = error_df[error_df['medium'] == medium]['abs_error']
        bp = ax.boxplot([data], positions=[i], widths=0.6, patch_artist=True)
        bp['boxes'][0].set_facecolor(colors.get(medium, 'gray'))
    ax.set_xticks(range(len(mediums)))
    ax.set_xticklabels(mediums)
    ax.set_ylabel('Absolute Error')
    ax.set_title('Prediction Error by Medium')
    
    # Scatter: actual vs predicted colored by error
    ax = axes[1]
    scatter = ax.scatter(error_df['y_true'], error_df['y_pred'], 
                         c=error_df['abs_error'], cmap='Reds', 
                         s=80, edgecolors='black', linewidth=0.5, alpha=0.8)
    lims = [min(error_df['y_true'].min(), error_df['y_pred'].min()) - 5,
            max(error_df['y_true'].max(), error_df['y_pred'].max()) + 5]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2)
    ax.set_xlabel('Actual IE (%)')
    ax.set_ylabel('Predicted IE (%)')
    ax.set_title('Predictions Colored by Error Magnitude')
    plt.colorbar(scatter, ax=ax, label='Absolute Error')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "error_analysis_plots.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ Saved: error_analysis_plots.png")
    
    # 2. Identify worst predictions
    print("\n  Top 5 Worst Predictions:")
    print("-" * 50)
    worst = error_df.nlargest(5, 'abs_error')[['y_true', 'y_pred', 'abs_error', 'medium', 'pH', 'Conc']]
    print(worst.to_string(index=False))
    
    # 3. Error vs feature values
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(FEATURE_COLUMNS):
        ax = axes[i]
        ax.scatter(error_df[feature], error_df['residual'], 
                   alpha=0.6, c='#2E86AB', edgecolors='black', linewidth=0.3)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
        ax.set_xlabel(feature)
        ax.set_ylabel('Residual')
        ax.set_title(f'Residual vs {feature}')
    
    plt.suptitle('Residual Analysis by Feature\n(Look for patterns indicating model bias)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_vs_features.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ Saved: residual_vs_features.png")
    
    return error_df


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*60)
    print("STEP 8: MODEL EVALUATION & INTERPRETATION")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    X_trainval, y_trainval, X_test, y_test, test_df = load_data()
    print(f"  Train+Val samples: {len(X_trainval)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Load best params and train model
    print("\nTraining final model with best hyperparameters...")
    params = load_best_model_params()
    model = train_final_model(X_trainval, y_trainval, params)
    print(f"  ✓ Model trained: RandomForest(n_estimators={params['n_estimators']}, "
          f"max_depth={params['max_depth']})")
    
    # Save model for later use
    joblib.dump(model, OUTPUT_DIR / "final_model.joblib")
    print(f"  ✓ Saved: final_model.joblib")
    
    # Run analyses
    shap_values = run_shap_analysis(model, X_trainval, X_test)
    run_partial_dependence(model, X_trainval)
    error_df = run_error_analysis(model, X_test, y_test, test_df)
    
    # Summary
    print("\n" + "="*60)
    print("✓ INTERPRETATION COMPLETE!")
    print("="*60)
    print(f"\nOutputs saved in: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print("\nFiles created:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
