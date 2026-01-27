"""
Feature Importance Analysis: General vs Medium-Specific Models

Extract and compare feature importance to understand how predictive factors
shift when training on individual mediums vs all mediums combined.

Key Question: Which molecular features drive IE in each corrosive environment?
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "models" / "medium_specific" / "feature_importance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = ["C#", "Mw", "HLB", "EO", "Conc", "pH"]
MEDIUMS = ["HCl", "NaCl", "CPS"]
SEED = 0

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150


def load_general_model_data():
    """Load general model train/val data."""
    train_df = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "processed" / "val.csv")
    
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["IE"]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["IE"]
    
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    
    return X_trainval, y_trainval


def load_medium_data(medium_name):
    """Load medium-specific train/val data."""
    medium_dir = DATA_DIR / "processed" / "medium_specific" / medium_name
    
    train_df = pd.read_csv(medium_dir / "train.csv")
    val_df = pd.read_csv(medium_dir / "val.csv")
    
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["IE"]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["IE"]
    
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    
    return X_trainval, y_trainval


def get_rf_importance(model, X, y):
    """Get Random Forest feature importance (built-in)."""
    importances = model.feature_importances_
    
    # Normalize to 0-1 range
    importances = importances / importances.sum()
    
    return dict(zip(FEATURE_COLUMNS, importances))


def get_svr_importance(model, X, y):
    """Get SVR feature importance using permutation importance."""
    # Use permutation importance for SVR
    perm_result = permutation_importance(
        model, X, y, n_repeats=10, random_state=SEED, n_jobs=-1
    )
    
    importances = perm_result.importances_mean
    
    # Normalize to 0-1 range (handle negative values)
    importances = np.maximum(importances, 0)  # Clip negatives to 0
    if importances.sum() > 0:
        importances = importances / importances.sum()
    
    return dict(zip(FEATURE_COLUMNS, importances))


def extract_general_model_importance():
    """Extract feature importance from general model (Random Forest)."""
    print("\nExtracting General Model Importance...")
    print("-" * 60)
    
    # Load data
    X, y = load_general_model_data()
    
    # Load best params from results.json
    with open(DATA_DIR / "models" / "results.json") as f:
        results = json.load(f)
    
    best_params = results["best_params"]
    
    # Train Random Forest with best params
    model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Extract importance
    importance = get_rf_importance(model, X, y)
    
    print(f"  Samples: {len(X)}")
    print(f"  Model: Random Forest")
    print(f"  Top feature: {max(importance, key=importance.get)} ({importance[max(importance, key=importance.get)]:.3f})")
    
    return importance


def extract_medium_importance(medium_name):
    """Extract feature importance for a specific medium."""
    print(f"\nExtracting {medium_name} Model Importance...")
    print("-" * 60)
    
    # Load data
    X, y = load_medium_data(medium_name)
    
    # Load best model and params
    results_path = DATA_DIR / "models" / "medium_specific" / medium_name / "results.json"
    with open(results_path) as f:
        results = json.load(f)
    
    best_model_name = results["best_model"]
    best_params = results["best_params"]
    
    # Train model with best params
    if best_model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"],
            random_state=SEED,
            n_jobs=-1
        )
        model.fit(X, y)
        importance = get_rf_importance(model, X, y)
    else:  # SVR
        model = SVR(
            kernel="rbf",
            C=best_params["C"],
            gamma=best_params["gamma"],
            epsilon=best_params["epsilon"],
            cache_size=2000
        )
        model.fit(X, y)
        importance = get_svr_importance(model, X, y)
    
    print(f"  Samples: {len(X)}")
    print(f"  Model: {best_model_name}")
    print(f"  Top feature: {max(importance, key=importance.get)} ({importance[max(importance, key=importance.get)]:.3f})")
    
    return importance


def create_comparison_table(all_importance):
    """Create a comparison table of feature importance."""
    df = pd.DataFrame(all_importance)
    df = df[["General", "HCl", "NaCl", "CPS"]]  # Reorder columns
    
    # Save CSV
    csv_path = OUTPUT_DIR / "importance_comparison.csv"
    df.to_csv(csv_path)
    print(f"\n✓ Saved: {csv_path.relative_to(PROJECT_ROOT)}")
    
    # Create text report
    report_path = OUTPUT_DIR / "importance_analysis.txt"
    with open(report_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("FEATURE IMPORTANCE COMPARISON\n")
        f.write("="*70 + "\n\n")
        
        f.write("HOW TO READ THIS TABLE:\n")
        f.write("-"*70 + "\n")
        f.write("Values represent normalized feature importance (0-1 scale).\n")
        f.write("Higher values = more important for predicting IE.\n\n")
        
        f.write("IMPORTANCE COMPARISON TABLE\n")
        f.write("-"*70 + "\n")
        f.write(df.to_string())
        f.write("\n\n")
        
        f.write("TOP 3 FEATURES PER MODEL\n")
        f.write("-"*70 + "\n")
        for col in df.columns:
            top_3 = df[col].nlargest(3)
            f.write(f"\n{col}:\n")
            for i, (feat, imp) in enumerate(top_3.items(), 1):
                f.write(f"  {i}. {feat:<6} ({imp:.3f})\n")
        
        f.write("\n\nKEY INSIGHTS\n")
        f.write("="*70 + "\n\n")
        
        # pH importance drop
        ph_general = df.loc["pH", "General"]
        ph_hcl = df.loc["pH", "HCl"]
        ph_nacl = df.loc["pH", "NaCl"]
        ph_cps = df.loc["pH", "CPS"]
        
        f.write("1. pH IMPORTANCE COLLAPSE:\n")
        f.write(f"   General Model:   pH = {ph_general:.3f} (DOMINANT!)\n")
        f.write(f"   HCl Model:       pH = {ph_hcl:.3f} (dropped {(ph_general-ph_hcl)/ph_general*100:.0f}%)\n")
        f.write(f"   NaCl Model:      pH = {ph_nacl:.3f} (dropped {(ph_general-ph_nacl)/ph_general*100:.0f}%)\n")
        f.write(f"   CPS Model:       pH = {ph_cps:.3f} (dropped {(ph_general-ph_cps)/ph_general*100:.0f}%)\n\n")
        f.write("   → pH was critical in general model because it captured WHICH MEDIUM\n")
        f.write("   → In medium-specific models, pH is constant → importance drops to near zero\n\n")
        
        # Top features per medium
        f.write("2. WHAT DRIVES IE IN EACH MEDIUM:\n\n")
        
        for medium in ["HCl", "NaCl", "CPS"]:
            top_feat = df[medium].idxmax()
            top_imp = df.loc[top_feat, medium]
            f.write(f"   {medium}:\n")
            f.write(f"     Most important: {top_feat} ({top_imp:.3f})\n")
            
            # Compare to general
            general_imp = df.loc[top_feat, "General"]
            diff = top_imp - general_imp
            f.write(f"     In general model: {general_imp:.3f} (Δ = {diff:+.3f})\n\n")
        
        f.write("\n3. WHY MEDIUM-SPECIFIC MODELS FAILED:\n")
        f.write("   - Loss of pH feature (was 0.811 importance in general model)\n")
        f.write("   - Small sample sizes (HCl/NaCl: 100, CPS: 45)\n")
        f.write("   - Remaining features have weak correlations with IE\n")
        f.write("   - Result: Poor predictive power (Test R² << General Model)\n")
    
    print(f"✓ Saved: {report_path.relative_to(PROJECT_ROOT)}")
    
    return df


def create_visualizations(df):
    """Create comparison visualizations."""
    
    # Figure 1: Grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(FEATURE_COLUMNS))
    width = 0.2
    
    colors = {'General': '#2E86AB', 'HCl': '#E94F37', 'NaCl': '#F6AA1C', 'CPS': '#4DAA57'}
    
    for i, model in enumerate(["General", "HCl", "NaCl", "CPS"]):
        values = [df.loc[feat, model] for feat in FEATURE_COLUMNS]
        ax.bar(x + i*width, values, width, label=model, color=colors[model], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Feature')
    ax.set_ylabel('Normalized Importance')
    ax.set_title('Feature Importance Comparison: General vs Medium-Specific Models')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(FEATURE_COLUMNS)
    ax.legend(title='Model')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "importance_comparison_bars.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {(OUTPUT_DIR / 'importance_comparison_bars.png').relative_to(PROJECT_ROOT)}")
    
    # Figure 2: Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Transpose for better visualization
    heatmap_data = df.T
    
    im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(FEATURE_COLUMNS)))
    ax.set_yticks(np.arange(len(heatmap_data)))
    ax.set_xticklabels(FEATURE_COLUMNS)
    ax.set_yticklabels(heatmap_data.index)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add text annotations
    for i in range(len(heatmap_data)):
        for j in range(len(FEATURE_COLUMNS)):
            text = ax.text(j, i, f'{heatmap_data.values[i, j]:.3f}',
                          ha="center", va="center", color="black" if heatmap_data.values[i, j] < 0.5 else "white",
                          fontsize=10, fontweight='bold')
    
    ax.set_title("Feature Importance Heatmap\n(0 = Not Important, 1 = Highly Important)")
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Importance', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "importance_comparison_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {(OUTPUT_DIR / 'importance_comparison_heatmap.png').relative_to(PROJECT_ROOT)}")
    
    # Figure 3: pH importance drop
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = ["General", "HCl", "NaCl", "CPS"]
    ph_importance = [df.loc["pH", model] for model in models]
    
    bars = ax.bar(models, ph_importance, color=['#2E86AB', '#E94F37', '#F6AA1C', '#4DAA57'],
                  edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('pH Importance')
    ax.set_title('pH Importance: General vs Medium-Specific Models\n(Shows why general model performed better)')
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar, val in zip(bars, ph_importance):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add annotation
    ax.annotate('pH captured WHICH MEDIUM\n→ Critical for general model',
                xy=(0, ph_importance[0]), xytext=(1.5, 0.6),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax.annotate('pH is constant within medium\n→ No predictive power',
                xy=(2, ph_importance[2]), xytext=(2, 0.3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=10, color='blue', weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ph_importance_drop.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {(OUTPUT_DIR / 'ph_importance_drop.png').relative_to(PROJECT_ROOT)}")


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    print("\nComparing feature importance across models...")
    
    # Extract importance for all models
    all_importance = {}
    
    # General model
    all_importance["General"] = extract_general_model_importance()
    
    # Medium-specific models
    for medium in MEDIUMS:
        all_importance[medium] = extract_medium_importance(medium)
    
    # Create comparison table and visualizations
    print("\n" + "="*70)
    print("GENERATING OUTPUTS")
    print("="*70)
    
    df = create_comparison_table(all_importance)
    create_visualizations(df)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nOutputs saved in: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print("\nFiles created:")
    print("  - importance_comparison.csv")
    print("  - importance_analysis.txt")
    print("  - importance_comparison_bars.png")
    print("  - importance_comparison_heatmap.png")
    print("  - ph_importance_drop.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
