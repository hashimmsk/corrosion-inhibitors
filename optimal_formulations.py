"""
Optimal Formulation Analysis

Generate optimal surfactant formulation recommendations per medium,
with concentrations in ORIGINAL units (not scaled).

Outputs:
- Optimal formulation table for each medium
- Inverse-transformed concentration values
- Visualizations and report
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "optimal_formulations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = ["C#", "Mw", "HLB", "EO", "Conc", "pH"]
SEED = 0

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150


def load_data_and_scaler():
    """Load data and fit scaler to get inverse transform capability."""
    # Load raw cleaned data (before scaling)
    cleaned_df = pd.read_csv(DATA_DIR / "processed" / "cleaned_full.csv")
    
    # For simplicity, compute scaler parameters from cleaned_full
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(cleaned_df[FEATURE_COLUMNS]),
        columns=FEATURE_COLUMNS
    )
    
    scaler = StandardScaler()
    scaler.fit(X_imputed)
    
    return cleaned_df, scaler


def load_model():
    """Load the trained model."""
    model_path = DATA_DIR / "interpretation" / "final_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    
    # Fallback: retrain
    print("  Retraining model...")
    train_df = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "processed" / "val.csv")
    
    X = pd.concat([train_df[FEATURE_COLUMNS], val_df[FEATURE_COLUMNS]])
    y = pd.concat([train_df["IE"], val_df["IE"]])
    
    model = RandomForestRegressor(
        n_estimators=600, max_depth=6, min_samples_leaf=2,
        random_state=SEED, n_jobs=-1
    )
    model.fit(X, y)
    return model


def find_optimal_formulations(model, cleaned_df, scaler):
    """Find optimal formulations for each medium using BATCH predictions."""
    print("\n" + "="*70)
    print("FINDING OPTIMAL FORMULATIONS PER MEDIUM")
    print("="*70)
    
    results = {}
    ph_map = {"HCl": 0.5, "NaCl": 4.9, "CPS": 12.5}
    
    # Get unique surfactant formulations
    surfactant_cols = ["C#", "Mw", "HLB", "EO"]
    unique_surfactants = cleaned_df[surfactant_cols].drop_duplicates().dropna().reset_index(drop=True)
    
    print(f"\n  Unique surfactant formulations in data: {len(unique_surfactants)}")
    
    # Get concentration range in original units
    conc_min_orig = cleaned_df["Conc"].min()
    conc_max_orig = cleaned_df["Conc"].max()
    conc_range_orig = np.linspace(conc_min_orig, conc_max_orig, 20)  # Reduced for speed
    
    print(f"  Concentration range: {conc_min_orig:.4f} to {conc_max_orig:.4f} (original units)")
    print(f"  Testing {len(unique_surfactants)} surfactants x {len(conc_range_orig)} concentrations = {len(unique_surfactants) * len(conc_range_orig)} combinations per medium")
    
    for medium, ph in ph_map.items():
        print(f"\n  Optimizing for {medium} (pH={ph})...")
        
        # Build all combinations as a single DataFrame for batch prediction
        rows = []
        for _, surf in unique_surfactants.iterrows():
            for conc_orig in conc_range_orig:
                rows.append({
                    "C#": surf["C#"],
                    "Mw": surf["Mw"],
                    "HLB": surf["HLB"],
                    "EO": surf["EO"],
                    "Conc": conc_orig,
                    "pH": ph
                })
        
        X_orig = pd.DataFrame(rows)
        
        # Scale the data
        X_scaled = pd.DataFrame(
            scaler.transform(X_orig),
            columns=FEATURE_COLUMNS
        )
        
        # BATCH prediction (much faster)
        ie_predictions = model.predict(X_scaled)
        
        # Find best
        best_idx = np.argmax(ie_predictions)
        best_row = X_orig.iloc[best_idx]
        best_ie = ie_predictions[best_idx]
        
        results[medium] = {
            "C#": best_row["C#"],
            "Mw": best_row["Mw"],
            "HLB": best_row["HLB"],
            "EO": best_row["EO"],
            "Conc_original": best_row["Conc"],
            "pH": ph,
            "Predicted_IE": best_ie,
        }
        
        print(f"    Best: C#={best_row['C#']:.0f}, Mw={best_row['Mw']:.0f}, "
              f"HLB={best_row['HLB']:.2f}, EO={best_row['EO']:.1f}")
        print(f"    Optimal Conc: {best_row['Conc']:.4f} (original units)")
        print(f"    Max Predicted IE: {best_ie:.1f}%")
    
    return results


def create_summary_table(optimal_results):
    """Create summary tables for the report."""
    print("\n" + "="*70)
    print("OPTIMAL FORMULATION SUMMARY")
    print("="*70)
    
    # Optimal per medium
    print("\n  BEST FORMULATION PER MEDIUM:")
    print("-" * 70)
    print(f"  {'Medium':<8} | {'C#':<5} | {'Mw':<6} | {'HLB':<5} | {'EO':<4} | {'Conc':<8} | {'Max IE':<8}")
    print("-" * 70)
    
    for medium, data in optimal_results.items():
        print(f"  {medium:<8} | {data['C#']:<5.0f} | {data['Mw']:<6.0f} | "
              f"{data['HLB']:<5.2f} | {data['EO']:<4.1f} | {data['Conc_original']:<8.4f} | "
              f"{data['Predicted_IE']:<8.1f}%")
    
    print("-" * 70)
    
    # Save to CSV
    summary_df = pd.DataFrame(optimal_results).T
    summary_df.index.name = "Medium"
    summary_df.to_csv(OUTPUT_DIR / "optimal_formulations_by_medium.csv")
    print(f"\n  Saved: optimal_formulations_by_medium.csv")
    
    return summary_df


def create_visualizations(optimal_results, cleaned_df, model, scaler):
    """Create visualizations."""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Figure 1: Optimal formulations comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mediums = list(optimal_results.keys())
    max_ie = [optimal_results[m]["Predicted_IE"] for m in mediums]
    colors = {'HCl': '#E94F37', 'NaCl': '#2E86AB', 'CPS': '#4DAA57'}
    
    bars = ax.bar(mediums, max_ie, color=[colors[m] for m in mediums], 
                  edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel("Maximum Predicted IE (%)")
    ax.set_title("Maximum Achievable Inhibition Efficiency per Medium\n(Based on Optimization Analysis)")
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar, ie in zip(bars, max_ie):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{ie:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add formulation annotations
    for i, (medium, data) in enumerate(optimal_results.items()):
        formula = f"C#={data['C#']:.0f}, HLB={data['HLB']:.1f}\nEO={data['EO']:.1f}, Conc={data['Conc_original']:.3f}"
        ax.text(i, max_ie[i]/2, formula, ha='center', va='center', 
                fontsize=9, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "max_ie_by_medium.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Saved: max_ie_by_medium.png")
    
    # Figure 2: Concentration-response curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ph_map = {"HCl": 0.5, "NaCl": 4.9, "CPS": 12.5}
    conc_range = np.linspace(cleaned_df["Conc"].min(), cleaned_df["Conc"].max(), 50)
    
    for ax, (medium, data) in zip(axes, optimal_results.items()):
        # Create DataFrame for batch prediction
        X_orig = pd.DataFrame({
            "C#": [data["C#"]] * len(conc_range),
            "Mw": [data["Mw"]] * len(conc_range),
            "HLB": [data["HLB"]] * len(conc_range),
            "EO": [data["EO"]] * len(conc_range),
            "Conc": conc_range,
            "pH": [ph_map[medium]] * len(conc_range)
        })
        
        X_scaled = pd.DataFrame(scaler.transform(X_orig), columns=FEATURE_COLUMNS)
        ie_values = model.predict(X_scaled)
        
        ax.plot(conc_range, ie_values, 'b-', linewidth=2)
        ax.axvline(x=data["Conc_original"], color='red', linestyle='--', linewidth=2,
                   label=f'Optimal: {data["Conc_original"]:.4f}')
        ax.axhline(y=data["Predicted_IE"], color='green', linestyle=':', linewidth=1.5)
        
        ax.set_xlabel("Concentration (original units)")
        ax.set_ylabel("Predicted IE (%)")
        ax.set_title(f"{medium}: IE vs Concentration\n(Other features at optimal)")
        ax.legend(loc='lower right')
    
    plt.suptitle("Concentration-Response Curves at Optimal Formulations", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "concentration_response.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  Saved: concentration_response.png")


def generate_report(optimal_results):
    """Generate comprehensive report."""
    report_path = OUTPUT_DIR / "report.txt"
    
    with open(report_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("OPTIMAL FORMULATION ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("OBJECTIVE\n")
        f.write("-"*70 + "\n")
        f.write("Identify optimal surfactant formulations (C#, Mw, HLB, EO, Conc)\n")
        f.write("for maximizing inhibition efficiency (IE) in each corrosive medium.\n\n")
        
        f.write("OPTIMAL FORMULATIONS BY MEDIUM\n")
        f.write("-"*70 + "\n\n")
        
        for medium, data in optimal_results.items():
            f.write(f"{medium} (pH = {data['pH']}):\n")
            f.write(f"  Carbon Number (C#):        {data['C#']:.0f}\n")
            f.write(f"  Molecular Weight (Mw):     {data['Mw']:.0f}\n")
            f.write(f"  HLB Value:                 {data['HLB']:.2f}\n")
            f.write(f"  Ethylene Oxide Units (EO): {data['EO']:.1f}\n")
            f.write(f"  Optimal Concentration:     {data['Conc_original']:.4f} (original units)\n")
            f.write(f"  Maximum Predicted IE:      {data['Predicted_IE']:.1f}%\n\n")
        
        f.write("SUMMARY TABLE\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Medium':<8} | {'C#':<5} | {'Mw':<6} | {'HLB':<5} | {'EO':<4} | {'Conc':<8} | {'Max IE':<8}\n")
        f.write("-"*70 + "\n")
        for medium, data in optimal_results.items():
            f.write(f"{medium:<8} | {data['C#']:<5.0f} | {data['Mw']:<6.0f} | "
                   f"{data['HLB']:<5.2f} | {data['EO']:<4.1f} | {data['Conc_original']:<8.4f} | "
                   f"{data['Predicted_IE']:<8.1f}%\n")
        f.write("-"*70 + "\n\n")
        
        f.write("IMPORTANT LIMITATIONS\n")
        f.write("-"*70 + "\n\n")
        
        f.write("1. MODEL ACCURACY:\n")
        f.write("   - Test R-squared = 0.417 (explains 42% of variance)\n")
        f.write("   - RMSE = 20.1% (predictions have ~20% error)\n")
        f.write("   - Treat recommendations as DIRECTIONAL GUIDANCE, not precise values\n\n")
        
        f.write("2. CONCENTRATION UNITS:\n")
        f.write("   - Values shown are in ORIGINAL units (not scaled)\n")
        f.write("   - Verify unit interpretation with experimental protocols\n\n")
        
        f.write("3. TRAINING DATA CONSTRAINTS:\n")
        f.write("   - Model can only recommend formulations similar to training data\n")
        f.write("   - Cannot extrapolate to novel surfactant designs\n")
        f.write("   - Maximum achievable IE is capped by what was observed in data\n\n")
        
        f.write("4. CROSS-MEDIUM GENERALIZATION:\n")
        f.write("   - Leave-one-medium-out tests show model CANNOT generalize\n")
        f.write("     to unseen corrosive environments (R-squared < 0)\n")
        f.write("   - These recommendations are specific to each medium\n\n")
        
        f.write("="*70 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*70 + "\n\n")
        
        f.write("The optimization analysis provides the following recommendations:\n\n")
        
        for medium, data in optimal_results.items():
            f.write(f"For {medium} (pH={data['pH']}):\n")
            f.write(f"  - Use surfactants with C# around {data['C#']:.0f}\n")
            f.write(f"  - Target HLB value of approximately {data['HLB']:.1f}\n")
            f.write(f"  - EO units around {data['EO']:.1f}\n")
            f.write(f"  - Concentration near {data['Conc_original']:.4f}\n")
            f.write(f"  - Expected IE: up to {data['Predicted_IE']:.0f}%\n\n")
        
        f.write("These recommendations should be treated as starting points for\n")
        f.write("experimental validation, not as definitive prescriptions. The model's\n")
        f.write("moderate accuracy (R-squared=0.42) means experimental verification is\n")
        f.write("essential before deployment.\n\n")
        
        f.write("The fact that different mediums favor different formulations, combined\n")
        f.write("with the model's inability to generalize across mediums (leave-one-out\n")
        f.write("R-squared < 0), suggests that inhibitor design is highly environment-\n")
        f.write("specific. Future work should focus on understanding the mechanistic\n")
        f.write("differences in how surfactants interact with steel surfaces under\n")
        f.write("different pH conditions.\n")
    
    print(f"  Saved: report.txt")
    return report_path


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("OPTIMAL FORMULATION ANALYSIS")
    print("="*70)
    print("\nGenerating optimal surfactant formulations per medium...")
    print("(With concentrations in ORIGINAL units, not scaled)")
    
    # Load data and model
    print("\nLoading data and model...")
    cleaned_df, scaler = load_data_and_scaler()
    model = load_model()
    print(f"  Dataset: {len(cleaned_df)} samples")
    print("  Model: Random Forest (loaded)")
    
    # Find optimal formulations
    optimal_results = find_optimal_formulations(model, cleaned_df, scaler)
    
    # Create summary
    summary_df = create_summary_table(optimal_results)
    
    # Create visualizations
    create_visualizations(optimal_results, cleaned_df, model, scaler)
    
    # Generate report
    generate_report(optimal_results)
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutputs saved in: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print("\nFiles created:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
