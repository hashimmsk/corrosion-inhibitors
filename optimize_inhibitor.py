"""
Step 9: Optimization & Design Use-Case

This script provides optimization tools for inhibitor design:
1. Dosage Optimization - Find optimal concentration for a given surfactant
2. Formulation Recommendation - Suggest best surfactant properties for target IE
3. Sensitivity Analysis - How does changing each parameter affect IE?

Outputs saved to: data/optimization/
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "optimization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = ["C#", "Mw", "HLB", "EO", "Conc", "pH"]
SEED = 0

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150


def load_model():
    """Load the trained model."""
    model_path = DATA_DIR / "interpretation" / "final_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    
    # Fallback: retrain if not found
    print("  Model not found, retraining...")
    train_df = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "processed" / "val.csv")
    
    X = pd.concat([train_df[FEATURE_COLUMNS], val_df[FEATURE_COLUMNS]])
    y = pd.concat([train_df["IE"], val_df["IE"]])
    
    with open(DATA_DIR / "models" / "results.json") as f:
        params = json.load(f)["best_params"]
    
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def load_feature_bounds():
    """Get realistic bounds for each feature from training data."""
    train_df = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "processed" / "val.csv")
    df = pd.concat([train_df, val_df])
    
    bounds = {}
    for col in FEATURE_COLUMNS:
        bounds[col] = (df[col].min(), df[col].max())
    
    return bounds


def get_typical_surfactants():
    """Get examples of typical surfactant formulations from the dataset."""
    cleaned_df = pd.read_csv(DATA_DIR / "processed" / "cleaned_full.csv")
    
    # Group by unique surfactant (same C#, Mw, HLB, EO)
    surfactant_cols = ["C#", "Mw", "HLB", "EO"]
    unique_surfactants = cleaned_df.groupby(surfactant_cols).agg({
        "IE": "mean",
        "Conc": "median",
        "pH": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.median()
    }).reset_index()
    
    return unique_surfactants


# ============================================================
# 1. DOSAGE OPTIMIZATION
# ============================================================
def optimize_concentration(model, surfactant_props, target_medium="HCl"):
    """
    Find optimal concentration for a given surfactant to maximize IE.
    
    Args:
        model: Trained model
        surfactant_props: dict with C#, Mw, HLB, EO values
        target_medium: 'HCl' (pH=0.5), 'NaCl' (pH=4.9), or 'CPS' (pH=12.5)
    
    Returns:
        optimal_conc, max_ie, optimization_result
    """
    # Set pH based on medium
    ph_map = {"HCl": 0.5, "NaCl": 4.9, "CPS": 12.5}
    ph = ph_map.get(target_medium, 4.9)
    
    bounds = load_feature_bounds()
    conc_bounds = bounds["Conc"]
    
    def objective(conc):
        # Create feature vector
        X = np.array([[
            surfactant_props["C#"],
            surfactant_props["Mw"],
            surfactant_props["HLB"],
            surfactant_props["EO"],
            conc[0],
            ph
        ]])
        # Minimize negative IE (maximize IE)
        return -model.predict(X)[0]
    
    # Run optimization
    result = minimize(
        objective,
        x0=[(conc_bounds[0] + conc_bounds[1]) / 2],  # Start at middle
        bounds=[conc_bounds],
        method='L-BFGS-B'
    )
    
    optimal_conc = result.x[0]
    max_ie = -result.fun
    
    return optimal_conc, max_ie, result


def run_dosage_optimization(model):
    """Run dosage optimization for example surfactants."""
    print("\n" + "="*60)
    print("1. DOSAGE OPTIMIZATION")
    print("="*60)
    print("\nFinding optimal concentration to maximize IE...")
    
    # Get example surfactants
    surfactants = get_typical_surfactants()
    top_surfactants = surfactants.nlargest(5, 'IE')
    
    results = []
    
    for medium in ["HCl", "NaCl", "CPS"]:
        print(f"\n  Medium: {medium}")
        print("-" * 50)
        
        for _, row in top_surfactants.iterrows():
            props = {
                "C#": row["C#"],
                "Mw": row["Mw"],
                "HLB": row["HLB"],
                "EO": row["EO"]
            }
            
            opt_conc, max_ie, _ = optimize_concentration(model, props, medium)
            
            results.append({
                "C#": props["C#"],
                "Mw": props["Mw"],
                "HLB": props["HLB"],
                "EO": props["EO"],
                "Medium": medium,
                "Optimal_Conc": opt_conc,
                "Max_IE": max_ie
            })
            
            print(f"    C#={props['C#']:.0f}, Mw={props['Mw']:.0f}: "
                  f"Optimal Conc={opt_conc:.4f} → IE={max_ie:.1f}%")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "dosage_optimization_results.csv", index=False)
    print(f"\n  ✓ Saved: dosage_optimization_results.csv")
    
    return results_df


# ============================================================
# 2. FORMULATION RECOMMENDATION
# ============================================================
def find_optimal_formulation(model, target_ie=80, target_medium="HCl"):
    """
    Find surfactant formulation that achieves target IE.
    Uses grid search over existing surfactant properties + concentration optimization.
    
    Args:
        model: Trained model
        target_ie: Desired inhibition efficiency (%)
        target_medium: 'HCl', 'NaCl', or 'CPS'
    
    Returns:
        optimal_params, achieved_ie
    """
    bounds_dict = load_feature_bounds()
    ph_map = {"HCl": 0.5, "NaCl": 4.9, "CPS": 12.5}
    ph = ph_map.get(target_medium, 4.9)
    
    # Get existing surfactant combinations from data
    cleaned_df = pd.read_csv(DATA_DIR / "processed" / "cleaned_full.csv")
    surfactant_cols = ["C#", "Mw", "HLB", "EO"]
    unique_surfactants = cleaned_df[surfactant_cols].drop_duplicates()
    
    best_result = None
    best_diff = float('inf')
    
    # Grid search over surfactants and concentration values
    conc_values = np.linspace(bounds_dict["Conc"][0], bounds_dict["Conc"][1], 20)
    
    for _, surf in unique_surfactants.iterrows():
        for conc in conc_values:
            X = np.array([[surf["C#"], surf["Mw"], surf["HLB"], surf["EO"], conc, ph]])
            predicted_ie = model.predict(X)[0]
            diff = abs(predicted_ie - target_ie)
            
            if diff < best_diff:
                best_diff = diff
                best_result = {
                    "C#": surf["C#"],
                    "Mw": surf["Mw"],
                    "HLB": surf["HLB"],
                    "EO": surf["EO"],
                    "Conc": conc,
                    "pH": ph
                }
                achieved_ie = predicted_ie
    
    return best_result, achieved_ie


def run_formulation_recommendation(model):
    """Find optimal formulations for different target IE values."""
    print("\n" + "="*60)
    print("2. FORMULATION RECOMMENDATION")
    print("="*60)
    print("\nFinding optimal surfactant formulations for target IE...")
    
    results = []
    
    for medium in ["HCl", "NaCl", "CPS"]:
        print(f"\n  Medium: {medium}")
        print("-" * 50)
        
        for target_ie in [60, 70, 80, 90]:
            try:
                params, achieved_ie = find_optimal_formulation(model, target_ie, medium)
                
                results.append({
                    "Medium": medium,
                    "Target_IE": target_ie,
                    "Achieved_IE": achieved_ie,
                    **params
                })
                
                print(f"    Target {target_ie}% → Achieved {achieved_ie:.1f}%")
                print(f"      C#={params['C#']:.1f}, Mw={params['Mw']:.0f}, "
                      f"HLB={params['HLB']:.2f}, EO={params['EO']:.1f}, "
                      f"Conc={params['Conc']:.4f}")
            except Exception as e:
                print(f"    Target {target_ie}%: Optimization failed - {e}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "formulation_recommendations.csv", index=False)
    print(f"\n  ✓ Saved: formulation_recommendations.csv")
    
    return results_df


# ============================================================
# 3. SENSITIVITY ANALYSIS
# ============================================================
def run_sensitivity_analysis(model):
    """Analyze how changing each parameter affects predicted IE."""
    print("\n" + "="*60)
    print("3. SENSITIVITY ANALYSIS")
    print("="*60)
    print("\nComputing sensitivity curves for each feature...")
    
    bounds = load_feature_bounds()
    
    # Use median values as baseline
    train_df = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    baseline = {col: train_df[col].median() for col in FEATURE_COLUMNS}
    
    # Create sensitivity plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    sensitivity_data = {}
    
    for i, feature in enumerate(FEATURE_COLUMNS):
        ax = axes[i]
        
        # Create range of values for this feature
        feat_min, feat_max = bounds[feature]
        feat_range = np.linspace(feat_min, feat_max, 50)
        
        # Predict IE for each value (holding others at baseline)
        ie_predictions = []
        for val in feat_range:
            X = np.array([[
                val if feature == "C#" else baseline["C#"],
                val if feature == "Mw" else baseline["Mw"],
                val if feature == "HLB" else baseline["HLB"],
                val if feature == "EO" else baseline["EO"],
                val if feature == "Conc" else baseline["Conc"],
                val if feature == "pH" else baseline["pH"],
            ]])
            ie_predictions.append(model.predict(X)[0])
        
        ie_predictions = np.array(ie_predictions)
        sensitivity_data[feature] = {
            'range': feat_range.tolist(),
            'predictions': ie_predictions.tolist()
        }
        
        # Plot
        ax.plot(feat_range, ie_predictions, 'b-', linewidth=2)
        ax.fill_between(feat_range, ie_predictions.min(), ie_predictions, alpha=0.3)
        ax.axhline(y=baseline["IE"] if "IE" in baseline else ie_predictions.mean(), 
                   color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel(feature)
        ax.set_ylabel('Predicted IE (%)')
        ax.set_title(f'Sensitivity: {feature}')
        
        # Compute sensitivity (range of IE / range of feature)
        ie_range = ie_predictions.max() - ie_predictions.min()
        print(f"  {feature}: IE varies from {ie_predictions.min():.1f}% to {ie_predictions.max():.1f}% "
              f"(range: {ie_range:.1f}%)")
    
    plt.suptitle('Sensitivity Analysis\n(Effect of Each Feature on Predicted IE, Others at Median)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sensitivity_analysis.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n  ✓ Saved: sensitivity_analysis.png")
    
    # Save data
    with open(OUTPUT_DIR / "sensitivity_data.json", "w") as f:
        json.dump(sensitivity_data, f, indent=2)
    print(f"  ✓ Saved: sensitivity_data.json")
    
    return sensitivity_data


# ============================================================
# 4. INTERACTIVE PREDICTOR (for future use)
# ============================================================
def predict_ie(model, c_num, mw, hlb, eo, conc, ph):
    """
    Predict IE for a given formulation.
    
    Example usage:
        from optimize_inhibitor import load_model, predict_ie
        model = load_model()
        ie = predict_ie(model, C#=12, Mw=500, HLB=10, EO=5, Conc=0.01, pH=0.5)
    """
    X = np.array([[c_num, mw, hlb, eo, conc, ph]])
    return model.predict(X)[0]


def create_prediction_table(model):
    """Create a reference table of predictions for common scenarios."""
    print("\n" + "="*60)
    print("4. PREDICTION REFERENCE TABLE")
    print("="*60)
    
    # Get unique surfactants from data
    cleaned_df = pd.read_csv(DATA_DIR / "processed" / "cleaned_full.csv")
    surfactant_cols = ["C#", "Mw", "HLB", "EO"]
    unique_surfactants = cleaned_df[surfactant_cols].drop_duplicates().head(10)
    
    # Predict for each surfactant at different conditions
    predictions = []
    
    for _, surf in unique_surfactants.iterrows():
        for medium, ph in [("HCl", 0.5), ("NaCl", 4.9), ("CPS", 12.5)]:
            for conc in [0.001, 0.01, 0.1]:
                ie = predict_ie(model, surf["C#"], surf["Mw"], surf["HLB"], surf["EO"], conc, ph)
                predictions.append({
                    "C#": surf["C#"],
                    "Mw": surf["Mw"],
                    "HLB": surf["HLB"],
                    "EO": surf["EO"],
                    "Medium": medium,
                    "Conc": conc,
                    "Predicted_IE": ie
                })
    
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(OUTPUT_DIR / "prediction_reference_table.csv", index=False)
    print(f"\n  ✓ Saved: prediction_reference_table.csv")
    print(f"  Contains {len(pred_df)} predictions for {len(unique_surfactants)} surfactants")
    
    return pred_df


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*60)
    print("STEP 9: OPTIMIZATION & DESIGN USE-CASE")
    print("="*60)
    
    # Load model
    print("\nLoading trained model...")
    model = load_model()
    print("  ✓ Model loaded")
    
    # Run optimizations
    dosage_results = run_dosage_optimization(model)
    formulation_results = run_formulation_recommendation(model)
    sensitivity_data = run_sensitivity_analysis(model)
    prediction_table = create_prediction_table(model)
    
    # Summary
    print("\n" + "="*60)
    print("✓ OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"\nOutputs saved in: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print("\nFiles created:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    print("""
    # Load model and predict IE for a custom formulation:
    from optimize_inhibitor import load_model, predict_ie
    
    model = load_model()
    ie = predict_ie(model, 
                    c_num=12,      # Carbon number
                    mw=500,        # Molecular weight
                    hlb=10,        # HLB value
                    eo=5,          # Ethylene oxide units
                    conc=0.01,     # Concentration
                    ph=0.5)        # pH (0.5 for HCl, 4.9 for NaCl, 12.5 for CPS)
    print(f"Predicted IE: {ie:.1f}%")
    """)


if __name__ == "__main__":
    main()
