"""
Generate publication-quality figures for the ML schematic visualization.
These figures will be provided to Gemini to improve the project visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Set style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.dpi'] = 150

# Paths
DATA_DIR = Path("/Users/hashi/Desktop/corrosion-inhibitors/data")
OUTPUT_DIR = DATA_DIR / "viz_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
cleaned_df = pd.read_csv(DATA_DIR / "processed" / "cleaned_full.csv")
test_preds = pd.read_csv(DATA_DIR / "models" / "test_predictions.csv")
with open(DATA_DIR / "models" / "results.json") as f:
    results = json.load(f)
feature_importance = pd.read_csv(DATA_DIR / "feature_importance" / "baseline_importance.csv", index_col=0)

# ============================================================
# FIGURE 1: Model Performance Comparison Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

models = ['Random Forest', 'SVR']
val_r2 = [0.693, 0.556]
val_rmse = [20.4, 24.6]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, val_r2, width, label='R² Score', color='#2E86AB', edgecolor='black')
bars2 = ax.bar(x + width/2, [r/30 for r in val_rmse], width, label='RMSE (scaled ÷30)', color='#E94F37', edgecolor='black')

ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison\n(Validation Set)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='upper right')
ax.set_ylim(0, 1.0)

# Add value labels
for bar, val in zip(bars1, val_r2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
for bar, val in zip(bars2, val_rmse):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison_bar.png", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

# ============================================================
# FIGURE 2: Predicted vs Actual Scatter Plot
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))

# Color by medium
colors = {'HCl': '#E94F37', 'NaCl': '#2E86AB', 'CPS': '#4DAA57'}
for medium in test_preds['medium'].unique():
    mask = test_preds['medium'] == medium
    ax.scatter(test_preds.loc[mask, 'y_true'], test_preds.loc[mask, 'y_pred'],
               c=colors.get(medium, 'gray'), label=medium, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)

# Perfect prediction line
lims = [min(test_preds['y_true'].min(), test_preds['y_pred'].min()) - 5,
        max(test_preds['y_true'].max(), test_preds['y_pred'].max()) + 5]
ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Prediction', linewidth=2)

ax.set_xlabel('Experimental IE (%)')
ax.set_ylabel('Predicted IE (%)')
ax.set_title('Random Forest: Predicted vs Experimental\n(Test Set, R² = 0.417)')
ax.legend(loc='upper left')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "predicted_vs_actual.png", dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ============================================================
# FIGURE 3: Feature Importance Horizontal Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

# Use mean importance from the CSV
importance_mean = feature_importance['Mean'].sort_values(ascending=True)
colors_fi = plt.cm.Blues(np.linspace(0.4, 0.9, len(importance_mean)))

bars = ax.barh(importance_mean.index, importance_mean.values, color=colors_fi, edgecolor='black')
ax.set_xlabel('Normalized Importance (Mean)')
ax.set_title('Feature Importance Ranking')
ax.set_xlim(0, 1)

# Add value labels
for bar, val in zip(bars, importance_mean.values):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
            f'{val:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance_bar.png", dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ============================================================
# FIGURE 4: IE Distribution by Medium (Box Plot)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

medium_order = ['HCl', 'NaCl', 'CPS']
palette = {'HCl': '#E94F37', 'NaCl': '#2E86AB', 'CPS': '#4DAA57'}

# Filter to only mediums in data
available_mediums = [m for m in medium_order if m in cleaned_df['medium'].values]
plot_data = cleaned_df[cleaned_df['medium'].isin(available_mediums)]

sns.boxplot(data=plot_data, x='medium', y='IE', order=available_mediums, 
            palette=palette, ax=ax, linewidth=1.5, showfliers=False)
ax.set_xlabel('Corrosive Medium')
ax.set_ylabel('Inhibition Efficiency (%)')
ax.set_title('IE Distribution Across Corrosive Mediums')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ie_by_medium_boxplot.png", dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ============================================================
# FIGURE 5: Correlation Heatmap (Features Only)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

features = ['C#', 'Mw', 'HLB', 'EO', 'Conc', 'pH', 'IE']
corr_matrix = cleaned_df[features].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
ax.set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap_clean.png", dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ============================================================
# FIGURE 6: IE vs Concentration by Medium
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

for medium in available_mediums:
    mask = cleaned_df['medium'] == medium
    data = cleaned_df[mask].groupby('Conc')['IE'].mean().reset_index()
    ax.plot(data['Conc'], data['IE'], 'o-', label=medium, color=palette[medium], 
            markersize=8, linewidth=2, markeredgecolor='black', markeredgewidth=0.5)

ax.set_xlabel('Concentration (mM)')
ax.set_ylabel('Mean IE (%)')
ax.set_title('Inhibition Efficiency vs Concentration')
ax.legend(title='Medium')
ax.set_xscale('log')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ie_vs_concentration.png", dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ============================================================
# FIGURE 7: Residual Distribution
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

ax.hist(test_preds['residual'], bins=15, color='#2E86AB', edgecolor='black', alpha=0.8)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax.axvline(x=test_preds['residual'].mean(), color='orange', linestyle='-', linewidth=2, 
           label=f'Mean: {test_preds["residual"].mean():.1f}')

ax.set_xlabel('Residual (Actual - Predicted)')
ax.set_ylabel('Frequency')
ax.set_title('Prediction Residual Distribution\n(Test Set)')
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residual_distribution.png", dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ============================================================
# Summary Stats for Gemini
# ============================================================
print("="*60)
print("FIGURES GENERATED FOR GEMINI VISUALIZATION")
print("="*60)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nFiles created:")
for f in OUTPUT_DIR.glob("*.png"):
    print(f"  - {f.name}")

print("\n" + "="*60)
print("KEY PROJECT STATISTICS FOR GEMINI")
print("="*60)
print(f"\n📊 Dataset:")
print(f"   - Total samples: {len(cleaned_df)}")
print(f"   - Features: C#, Mw, HLB, EO, Conc, pH")
print(f"   - Target: IE (Inhibition Efficiency %)")
print(f"   - Mediums: {', '.join(cleaned_df['medium'].unique())}")

print(f"\n🔬 pH Values in data:")
print(f"   - Unique values: {sorted(cleaned_df['pH'].unique())}")

print(f"\n🎯 Best Model: Random Forest")
print(f"   - Val R²: 0.693 | Val RMSE: 20.4")
print(f"   - Test R²: 0.417 | Test RMSE: 20.1")
print(f"   - Hyperparameters: n_estimators=600, max_depth=6, min_samples_leaf=2")

print(f"\n📈 Feature Importance (by Mean):")
for feat, imp in feature_importance['Mean'].sort_values(ascending=False).items():
    print(f"   - {feat}: {imp:.3f}")

print("\n" + "="*60)
