"""
Medium-Specific Preprocessing Pipeline

This script splits the dataset by corrosive medium (HCl, NaCl, CPS) and applies
preprocessing to each subset independently. This allows comparison of:
1. Whether medium-specific models outperform the general model
2. How feature importance changes across different corrosive environments

Directory Structure Created:
    data/medium_specific/
        ├── HCl/
        │   ├── processed/
        │   │   ├── cleaned_full.csv
        │   │   ├── train.csv
        │   │   ├── val.csv
        │   │   └── test.csv
        │   └── eda/
        │       └── medium_stats.txt
        ├── NaCl/
        │   └── ...
        └── CPS/
            └── ...
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from preprocessing import PreprocessingConfig, clean_dataset, preprocess_dataset

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATASET_PATH = PROJECT_ROOT / "dataset.csv"
OUTPUT_BASE = PROJECT_ROOT / "data" / "medium_specific"

MEDIUMS = ["HCl", "NaCl", "CPS"]

def create_directory_structure():
    """Create organized directory structure for medium-specific data."""
    for medium in MEDIUMS:
        medium_dir = OUTPUT_BASE / medium
        (medium_dir / "processed").mkdir(parents=True, exist_ok=True)
        (medium_dir / "eda").mkdir(parents=True, exist_ok=True)
    print(f"✓ Directory structure created at: {OUTPUT_BASE}")

def load_and_split_by_medium():
    """Load dataset and split by medium."""
    print(f"\nLoading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    
    medium_data = {}
    for medium in MEDIUMS:
        subset = df[df['medium'] == medium].copy()
        medium_data[medium] = subset
        print(f"  - {medium}: {len(subset)} samples")
    
    return medium_data

def generate_medium_eda(medium_name, df_raw, df_cleaned, splits, output_dir):
    """Generate quick EDA report for a specific medium."""
    eda_file = output_dir / "eda" / "medium_stats.txt"
    
    with open(eda_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"MEDIUM-SPECIFIC EDA: {medium_name}\n")
        f.write("="*70 + "\n\n")
        
        # Basic stats
        f.write("DATASET OVERVIEW\n")
        f.write("-"*70 + "\n")
        f.write(f"Total samples (raw): {len(df_raw)}\n")
        f.write(f"Total samples (cleaned): {len(df_cleaned)}\n")
        f.write(f"Train samples: {len(splits['X_train'])}\n")
        f.write(f"Val samples: {len(splits['X_val'])}\n")
        f.write(f"Test samples: {len(splits['X_test'])}\n\n")
        
        # IE statistics
        f.write("INHIBITION EFFICIENCY (IE) STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Mean IE: {df_cleaned['IE'].mean():.2f}%\n")
        f.write(f"Std Dev: {df_cleaned['IE'].std():.2f}%\n")
        f.write(f"Min IE: {df_cleaned['IE'].min():.2f}%\n")
        f.write(f"Max IE: {df_cleaned['IE'].max():.2f}%\n")
        f.write(f"Median IE: {df_cleaned['IE'].median():.2f}%\n\n")
        
        # pH distribution
        f.write("pH DISTRIBUTION\n")
        f.write("-"*70 + "\n")
        ph_counts = df_cleaned['pH'].value_counts().sort_index()
        for ph, count in ph_counts.items():
            f.write(f"pH {ph}: {count} samples ({count/len(df_cleaned)*100:.1f}%)\n")
        f.write(f"Mean pH: {df_cleaned['pH'].mean():.2f}\n\n")
        
        # Feature statistics
        f.write("FEATURE STATISTICS (Cleaned Data)\n")
        f.write("-"*70 + "\n")
        features = ['C#', 'Mw', 'HLB', 'EO', 'Conc', 'pH']
        for feat in features:
            if feat in df_cleaned.columns:
                f.write(f"\n{feat}:\n")
                f.write(f"  Mean: {df_cleaned[feat].mean():.2f}\n")
                f.write(f"  Std:  {df_cleaned[feat].std():.2f}\n")
                f.write(f"  Min:  {df_cleaned[feat].min():.2f}\n")
                f.write(f"  Max:  {df_cleaned[feat].max():.2f}\n")
        
        # Correlation with IE
        f.write("\n\nCORRELATION WITH IE (Inhibition Efficiency)\n")
        f.write("-"*70 + "\n")
        for feat in features:
            if feat in df_cleaned.columns:
                corr = df_cleaned[feat].corr(df_cleaned['IE'])
                f.write(f"{feat:10s}: {corr:+.3f}\n")
    
    print(f"    ✓ EDA report: {eda_file.relative_to(PROJECT_ROOT)}")

def preprocess_medium(medium_name, df_medium):
    """Preprocess data for a specific medium."""
    print(f"\n{'='*70}")
    print(f"PREPROCESSING: {medium_name}")
    print(f"{'='*70}")
    
    output_dir = OUTPUT_BASE / medium_name
    
    # Save raw medium subset
    raw_path = output_dir / f"{medium_name.lower()}_raw.csv"
    df_medium.to_csv(raw_path, index=False)
    print(f"  ✓ Raw subset saved: {raw_path.relative_to(PROJECT_ROOT)}")
    
    # Apply preprocessing using the existing pipeline
    # First, we need to save a temporary file for preprocessing
    temp_path = output_dir / "temp_for_preprocessing.csv"
    df_medium.to_csv(temp_path, index=False)
    
    config = PreprocessingConfig(
        dataset_path=str(temp_path),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=0,
    )
    
    # Run preprocessing
    processed = preprocess_dataset(config)
    
    # Save cleaned full dataset
    cleaned_path = output_dir / "processed" / "cleaned_full.csv"
    processed["cleaned_df"].to_csv(cleaned_path, index=False)
    print(f"  ✓ Cleaned data: {cleaned_path.relative_to(PROJECT_ROOT)}")
    
    # Save train/val/test splits
    for split_name in ['train', 'val', 'test']:
        split_data = processed['splits'][split_name]
        
        # Combine features and target
        df_out = split_data['X_scaled'].copy()
        df_out['IE'] = split_data['y'].values
        
        # Add pH_original for interpretability
        df_out['pH_original'] = split_data['X_raw']['pH'].values
        
        # Add medium column
        df_out['medium'] = medium_name
        
        split_path = output_dir / "processed" / f"{split_name}.csv"
        df_out.to_csv(split_path, index=False)
        print(f"  ✓ {split_name.capitalize()} split: {split_path.relative_to(PROJECT_ROOT)}")
    
    # Generate EDA report
    splits = {
        'X_train': processed['splits']['train']['X_raw'],
        'X_val': processed['splits']['val']['X_raw'],
        'X_test': processed['splits']['test']['X_raw'],
    }
    generate_medium_eda(medium_name, df_medium, processed["cleaned_df"], splits, output_dir)
    
    # Clean up temp file
    temp_path.unlink()
    
    return processed

def create_combined_summary():
    """Create a summary comparing all three mediums."""
    summary_path = OUTPUT_BASE / "SUMMARY_all_mediums.txt"
    
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MEDIUM-SPECIFIC PREPROCESSING SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("This preprocessing splits the original dataset into three subsets\n")
        f.write("based on corrosive medium, then applies independent preprocessing\n")
        f.write("(cleaning, train/val/test split, scaling) to each subset.\n\n")
        
        f.write("PURPOSE:\n")
        f.write("-"*70 + "\n")
        f.write("1. Train medium-specific Random Forest and SVR models\n")
        f.write("2. Compare performance vs. general model (trained on all mediums)\n")
        f.write("3. Analyze how feature importance varies across corrosive environments\n\n")
        
        f.write("COMPARISON TABLE\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Medium':<10} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8} {'Mean IE':<12} {'pH Range':<12}\n")
        f.write("-"*70 + "\n")
        
        for medium in MEDIUMS:
            medium_dir = OUTPUT_BASE / medium
            
            # Load data
            cleaned = pd.read_csv(medium_dir / "processed" / "cleaned_full.csv")
            train = pd.read_csv(medium_dir / "processed" / "train.csv")
            val = pd.read_csv(medium_dir / "processed" / "val.csv")
            test = pd.read_csv(medium_dir / "processed" / "test.csv")
            
            mean_ie = cleaned['IE'].mean()
            ph_min = cleaned['pH'].min()
            ph_max = cleaned['pH'].max()
            ph_range = f"{ph_min:.1f}-{ph_max:.1f}" if ph_min != ph_max else f"{ph_min:.1f}"
            
            f.write(f"{medium:<10} {len(cleaned):<8} {len(train):<8} {len(val):<8} {len(test):<8} {mean_ie:<12.2f} {ph_range:<12}\n")
        
        f.write("\n\nNEXT STEPS:\n")
        f.write("-"*70 + "\n")
        f.write("1. Run medium_specific_training.py to train models for each medium\n")
        f.write("2. Compare results with general model (data/models/results.json)\n")
        f.write("3. Analyze feature importance differences across mediums\n")
        f.write("4. Generate comparative visualizations\n\n")
        
        f.write("FILES GENERATED:\n")
        f.write("-"*70 + "\n")
        for medium in MEDIUMS:
            f.write(f"\n{medium}/\n")
            f.write(f"  processed/\n")
            f.write(f"    - cleaned_full.csv (full cleaned dataset)\n")
            f.write(f"    - train.csv (70% split)\n")
            f.write(f"    - val.csv (15% split)\n")
            f.write(f"    - test.csv (15% split)\n")
            f.write(f"  eda/\n")
            f.write(f"    - medium_stats.txt (statistics & correlations)\n")
    
    print(f"\n{'='*70}")
    print(f"✓ Combined summary: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"{'='*70}")

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("MEDIUM-SPECIFIC PREPROCESSING PIPELINE")
    print("="*70)
    
    # Step 1: Create directory structure
    create_directory_structure()
    
    # Step 2: Load and split by medium
    medium_data = load_and_split_by_medium()
    
    # Step 3: Preprocess each medium independently
    results = {}
    for medium_name, df_medium in medium_data.items():
        results[medium_name] = preprocess_medium(medium_name, df_medium)
    
    # Step 4: Create combined summary
    create_combined_summary()
    
    print("\n" + "="*70)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*70)
    print("\nAll medium-specific datasets are ready for model training.")
    print(f"Location: {OUTPUT_BASE.relative_to(PROJECT_ROOT)}")
    print("\nNext: Run medium-specific model training script.")

if __name__ == "__main__":
    main()
