# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 13:02:53 2025

@author: ma'wei'bin
"""

import pandas as pd
import smogn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ===== Custom augmentation function definitions (fix negative value issues and formula sum constraints) =====
def normalize_formula(row, formula_cols):
    """Ensure formula features sum does not exceed 100, and scale proportionally"""
    total = row[formula_cols].sum()
    if total > 100:
        # Scale proportionally so sum equals 100
        scaling_factor = 100 / total
        row[formula_cols] = row[formula_cols] * scaling_factor
    return row

def generate_extreme_samples(df, target_col, num_needed, threshold, formula_cols):
    """Generate extreme samples in the long-tail region (ensure no negative values and formula sum ≤ 100)"""
    print(f"Using custom method to generate {num_needed} extreme samples")
    
    # Create position index (avoid using label index)
    df = df.reset_index(drop=True)
    
    # Feature columns (excluding temporary index)
    feature_cols = [col for col in df.columns if col not in ['original_index']]
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    
    # Use KNN
    knn = NearestNeighbors(n_neighbors=min(5, len(df)))
    knn.fit(scaled_features)
    
    synthetic_samples = []
    attempts = 0
    max_attempts = num_needed * 3
    
    # Record minimum value for each column (to prevent negative values)
    min_values = {}
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.float32]:
            min_values[col] = max(0, df[col].min())  # Ensure minimum value is non-negative
    
    while len(synthetic_samples) < num_needed and attempts < max_attempts:
        attempts += 1
        
        # Randomly select a tail sample (using position index)
        seed_idx = np.random.randint(0, len(df))
        seed = df.iloc[seed_idx]
        
        # Find nearest neighbors
        _, indices = knn.kneighbors([scaled_features[seed_idx]], n_neighbors=min(5, len(df)))
        
        # Randomly select a neighbor
        neighbor_idx = np.random.choice(indices[0][1:])
        neighbor = df.iloc[neighbor_idx]
        
        # Interpolate between seed and neighbor (biased towards extreme values)
        alpha = np.random.uniform(0.0, 0.5)
        
        new_sample = seed[feature_cols].copy()
        for col in feature_cols:
            if df[col].dtype in [np.float64, np.float32]:
                # For target variable, more likely to generate higher values
                if col == target_col:
                    higher_value = max(seed[col], neighbor[col])
                    new_sample[col] = higher_value * (1 + np.random.uniform(0, 0.3))
                else:
                    new_sample[col] = alpha * seed[col] + (1 - alpha) * neighbor[col]
                
                # Add noise (ensure non-negative)
                noise = np.random.normal(0, 0.1 * df[col].std())
                new_val = new_sample[col] + noise
                
                # Ensure values are non-negative and reasonable
                if new_val < min_values[col]:
                    # If below minimum, use random multiple of minimum value
                    new_val = min_values[col] * np.random.uniform(0.8, 1.2)
                new_sample[col] = max(min_values[col], new_val)
            else:
                # For categorical variables, randomly choose seed or neighbor value
                new_sample[col] = seed[col] if np.random.rand() > 0.5 else neighbor[col]
        
        # Ensure formula sum does not exceed 100
        if formula_cols:
            new_sample = normalize_formula(new_sample, formula_cols)
        
        # Ensure in long-tail region
        if new_sample[target_col] < threshold * 1.1:
            continue
        
        new_sample['source'] = 'synthetic'
        synthetic_samples.append(new_sample)
    
    return pd.DataFrame(synthetic_samples)

def generate_normal_samples(df, target_col, num_needed, formula_cols):
    """Generate normal samples in non-long-tail region (ensure no negative values and formula sum ≤ 100)"""
    print(f"Using custom method to generate {num_needed} normal samples")
    
    # Create position index (avoid using label index)
    df = df.reset_index(drop=True)
    
    feature_cols = [col for col in df.columns if col not in ['original_index']]
    
    # Record minimum value for each column (to prevent negative values)
    min_values = {}
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.float32]:
            min_values[col] = max(0, df[col].min())  # Ensure minimum value is non-negative
    
    synthetic_samples = []
    for _ in range(num_needed):
        # Randomly select a seed sample (using position index)
        seed_idx = np.random.randint(0, len(df))
        seed = df.iloc[seed_idx]
        
        # Randomly select a neighbor (using position index)
        neighbor_idx = np.random.choice([i for i in range(len(df)) if i != seed_idx])
        neighbor = df.iloc[neighbor_idx]
        
        # Interpolate between seed and neighbor
        alpha = np.random.rand()
        new_sample = seed[feature_cols].copy()
        for col in feature_cols:
            if df[col].dtype in [np.float64, np.float32]:
                # Interpolation calculation
                new_val = alpha * seed[col] + (1 - alpha) * neighbor[col]
                
                # Add noise (ensure non-negative)
                noise = np.random.normal(0, 0.05 * df[col].std())
                new_val += noise
                
                # Ensure values are non-negative and reasonable
                if new_val < min_values[col]:
                    # If below minimum, use random multiple of minimum value
                    new_val = min_values[col] * np.random.uniform(0.8, 1.2)
                new_sample[col] = max(min_values[col], new_val)
            else:
                # For categorical variables, randomly choose seed or neighbor value
                new_sample[col] = seed[col] if np.random.rand() > 0.5 else neighbor[col]
        
        # Ensure formula sum does not exceed 100
        if formula_cols:
            new_sample = normalize_formula(new_sample, formula_cols)
        
        new_sample['source'] = 'synthetic'
        synthetic_samples.append(new_sample)
    
    return pd.DataFrame(synthetic_samples)

# 1. Read data
df = pd.read_excel("phrr1pp1.xlsx")
df.columns = [col.strip() for col in df.columns]

# 2. Remove rows with NaN
df_clean = df.dropna().reset_index(drop=True)
original_df = df_clean.copy()
print(f"Original data rows: {len(original_df)}")

# 3. Protect ID column
if 'NO' in original_df.columns:
    # Create ID mapping dictionary
    no_mapping = dict(zip(original_df.index, original_df['NO']))
    print("✅ ID mapping dictionary created")
    
    # Remove ID column for augmentation
    df_clean = df_clean.drop(columns=['NO'])
else:
    no_mapping = {}
    print("⚠️ ID column (NO) not found, will create new IDs")

# 4. Identify formula feature columns (sum should not exceed 100%)
# Assume formula feature column names contain specific keywords like "content", "ratio", "%"
formula_cols = [col for col in df_clean.columns if any(keyword in col for keyword in ['content', 'ratio', '%'])]
if not formula_cols:
    # If not found, try to identify numeric columns (excluding target column)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    target_col = 'PHRR（Kw/m2）'
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    formula_cols = numeric_cols[:5]  # Assume first 5 numeric columns are formula features
    print(f"⚠️ No clear formula feature columns detected, using first 5 numeric columns: {formula_cols}")

print(f"Formula feature columns: {formula_cols}")

# 5. Analyze target variable distribution - identify long-tail region
target_col = 'PHRR（Kw/m2）'
plt.figure(figsize=(10, 6))
plt.hist(df_clean[target_col], bins=30, alpha=0.7, color='blue')
plt.title('Target Variable Distribution - Long-tail Analysis')
plt.xlabel(target_col)
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('target_distribution.png')
print("✅ Target variable distribution plot saved")

# Calculate distribution percentiles
percentiles = np.percentile(df_clean[target_col], [90, 95, 99])
print(f"Distribution percentiles: 90% = {percentiles[0]:.2f}, 95% = {percentiles[1]:.2f}, 99% = {percentiles[2]:.2f}")

# 6. Long-tail enhancement parameter configuration
TAIL_PERCENTILE = 90  # Define long-tail region (above 90th percentile)
TAIL_MULTIPLIER = 10  # Long-tail region enhancement multiplier
NON_TAIL_MULTIPLIER = 2  # Non-long-tail region enhancement multiplier

# 7. Identify long-tail samples
tail_threshold = np.percentile(df_clean[target_col], TAIL_PERCENTILE)
tail_mask = df_clean[target_col] > tail_threshold
tail_samples = df_clean[tail_mask]
non_tail_samples = df_clean[~tail_mask]

print(f"Long-tail region samples: {len(tail_samples)} (threshold: {tail_threshold:.2f})")
print(f"Non-long-tail region samples: {len(non_tail_samples)}")

# 8. Long-tail region enhancement - compatible with different smogn versions
def augment_tail_data(df, target_col, multiplier, original_indices, formula_cols):
    """Enhanced function specifically designed for long-tail region"""
    # Calculate number of samples needed
    num_needed = max(0, multiplier * len(df) - len(df))
    if num_needed == 0:
        return pd.DataFrame()
    
    print(f"Long-tail region needs to generate {num_needed} samples")
    
    # Add temporary index column for subsequent matching
    df = df.copy()
    df['original_index'] = original_indices
    
    # Execute SMOGN enhancement - compatible with different versions
    try:
        # Exclude temporary index column
        feature_cols = [col for col in df.columns if col != 'original_index']
        
        # Try using new version parameters
        try:
            aug_data = smogn.smoter(
                data=df[feature_cols],
                y=target_col,
                k=8,  # More neighbors
                samp_method='extreme',  # Focus on extreme values
                rel_thres=0.6,  # Lower threshold
                rel_method='auto',
                under_samp=False
            )
        except TypeError:
            # Old version parameters
            aug_data = smogn.smoter(
                data=df[feature_cols],
                y=target_col,
                k=8,
                samp_method='extreme',
                rel_thres=0.6,
                rel_method='auto'
            )
        
        # Add source marker
        aug_data['source'] = 'synthetic'
        aug_data['original_index'] = np.nan  # Newly generated samples have no original index
        
        print(f"SMOGN generated long-tail samples: {len(aug_data)}")
        
        # Ensure SMOGN generated samples also have no negative values
        for col in feature_cols:
            if df[col].dtype in [np.float64, np.float32]:
                min_val = max(0, df[col].min())
                aug_data[col] = aug_data[col].apply(lambda x: max(min_val, x))
        
        # Ensure formula sum does not exceed 100
        if formula_cols:
            aug_data = aug_data.apply(lambda row: normalize_formula(row, formula_cols), axis=1)
        
        # If not enough, supplement with other methods
        if len(aug_data) < num_needed:
            print(f"Supplementing with {num_needed - len(aug_data)} long-tail samples")
            additional = generate_extreme_samples(df, target_col, num_needed - len(aug_data), tail_threshold, formula_cols)
            aug_data = pd.concat([aug_data, additional], ignore_index=True)
        
        return aug_data
    
    except Exception as e:
        print(f"SMOGN long-tail enhancement failed: {str(e)}")
        return generate_extreme_samples(df, target_col, num_needed, tail_threshold, formula_cols)

# 9. Non-long-tail region enhancement - compatible with different smogn versions
def augment_non_tail_data(df, target_col, multiplier, original_indices, formula_cols):
    """Non-long-tail region enhancement function"""
    num_needed = max(0, multiplier * len(df) - len(df))
    if num_needed == 0:
        return pd.DataFrame()
    
    print(f"Non-long-tail region needs to generate {num_needed} samples")
    
    # Add temporary index column
    df = df.copy()
    df['original_index'] = original_indices
    
    try:
        feature_cols = [col for col in df.columns if col != 'original_index']
        
        # Try using new version parameters
        try:
            aug_data = smogn.smoter(
                data=df[feature_cols],
                y=target_col,
                k=5,
                samp_method='balance',
                rel_thres=0.8,
                rel_method='auto',
                under_samp=False
            )
        except TypeError:
            # Old version parameters
            aug_data = smogn.smoter(
                data=df[feature_cols],
                y=target_col,
                k=5,
                samp_method='balance',
                rel_thres=0.8,
                rel_method='auto'
            )
        
        aug_data['source'] = 'synthetic'
        aug_data['original_index'] = np.nan
        
        print(f"SMOGN generated non-long-tail samples: {len(aug_data)}")
        
        # Ensure SMOGN generated samples also have no negative values
        for col in feature_cols:
            if df[col].dtype in [np.float64, np.float32]:
                min_val = max(0, df[col].min())
                aug_data[col] = aug_data[col].apply(lambda x: max(min_val, x))
        
        # Ensure formula sum does not exceed 100
        if formula_cols:
            aug_data = aug_data.apply(lambda row: normalize_formula(row, formula_cols), axis=1)
        
        if len(aug_data) < num_needed:
            print(f"Supplementing with {num_needed - len(aug_data)} non-long-tail samples")
            additional = generate_normal_samples(df, target_col, num_needed - len(aug_data), formula_cols)
            aug_data = pd.concat([aug_data, additional], ignore_index=True)
        
        return aug_data
    
    except Exception as e:
        print(f"SMOGN non-long-tail enhancement failed: {str(e)}")
        return generate_normal_samples(df, target_col, num_needed, formula_cols)

# 10. Execute partitioned enhancement
print("\n===== Starting Long-tail Region Enhancement =====")
tail_augmented = augment_tail_data(
    tail_samples, 
    target_col, 
    TAIL_MULTIPLIER,
    original_indices=tail_samples.index,
    formula_cols=formula_cols
)

print("\n===== Starting Non-long-tail Region Enhancement =====")
non_tail_augmented = augment_non_tail_data(
    non_tail_samples, 
    target_col, 
    NON_TAIL_MULTIPLIER,
    original_indices=non_tail_samples.index,
    formula_cols=formula_cols
)

# 11. Combine all data
all_synthetic = pd.concat([tail_augmented, non_tail_augmented], ignore_index=True) if not tail_augmented.empty and not non_tail_augmented.empty else (
    tail_augmented if not tail_augmented.empty else non_tail_augmented
)

# 12. Restore original data (including ID column)
# Add source marker to original data
original_df['source'] = 'original'

# Combine original data and enhanced data
final_df = pd.concat([original_df, all_synthetic], ignore_index=True)

# 13. Handle ID column
if no_mapping:
    # Step 1: Restore original IDs for original data
    final_df.loc[final_df['source'] == 'original', 'NO'] = final_df.loc[
        final_df['source'] == 'original', :].index.map(lambda x: no_mapping.get(x, x))
    
    # Step 2: Generate new IDs for synthetic data
    max_no = max(no_mapping.values()) if no_mapping else 0
    synth_indices = final_df[final_df['source'] == 'synthetic'].index
    final_df.loc[synth_indices, 'NO'] = range(max_no + 1, max_no + 1 + len(synth_indices))
else:
    # If no original IDs, create completely new IDs
    final_df['NO'] = range(1, len(final_df) + 1)

# Ensure IDs are integers
final_df['NO'] = final_df['NO'].astype(int)

# 14. Final check: Ensure all numeric features are non-negative and formula sum ≤ 100
for col in final_df.columns:
    if final_df[col].dtype in [np.float64, np.float32]:
        min_val = final_df[col].min()
        if min_val < 0:
            print(f"⚠️ Warning: Column {col} still has negative values ({min_val:.4f}), correcting...")
            # Use original data's minimum non-negative value as baseline
            base_val = max(0, original_df[col].min())
            final_df[col] = final_df[col].apply(lambda x: max(base_val, x))

# Ensure formula sum does not exceed 100
if formula_cols:
    print("Checking formula sum for all samples...")
    formula_sums = final_df[formula_cols].sum(axis=1)
    over_100 = formula_sums > 100
    if over_100.any():
        print(f"⚠️ Found {over_100.sum()} samples with formula sum exceeding 100, correcting...")
        # Correct these samples
        for idx in final_df[over_100].index:
            final_df.loc[idx, formula_cols] = normalize_formula(final_df.loc[idx], formula_cols)
    
    # Verify correction results
    formula_sums_after = final_df[formula_cols].sum(axis=1)
    print(f"Formula sum check completed: Maximum={formula_sums_after.max():.2f}, Minimum={formula_sums_after.min():.2f}")

# 15. Verify enhancement effect
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(original_df[target_col], bins=30, alpha=0.7, color='blue', label='Original Data')
plt.title('Original Data Distribution')
plt.xlabel(target_col)

plt.subplot(1, 2, 2)
plt.hist(final_df[final_df['source'] == 'synthetic'][target_col], bins=30, alpha=0.7, color='red', label='Synthetic Data')
plt.title('Synthetic Data Distribution')
plt.xlabel(target_col)
plt.savefig('enhancement_comparison.png')
print("✅ Enhancement effect comparison plot saved")

# 16. Save results
final_df.to_excel("PHRR_LongTail_Enhanced1.xlsx", index=False)
print(f"\n✅ Enhancement completed! Final dataset: {len(final_df)} rows")
print(f"  Original data: {len(original_df)} rows")
print(f"  Synthetic data: {len(all_synthetic)} rows" if 'all_synthetic' in locals() else "  Synthetic data: 0 rows")
if len(tail_samples) > 0:
    print(f"  Long-tail region enhancement multiplier: {len(tail_augmented)/len(tail_samples):.1f}x")
else:
    print("  Long-tail region: no samples")
if len(non_tail_samples) > 0:
    print(f"  Non-long-tail region enhancement multiplier: {len(non_tail_augmented)/len(non_tail_samples):.1f}x")
else:
    print("  Non-long-tail region: no samples")
print("📁 Results saved to PHRR_LongTail_Enhanced1.xlsx")