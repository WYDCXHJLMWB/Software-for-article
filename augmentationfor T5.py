# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 12:50:52 2025

@author: ma'wei'bin
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def global_nonnegative_augmentation(df, n_samples=100, noise_scale=0.05, 
                                   exclude_cols=None, non_negative_cols='all',
                                   scaling=True, random_state=None):
    """
    Perform non-negative noise interpolation to generate new samples for the entire dataset
    
    Parameters:
    - df: Original data DataFrame
    - n_samples: Number of samples to generate
    - noise_scale: Noise amplitude (0.01-0.1)
    - exclude_cols: List of column names to exclude (e.g., ['ID', 'source'])
    - non_negative_cols: 'all' or list of column names that need to remain non-negative
    - scaling: Whether to perform normalization (recommended True)
    - random_state: Random seed (for reproducibility)
    
    Returns:
    - Newly generated samples DataFrame
    """
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Set default excluded columns
    if exclude_cols is None:
        exclude_cols = ['source', 'NO']
    else:
        exclude_cols = list(set(exclude_cols + ['source']))
    
    # Get feature columns (excluding specified columns)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Determine non-negative columns
    if non_negative_cols == 'all':
        non_negative_cols = feature_cols
    elif not isinstance(non_negative_cols, list):
        raise ValueError("non_negative_cols should be 'all' or a list of column names")
    
    # Exception handling: insufficient samples
    if len(df) < 2:
        raise ValueError(f"Insufficient dataset samples ({len(df)}), at least 2 samples are required for interpolation")
    
    # Data normalization (0-1 range)
    if scaling:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=feature_cols)
    else:
        scaled_df = df[feature_cols].copy()
    
    # Calculate standard deviation of feature columns (avoid division by zero)
    feature_stds = scaled_df.std().replace(0, 1e-6)
    
    new_samples = []
    
    # Generate new samples
    for _ in range(n_samples):
        # Randomly select two different samples
        idx1, idx2 = np.random.choice(len(df), 2, replace=False)
        row1 = scaled_df.iloc[idx1].values
        row2 = scaled_df.iloc[idx2].values
        
        # Linear interpolation (ensure within 0-1 range)
        alpha = np.random.rand()
        interp = alpha * row1 + (1 - alpha) * row2
        
        # Add proportional noise (limit to 0-1 range)
        noise = np.random.normal(0, noise_scale, size=interp.shape) * feature_stds.values
        new_sample = np.clip(interp + noise, 0, 1)
        
        # Ensure 'Matrix type' column values are 0 or 1
        if 'Matrix type' in feature_cols:
            new_sample[feature_cols.index('Matrix type')] = np.random.choice([0, 1])
        
        # Add to sample list
        new_samples.append(new_sample)
    
    # Create new DataFrame
    df_scaled_new = pd.DataFrame(new_samples, columns=feature_cols)
    
    # Inverse transform normalized data
    if scaling:
        df_new = pd.DataFrame(scaler.inverse_transform(df_scaled_new), columns=feature_cols)
    else:
        df_new = df_scaled_new
    
    # Add excluded columns and set default values
    for col in exclude_cols:
        if col == 'source':
            df_new[col] = 'synthetic'
        else:
            # Maintain original data type
            original_dtype = df[col].dtype
            df_new[col] = pd.Series(dtype=original_dtype)
    
    # Ensure global non-negativity
    for col in non_negative_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].clip(lower=0)
    
    # Ensure consistent column order
    df_new = df_new[df.columns]
    
    return df_new

def process_data(input_file, output_file, n_samples=100, noise_scale=0.05, 
                exclude_cols=['NO'], non_negative_cols='all', random_state=42):
    """
    Complete data processing pipeline
    """
    print(f"📊 Starting data processing: {input_file}")
    
    # Read data
    df = pd.read_excel(input_file)
    df.columns = [col.strip() for col in df.columns]  # Clean column name spaces
    print(f"Original data dimensions: {df.shape}")
    
    # Data preprocessing
    df_clean = df.dropna().reset_index(drop=True)
    df_clean['source'] = 'original'
    print(f"Cleaned data: {len(df_clean)} rows")
    
    # Augment data
    print(f"🔧 Generating {n_samples} augmented samples...")
    df_aug = global_nonnegative_augmentation(
        df_clean,
        n_samples=n_samples,
        noise_scale=noise_scale,
        exclude_cols=exclude_cols,
        non_negative_cols=non_negative_cols,
        scaling=True,
        random_state=random_state
    )
    
    # Combine data
    df_combined = pd.concat([df_clean, df_aug], ignore_index=True)
    
    # Save results
    df_combined.to_excel(output_file, index=False)
    
    print(f"✅ Processing completed! Original samples: {len(df_clean)}, New samples: {len(df_aug)}, Total: {len(df_combined)}")
    print(f"💾 Results saved to: {output_file}")
    
    return df_combined

# Usage example
if __name__ == "__main__":
    # Configuration parameters
    INPUT_FILE = "T5%.xlsx"
    OUTPUT_FILE = "T5%A.xlsx"
    
    # Execute processing
    augmented_data = process_data(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        n_samples=64,  # Generate 64 new samples
        noise_scale=0.03,  # Medium noise level
        exclude_cols=['NO'],  # Excluded identifier columns
        non_negative_cols='all',  # All columns remain non-negative
        random_state=42  # Fixed random seed for reproducibility
    )
    
    # Display first few rows
    print("\nAugmented data preview:")
    print(augmented_data.head())