"""
Fitbit Large Dataset Generator
Generates a 500,000+ row synthetic dataset based on the original Fitbit data distribution.
Outputs in CSV and Apache Parquet (standard big-data format for AWS Athena/S3).
"""

import os
import numpy as np
import pandas as pd

def generate_large_dataset():
    input_path = "Fitbit_dataset.csv"
    output_csv = "Fitbit_large_dataset.csv"
    output_parquet = "Fitbit_large_dataset.parquet"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source dataset not found at {input_path}")

    print("Reading source dataset...")
    df = pd.read_csv(input_path)

    # Clean index if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    original_rows = len(df)
    target_rows = 500000
    multiplication_factor = int(np.ceil(target_rows / original_rows))

    print(f"Original rows: {original_rows}")
    print(f"Scaling up by a factor of {multiplication_factor} to reach ~{target_rows} rows...")

    # Replicate the dataframe
    large_df = pd.concat([df] * multiplication_factor, ignore_index=True)
    
    # We only keep up to target_rows
    large_df = large_df.iloc[:target_rows].copy()

    # Introduce small statistical variations (noise) to numerical columns to ensure uniqueness
    # while preserving correlations and distributions.
    print("Simulating variance (adding statistical noise)...")
    numerical_cols = [
        'Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
        'Session_Duration (hours)', 'Fat_Percentage', 'Water_Intake (liters)',
        'BMI', 'Base_MET', 'HR_Intensity', 'Effective_MET', 'Calories_Burned (kcal)'
    ]

    np.random.seed(42)
    for col in numerical_cols:
        col_std = large_df[col].std()
        # Add normal distribution noise (scaled to 1.5% of standard deviation to preserve patterns)
        noise = np.random.normal(0, col_std * 0.015, size=len(large_df))
        
        # Apply noise
        large_df[col] = large_df[col] + noise
        
        # Keep bounds realistic
        if col in ['Age', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Workout_Frequency (days/week)', 'Experience_Level']:
            large_df[col] = large_df[col].round().astype(int)
        elif col in ['Weight (kg)', 'Fat_Percentage', 'Water_Intake (liters)', 'Calories_Burned (kcal)']:
            large_df[col] = large_df[col].round(2)
        elif col in ['Height (m)', 'BMI', 'Base_MET', 'HR_Intensity', 'Effective_MET']:
            large_df[col] = large_df[col].round(4)

        # Make sure no negative values exist where they shouldn't
        if col in ['Age', 'Weight (kg)', 'Height (m)', 'Avg_BPM', 'Resting_BPM', 'Session_Duration (hours)', 'Water_Intake (liters)', 'Calories_Burned (kcal)']:
            large_df[col] = large_df[col].clip(lower=df[col].min() * 0.9, upper=df[col].max() * 1.1)

    # Recalculate physical dependencies exactly to avoid breaking logical consistency
    print("Enforcing logical physical consistency (BMI & METs)...")
    large_df['BMI'] = large_df['Weight (kg)'] / (large_df['Height (m)'] ** 2)
    large_df['Effective_MET'] = large_df['Base_MET'] * large_df['HR_Intensity']
    
    # Round metrics for cleanliness
    large_df['BMI'] = large_df['BMI'].round(2)
    large_df['Effective_MET'] = large_df['Effective_MET'].round(4)

    # Save to CSV
    print(f"Writing to CSV format: {output_csv}...")
    large_df.to_csv(output_csv, index=False)
    print(f"Saved CSV. Size: {os.path.getsize(output_csv) / (1024*1024):.2f} MB")

    # Save to Parquet (standard compressed columnar format for AWS Athena/Redshift/S3)
    print(f"Writing to Parquet format: {output_parquet}...")
    large_df.to_parquet(output_parquet, compression='snappy', index=False)
    print(f"Saved Parquet. Size: {os.path.getsize(output_parquet) / (1024*1024):.2f} MB")

    print("Large dataset generation completed successfully!")

if __name__ == "__main__":
    generate_large_dataset()
