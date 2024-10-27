import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import chardet
from tqdm import tqdm
from datetime import datetime

def detect_encoding(file_path):
    """
    Detect the encoding of a file using chardet.
    
    Parameters:
    file_path (str): Path to the file
    
    Returns:
    str: Detected encoding
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read(10000)  # Read first 10000 bytes
        result = chardet.detect(raw_data)
        return result['encoding']

def load_csv_files(directory_path, limit=None):
    """
    Load CSV files from a directory into a single DataFrame with a file limit option.
    
    Parameters:
    directory_path (str): Path to directory containing CSV files
    limit (int): Maximum number of files to process. If None, process all files
    
    Returns:
    pd.DataFrame: Combined DataFrame from loaded CSV files
    """
    print("Loading and combining CSV files...")
    all_dataframes = []
    
    # Get list of all CSV files
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    if not csv_files:
        raise ValueError(f"No CSV files found in {directory_path}")
    
    # Apply limit if specified
    if limit is not None:
        csv_files = csv_files[:limit]
        print(f"Processing first {limit} files out of {len(csv_files)} total files")
    
    # Process files with progress bar
    for filename in tqdm(csv_files, desc="Processing CSV files"):
        file_path = os.path.join(directory_path, filename)
        
        try:
            # First try UTF-8
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            try:
                # Try latin-1
                df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
            except UnicodeDecodeError:
                try:
                    # Try detected encoding
                    detected_encoding = detect_encoding(file_path)
                    df = pd.read_csv(file_path, encoding=detected_encoding, low_memory=False)
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")
                    continue
        
        # Add source file information
        df['source_file'] = filename
        all_dataframes.append(df)
        print(f"Successfully loaded {filename} with {len(df)} rows")
        
        # Optional: Print memory usage
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # Convert to MB
        print(f"Memory usage for {filename}: {memory_usage:.2f} MB")
    
    if not all_dataframes:
        raise ValueError("No DataFrames were successfully loaded")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Print final statistics
    print(f"\nProcessed {len(csv_files)} files")
    print(f"Total combined rows: {len(combined_df):,}")
    print(f"Total combined columns: {len(combined_df.columns)}")
    total_memory = combined_df.memory_usage(deep=True).sum() / 1024**2
    print(f"Total memory usage: {total_memory:.2f} MB")
    
    return combined_df

def analyze_ppp_data(directory_path):
    """
    Perform comprehensive exploratory data analysis on PPP loan dataset.
    """
    # Load and combine the data
    df = load_csv_files(directory_path, 5)
    
    # Basic dataset information
    print("\n=== Basic Dataset Information ===")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}: {df[col].dtype}")
    
    # Missing values analysis
    print("\n=== Missing Values Analysis ===")
    missing_info = pd.DataFrame({
        'Missing Values': df.isnull().sum(),
        'Percentage Missing': (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('Percentage Missing', ascending=False)
    print(missing_info[missing_info['Missing Values'] > 0])
    
    # Numerical columns analysis
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print("\n=== Numerical Columns Statistics ===")
    print(df[numerical_cols].describe())
    
    # Correlation analysis with JobsReported
    if 'JobsReported' in df.columns:
        # Convert potential string columns to numeric
        numeric_df = df.copy()
        amount_columns = ['InitialApprovalAmount', 'CurrentApprovalAmount', 'UndisbursedAmount',
                         'UTILITIES_PROCEED', 'PAYROLL_PROCEED', 'MORTGAGE_INTEREST_PROCEED',
                         'RENT_PROCEED', 'REFINANCE_EIDL_PROCEED', 'HEALTH_CARE_PROCEED',
                         'DEBT_INTEREST_PROCEED', 'ForgivenessAmount']
        
        for col in amount_columns:
            if col in df.columns:
                numeric_df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate correlations for numeric columns only
        numeric_cols = numeric_df.select_dtypes(include=['int64', 'float64']).columns
        correlations = numeric_df[numeric_cols].corr()['JobsReported'].sort_values(ascending=False)
        print("\n=== Correlations with JobsReported ===")
        print(correlations)
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap of Numerical Variables')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    print("\n=== Categorical Columns Analysis ===")
    for col in categorical_cols:
        n_unique = df[col].nunique()
        print(f"\n{col}:")
        print(f"Number of unique values: {n_unique}")
        if n_unique < 20:  # Only show value counts for columns with fewer unique values
            print("Value counts:")
            print(df[col].value_counts(normalize=True).head())
    
    # NAICS code analysis
    if 'NAICSCode' in df.columns:
        print("\n=== NAICS Code Analysis ===")
        naics_summary = df.groupby('NAICSCode').agg({
            'JobsReported': ['count', 'mean', 'std'],
            'InitialApprovalAmount': ['mean']  # Changed from LoanAmount to InitialApprovalAmount
        }).round(2)
        print(naics_summary.sort_values(('JobsReported', 'count'), ascending=False).head())
    
    # Distribution of JobsReported
    if 'JobsReported' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='JobsReported', bins=50)
        plt.title('Distribution of Jobs Reported')
        plt.savefig('jobs_distribution.png')
        plt.close()
        
        # Basic statistics for JobsReported
        print("\n=== JobsReported Statistics ===")
        jobs_reported = pd.to_numeric(df['JobsReported'], errors='coerce')
        print(f"Mean: {jobs_reported.mean():.2f}")
        print(f"Median: {jobs_reported.median():.2f}")
        print(f"Std Dev: {jobs_reported.std():.2f}")
        print(f"Skewness: {jobs_reported.skew():.2f}")
        
        # Check for outliers
        z_scores = np.abs(stats.zscore(jobs_reported.fillna(jobs_reported.median())))
        outliers_count = len(z_scores[z_scores > 3])
        print(f"\nNumber of outliers (z-score > 3): {outliers_count}")
    
    # Save the combined DataFrame
    df.to_csv('combined_ppp_data' + datetime.now().strftime('%Y%m%d%H%M%S') + '.csv', index=False)
    print("\nSaved combined dataset to 'combined_ppp_data.csv'")
    
    return {
        'df': df,
        'missing_info': missing_info,
        'correlations': correlations if 'JobsReported' in df.columns else None,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols
    }

if __name__ == "__main__":
    # Replace with your actual directory path
    directory_path = "data/raw"
    results = analyze_ppp_data(directory_path)