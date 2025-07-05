import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_vae_final_dataset():
    """
    Comprehensive analysis of the vae_hitters_final.csv dataset
    """
    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS OF VAE_HITTERS_FINAL.CSV")
    print("=" * 80)
    
    # Load the dataset
    print("\nLoading dataset...")
    df = pd.read_csv('vae_hitters_final.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    # Basic info
    print("\n" + "=" * 50)
    print("BASIC DATASET INFORMATION")
    print("=" * 50)
    
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Data types:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"  {dtype}: {count} columns")
    
    # Column categories
    print("\n" + "=" * 50)
    print("COLUMN CATEGORIES")
    print("=" * 50)
    
    # Categorize columns
    baseball_stats = [col for col in df.columns if col in [
        'hand_speed_max', 'bat_speed_max', 'rot_acc_max',
        'sixty_time', 'thirty_time', 'ten_yard_time', 'run_speed_max',
        'exit_velo_max', 'exit_velo_avg', 'distance_max', 'sweet_spot_p',
        'hard_hit_p'
    ]]
    
    fielding_stats = [col for col in df.columns if col in [
        'c_velo', 'inf_velo', 'of_velo', 'p_velo'
    ]]
    
    missing_indicators = [col for col in df.columns if col.startswith('missing_')]
    
    demographic_cols = [col for col in df.columns if col in [
        'age', 'height', 'weight', 'high_school', 'state', 'position'
    ]]
    
    other_cols = [col for col in df.columns if col not in 
                  baseball_stats + fielding_stats + missing_indicators + demographic_cols]
    
    print(f"Baseball statistics: {len(baseball_stats)} columns")
    print(f"Fielding statistics: {len(fielding_stats)} columns")
    print(f"Missing indicators: {len(missing_indicators)} columns")
    print(f"Demographic columns: {len(demographic_cols)} columns")
    print(f"Other columns: {len(other_cols)} columns")
    
    # NaN Analysis
    print("\n" + "=" * 50)
    print("NaN VALUES ANALYSIS")
    print("=" * 50)
    
    # Calculate NaN counts for all columns
    nan_counts = df.isnull().sum()
    nan_percentages = (df.isnull().sum() / len(df)) * 100
    
    print("NaN counts and percentages for all columns:")
    print("-" * 60)
    
    for col in df.columns:
        count = nan_counts[col]
        percentage = nan_percentages[col]
        if count > 0:
            print(f"{col:25} | {count:6} NaN | {percentage:5.1f}%")
        else:
            print(f"{col:25} | {count:6} NaN | {percentage:5.1f}% ✓")
    
    # Summary of NaN patterns
    print(f"\nColumns with no NaN values: {(nan_counts == 0).sum()}")
    print(f"Columns with some NaN values: {(nan_counts > 0).sum()}")
    print(f"Columns with >50% NaN values: {(nan_percentages > 50).sum()}")
    print(f"Columns with >90% NaN values: {(nan_percentages > 90).sum()}")
    
    # Baseball Statistics Analysis
    print("\n" + "=" * 50)
    print("BASEBALL STATISTICS ANALYSIS")
    print("=" * 50)
    
    if baseball_stats:
        print("\nBaseball statistics summary:")
        print("-" * 80)
        print(f"{'Feature':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'NaN':<6} {'NaN%':<6}")
        print("-" * 80)
        
        for stat in baseball_stats:
            if stat in df.columns:
                data = df[stat]
                mean_val = data.mean()
                std_val = data.std()
                min_val = data.min()
                max_val = data.max()
                nan_count = data.isnull().sum()
                nan_pct = (nan_count / len(data)) * 100
                
                print(f"{stat:<20} {mean_val:<10.2f} {std_val:<10.2f} {min_val:<10.2f} {max_val:<10.2f} {nan_count:<6} {nan_pct:<6.1f}")
    
    # Fielding Statistics Analysis
    print("\n" + "=" * 50)
    print("FIELDING STATISTICS ANALYSIS")
    print("=" * 50)
    
    if fielding_stats:
        print("\nFielding statistics summary:")
        print("-" * 80)
        print(f"{'Feature':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'NaN':<6} {'NaN%':<6}")
        print("-" * 80)
        
        for stat in fielding_stats:
            if stat in df.columns:
                data = df[stat]
                mean_val = data.mean()
                std_val = data.std()
                min_val = data.min()
                max_val = data.max()
                nan_count = data.isnull().sum()
                nan_pct = (nan_count / len(data)) * 100
                
                print(f"{stat:<15} {mean_val:<10.2f} {std_val:<10.2f} {min_val:<10.2f} {max_val:<10.2f} {nan_count:<6} {nan_pct:<6.1f}")
    
    # Missing Indicators Analysis
    print("\n" + "=" * 50)
    print("MISSING INDICATORS ANALYSIS")
    print("=" * 50)
    
    if missing_indicators:
        print("\nMissing indicators summary:")
        print("-" * 50)
        print(f"{'Indicator':<25} {'Missing Count':<15} {'Missing %':<10}")
        print("-" * 50)
        
        for indicator in missing_indicators:
            missing_count = df[indicator].sum()
            missing_pct = (missing_count / len(df)) * 100
            print(f"{indicator:<25} {missing_count:<15} {missing_pct:<10.1f}")
        
        # Analyze number_of_missing distribution
        if 'number_of_missing' in df.columns:
            print(f"\n'number_of_missing' distribution:")
            print(f"  Mean: {df['number_of_missing'].mean():.2f}")
            print(f"  Median: {df['number_of_missing'].median():.2f}")
            print(f"  Std: {df['number_of_missing'].std():.2f}")
            print(f"  Min: {df['number_of_missing'].min()}")
            print(f"  Max: {df['number_of_missing'].max()}")
            
            # Show distribution
            missing_dist = df['number_of_missing'].value_counts().sort_index()
            print(f"\n  Distribution:")
            for count, freq in missing_dist.head(10).items():
                print(f"    {count} missing: {freq} players ({freq/len(df)*100:.1f}%)")
    
    # Demographic Analysis
    print("\n" + "=" * 50)
    print("DEMOGRAPHIC ANALYSIS")
    print("=" * 50)
    
    for col in demographic_cols:
        if col in df.columns:
            print(f"\n{col.upper()}:")
            if df[col].dtype in ['object', 'string']:
                # Categorical data
                value_counts = df[col].value_counts()
                print(f"  Total unique values: {df[col].nunique()}")
                print(f"  NaN count: {df[col].isnull().sum()}")
                print(f"  Top 5 values:")
                for val, count in value_counts.head(5).items():
                    print(f"    {val}: {count} ({count/len(df)*100:.1f}%)")
            else:
                # Numerical data
                data = df[col]
                print(f"  Mean: {data.mean():.2f}")
                print(f"  Median: {data.median():.2f}")
                print(f"  Std: {data.std():.2f}")
                print(f"  Min: {data.min()}")
                print(f"  Max: {data.max()}")
                print(f"  NaN count: {data.isnull().sum()}")
    
    # Position Analysis
    if 'position' in df.columns:
        print("\n" + "=" * 50)
        print("POSITION ANALYSIS")
        print("=" * 50)
        
        position_counts = df['position'].value_counts()
        print(f"Total positions: {df['position'].nunique()}")
        print(f"Position distribution:")
        for pos, count in position_counts.items():
            print(f"  {pos}: {count} players ({count/len(df)*100:.1f}%)")
    
    # Correlation Analysis for Baseball Stats
    print("\n" + "=" * 50)
    print("CORRELATION ANALYSIS (Baseball Statistics)")
    print("=" * 50)
    
    if len(baseball_stats) > 1:
        # Create correlation matrix for baseball stats
        baseball_data = df[baseball_stats].dropna()
        if len(baseball_data) > 0:
            corr_matrix = pd.DataFrame(baseball_data).corr()
            
            # Find highest correlations
            print("Top 10 highest correlations:")
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if not pd.isna(corr_val):
                        correlations.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            
            for i, (col1, col2, corr) in enumerate(correlations[:10]):
                print(f"  {i+1:2}. {col1:20} ↔ {col2:20} : {corr:6.3f}")
    
    # Data Quality Assessment
    print("\n" + "=" * 50)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 50)
    
    # Check for extreme values in baseball stats
    print("\nExtreme values check (beyond 3 standard deviations):")
    for stat in baseball_stats:
        if stat in df.columns:
            data = df[stat].dropna()
            if len(data) > 0:
                mean_val = data.mean()
                std_val = data.std()
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                
                extreme_low = (data < lower_bound).sum()
                extreme_high = (data > upper_bound).sum()
                
                if extreme_low > 0 or extreme_high > 0:
                    print(f"  {stat}: {extreme_low} below, {extreme_high} above 3σ range")
    
    # Summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    
    print(f"Total players: {len(df)}")
    print(f"Players with complete baseball stats: {(df['number_of_missing'] == 0).sum()} ({(df['number_of_missing'] == 0).sum()/len(df)*100:.1f}%)")
    print(f"Players with some missing baseball stats: {(df['number_of_missing'] > 0).sum()} ({(df['number_of_missing'] > 0).sum()/len(df)*100:.1f}%)")
    
    # VAE effectiveness
    print(f"\nVAE Effectiveness:")
    print(f"  Baseball stats filled by VAE: {len([col for col in baseball_stats if col in df.columns and df[col].isnull().sum() == 0])}")
    print(f"  Fielding stats still missing: {len([col for col in fielding_stats if col in df.columns and df[col].isnull().sum() > 0])}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    analyze_vae_final_dataset() 