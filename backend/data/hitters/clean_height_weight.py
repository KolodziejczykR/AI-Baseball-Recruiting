import pandas as pd

def clean_height_weight():
    """
    Remove rows with NaN values in height or weight columns
    """
    print("Loading vae_hitters_final.csv...")
    df = pd.read_csv('vae_hitters_final.csv')
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original rows: {len(df)}")
    
    # Check NaN counts before cleaning
    height_nan = df['height'].isnull().sum()
    weight_nan = df['weight'].isnull().sum()
    
    print(f"\nNaN counts before cleaning:")
    print(f"  Height NaN: {height_nan}")
    print(f"  Weight NaN: {weight_nan}")
    
    # Remove rows with NaN in height or weight
    df_cleaned = df.dropna(subset=['height', 'weight'])
    
    print(f"\nAfter removing rows with NaN in height or weight:")
    print(f"  Rows removed: {len(df) - len(df_cleaned)}")
    print(f"  Remaining rows: {len(df_cleaned)}")
    print(f"  New dataset shape: {df_cleaned.shape}")
    
    # Verify no NaN values remain in height and weight
    height_nan_after = df_cleaned['height'].isnull().sum()
    weight_nan_after = df_cleaned['weight'].isnull().sum()
    
    print(f"\nNaN counts after cleaning:")
    print(f"  Height NaN: {height_nan_after}")
    print(f"  Weight NaN: {weight_nan_after}")
    
    # Save the cleaned dataset
    output_file = 'vae_hitters_final_cleaned.csv'
    df_cleaned.to_csv(output_file, index=False)
    
    print(f"\nCleaned dataset saved as: {output_file}")
    
    # Show some statistics of the cleaned data
    print(f"\nCleaned dataset statistics:")
    print(f"  Height - Mean: {df_cleaned['height'].mean():.2f}, Std: {df_cleaned['height'].std():.2f}")
    print(f"  Weight - Mean: {df_cleaned['weight'].mean():.2f}, Std: {df_cleaned['weight'].std():.2f}")
    
    return df_cleaned

if __name__ == "__main__":
    clean_height_weight() 