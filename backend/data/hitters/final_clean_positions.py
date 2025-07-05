import pandas as pd

def final_clean_positions():
    """
    Clean the segmented datasets to keep only specific positions:
    - Infielders: 1B, 2B, 3B, SS only
    - Outfielders: OF only  
    - Catchers: C only
    """
    
    # Clean infielders - keep only 1B, 2B, 3B, SS
    print("Cleaning infielders dataset...")
    infielders = pd.read_csv('vae_infielders.csv')
    original_inf = len(infielders)
    infield_positions = ['1B', '2B', '3B', 'SS']
    infielders_cleaned = infielders[infielders['primary_position'].isin(infield_positions)]
    infielders_cleaned.to_csv('vae_infielders.csv', index=False)
    print(f"  Infielders: {original_inf} -> {len(infielders_cleaned)} rows")
    
    # Clean outfielders - keep only OF
    print("Cleaning outfielders dataset...")
    outfielders = pd.read_csv('vae_outfielders.csv')
    original_of = len(outfielders)
    outfielders_cleaned = outfielders[outfielders['primary_position'] == 'OF']
    outfielders_cleaned.to_csv('vae_outfielders.csv', index=False)
    print(f"  Outfielders: {original_of} -> {len(outfielders_cleaned)} rows")
    
    # Clean catchers - keep only C
    print("Cleaning catchers dataset...")
    catchers = pd.read_csv('vae_catchers.csv')
    original_c = len(catchers)
    catchers_cleaned = catchers[catchers['primary_position'] == 'C']
    catchers_cleaned.to_csv('vae_catchers.csv', index=False)
    print(f"  Catchers: {original_c} -> {len(catchers_cleaned)} rows")
    
    print("\n=== FINAL POSITION DISTRIBUTIONS ===")
    
    # Print final distributions
    print("Infielders positions:")
    inf_positions = infielders_cleaned['primary_position'].value_counts().sort_index()
    for pos, count in inf_positions.items():
        print(f"  {pos}: {count}")
    
    print("\nOutfielders positions:")
    of_positions = outfielders_cleaned['primary_position'].value_counts().sort_index()
    for pos, count in of_positions.items():
        print(f"  {pos}: {count}")
    
    print("\nCatchers positions:")
    c_positions = catchers_cleaned['primary_position'].value_counts().sort_index()
    for pos, count in c_positions.items():
        print(f"  {pos}: {count}")

if __name__ == "__main__":
    final_clean_positions() 