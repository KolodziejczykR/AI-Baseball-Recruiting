import pandas as pd

# Use the absolute path to the CSV file
csv_path = '/Users/ryankolodziejczyk/Documents/AI Baseball Recruitment/code/backend/data/pitchers/pitch_by_position.csv'
df = pd.read_csv(csv_path)

# Print unique values for primary_position
print('Unique values for primary_position:')
print(df['primary_position'].unique())
print('\n')

# Print count and percentage of NaN values for each column
print('NaN (missing) values per column:')
nan_count = df.isna().sum()
nan_percent = (nan_count / len(df)) * 100
nan_summary = pd.DataFrame({'NaN Count': nan_count, 'NaN %': nan_percent.round(2)})
print(nan_summary)

# Print total number of rows
print(f'\nTotal rows: {len(df)}')

# Print a summary of the dataframe
df.info()

# --- Feature Engineering ---
# 1. num_pitches: count of non-null velo range columns per row
velo_cols = [
    'FastballVelo Range',
    'Changeup Velo Range',
    'Curveball Velo Range',
    'Slider Velo Range'
]
df['num_pitches'] = df[velo_cols].notna().sum(axis=1)

# 2. Pitch velo difference features
# fb_ch_velo_diff: FastballVelo Range - Changeup Velo Range
# fb_cb_velo_diff: FastballVelo Range - Curveball Velo Range
# fb_sl_velo_diff: FastballVelo Range - Slider Velo Range
def safe_diff(a, b):
    return a - b if pd.notna(a) and pd.notna(b) else pd.NA

df['fb_ch_velo_diff'] = df.apply(lambda row: safe_diff(row['FastballVelo Range'], row['Changeup Velo Range']), axis=1)
df['fb_cb_velo_diff'] = df.apply(lambda row: safe_diff(row['FastballVelo Range'], row['Curveball Velo Range']), axis=1)
df['fb_sl_velo_diff'] = df.apply(lambda row: safe_diff(row['FastballVelo Range'], row['Slider Velo Range']), axis=1)

# 3. Add missingness flags for each pitch feature
pitch_features = [
    'FastballVelocity (max)', 'FastballVelo Range', 'FastballSpin Rate (avg)',
    'Changeup Velo Range', 'Changeup Spin Rate (avg)',
    'Curveball Velo Range', 'Curveball Spin Rate (avg)',
    'Slider Velo Range', 'Slider Spin Rate (avg)'
]
for col in pitch_features:
    flag_col = f'{col}_missing'
    df[flag_col] = df[col].isna().astype(int)

# --- Report ---
print('\n--- Feature Engineering Report ---')
print('1. Added num_pitches: Number of non-null pitch velo ranges per row.')
print('2. Added pitch velo difference features:')
print('   - fb_ch_velo_diff: FastballVelo Range - Changeup Velo Range')
print('   - fb_cb_velo_diff: FastballVelo Range - Curveball Velo Range')
print('   - fb_sl_velo_diff: FastballVelo Range - Slider Velo Range')
print('3. Added missingness flags for each pitch feature (1 = missing, 0 = present):')
for col in pitch_features:
    print(f'   - {col}_missing')

# Optionally, save the new dataframe to a new CSV
df.to_csv('/Users/ryankolodziejczyk/Documents/AI Baseball Recruitment/code/backend/data/pitchers/pitch_by_position_with_features.csv', index=False) 