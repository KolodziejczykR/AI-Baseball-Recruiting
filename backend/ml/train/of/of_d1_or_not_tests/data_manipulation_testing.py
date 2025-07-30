import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, fbeta_score, f1_score
import xgboost as xgb
import optuna

# Load data
csv_path = '/Users/ryankolodziejczyk/Documents/AI Baseball Recruitment/code/backend/data/hitters/vae_outfielders.csv'
og_df = pd.read_csv(csv_path)

# Remove problematic columns with too many missing values
exclude_columns = ['hard_hit_p', 'position_velo']  # Keep inf_velo

# Also remove non-predictive columns and data leakage columns
non_predictive = ['Unnamed: 0', 'name', 'link', 'commitment', 'college_location', 
                    'conf_short', 'committment_group', 'high_school', 'class', 'positions',
                    'player_section_of_region', 'confidence', 'age', 'player_state']

# Remove data leakage columns that directly indicate the target
leakage_columns = ['division', 'conference']

# Remove missing indicator columns (we'll handle missing values differently)
missing_indicator_cols = [col for col in og_df.columns if col.startswith('missing_')]
exclude_columns.extend(missing_indicator_cols)

og_df['d1_or_not'] = (og_df['three_section_commit_group'].str.lower() != 'non d1').astype(int)
exclude_columns.append('three_section_commit_group')

keep_columns = [col for col in og_df.columns if col not in exclude_columns]
og_df = pd.DataFrame(og_df[keep_columns])

# The scraper flipped the order of 'throwing_hand' and 'hitting_handedness', so swap their values
og_df['throwing_hand'], og_df['hitting_handedness'] = og_df['hitting_handedness'], og_df['throwing_hand']

og_df['throwing_hand'] = og_df['throwing_hand'].str.strip()
og_df['hitting_handedness'] = og_df['hitting_handedness'].str.strip()

valid_throwing_hands = ['L', 'R']
valid_hitting_hands = ['R', 'L', 'S']

mask_hitting = og_df['hitting_handedness'].isin(valid_hitting_hands)
mask_throwing = og_df['throwing_hand'].isin(valid_throwing_hands)
og_df = og_df[mask_hitting & mask_throwing]

og_df.to_csv('/Users/ryankolodziejczyk/Documents/AI Baseball Recruitment/code/backend/data/hitters/of_feat_eng_d1_or_not.csv', index=False)
