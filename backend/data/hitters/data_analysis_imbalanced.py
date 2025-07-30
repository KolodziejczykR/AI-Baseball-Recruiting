import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import make_scorer, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from lightgbm import LGBMClassifier
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import optuna



# Use the absolute path to the CSV file
csv_path = '/Users/ryankolodziejczyk/Documents/AI Baseball Recruitment/code/backend/data/hitters/vae_infielders.csv'

df = pd.read_csv(csv_path)
        
# Remove problematic columns with too many missing values
exclude_columns = ['hard_hit_p', 'position_velo']  # Keep inf_velo

# Also remove non-predictive columns and data leakage columns
non_predictive = ['Unnamed: 0', 'name', 'link', 'commitment', 'college_location', 
                    'conf_short', 'committment_group', 'high_school', 'class', 'positions',
                    'player_section_of_region', 'confidence', 'age', 'player_state']

# Remove data leakage columns that directly indicate the target
leakage_columns = ['division', 'conference']

exclude_columns.extend(non_predictive)
exclude_columns.extend(leakage_columns)

# Remove columns with too many missing values
missing_threshold = 0.5  # 50% missing
missing_counts = df.isnull().sum() / len(df)
high_missing_cols = missing_counts[missing_counts > missing_threshold].index.tolist()
exclude_columns.extend(high_missing_cols)

# Remove missing indicator columns (we'll handle missing values differently)
missing_indicator_cols = [col for col in df.columns if col.startswith('missing_')]
exclude_columns.extend(missing_indicator_cols)

df['d1_or_not'] = (df['three_section_commit_group'].str.lower() != 'non d1').astype(int)
exclude_columns.append('three_section_commit_group')

# Keep only relevant columns
keep_columns = [col for col in df.columns if col not in exclude_columns]
df = pd.DataFrame(df[keep_columns])

# The scraper flipped the order of 'throwing_hand' and 'hitting_handedness', so swap their values
df['throwing_hand'], df['hitting_handedness'] = df['hitting_handedness'], df['throwing_hand']

df['throwing_hand'] = df['throwing_hand'].str.strip()
df['hitting_handedness'] = df['hitting_handedness'].str.strip()

valid_throwing_hands = ['L', 'R']
valid_hitting_hands = ['R', 'L', 'S']

mask_hitting = df['hitting_handedness'].isin(valid_hitting_hands)
mask_throwing = df['throwing_hand'].isin(valid_throwing_hands)
df = df[mask_hitting & mask_throwing]

df_no_encoded = df.copy() # for checking unique values and making sure hit/throw are correct


# 11475 non d1, 4264 d1
# print(pd.Series(df['d1_or_not']).value_counts())

# Identify categorical columns (object or category dtype, except the target)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if 'd1_or_not' in categorical_cols:
    categorical_cols.remove('d1_or_not')

# One-hot encode categorical columns
if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Remove columns with all missing values
df_no_na = df.dropna(axis=0, how='any')

"""
# Separate features and target
X = df_no_na.drop(columns=['d1_or_not'])
y = df_no_na['d1_or_not']

# 80/20 train-test split, stratify to preserve class balance
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert back to pandas objects
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

unique, counts = np.unique(y_train, return_counts=True)
# print(f"y_train distribution:\n{unique} {counts}")

sm = SMOTE(random_state=42, sampling_strategy='auto')

# Apply SMOTE to balance the dataset
print("Applying SMOTE...")
try:
    resampled_data = sm.fit_resample(x_train, y_train)
    x_train_res = resampled_data[0]
    y_train_res = resampled_data[1]
    print("SMOTE completed successfully")
except Exception as e:
    print(f"SMOTE failed: {e}")
    # Fallback: use original data
    x_train_res = x_train
    y_train_res = y_train

unique_res, counts_res = np.unique(y_train_res, return_counts=True)
print(f"y_train_res distribution after SMOTE:\n{unique_res} {counts_res}")

# SMOTE data to numpy arrays
x_train_res = x_train_res.to_numpy().astype(np.float64)
y_train_res = y_train_res.to_numpy().astype(np.int64)

# Convert to numpy arrays for sklearn compatibility
x_train = x_train.to_numpy().astype(np.float64)
y_train = y_train.to_numpy().astype(np.int64)
x_test = x_test.to_numpy().astype(np.float64)
y_test = y_test.to_numpy().astype(np.int64)

clf = LGBMClassifier(random_state=42, verbose=-1, is_unbalance=True).fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
print("Confusion Matrix (original):")
print(confusion_matrix(y_test, y_pred))


# Calculate and print the No Information Rate (NIR)
# NIR is the accuracy that would be achieved by always predicting the most frequent class in y_test
most_common_count = Counter(y_test).most_common(1)[0][1]
nir = most_common_count / len(y_test)
print(f"No Information Rate (NIR): {nir:.4f}")

# CV validation for weights, lightgbm
recall_scorer = make_scorer(recall_score, pos_label=1)
weights = np.linspace(0.5, 1.0, 10)  # Keep ratios <= 1.0 to avoid NaN issues
pipe = ImbPipeline([('smoteenn', SMOTEENN(random_state=42)), ('lgbm', LGBMClassifier(random_state=42, verbose=-1))])

gsc = GridSearchCV(
    estimator=pipe, 
    param_grid={'smoteenn__sampling_strategy': weights}, 
    cv=5, 
    scoring=recall_scorer
)

grid_result = gsc.fit(x_train, y_train)

print("Best parameters set:")
print(grid_result.best_params_)
print("Best score:")
print(grid_result.best_score_)

weight_f1_score_df = pl.DataFrame({'score': grid_result.cv_results_['mean_test_score'], 'weight': weights})
print(weight_f1_score_df)

pipe2 = ImbPipeline([('smoteenn', SMOTEENN(random_state=42, sampling_strategy=grid_result.best_params_['smoteenn__sampling_strategy'])), ('lgbm', LGBMClassifier(random_state=42, verbose=-1))])
pipe2.fit(x_train, y_train)
y_pred2 = pipe2.predict(x_test)

print("Classification Report:")
print(classification_report(y_test, y_pred2))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred2))


print("x_train shape:", x_train.shape)
print("Feature names:", X.columns.tolist())

import matplotlib.pyplot as plt
import seaborn as sns

x_train_df = pd.DataFrame(x_train, columns=X.columns)
# Correlation matrix / heatmap
corr_matrix = x_train_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
            xticklabels=x_train_df.columns.tolist(), yticklabels=x_train_df.columns.tolist())
plt.title("Correlation Heatmap of Features in x_train")
plt.tight_layout()
plt.show()

# Plot KDE for continuous features by class

continuous_cols = df_no_na.select_dtypes(include=[np.number]).columns.tolist()
continuous_cols = [col for col in continuous_cols if col != 'd1_or_not']

for col in continuous_cols: 
    plt.figure(figsize=(8, 6))
    # Use the correct DataFrame and target
    df_no_na.loc[df_no_na['d1_or_not'] == 0, col].plot(kind='kde', label='Non-D1')
    df_no_na.loc[df_no_na['d1_or_not'] == 1, col].plot(kind='kde', label='D1')
    plt.title(f'Distribution of {col} by Class')
    plt.legend()
    plt.show()
"""

# has correct hit/pitch sides and no missing values
feature_engineering = df_no_na.copy()

# create new features
feature_engineering['velo_by_inf'] = feature_engineering.exit_velo_max * feature_engineering.inf_velo
feature_engineering['arm_swing'] = feature_engineering.inf_velo * feature_engineering.hand_speed_max
feature_engineering['rot_efficiency'] = feature_engineering.rot_acc_max / feature_engineering.sixty_time

feature_engineering['power_speed'] = feature_engineering.exit_velo_max / feature_engineering.sixty_time
feature_engineering['sweet_power'] = feature_engineering.sweet_spot_p * feature_engineering.exit_velo_max
feature_engineering['velo_dist'] = feature_engineering.exit_velo_max * feature_engineering.distance_max
feature_engineering['accel_power'] = feature_engineering.rot_acc_max * feature_engineering.exit_velo_max

feature_engineering['speed_ratio'] = feature_engineering.run_speed_max / feature_engineering.sixty_time
feature_engineering['accel_per_10'] = feature_engineering.rot_acc_max / feature_engineering.ten_yard_time
feature_engineering['accel_per_10_sq'] = feature_engineering['accel_per_10'] ** 2

feature_engineering['sixty_inv'] = 1 / feature_engineering['sixty_time']
feature_engineering['sweet_mid'] = ((feature_engineering['sweet_spot_p'] > 50) & (feature_engineering['sweet_spot_p'] < 75))

feature_engineering['height_weight'] = feature_engineering.height * feature_engineering.weight

# Separate features and target
fX = feature_engineering.drop(columns=['d1_or_not'])
fy = feature_engineering['d1_or_not']

# 80/20 train-test split, stratify to preserve class balance
fx_train, fx_test, fy_train, fy_test = train_test_split(
    fX, fy, test_size=0.2, random_state=42, stratify=fy
)

# Convert to pandas DataFrames for better visualization
fx_train = pd.DataFrame(fx_train)
fx_test = pd.DataFrame(fx_test)
fy_train = pd.Series(fy_train)
fy_test = pd.Series(fy_test)

"""
# Replace inf/-inf with nan, then drop rows with nan
fx_train = pd.DataFrame(fx_train, columns=feature_engineering.drop(columns=['d1_or_not']).columns)
fx_train.replace([np.inf, -np.inf], np.nan, inplace=True)
fx_train.dropna(axis=0, how='any', inplace=True)
fy_train = fy_train[:len(fx_train)]  # Ensure y matches X after dropping rows


# SMOTE has slightly better results, but significantly reduced correct D! predictions
# which are important for the model to be able to predict correctly

f_resampled_data = sm.fit_resample(fx_train, fy_train)
fx_train_smote = f_resampled_data[0]
fy_train_smote = f_resampled_data[1]
print("SMOTE completed successfully")
fx_train_smote = fx_train_smote.to_numpy().astype(np.float64)
fy_train_smote = fy_train_smote.to_numpy().astype(np.int64)
"""

# Convert to numpy arrays for sklearn compatibility
fx_train = fx_train.to_numpy().astype(np.float64)
fy_train = fy_train.to_numpy().astype(np.int64)
fx_test = fx_test.to_numpy().astype(np.float64)
fy_test = fy_test.to_numpy().astype(np.int64)


f_clf = LGBMClassifier(random_state=42, verbose=-1, is_unbalance=True).fit(fx_train, fy_train)

y_pred = f_clf.predict(fx_test)
print(classification_report(fy_test, y_pred))
print("Confusion Matrix (engineered):")
print(confusion_matrix(fy_test, y_pred))

importances = f_clf.feature_importances_
feature_names = feature_engineering.drop(columns=['d1_or_not']).columns

# Print feature importances sorted from most to least important
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("\nFeature Importances (new model):")
print(importance_df)



                    