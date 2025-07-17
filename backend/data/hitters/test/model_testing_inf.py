import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Use the absolute path to the CSV file
csv_path = '/Users/ryankolodziejczyk/Documents/AI Baseball Recruitment/code/backend/data/hitters/test/inf_feat_eng.csv'

# Load data
df = pd.read_csv(csv_path)

X = df.drop(columns=['d1_or_not', 'exit_velo_avg', 'velo_by_inf', 'height_weight', 'power_speed', 'sweet_power', 'velo_dist', 'accel_power', 'speed_ratio', 'accel_per_10', 'accel_per_10_sq', 'sixty_inv', 'sweet_mid', 'arm_swing', 'rot_efficiency', 'run_speed_max', 'ten_yard_time', 'thirty_time'])
y = df['d1_or_not']

# Data cleaning: Handle infinite values and outliers
print("Data shape before cleaning:", X.shape)
print("Infinite values in features:", np.isinf(X).sum().sum())
print("NaN values in features:", X.isna().sum().sum())

# Replace infinite values with NaN, then fill with median
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Additional outlier handling: clip extreme values
for col in X.columns:
    if X[col].dtype in ['float64', 'int64']:
        Q1 = X[col].quantile(0.01)
        Q3 = X[col].quantile(0.99)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)

print("Data shape after cleaning:", X.shape)
print("Infinite values after cleaning:", np.isinf(X).sum().sum())
print("NaN values after cleaning:", X.isna().sum().sum())

# Split data for final evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Initialize 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# LightGBM model
print("=" * 50)
print("LIGHTGBM MODEL")
print("=" * 50)

lgb_model = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    verbosity=-1,
    is_unbalance=True
)

# Perform 5-fold cross-validation
lgb_cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=cv, scoring='f1')

#Train final LightGBM model on full training data
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)

# XGBoost model
print("\n" + "=" * 50)
print("XGBOOST MODEL")
print("=" * 50)

# Compute scale_pos_weight for XGBoost based on class distribution
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
scale_pos_weight = class_weights[1] / class_weights[0] if len(class_weights) == 2 else 1

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='logloss',
    verbosity=0,
    scale_pos_weight=scale_pos_weight,
    missing=np.nan
)

# Perform 5-fold cross-validation
xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='f1')
# Train final XGBoost model on full training data
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# Model comparison
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
print(f"LightGBM Test F1Score: {f1_score(y_test, lgb_pred):.4f}")
print(f"XGBoost Test F1Score: {f1_score(y_test, xgb_pred):.4f}")
print(f"LightGBM Test Accuracy: {accuracy_score(y_test, lgb_pred):.4f}")
print(f"XGBoost Test Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")


from statsmodels.stats.outliers_influence import variance_inflation_factor

print("\n" + "=" * 50)
print("VIF SCORES FOR XGBOOST MODEL FEATURES")
print("=" * 50)

# 1) Build DataFrame
X_train_vif = pd.DataFrame(X_train, columns=X.columns)

# 2) One-hot encode any non-numeric cols (optional, if you want VIF on dummies)
cat_cols = X_train_vif.select_dtypes(include=['object','category']).columns
if len(cat_cols):
    X_train_vif = pd.get_dummies(
        X_train_vif,
        columns=cat_cols,
        drop_first=True
    )

# 3) Keep only numeric columns
X_num = X_train_vif.select_dtypes(include=[np.number])

# 4) Drop rows with NaN or infinite values
X_num = X_num.replace([np.inf, -np.inf], np.nan).dropna()

# 5) Compute VIFs
vif_data = []
for i, col in enumerate(X_num.columns):
    vif = variance_inflation_factor(X_num.values, i)
    vif_data.append((col, vif))

# 6) Present results sorted by VIF descending
vif_df = (
    pd.DataFrame(vif_data, columns=['feature','VIF'])  # type: ignore
      .sort_values('VIF', ascending=False)
      .reset_index(drop=True)
)
print(vif_df.round(2))

