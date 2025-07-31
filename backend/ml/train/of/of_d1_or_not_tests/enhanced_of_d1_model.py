import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, fbeta_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

print("=" * 80)
print("ENHANCED OUTFIELD D1 MODEL - TARGETING 75%+ ACCURACY")
print("Addressing class imbalance and feature optimization")
print("=" * 80)

# Load data
csv_path = '/Users/ryankolodziejczyk/Documents/AI Baseball Recruitment/code/backend/data/hitters/of_feat_eng_d1_or_not.csv'
df = pd.read_csv(csv_path)

print(f"Loaded outfield dataset with {len(df)} players")
print(f"D1 Distribution: {df['d1_or_not'].value_counts()}")
print(f"D1 Rate: {df['d1_or_not'].mean():.2%}")
print(f"Class imbalance ratio: {(df['d1_or_not'] == 0).sum() / (df['d1_or_not'] == 1).sum():.1f}:1")

# ============================================================================
# ENHANCED DATA ANALYSIS AND FEATURE ENGINEERING
# ============================================================================
print("\n🔍 Enhanced Data Analysis:")

# Deep dive into missing patterns
missing_pattern = df.groupby('d1_or_not')['of_velo'].agg(['count', lambda x: x.isna().sum(), 'mean', 'std'])
missing_pattern.columns = ['total', 'missing', 'mean_velo', 'std_velo']
missing_pattern['missing_pct'] = missing_pattern['missing'] / missing_pattern['total'] * 100
print("Missing of_velo patterns by class:")
print(missing_pattern)

# Drop rows with missing 'of_velo' or 'player_region'
df = df.dropna(subset=['of_velo', 'player_region'])

# Create categorical encodings
df = pd.get_dummies(df, columns=['player_region', 'throwing_hand', 'hitting_handedness'], 
                   prefix_sep='_', drop_first=True)

# ============================================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================================
print("\n🔧 Enhanced Feature Engineering:")

# Core athletic metrics
df['power_speed'] = df['exit_velo_max'] / df['sixty_time']
df['of_velo_sixty_ratio'] = df['of_velo'] / df['sixty_time']
df['height_weight'] = df['height'] * df['weight']

# Enhanced outfield-specific features
df['of_arm_strength'] = (df['of_velo'] >= df['of_velo'].quantile(0.75)).astype(int)
df['of_arm_plus'] = (df['of_velo'] >= df['of_velo'].quantile(0.60)).astype(int)
df['exit_velo_elite'] = (df['exit_velo_max'] >= df['exit_velo_max'].quantile(0.75)).astype(int)
df['speed_elite'] = (df['sixty_time'] <= df['sixty_time'].quantile(0.25)).astype(int)

# Position-specific percentiles (more granular)
for col in ['exit_velo_max', 'of_velo', 'sixty_time', 'height', 'weight']:
    if col == 'sixty_time':  # Lower is better
        df[f'{col}_percentile'] = (1 - df[col].rank(pct=True)) * 100
    else:  # Higher is better
        df[f'{col}_percentile'] = df[col].rank(pct=True) * 100

# D1 thresholds (refined based on actual data distribution)
p75_exit = df['exit_velo_max'].quantile(0.75)
p75_of = df['of_velo'].quantile(0.75)
p25_sixty = df['sixty_time'].quantile(0.25)

df['d1_exit_velo_threshold'] = (df['exit_velo_max'] >= max(88, p75_exit * 0.95)).astype(int)
df['d1_arm_threshold'] = (df['of_velo'] >= max(80, p75_of * 0.9)).astype(int)
df['d1_speed_threshold'] = (df['sixty_time'] <= min(7.2, p25_sixty * 1.1)).astype(int)
df['d1_size_threshold'] = ((df['height'] >= 70) & (df['weight'] >= 165)).astype(int)

# Multi-tool analysis
df['tool_count'] = (df['exit_velo_elite'] + df['of_arm_strength'] + 
                   df['speed_elite'] + df['d1_size_threshold'])
df['is_multi_tool'] = (df['tool_count'] >= 2).astype(int)

# Advanced composites
df['athletic_index'] = (
    df['exit_velo_max_percentile'] * 0.3 +
    df['of_velo_percentile'] * 0.25 + 
    df['sixty_time_percentile'] * 0.25 +
    df['height_percentile'] * 0.1 +
    df['weight_percentile'] * 0.1
)

df['tools_athlete'] = df['tool_count'] * df['athletic_index']

df['d1_composite_score'] = (
    df['d1_exit_velo_threshold'] * 0.35 +
    df['d1_speed_threshold'] * 0.3 +
    df['d1_arm_threshold'] * 0.25 +
    df['d1_size_threshold'] * 0.1
)

# Key interactions (based on feature importance from previous model)
df['power_speed_sq'] = df['power_speed'] ** 2
df['of_power_combo'] = df['exit_velo_max'] * df['of_velo']
df['arm_speed_combo'] = df['of_arm_plus'] * df['speed_elite']

# Regional advantages (if they exist)
if 'player_region_West' in df.columns:
    df['west_speed_advantage'] = df['player_region_West'] * df['speed_elite']
if 'player_region_South' in df.columns:
    df['south_power_tradition'] = df['player_region_South'] * df['exit_velo_elite']

print(f"✓ Created {len(df.columns) - 10} enhanced features")

# ============================================================================
# PREPARE DATA WITH CLASS BALANCING OPTIONS
# ============================================================================
print("\n📊 Preparing data with class balancing:")

drop_cols = ['d1_or_not', 'primary_position']
X = df.drop(columns=drop_cols)
y = df['d1_or_not']

# Clean data
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Outlier clipping
for col in X.select_dtypes(include=[np.number]).columns:
    Q1, Q3 = X[col].quantile(0.01), X[col].quantile(0.99)
    IQR = Q3 - Q1
    X[col] = X[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

print(f'Final feature count: {X.shape[1]}')
print(f'Total NA values after cleaning: {X.isna().sum().sum()}')

# Data splits
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42
)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
print(f"Train D1 rate: {y_train.mean():.2%}")

# ============================================================================
# MULTIPLE MODELING APPROACHES
# ============================================================================
print("\n🚀 Testing Multiple Modeling Approaches:")

beta = 0.7
models_results = {}

# Approach 1: XGBoost with class balancing
print("\n1. XGBoost with scale_pos_weight optimization:")

def xgb_objective(trial):
    # Calculate optimal scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'learning_rate': trial.suggest_float('eta', 1e-3, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', scale_pos_weight * 0.5, scale_pos_weight * 2.0),
    }
    
    model = xgb.XGBClassifier(**params, n_estimators=500, random_state=42, missing=np.nan)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    
    return float(fbeta_score(y_val, preds, beta=beta))

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(xgb_objective, n_trials=50, show_progress_bar=True, n_jobs=-1)

xgb_model = xgb.XGBClassifier(**study_xgb.best_params, n_estimators=500, 
                              use_label_encoder=False, eval_metric='logloss',
                              random_state=42, missing=np.nan)
xgb_model.fit(X_train_val, y_train_val, verbose=False)

# Approach 2: LightGBM
print("\n2. LightGBM with class balancing:")

def lgb_objective(trial):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', scale_pos_weight * 0.5, scale_pos_weight * 2.0),
        'verbose': -1,
        'random_state': 42
    }
    
    model = lgb.LGBMClassifier(**params, n_estimators=500)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    preds = model.predict(X_val)
    
    return float(fbeta_score(y_val, preds, beta=beta))

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(lgb_objective, n_trials=50, show_progress_bar=True, n_jobs=-1)

lgb_model = lgb.LGBMClassifier(**study_lgb.best_params, n_estimators=500, verbose=-1)
lgb_model.fit(X_train_val, y_train_val)

# Approach 3: CatBoost with class balancing
print("\n3. CatBoost with class balancing:")

def cat_objective(trial):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    params = {
        'objective': 'Logloss',
        'eval_metric': 'AUC',
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 1),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', scale_pos_weight * 0.5, scale_pos_weight * 2.0),
        'verbose': False,
        'random_state': 42
    }
    
    model = cb.CatBoostClassifier(**params, iterations=500)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    
    return float(fbeta_score(y_val, preds, beta=beta))

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(cat_objective, n_trials=50, show_progress_bar=True, n_jobs=-1)

cat_model = cb.CatBoostClassifier(**study_cat.best_params, iterations=500, verbose=False)
cat_model.fit(X_train_val, y_train_val)

# Approach 4: SVM with class balancing
print("\n4. SVM with class balancing:")

# Scale features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_train_val_scaled = scaler.fit_transform(X_train_val)
X_test_scaled = scaler.transform(X_test)

from sklearn.svm import SVC

def svm_objective(trial):
    params = {
        'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'class_weight': 'balanced',
        'probability': True,
        'random_state': 42
    }
    if params['kernel'] == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 4)
    
    model = SVC(**params)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_val_scaled)
    
    return float(fbeta_score(y_val, preds, beta=beta))

study_svm = optuna.create_study(direction='maximize')
study_svm.optimize(svm_objective, n_trials=30, show_progress_bar=True, n_jobs=4)

svm_model = SVC(**study_svm.best_params, probability=True)
svm_model.fit(X_train_val_scaled, y_train_val)

# ============================================================================
# EVALUATE ALL APPROACHES
# ============================================================================
print("\n" + "=" * 60)
print("MODEL COMPARISON ON TEST SET")
print("=" * 60)

models = {
    'XGBoost_Balanced': xgb_model,
    'LightGBM_Balanced': lgb_model, 
    'CatBoost_Balanced': cat_model,
    'SVM_Balanced': svm_model
}

best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    if name == 'SVM_Balanced':
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fbeta = fbeta_score(y_test, y_pred, beta=beta)
    
    models_results[name] = {
        'accuracy': accuracy,
        'f1': f1,
        'fbeta': fbeta,
        'predictions': y_pred
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  F{beta}-Score: {fbeta:.4f}")
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fp_fn_ratio = fp / (fn + 0.001)
    print(f"  FP:FN Ratio: {fp_fn_ratio:.2f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name

# ============================================================================
# ENSEMBLE MODEL WITH SPECIFIED WEIGHTS
# ============================================================================
print("\n" + "=" * 60)
print("ENSEMBLE MODEL (XGB=0.3, LGB=0.3, CAT=0.2, SVM=0.2)")
print("=" * 60)

from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

# Create SVM pipeline for ensemble (handles scaling)
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(**study_svm.best_params, probability=True))
])

# Create ensemble with specified weights
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model),
        ('svm', svm_pipeline)
    ],
    voting='soft',
    weights=[0.3, 0.3, 0.2, 0.2]
)

# Train ensemble on full train+val data
ensemble.fit(X_train_val, y_train_val)

# Evaluate ensemble
y_pred_ensemble = ensemble.predict(X_test)

ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
ensemble_f1 = f1_score(y_test, y_pred_ensemble)
ensemble_fbeta = fbeta_score(y_test, y_pred_ensemble, beta=beta)

print(f"\nEnsemble Performance:")
print(f"  Accuracy: {ensemble_accuracy:.4f}")
print(f"  F1-Score: {ensemble_f1:.4f}")
print(f"  F{beta}-Score: {ensemble_fbeta:.4f}")

# Ensemble confusion matrix and classification report
print(f"\nEnsemble Classification Report:")
print(classification_report(y_test, y_pred_ensemble))

print("Ensemble Confusion Matrix:")
ensemble_cm = confusion_matrix(y_test, y_pred_ensemble)
print(ensemble_cm)

tn, fp, fn, tp = ensemble_cm.ravel()
fp_fn_ratio = fp / (fn + 0.001)
print(f"Ensemble FP:FN Ratio: {fp_fn_ratio:.2f}")

# ============================================================================
# LIGHTGBM MODEL DETAILED RESULTS
# ============================================================================
print("\n" + "=" * 60)
print("LIGHTGBM_BALANCED MODEL DETAILED RESULTS")
print("=" * 60)

lgb_pred = models_results['LightGBM_Balanced']['predictions']
print(f"LightGBM Classification Report:")
print(classification_report(y_test, lgb_pred))

print("LightGBM Confusion Matrix:")
lgb_cm = confusion_matrix(y_test, lgb_pred)
print(lgb_cm)

# ============================================================================
# BEST MODEL ANALYSIS
# ============================================================================
print(f"\n🏆 BEST INDIVIDUAL MODEL: {best_model_name} with {best_accuracy:.4f} accuracy")
print(f"🎯 ENSEMBLE MODEL: {ensemble_accuracy:.4f} accuracy")

# No Information Rate comparison
most_freq_class = y_test.value_counts(normalize=True).max()
print(f"No Information Rate: {most_freq_class:.4f}")
print(f"Improvement over no-info: {best_accuracy - most_freq_class:+.4f}")

if best_accuracy >= most_freq_class:
    print("✅ SUCCESS: Beat the no information rate!")
else:
    print(f"⚠️ Still {most_freq_class - best_accuracy:.4f} below no-info rate")

# Feature importance from best model
best_model = models[best_model_name]
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 15 Features ({best_model_name}):")
    print(importance_df.head(15))

print(f"\n📈 Performance Summary:")
print(f"Previous simplified model: 0.7077 accuracy")
print(f"Enhanced model: {best_accuracy:.4f} accuracy")
print(f"Improvement: {best_accuracy - 0.7077:+.4f}")

print(f"\n🎯 Next Steps if still below 75%:")
print("- Investigate data quality issues further")
print("- Consider ensemble of best performing models")
print("- Add external data sources if available")
print("- Focus on feature selection/elimination")