# D1 Classification Models - Production Deployment

## Overview

This directory contains production-ready ensemble models for classifying baseball players as D1-eligible or Non-D1. The models achieve **79.16% test accuracy** using a weighted ensemble approach that combines predictions from multiple optimized algorithms.

## Model Architecture

### **Weighted Ensemble Composition**:
- **XGBoost**: 35% weight - Strong gradient boosting performance
- **CatBoost**: 35% weight - Excellent categorical feature handling  
- **LightGBM**: 25% weight - Fast and efficient gradient boosting
- **SVM**: 5% weight - Support vector classification with RBF kernel

### **Voting Strategy**: 
- **Soft Voting** - Combines probability predictions from all models
- **Weighted Average** - Each model contributes based on validation performance
- **Final Prediction** - Ensemble probability ≥ 0.5 → D1 classification

## Model Files

### **Core Production Files**:
- `ensemble_model.pkl` - **Main classifier** - VotingClassifier with all 4 models
- `ensemble_scaler.pkl` - **Feature scaler** - StandardScaler for SVM preprocessing
- `ensemble_metadata.pkl` - **Configuration** - Performance metrics, feature names, weights

### **Individual Model Components**:
- `xgboost_model.pkl` + `xgboost_params.pkl` - XGBoost classifier & hyperparameters
- `catboost_model.pkl` + `catboost_params.pkl` - CatBoost classifier & hyperparameters  
- `lightgbm_model.pkl` + `lightgbm_params.pkl` - LightGBM classifier & hyperparameters
- `svm_model.pkl` + `svm_params.pkl` - SVM classifier & hyperparameters

## Performance Metrics

### **Test Set Performance**:
- **Accuracy**: 79.16%
- **F1-Score**: 0.6584
- **F-beta Score**: 0.7177 (β=0.7, emphasizes recall for recruiting)
- **Precision**: ~68%
- **Recall**: ~65%

### **Cross-Validation Results**:
- **5-fold CV F1**: 0.7033 (±0.0387)
- **5-fold CV Accuracy**: 0.7964 (±0.0183)
- **Consistent performance** across validation folds

### **Individual Model Contributions**:
| Model | Test Accuracy | Test F1 | Test F-beta | Weight |
|-------|---------------|---------|-------------|--------|
| XGBoost | 78.41% | 0.6427 | 0.7086 | 35% |
| CatBoost | 78.50% | 0.6463 | 0.7105 | 35% |
| LightGBM | 78.67% | 0.6503 | 0.7134 | 25% |
| SVM | 77.39% | 0.6188 | 0.6923 | 5% |
| **Ensemble** | **79.16%** | **0.6584** | **0.7177** | **100%** |

## Feature Requirements

### **Input Features** (41 total):
The models expect these engineered features in exact order:

#### **Raw Metrics**:
- `height`, `weight`, `sixty_time`, `exit_velo_max`, `inf_velo`, `power_speed`

#### **Position Indicators**:
- `primary_position_SS`, `primary_position_2B`, `primary_position_3B`

#### **Handedness & Region**:
- `hitting_handedness_R`, `throwing_handedness_R`
- `player_region_West`, `player_region_South`, `player_region_Northeast`, `player_region_Midwest`

#### **Engineered Athletic Features**:
- `exit_and_inf_velo_ss`, `west_coast_ss`, `all_around_ss`
- `inf_velo_x_velo_by_inf`, `inf_velo_sq`, `velo_by_inf_sq`
- `inf_velo_x_velo_by_inf_sq`, `inf_velo_x_velo_by_inf_cubed`
- `exit_inf_velo_inv`, `inf_velo_sixty_ratio`, `inf_velo_sixty_ratio_sq`

#### **Additional Derived Features**:
- Various composite athletic indices, percentiles, and binary thresholds
- Complete feature list available in `ensemble_metadata.pkl`

### **Data Preprocessing Requirements**:
1. **Missing Values**: Fill with 0 (models trained with this approach)
2. **Infinite Values**: Replace inf/-inf with NaN, then fill with 0
3. **Outlier Clipping**: Apply 1st-99th percentile clipping with 1.5×IQR bounds
4. **Feature Scaling**: SVM component requires StandardScaler (provided in `ensemble_scaler.pkl`)

## Usage Instructions

### **Loading Models**:
```python
import joblib
import pandas as pd
import numpy as np

# Load ensemble components
ensemble_model = joblib.load('ensemble_model.pkl')
ensemble_scaler = joblib.load('ensemble_scaler.pkl')  # Used internally by SVM
ensemble_metadata = joblib.load('ensemble_metadata.pkl')

# Get model configuration
feature_columns = ensemble_metadata['feature_columns']
test_accuracy = ensemble_metadata['test_accuracy']
model_weights = ensemble_metadata['weights']  # [0.35, 0.25, 0.35, 0.05]
```

### **Making Predictions**:
```python
def predict_d1_eligibility(player_features):
    \"\"\"
    Predict D1 eligibility for a player
    
    Args:
        player_features: DataFrame or array with 41 engineered features
        
    Returns:
        dict: prediction, probability, confidence level
    \"\"\"
    # Ensure correct feature order
    if isinstance(player_features, pd.DataFrame):
        player_features = player_features[ensemble_metadata['feature_columns']]
    
    # Get probability prediction
    d1_probability = ensemble_model.predict_proba([player_features])[0][1]
    d1_prediction = d1_probability >= 0.5
    
    # Determine confidence level
    if d1_probability >= 0.8 or d1_probability <= 0.2:
        confidence = 'High'
    elif d1_probability >= 0.65 or d1_probability <= 0.35:
        confidence = 'Medium'
    else:
        confidence = 'Low'
    
    return {
        'prediction': 'D1' if d1_prediction else 'Non-D1',
        'probability': d1_probability,
        'confidence': confidence,
        'binary_prediction': d1_prediction
    }

# Example usage
result = predict_d1_eligibility(player_features)
print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']})")
print(f"D1 Probability: {result['probability']:.3f}")
```

### **Batch Predictions**:
```python
def predict_d1_batch(players_df):
    \"\"\"Predict D1 eligibility for multiple players\"\"\"
    # Ensure correct feature order
    features = players_df[ensemble_metadata['feature_columns']]
    
    # Get predictions
    probabilities = ensemble_model.predict_proba(features)[:, 1]
    predictions = probabilities >= 0.5
    
    # Add results to dataframe
    results_df = players_df.copy()
    results_df['d1_probability'] = probabilities
    results_df['d1_prediction'] = predictions
    results_df['d1_prediction_label'] = results_df['d1_prediction'].map({True: 'D1', False: 'Non-D1'})
    
    return results_df
```

## Model Training Details

### **Hyperparameter Optimization**:
- **Framework**: Optuna with TPESampler
- **Trials**: 100 trials per model
- **Optimization Metric**: F-beta score (β=0.7) - emphasizes recall for recruiting
- **Cross-Validation**: Used for hyperparameter selection
- **Final Training**: Full train+validation set for production models

### **Key Hyperparameters**:
- **XGBoost**: Learning rate ~0.05, max_depth ~6, balanced class weights
- **CatBoost**: Learning rate ~0.08, depth ~8, bootstrap sampling
- **LightGBM**: Learning rate ~0.06, num_leaves ~150, feature sampling
- **SVM**: RBF kernel, C ~10, balanced class weights

### **Data Splits**:
- **Training**: 64% (hyperparameter optimization)
- **Validation**: 16% (hyperparameter selection)  
- **Test**: 20% (final evaluation, completely held out)
- **No data leakage** between optimization and final evaluation

## Business Impact

### **Recruiting Advantages**:
- **79.16% accuracy** provides reliable D1 eligibility screening
- **F-beta score optimization** balances precision and recall for recruiting needs
- **Confidence levels** help prioritize recruiting efforts
- **Ensemble robustness** reduces overfitting and improves generalization

### **Operational Benefits**:
- **Production-ready**: Optimized for speed and reliability
- **Scalable**: Handles batch predictions efficiently
- **Interpretable**: Feature importance and probability scores provide insight
- **Consistent**: Standardized preprocessing and prediction pipeline

## Technical Notes

- **Random State**: All models use `random_state=42` for reproducibility
- **Missing Value Strategy**: Fill with 0 (consistent with training)
- **Scaling**: Only SVM requires scaling (handled internally by ensemble)
- **Memory Efficient**: Models optimized for production deployment
- **Thread Safe**: All models support concurrent predictions

## Troubleshooting

### **Common Issues**:
1. **Feature Mismatch**: Ensure exactly 41 features in correct order
2. **Missing Values**: Use fillna(0) for consistency with training
3. **Scale Issues**: Don't manually scale features - ensemble handles SVM scaling
4. **Version Compatibility**: Models trained with scikit-learn 1.0+, XGBoost 1.5+

### **Performance Monitoring**:
- Monitor prediction probabilities for drift
- Track confidence level distributions
- Compare batch accuracy against 79.16% baseline
- Alert if accuracy drops below 75% on validation data

---

*These D1 classification models provide the foundation for the complete baseball recruiting pipeline, feeding into hierarchical P4 prediction for comprehensive recruiting intelligence.*