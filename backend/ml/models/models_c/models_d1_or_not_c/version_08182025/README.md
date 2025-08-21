# Catcher D1 Classification Models - Production Deployment

## Overview

This directory contains production-ready catcher D1 classification models that predict whether a catcher prospect will play at the Division 1 level. The models use a **meta-learner ensemble approach** that combines LightGBM and Deep Neural Network predictions through an advanced meta-learning framework.

## Model Architecture

### **Meta-Learner Ensemble Composition**:
- **LightGBM**: Gradient boosting base model optimized for catcher-specific features
- **Deep Neural Network (MLP)**: Multi-layer perceptron for complex pattern recognition
- **Meta-Learner**: Random Forest or Logistic Regression that learns from base model predictions

### **Meta-Learning Strategy**: 
- **Stage 1**: Train base models (LightGBM + DNN) on training data
- **Stage 2**: Generate meta-features from base model predictions on validation data
- **Stage 3**: Train meta-learner on enhanced meta-features to make final predictions
- **Enhanced Meta-Features**: 13 engineered features including interactions, polynomials, and disagreement measures

## Model Files

### **Core Production Files**:
- `lightgbm_model.pkl` - **LightGBM base model** - Primary gradient boosting classifier
- `lightgbm_params.pkl` - **LightGBM hyperparameters** - Optimized parameters via Optuna
- `dnn_model.pkl` - **Deep Neural Network** - Multi-layer perceptron classifier
- `dnn_params.pkl` - **DNN hyperparameters** - Neural network architecture and parameters
- `meta_learner.pkl` - **Meta-learner model** - Final ensemble decision maker
- `meta_learner_params.pkl` - **Meta-learner parameters** - Meta-model configuration

### **Supporting Files**:
- `dnn_scaler.pkl` - **Feature scaler** - StandardScaler for DNN preprocessing (if used)
- `model_metadata.pkl` - **Model metadata** - Performance metrics, feature names, configuration
- `prediction_pipeline.py` - **Inference pipeline** - Production prediction interface
- `README.md` - **Documentation** - This comprehensive guide

## Performance Metrics

### **Test Set Performance**:
Performance metrics will be populated after model training completion.

### **Model Approach Benefits**:
- **Meta-Learning**: Learns optimal combination of base model predictions
- **Feature Engineering**: 13 enhanced meta-features capture model interactions
- **Catcher-Specific**: Optimized for catcher defensive and offensive metrics
- **Robust Ensemble**: Reduces overfitting through stacked generalization

### **Key Meta-Features**:
1. **Base Predictions**: LightGBM and DNN probability outputs
2. **Interactions**: LGB × DNN cross-model interactions
3. **Statistics**: Average, max, min confidence across models
4. **Disagreement**: Model uncertainty and disagreement measures
5. **Polynomials**: LGB^2, LGB^3, LGB^4 for non-linear relationships
6. **Exponential**: e^LGB for capturing extreme values

## Feature Requirements

### **Input Features** (Catcher-Specific):
The models expect these engineered features for catchers:

#### **Raw Metrics**:
- `height`, `weight`, `sixty_time`, `exit_velo_max`, `c_velo`, `pop_time`

#### **Handedness & Region**:
- `hitting_handedness_L`, `hitting_handedness_R`
- `throwing_hand_L`
- `player_region_Midwest`, `player_region_Northeast`, `player_region_South`, `player_region_West`

#### **Catcher-Specific Athletic Features**:
- `athletic_index_v2`: Composite athletic ability score
- `c_velo_percentile`: Catcher velocity percentile ranking
- `pop_time_percentile`: Pop time percentile ranking
- `defensive_consistency`: Pop time × catcher velocity × defensive ability
- `pop_efficiency`: Catcher velocity / (pop time)²

#### **Advanced Engineered Features**:
- `exit_velo_over_c_velo`: Hitting power relative to arm strength
- `pop_c_velo_ratio`: Pop time efficiency relative to arm strength
- `arm_athleticism_correlation`: Correlation between arm strength and overall athleticism
- `tool_synergy`: Multi-tool combination scoring
- `athletic_ceiling`: Maximum athletic potential estimate

### **Data Preprocessing Requirements**:
1. **Missing Values**: Fill with 0 (consistent with training approach)
2. **Infinite Values**: Replace inf/-inf with NaN, then fill with 0
3. **Outlier Handling**: Apply 1st-99th percentile clipping with 1.5×IQR bounds
4. **Feature Scaling**: DNN component may require StandardScaler (handled automatically)

## Usage Instructions

### **Loading Models**:
```python
import joblib
import pandas as pd
import numpy as np
from prediction_pipeline import predict_catcher_d1_probability

# Load model metadata
metadata = joblib.load('model_metadata.pkl')
print(f"Model accuracy: {metadata['test_accuracy']:.3f}")
print(f"Ensemble type: {metadata['ensemble_type']}")
print(f"Training date: {metadata['training_date']}")
```

### **Making Predictions**:
```python
# Example catcher player data
player_data = {
    'player_id': 'catcher_001',
    'height': 72,          # inches
    'weight': 190,         # pounds
    'sixty_time': 7.2,     # seconds
    'c_velo': 78,          # mph
    'pop_time': 2.0,       # seconds
    'exit_velo_max': 92,   # mph
    'throwing_hand': 'R',
    'hitting_handedness': 'R',
    'player_region': 'South'
}

# Get prediction
models_dir = '/path/to/models'
result = predict_catcher_d1_probability(player_data, models_dir)

print(f"D1 Prediction: {result['d1_prediction']}")
print(f"D1 Probability: {result['d1_probability']:.3f}")
print(f"Confidence: {result['confidence_level']}")
print(f"LightGBM: {result['components']['individual_models']['lightgbm']:.3f}")
print(f"DNN: {result['components']['individual_models']['dnn']:.3f}")
```

### **Batch Predictions**:
```python
from prediction_pipeline import predict_catcher_d1_batch

# DataFrame with multiple catchers
catchers_df = pd.DataFrame([
    {'height': 72, 'weight': 190, 'c_velo': 78, 'pop_time': 2.0, ...},
    {'height': 70, 'weight': 185, 'c_velo': 75, 'pop_time': 2.1, ...},
    # ... more catchers
])

# Get batch predictions
results_df = predict_catcher_d1_batch(catchers_df, models_dir)
print(results_df[['d1_prediction_label', 'd1_probability', 'confidence_level']])
```

## Model Training Details

### **Meta-Learning Architecture**:
- **Base Models**: Trained on 80% of data with 5-fold cross-validation
- **Meta-Features**: Generated from out-of-fold predictions to prevent overfitting
- **Meta-Learner**: Trained on meta-features using remaining 20% validation data
- **Final Models**: Retrained on full dataset for production deployment

### **Hyperparameter Optimization**:
- **Framework**: Optuna with TPESampler for efficient hyperparameter search
- **Trials**: 100+ trials per model component
- **Objective**: F-beta score (β=0.7) optimized for recruiting recall needs
- **Base Model Optimization**: Individual optimization for LightGBM and DNN
- **Meta-Learner Optimization**: Separate optimization for final ensemble layer

### **Key Hyperparameters**:
- **LightGBM**: Learning rate ~0.05-0.1, num_leaves ~50-200, feature sampling
- **DNN**: 2-3 hidden layers, 50-200 neurons per layer, dropout regularization
- **Meta-Learner**: Random Forest (n_estimators=100) or Logistic Regression

### **Data Splits**:
- **Training**: 64% (base model training with cross-validation)
- **Validation**: 16% (meta-feature generation)
- **Test**: 20% (final evaluation, completely held out)
- **No data leakage** between meta-learning stages

## Catcher-Specific Features

### **Defensive Metrics Priority**:
1. **Catcher Velocity (c_velo)**: Primary arm strength indicator (30% weight)
2. **Pop Time**: Throwing efficiency to second base (15% weight)
3. **Defensive Consistency**: Combined catching defensive ability
4. **Pop Efficiency**: Velocity-to-time ratio for throwing efficiency

### **Athletic Composite Scoring**:
- **Power**: Exit velocity max (25% weight)
- **Speed**: Sixty-yard dash time (20% weight)
- **Arm Strength**: Catcher velocity (30% weight)
- **Catching Skill**: Pop time efficiency (15% weight)
- **Size**: Height percentile (10% weight)

### **Position-Specific Insights**:
- Catchers require unique balance of defensive skills and offensive ability
- Pop time and catcher velocity are critical discriminators for D1 level
- Meta-learner captures complex interactions between defensive and offensive tools
- Regional adjustments account for different developmental environments

## Business Impact

### **Recruiting Advantages**:
- **Catcher-Specific Models**: Tailored to unique position requirements
- **Meta-Learning Sophistication**: Advanced ensemble captures subtle patterns
- **Confidence Scoring**: Helps prioritize recruiting efforts and resources
- **Comprehensive Evaluation**: Balances offensive and defensive capabilities

### **Technical Benefits**:
- **Production-Ready**: Optimized inference pipeline with error handling
- **Scalable**: Efficient batch processing for large prospect databases
- **Interpretable**: Individual model contributions and confidence measures
- **Robust**: Meta-learning reduces overfitting and improves generalization

## Technical Implementation

### **Meta-Learning Pipeline**:
1. **Feature Engineering**: Create catcher-specific composite features
2. **Base Model Training**: Train LightGBM and DNN with cross-validation
3. **Meta-Feature Generation**: Create 13 enhanced meta-features from base predictions
4. **Meta-Learner Training**: Train final ensemble on meta-features
5. **Production Deployment**: Streamlined inference with all components

### **Error Handling**:
- **Missing Features**: Automatic default value assignment
- **Invalid Data**: Robust preprocessing with outlier clipping
- **Model Loading**: Graceful fallback for missing model components
- **Prediction Bounds**: Probability outputs constrained to [0,1] range

## Troubleshooting

### **Common Issues**:
1. **Feature Mismatch**: Ensure all required catcher features are present
2. **Missing Values**: Use fillna(0) for consistency with training
3. **Scale Issues**: DNN scaling handled automatically if scaler available
4. **Performance Monitoring**: Track meta-learner confidence distributions

### **Performance Monitoring**:
- Monitor individual base model prediction distributions
- Track meta-learner confidence and disagreement patterns
- Compare ensemble accuracy against baseline single-model performance
- Alert if prediction confidence patterns shift significantly

---

*This catcher D1 classification model represents the most sophisticated ensemble approach in the baseball recruiting pipeline, specifically optimized for the unique requirements and evaluation criteria of catcher prospects.*