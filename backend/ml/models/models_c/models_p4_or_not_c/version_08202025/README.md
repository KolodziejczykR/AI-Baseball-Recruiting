# Catcher P4 Classification Model - Version 08/20/2025

## Model Overview

This directory contains a weighted ensemble model for predicting whether a catcher will play at the Power 4 (P4) level versus Non-Power 4 D1 level. This is the second stage of a two-stage hierarchical classification system.

### Performance Metrics
- **Accuracy**: 66.3% (vs 72% NIR)
- **P4 Detection Rate**: 36.8% (25/68 P4 players identified)
- **Non-P4 Accuracy**: 77.7% (136/175 correctly identified) 
- **F1 Score**: 0.38
- **Balanced Accuracy**: 57.2%
- **Optimal Threshold**: 0.35

## Model Architecture

### Ensemble Composition
The model uses a weighted ensemble of four algorithms:

1. **SVM** (38.8% weight) - Support Vector Machine with RBF kernel
2. **XGBoost** (22.1% weight) - Gradient boosting with tree-based learners
3. **LightGBM** (20.0% weight) - Gradient boosting with leaf-wise tree growth
4. **MLP** (19.2% weight) - Multi-layer perceptron neural network

*Weights are determined by squared CV performance scores to amplify differences between model performances.*

### Input Requirements
**Core Features:**
- `height` (inches)
- `weight` (pounds) 
- `sixty_time` (seconds)
- `exit_velo_max` (mph)
- `c_velo` (catcher velocity in mph)
- `pop_time` (seconds)
- `player_region` (categorical: Midwest/Northeast/South/West)
- `throwing_hand` (categorical: L/R)
- `hitting_handedness` (categorical: L/R/S)
- `d1_probability` (from D1 stage prediction, 0-1)

**Engineered Features:**
- 51 total features including composites, ratios, percentiles, and D1 interactions
- Top predictive features: BMI-swing correlations, power metrics, speed-size efficiency

## File Structure

```
version_08202025/
├── lightgbm_model.pkl      # LightGBM model
├── lightgbm_params.pkl     # LightGBM hyperparameters
├── xgboost_model.pkl       # XGBoost model  
├── xgboost_params.pkl      # XGBoost hyperparameters
├── mlp_model.pkl           # Neural network model
├── mlp_params.pkl          # MLP hyperparameters
├── svm_model.pkl           # Support Vector Machine model
├── svm_params.pkl          # SVM hyperparameters
├── feature_scaler.pkl      # StandardScaler for MLP/SVM
├── model_metadata.pkl      # Model configuration and metrics
├── prediction_pipeline.py  # Production prediction pipeline
└── README.md              # This file
```

## Usage

### Basic Prediction
```python
from prediction_pipeline import predict_catcher_p4_probability

player_data = {
    'height': 73.0,
    'weight': 195.0,
    'sixty_time': 6.9,
    'exit_velo_max': 99.0,
    'c_velo': 79.0,
    'pop_time': 1.9,
    'player_region': 'South',
    'throwing_hand': 'R',
    'hitting_handedness': 'R'
}

models_dir = '/path/to/version_08202025'
d1_probability = 0.75  # From D1 stage

result = predict_catcher_p4_probability(player_data, models_dir, d1_probability)

print(f"P4 Probability: {result.p4_probability:.1%}")
print(f"P4 Prediction: {result.p4_prediction}")
print(f"Confidence: {result.confidence}")
```

### Integration with Catcher Pipeline
```python
# This model is designed to be called after D1 prediction
from backend.ml.pipeline.catcher_pipeline import predict_catcher_classification

result = predict_catcher_classification(player_data)
# Uses this P4 model automatically for D1-predicted players
```

## Model Training Details

### Feature Engineering
The model uses extensive feature engineering including:

- **Power Metrics**: Exit velocity ratios, power per pound, BMI-swing correlations
- **Defensive Metrics**: Pop time efficiency, arm strength ratios, framing potential  
- **Athletic Composites**: Speed-size efficiency, overall athletic index
- **Elite Thresholds**: Binary indicators for elite tools (exit velo ≥98, c_velo ≥78, etc.)
- **D1 Interactions**: D1 probability crossed with core metrics for hierarchical learning

### Hyperparameter Optimization
- **Optuna** used for Bayesian optimization (30 trials for tree models, 20 for MLP/SVM)
- **Balanced scoring function** optimizing accuracy + P4 recall with precision penalty
- **Cross-validation** on 65% train, 15% validation, 20% test split

### Class Handling
- **Class imbalance**: 28% P4 rate in filtered data (D1 probability ≥ 0.3)
- **Moderate class weighting** applied to boost P4 detection while maintaining accuracy
- **No SMOTE** - removed to improve model stability

## Production Considerations

### Strengths
- Above-random performance for challenging classification task
- Reasonable P4 detection rate for user engagement
- Conservative approach reduces false positives
- Robust ensemble reduces overfitting risk

### Limitations  
- **Below 70% accuracy target** - suitable for initial production but needs improvement
- **Limited P4 recall** - misses ~63% of P4 players
- **Small P4 sample size** - only 68 P4 examples in test set
- **Feature complexity** - requires extensive preprocessing

### Recommendations
- **Use as screening tool** rather than definitive classifier
- **Combine with scout evaluation** for final decisions
- **Collect more P4 data** to improve future versions
- **Monitor performance** and retrain as data grows

## Model Versioning

**Version**: 08/20/2025  
**Training Data**: Filtered for D1 probability ≥ 0.3 (production realistic)  
**Random Seed**: 42  
**Class Distribution**: 175 Non-P4, 68 P4 (28% P4 rate)

## Contact
For questions about this model or integration support, refer to the main catcher pipeline documentation.