# P4 Outfielder Prediction Model v4_20250807_105039

## Model Overview
- **Type**: Hierarchical Ensemble (XGBoost + LightGBM + MLP + SVM)
- **Target**: Predict P4 college level probability for outfielders
- **Performance**: 73.3% accuracy, 69.3% P4 recall

## Files Description
- `xgb_model.pkl`: XGBoost model
- `lgb_model.pkl`: LightGBM model  
- `mlp_model.pkl`: Multi-layer Perceptron model
- `svm_model.pkl`: Support Vector Machine model
- `elite_model.pkl`: Elite player detection model
- `scaler.pkl`: Feature scaler for MLP and SVM
- `model_config.json`: Model parameters and performance metrics
- `feature_metadata.json`: Feature lists and preprocessing information
- `prediction_pipeline.py`: Complete prediction function for production use

## Required Input Features
```python
player_data = {
    'height': float,          # inches
    'weight': float,          # pounds
    'sixty_time': float,      # seconds
    'exit_velo_max': float,   # mph
    'of_velo': float,         # mph (outfield velocity)
    'player_region': str,     # Geographic region
    'throwing_hand': str,     # 'Left' or 'Right'
    'hitting_handedness': str # 'Left', 'Right', or 'Switch'
}
```

## Usage
```python
from prediction_pipeline import predict_outfielder_p4_probability

result = predict_outfielder_p4_probability(
    player_data=player_data,
    models_dir='/Users/ryankolodziejczyk/Documents/AI Baseball Recruitment/code/backend/ml/models/models_of/models_p4_or_not_of/v4_20250807_105039'
)

print(f"P4 Probability: {result['p4_probability']:.1%}")
print(f"P4 Prediction: {result['p4_prediction']}")
```

## Model Performance
- **Test Accuracy**: 73.3%
- **P4 Recall**: 69.3%
- **P4 Precision**: 57.1%
- **F1 Score**: 0.627

## Hierarchical Strategy
- **Elite Players**: 0.380 threshold (140 players, 30.1%)
- **Non-Elite Players**: 0.400 threshold (325 players, 69.9%)

## Production Notes
- Model optimized for P4 recruitment pipeline
- Uses performance-weighted ensemble for robust predictions
- Includes confidence levels and model component breakdown
- Requires D1 model integration for optimal performance
