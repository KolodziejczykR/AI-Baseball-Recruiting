# Outfielder D1 Prediction Model v2_20250804_192917

## Model Overview
- **Type**: Hierarchical Ensemble (Elite Detection + XGBoost + DNN + LightGBM + SVM)
- **Target**: Predict D1 college level probability for outfielders
- **Performance**: 75.0% accuracy, 55.4% D1 recall

## Files Description
- `elite_model.pkl`: Elite outfielder detection model
- `*_full_model.pkl`: Models trained on full dataset for hierarchical prediction
- `scaler_*.pkl`: Feature scalers for neural network and SVM models
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
from prediction_pipeline import predict_outfielder_d1_probability

result = predict_outfielder_d1_probability(
    player_data=player_data,
    models_dir='/Users/ryankolodziejczyk/Documents/AI Baseball Recruitment/code/backend/ml/models/models_of/models_d1_or_not_of/v2_20250804_192917'
)

print(f"D1 Probability: {result['d1_probability']:.1%}")
print(f"Prediction: {result['d1_prediction']}")
```

## Model Performance
- **Accuracy**: 75.0%
- **D1 Recall**: 55.4%
- **D1 Precision**: 50.8%
- **FP:FN Ratio**: 1.20 (prefers over-recruiting to missing talent)

## Production Notes
- Model is optimized for recruiting pipeline (first-stage filter)
- Threshold optimized for 74%+ accuracy while maintaining 50%+ D1 recall
- Uses performance-weighted ensemble for robust predictions
- Includes confidence levels and model component breakdown
