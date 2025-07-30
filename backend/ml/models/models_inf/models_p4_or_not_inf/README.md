# Power 4 Prediction Models - Production Deployment

## Model Files:

### Core Models:
- `elite_detection_xgboost_model.pkl` - Detects elite D1 players (97% accuracy)
- `p4_ensemble_xgboost_model.pkl` - XGBoost P4 classifier (30% weight)
- `p4_ensemble_catboost_model.pkl` - CatBoost P4 classifier (35% weight)  
- `p4_ensemble_lightgbm_model.pkl` - LightGBM P4 classifier (17.5% weight)
- `p4_ensemble_svm_model.pkl` - SVM P4 classifier (17.5% weight)

### Supporting Files:
- `feature_scaler_for_svm.pkl` - StandardScaler for SVM preprocessing
- `model_config_and_metadata.pkl` - Thresholds, feature lists, performance metrics
- `production_prediction_pipeline.py` - Reference implementation code

## Performance Metrics:
- **Accuracy**: 78.7%
- **P4 Precision**: 86% (when model says P4, it's right 86% of the time)
- **P4 Recall**: 42% (captures 2 out of 5 actual P4 players)
- **Balanced Accuracy**: 69.2%

## Key Thresholds:
- **Elite Detection**: Players above composite score threshold are "elite D1"
- **P4 Prediction**: Hierarchical probability threshold for final classification
- **Confidence Levels**: High confidence when probability > 0.8 or < 0.2

## Feature Engineering Required:
Input data must include engineered features:
- Percentile rankings for all metrics
- Advanced ratios (power_per_pound, exit_to_sixty_ratio, etc.)
- Elite binary indicators
- Athletic composite scores

## Prediction Flow:
1. Engineer features from raw player data
2. Elite detection model → elite probability
3. Ensemble P4 models → P4 probability  
4. Hierarchical combination → final prediction
5. Apply threshold → binary P4/Non-P4 classification

## Usage Notes:
- Load all models once at startup for performance
- Feature engineering must match training pipeline exactly
- SVM requires feature scaling (use provided scaler)
- Models expect same 41 engineered features as training
