# ML Pipeline and API Structure

## Overview

This directory contains the machine learning pipelines and API endpoints for baseball player recruitment predictions. The system predicts college level classifications (Non D1, Non P4 D1, Power 4 D1) for three different positions: **Infielders**, **Outfielders**, and **Catchers**.

## Directory Structure

```
backend/ml/
├── pipeline/                    # ML prediction pipelines
│   ├── __init__.py
│   ├── infielder_pipeline.py   # Infielder prediction logic
│   ├── outfielder_pipeline.py  # Outfielder prediction logic
│   └── catcher_pipeline.py     # Catcher prediction logic
├── router/                     # FastAPI route handlers
│   ├── __init__.py
│   ├── infielder_router.py     # Infielder API endpoints
│   ├── outfielder_router.py    # Outfielder API endpoints
│   └── catcher_router.py       # Catcher API endpoints
├── models/                     # Trained ML models and artifacts
│   ├── lgb_infield_d1_or_not.pkl
│   ├── cb_infield_d1_p4_or_not.pkl
│   ├── xgb_outfield_d1_or_not.pkl
│   ├── xgb_outfield_d1_p4_or_not.pkl
│   ├── xgb_catcher_d1_or_not.pkl
│   ├── xgb_catcher_d1_p4_or_not.pkl
│   └── [scalers and encoders]
├── train/                      # Model training scripts
│   ├── train_infielder_d1_or_not.py
│   ├── train_infielder_d1_p4_or_not.py
│   ├── train_outfielder_d1_or_not.py
│   ├── train_outfielder_d1_p4_or_not.py
│   ├── train_catcher_d1_or_not.py
│   └── train_catcher_d1_p4_or_not.py
├── ml_router.py               # Main ML router aggregator
├── requirements_ml.txt        # ML-specific dependencies
└── README.md                 # This file
```

## Architecture

### Two-Stage Prediction System

All three position pipelines use a two-stage approach:

1. **Stage 1**: Predict D1 vs Non-D1
2. **Stage 2**: If D1 is predicted, predict Power 4 D1 vs Non-Power 4 D1

### Final Output Categories

- **Non D1**: Player is predicted to play at a non-Division 1 school
- **Non P4 D1**: Player is predicted to play at a Division 1 school that is not a Power 4 conference  
- **Power 4 D1**: Player is predicted to play at a Power 4 Division 1 school

## Position-Specific Models

### Infielders
- **D1 Model**: LightGBM (`lgb_infield_d1_or_not.pkl`)
- **Power 4 Model**: CatBoost (`cb_infield_d1_p4_or_not.pkl`)
- **Key Features**: `inf_velo` (infield velocity), standard hitting metrics
- **Positions**: SS, 2B, 3B, 1B

### Outfielders  
- **D1 Model**: XGBoost (`xgb_outfield_d1_or_not.pkl`)
- **Power 4 Model**: XGBoost (`xgb_outfield_d1_p4_or_not.pkl`)
- **Key Features**: `of_velo` (outfield velocity), speed metrics, power metrics
- **Positions**: CF, LF, RF, OF

### Catchers
- **D1 Model**: XGBoost (`xgb_catcher_d1_or_not.pkl`)
- **Power 4 Model**: XGBoost (`xgb_catcher_d1_p4_or_not.pkl`)
- **Key Features**: `c_velo` (catcher velocity), `pop_time`, standard hitting metrics
- **Positions**: C

## API Endpoints

### Base URLs
- **Infielders**: `/infielder/*`
- **Outfielders**: `/outfielder/*`  
- **Catchers**: `/catcher/*`

### Common Endpoints (for each position)

#### POST `/{position}/predict`
Predict college level for a player.

**Request Body Example:**
```json
{
  "height": 72.0,
  "weight": 180.0,
  "exit_velo_max": 88.0,
  "inf_velo": 78.0,        // infielder-specific
  "of_velo": 82.0,         // outfielder-specific  
  "c_velo": 78.0,          // catcher-specific
  "pop_time": 1.8,         // catcher-specific
  "throwing_hand": "R",
  "hitting_handedness": "R",
  "player_region": "West",
  "primary_position": "SS"  // or "CF", "C", etc.
}
```

**Response:**
```json
{
  "prediction": "Power 4 D1",
  "probabilities": {
    "Non D1": 0.15,
    "Non P4 D1": 0.20,
    "Power 4 D1": 0.65
  },
  "confidence": 0.65,
  "stage": "D1 + Power 4"
}
```

#### GET `/{position}/features`
Get information about required features.

#### GET `/{position}/health`
Health check endpoint.

#### GET `/{position}/example`
Get an example of valid input data.

## Usage Examples

### Python Pipeline Usage
```python
from backend.ml.pipeline.infielder_pipeline import InfielderPredictionPipeline
from backend.ml.pipeline.outfielder_pipeline import OutfielderPredictionPipeline
from backend.ml.pipeline.catcher_pipeline import CatcherPredictionPipeline

# Initialize pipelines
infielder_pipeline = InfielderPredictionPipeline(models_dir='models/')
outfielder_pipeline = OutfielderPredictionPipeline(models_dir='models/')
catcher_pipeline = CatcherPredictionPipeline(models_dir='models/')

# Example player data
player_data = {
    "height": 72.0,
    "weight": 180.0,
    "exit_velo_max": 88.0,
    "inf_velo": 78.0,
    "throwing_hand": "R",
    "hitting_handedness": "R",
    "player_region": "West",
    "primary_position": "SS"
}

# Get prediction
result = infielder_pipeline.predict(player_data)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### API Usage
```bash
# Test infielder prediction
curl -X POST "http://localhost:8000/infielder/predict" \
  -H "Content-Type: application/json" \
  -d '{"height": 72.0, "exit_velo_max": 88.0, "inf_velo": 78.0, "primary_position": "SS"}'

# Test outfielder prediction  
curl -X POST "http://localhost:8000/outfielder/predict" \
  -H "Content-Type: application/json" \
  -d '{"height": 73.0, "exit_velo_max": 92.0, "of_velo": 82.0, "primary_position": "CF"}'

# Test catcher prediction
curl -X POST "http://localhost:8000/catcher/predict" \
  -H "Content-Type: application/json" \
  -d '{"height": 72.0, "exit_velo_max": 88.0, "c_velo": 78.0, "pop_time": 1.8, "primary_position": "C"}'
```

## Testing

### Run All Tests
```bash
cd /path/to/project
python3 -m pytest tests/ -v
```

### Test Specific Components
```bash
# Test infielder pipeline
python3 -m pytest tests/test_infield_model_pipeline.py -v

# Test outfielder pipeline  
python3 -m pytest tests/test_outfielder_model_pipeline.py -v

# Test catcher pipeline
python3 -m pytest tests/test_catcher_model_pipeline.py -v

# Test API endpoints
python3 -m pytest tests/test_api.py -v
```

## Model Training

### Training Scripts Location
All training scripts are in the `train/` directory and include:
- Cross-validation reporting
- Model comparison (XGBoost, LightGBM, CatBoost)
- Feature importance analysis
- Performance metrics

### Running Training
```bash
cd backend/ml/train

# Train infielder models
python3 train_infielder_d1_or_not.py
python3 train_infielder_d1_p4_or_not.py

# Train outfielder models
python3 train_outfielder_d1_or_not.py  
python3 train_outfielder_d1_p4_or_not.py

# Train catcher models
python3 train_catcher_d1_or_not.py
python3 train_catcher_d1_p4_or_not.py
```

## Key Features

### Numerical Features (Common)
- `height`, `weight`: Physical measurements
- `hand_speed_max`, `bat_speed_max`: Batting metrics
- `rot_acc_max`: Rotational acceleration
- `sixty_time`, `thirty_time`, `ten_yard_time`: Speed metrics
- `run_speed_max`: Maximum running speed
- `exit_velo_max`, `exit_velo_avg`: Hitting power metrics
- `distance_max`: Maximum hit distance
- `sweet_spot_p`: Sweet spot percentage

### Position-Specific Features
- **Infielders**: `inf_velo` (infield velocity)
- **Outfielders**: `of_velo` (outfield velocity)  
- **Catchers**: `c_velo` (catcher velocity), `pop_time` (pop time)

### Categorical Features
- `throwing_hand`: L/R
- `hitting_handedness`: L/R/S
- `player_region`: Geographic region
- `primary_position`: Position code

## Error Handling

All pipelines include comprehensive error handling:
- **Model Loading**: Graceful fallback if models are missing
- **Input Validation**: Pydantic validation for API inputs
- **Missing Values**: Intelligent default values for missing features
- **Unseen Categories**: Mapping to most common category
- **Probability Normalization**: Ensures probabilities sum to 1.0

## Integration

### Main API Integration
The routers are integrated into the main FastAPI application in `backend/api/main.py`:
```python
from backend.ml.router.infielder_router import router as infielder_router
from backend.ml.router.outfielder_router import router as outfielder_router  
from backend.ml.router.catcher_router import router as catcher_router

app.include_router(infielder_router, prefix="/infielder")
app.include_router(outfielder_router, prefix="/outfielder")
app.include_router(catcher_router, prefix="/catcher")
```

### ML Router Aggregation
The `ml_router.py` file aggregates all position routers for centralized ML endpoint management.

## Performance Notes

- **Infielders**: LightGBM for D1 classification, CatBoost for Power 4
- **Outfielders**: XGBoost for both stages
- **Catchers**: XGBoost for both stages with catcher-specific features
- All models include cross-validation and performance monitoring
- Probability normalization ensures consistent output format

## Future Enhancements

- **Model Retraining Pipeline**: Automated retraining with new data
- **A/B Testing**: Model comparison capabilities
- **Real-time Monitoring**: Performance tracking and alerting
- **Additional Positions**: Pitcher predictions
- **Confidence Intervals**: Uncertainty quantification
- **Feature Importance API**: Dynamic feature analysis endpoints 