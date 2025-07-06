# Model Prediction Pipeline

## Overview

This README will have sections for every single type of model here.


# Infielder Prediction Pipeline

This pipeline uses a two-stage XGBoost model approach to predict the college level of infielder prospects.

## Overview

The pipeline works as follows:

1. **Stage 1**: Predict D1 vs Non-D1 using `xgb_infield_d1_or_not.pkl`
2. **Stage 2**: If D1 is predicted, then predict Power 4 D1 vs Non-Power 4 D1 using `xgb_infield_d1_p4_or_not.pkl`

## Final Output Categories

- **Non D1**: Player is predicted to play at a non-Division 1 school
- **Non P4 D1**: Player is predicted to play at a Division 1 school that is not a Power 4 conference
- **Power 4 D1**: Player is predicted to play at a Power 4 Division 1 school

## Files

### Core Pipeline
- `infielder_pipeline.py`: Main pipeline class that handles the two-stage prediction
- `infielder_router.py`: FastAPI router for the infielder prediction endpoints
- `test_pipeline.py`: Test script to verify the pipeline works correctly

### Models (in `models/` directory)
- `xgb_infield_d1_or_not.pkl`: XGBoost model for D1 vs Non-D1 classification
- `xgb_infield_d1_p4_or_not.pkl`: XGBoost model for Power 4 vs Non-Power 4 D1 classification
- `scalers.pkl`: StandardScaler for the D1 model
- `scalers_d1_p4.pkl`: StandardScaler for the Power 4 model
- `label_encoders.pkl`: Label encoders for categorical features (D1 model)
- `label_encoders_d1_p4.pkl`: Label encoders for categorical features (Power 4 model)

## Features Used

### Numerical Features
- `age`: Player age
- `height`: Player height in inches
- `weight`: Player weight in pounds
- `hand_speed_max`: Maximum hand speed (mph)
- `bat_speed_max`: Maximum bat speed (mph)
- `rot_acc_max`: Maximum rotational acceleration
- `sixty_time`: 60-yard dash time (seconds)
- `thirty_time`: 30-yard dash time (seconds)
- `ten_yard_time`: 10-yard dash time (seconds)
- `run_speed_max`: Maximum running speed (mph)
- `exit_velo_max`: Maximum exit velocity (mph)
- `exit_velo_avg`: Average exit velocity (mph)
- `distance_max`: Maximum hit distance (feet)
- `sweet_spot_p`: Sweet spot percentage (0-1)
- `inf_velo`: Infield velocity (mph)

### Categorical Features
- `player_state`: Player state
- `throwing_hand`: Throwing hand (L/R)
- `hitting_handedness`: Hitting handedness (L/R/S)
- `player_region`: Player region

## API Endpoints

### POST `/infielder/predict`
Predict the college level of an infielder.

**Request Body:**
```json
{
  "age": 17.5,
  "height": 72.0,
  "weight": 180.0,
  "hand_speed_max": 22.5,
  "bat_speed_max": 75.0,
  "rot_acc_max": 18.0,
  "sixty_time": 6.8,
  "thirty_time": 3.2,
  "ten_yard_time": 1.7,
  "run_speed_max": 22.0,
  "exit_velo_max": 88.0,
  "exit_velo_avg": 78.0,
  "distance_max": 320.0,
  "sweet_spot_p": 0.75,
  "inf_velo": 78.0,
  "player_state": "CA",
  "throwing_hand": "R",
  "hitting_handedness": "R",
  "player_region": "West"
}
```

**Response:**
```json
{
  "prediction": "Power 4 D1",
  "probabilities": {
    "Non D1": 0.15,
    "D1": 0.85,
    "Power 4 D1": 0.65,
    "Non P4 D1": 0.20
  },
  "confidence": 0.65,
  "stage": "D1 + Power 4 classification"
}
```

### GET `/infielder/features`
Get information about required features.

**Response:**
```json
{
  "required_features": ["age", "height", "weight", ...],
  "feature_info": {
    "numerical_features": [...],
    "categorical_features": [...],
    "descriptions": {...}
  }
}
```

### GET `/infielder/health`
Health check endpoint.

### GET `/infielder/example`
Get an example of valid input data.

## Usage Examples

### Python Usage
```python
from infielder_pipeline import InfielderPredictionPipeline

# Initialize pipeline
pipeline = InfielderPredictionPipeline(models_dir='models/')

# Player data
player_data = {
    "age": 17.5,
    "height": 72.0,
    "weight": 180.0,
    "hand_speed_max": 22.5,
    "bat_speed_max": 75.0,
    "rot_acc_max": 18.0,
    "sixty_time": 6.8,
    "thirty_time": 3.2,
    "ten_yard_time": 1.7,
    "run_speed_max": 22.0,
    "exit_velo_max": 88.0,
    "exit_velo_avg": 78.0,
    "distance_max": 320.0,
    "sweet_spot_p": 0.75,
    "inf_velo": 78.0,
    "player_state": "CA",
    "throwing_hand": "R",
    "hitting_handedness": "R",
    "player_region": "West"
}

# Get prediction
result = pipeline.predict(player_data)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Testing
```bash
# Run the test script
cd backend/ml
python test_pipeline.py
```

### Frontend Testing
Open `frontend/infielder_test.html` in a web browser to test the API with a user interface.

## Missing Value Handling

The pipeline handles missing values by:
- Using reasonable default values for numerical features
- Using the most common category for categorical features
- All features are optional - the pipeline will work with partial data

## Model Performance

The models were trained on infielder data with the following characteristics:
- D1 vs Non-D1 model: Binary classification
- Power 4 vs Non-Power 4 model: Binary classification (D1 players only)
- Both models use XGBoost with optimized hyperparameters
- Feature scaling and categorical encoding are applied consistently

## Integration

The pipeline is integrated into the main FastAPI application at `/infielder` endpoints. The router is included in `backend/api/main.py`.

## Error Handling

The pipeline includes comprehensive error handling:
- Model loading errors
- Input validation
- Missing value handling
- Unseen categorical values
- Network errors in API calls

## Future Enhancements

Potential improvements:
- Add confidence intervals
- Include feature importance analysis
- Support for other positions (pitchers, outfielders, catchers)
- Model retraining pipeline
- A/B testing capabilities
- Real-time model performance monitoring 