# Baseball Recruitment ML API Documentation

A comprehensive machine learning API for predicting college baseball recruitment levels using player statistics and performance metrics.

## 🎯 Overview

This API provides predictions for college baseball recruitment classification across three levels:
- **Non-D1**: Not predicted for Division 1 level
- **Non-P4 D1**: Division 1 but not Power 4 conference  
- **Power 4 D1**: Power 4 conference (top tier)

### Supported Position Types
- **Infielders**: SS, 2B, 3B, 1B
- **Outfielders**: CF, LF, RF, OF
- **Catchers**: C (coming soon)

## 🚀 Quick Start

### Base URL
```
http://localhost:8000
```

### Health Check
```bash
curl http://localhost:8000/ping
```

### Make a Prediction
```bash
curl -X POST "http://localhost:8000/infielder/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "height": 72,
    "weight": 180,
    "sixty_time": 6.8,
    "exit_velo_max": 88.0,
    "inf_velo": 78.0,
    "throwing_hand": "R",
    "hitting_handedness": "R", 
    "player_region": "West",
    "primary_position": "SS"
  }'
```

## 📋 API Endpoints

### Core Endpoints (All Positions)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/{position}/predict` | Make prediction for player |
| `GET` | `/{position}/features` | Get feature information |
| `GET` | `/{position}/health` | Check pipeline health |
| `GET` | `/{position}/example` | Get example input data |
| `GET` | `/ping` | API health check |

Replace `{position}` with: `infielder`, `outfielder`, or `catcher`

## 🔧 Input Requirements

### Required Fields (All Positions)

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `height` | `int` | 60-84 | Player height in inches |
| `weight` | `int` | 120-300 | Player weight in pounds |
| `sixty_time` | `float` | 5.0-10.0 | 60-yard dash time in seconds |
| `exit_velo_max` | `float` | 50-130 | Maximum exit velocity in mph |
| `primary_position` | `string` | Position-specific | Player's primary position |
| `hitting_handedness` | `string` | R, L, S | Hitting handedness |
| `throwing_hand` | `string` | R, L | Throwing hand |
| `player_region` | `string` | See regions | Geographic region |

### Position-Specific Required Fields

#### Infielders
- `inf_velo` (float, 50-100): Infield velocity in mph

#### Outfielders  
- `of_velo` (float, 50-110): Outfield velocity in mph

#### Catchers
- `c_velo` (float, 50-100): Catcher velocity in mph
- `pop_time` (float, 1.5-3.0): Pop time in seconds

### Valid Categorical Values

#### Positions
- **Infielders**: `SS`, `2B`, `3B`, `1B`
- **Outfielders**: `CF`, `LF`, `RF`, `OF`
- **Catchers**: `C`

#### Handedness
- `R`: Right-handed
- `L`: Left-handed
- `S`: Switch hitter (hitting only)

#### Regions
- `West`: Western United States
- `South`: Southern United States  
- `Northeast`: Northeastern United States
- `Midwest`: Midwestern United States

### Optional Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `hand_speed_max` | `float` | 10-35 | Maximum hand speed in mph |
| `bat_speed_max` | `float` | 50-100 | Maximum bat speed in mph |
| `rot_acc_max` | `float` | 5-30 | Maximum rotational acceleration |
| `thirty_time` | `float` | 2.5-5.0 | 30-yard dash time in seconds |
| `ten_yard_time` | `float` | 1.0-2.5 | 10-yard dash time in seconds |
| `run_speed_max` | `float` | 15-30 | Maximum running speed in mph |
| `exit_velo_avg` | `float` | 50-110 | Average exit velocity in mph |
| `distance_max` | `float` | 200-500 | Maximum hit distance in feet |
| `sweet_spot_p` | `float` | 0.0-1.0 | Sweet spot percentage |

## 📊 Response Format

### Successful Prediction

```json
{
  "final_prediction": "Power 4 D1",
  "final_category": 2,
  "d1_probability": 0.85,
  "p4_probability": 0.72,
  "probabilities": {
    "non_d1": 0.15,
    "d1_total": 0.85,
    "non_p4_d1": 0.238,
    "p4_d1": 0.612
  },
  "confidence": "High",
  "model_chain": "D1_then_P4",
  "d1_details": {
    "d1_prediction": 1,
    "d1_probability": 0.85,
    "confidence_level": "High",
    "model_votes": {
      "xgboost": 0.89,
      "lightgbm": 0.82,
      "catboost": 0.87,
      "svm": 0.81
    }
  },
  "p4_details": {
    "p4_prediction": 1,
    "p4_probability": 0.72,
    "confidence": "High",
    "is_elite_candidate": true
  },
  "player_info": {
    "player_type": "Elite Speed-Power Combo",
    "region": "West",
    "elite_candidate_d1": true,
    "elite_candidate_p4": true
  }
}\n```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `final_prediction` | Final classification string |
| `final_category` | Numeric category (0=Non-D1, 1=Non-P4 D1, 2=Power 4 D1) |
| `d1_probability` | Raw probability of being D1 level (0.0-1.0) |
| `p4_probability` | Raw probability of being Power 4 (conditional on D1) |
| `probabilities` | Breakdown of all category probabilities |
| `confidence` | Confidence level (Low, Medium, High) |
| `model_chain` | Models used ("D1_only" or "D1_then_P4") |
| `d1_details` | Detailed Stage 1 prediction results |
| `p4_details` | Detailed Stage 2 prediction results |
| `player_info` | Additional player analysis and metadata |

## ❌ Error Handling

### HTTP Status Codes

#### 200 - Success
Prediction completed successfully.

#### 422 - Unprocessable Entity
**Input validation failed**

Common causes:
- Missing required fields
- Invalid data types
- Values outside allowed ranges
- Invalid categorical values

```json
{
  "detail": [
    {
      "loc": ["height"],
      "msg": "ensure this value is greater than or equal to 60",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

#### 400 - Bad Request
**Application logic error**

Common causes:
- ML pipeline prediction failure
- Player object creation error
- Business logic violation

```json
{
  "detail": "Prediction failed: Model inference error"
}
```

#### 500 - Internal Server Error
**Server error**

Common causes:
- ML pipeline unavailable
- Unexpected internal error

```json
{
  "detail": "Prediction pipeline not available"
}
```

## 🔬 ML Pipeline Details

### Two-Stage Hierarchical Classification

Each position uses a sophisticated two-stage prediction system:

1. **Stage 1 (D1 Classification)**
   - Predicts whether player meets D1 standards
   - Uses ensemble of multiple models
   - Position-specific thresholds and architectures

2. **Stage 2 (P4 Classification)**
   - Only runs if Stage 1 predicts D1 level
   - Predicts Power 4 vs Non-Power 4 D1
   - Advanced elite detection algorithms

### Model Architectures

#### Infielders
- **Stage 1**: Weighted ensemble (XGBoost + LightGBM + CatBoost + SVM)
- **Stage 2**: CatBoost ensemble with elite detection

#### Outfielders
- **Stage 1**: XGBoost ensemble + DNN + SVM with elite detection
- **Stage 2**: XGBoost ensemble + MLP + SVM with elite detection

#### Model Performance
- Cross-validated accuracy: 85-92% across positions
- Precision/Recall optimized for recruitment use case
- Regular retraining on updated recruitment data

## 🔍 Usage Examples

### Example 1: High-Performing Infielder
```bash
curl -X POST "http://localhost:8000/infielder/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "height": 74,
    "weight": 195,
    "sixty_time": 6.5,
    "exit_velo_max": 95.0,
    "inf_velo": 85.0,
    "throwing_hand": "R",
    "hitting_handedness": "R",
    "player_region": "South",
    "primary_position": "SS",
    "bat_speed_max": 78.0,
    "distance_max": 380.0
  }'
```

### Example 2: Speed-Focused Outfielder
```bash
curl -X POST "http://localhost:8000/outfielder/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "height": 70,
    "weight": 165,
    "sixty_time": 6.3,
    "exit_velo_max": 82.0,
    "of_velo": 88.0,
    "throwing_hand": "L",
    "hitting_handedness": "L",
    "player_region": "West",
    "primary_position": "CF",
    "run_speed_max": 24.5
  }'
```

### Example 3: Get Feature Information
```bash
curl "http://localhost:8000/infielder/features"
```

### Example 4: Check Pipeline Health
```bash
curl "http://localhost:8000/outfielder/health"
```

## 🔧 Development & Testing

### Running the API
```bash
# Start the server
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
python3 -m pytest tests/test_api.py -v
```

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Rate Limiting
- Currently no rate limiting implemented
- Recommended for production: 100 requests/minute per IP

### Authentication
- Currently no authentication required
- Consider API key authentication for production

## 📈 Performance Considerations

### Response Times
- **Single prediction**: 100-300ms average
- **Batch predictions**: Not currently supported
- **Model loading**: ~2-5 seconds on startup

### Scalability
- Stateless design allows horizontal scaling
- Model files loaded once per instance
- Consider model caching for high-traffic scenarios

### Monitoring
- Comprehensive logging at INFO and DEBUG levels
- Performance metrics tracked per pipeline
- Error tracking with full stack traces

## 🛠️ Integration Guide

### Python Client Example
```python
import requests

def predict_player(position, player_data):
    url = f"http://localhost:8000/{position}/predict"
    response = requests.post(url, json=player_data)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Prediction failed: {response.text}")

# Usage
player = {
    "height": 72,
    "weight": 180,
    "sixty_time": 6.8,
    "exit_velo_max": 88.0,
    "inf_velo": 78.0,
    "throwing_hand": "R",
    "hitting_handedness": "R",
    "player_region": "West", 
    "primary_position": "SS"
}

result = predict_player("infielder", player)
print(f"Prediction: {result['final_prediction']}")
```

### JavaScript/Node.js Example
```javascript
async function predictPlayer(position, playerData) {
  const response = await fetch(`http://localhost:8000/${position}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(playerData)
  });
  
  if (response.ok) {
    return await response.json();
  } else {
    throw new Error(`Prediction failed: ${response.statusText}`);
  }
}

// Usage
const player = {
  height: 72,
  weight: 180,
  sixty_time: 6.8,
  exit_velo_max: 88.0,
  of_velo: 82.0,
  throwing_hand: "R",
  hitting_handedness: "R",
  player_region: "West",
  primary_position: "CF"
};

predictPlayer("outfielder", player)
  .then(result => console.log(`Prediction: ${result.final_prediction}`))
  .catch(error => console.error(error));
```

## 🔒 Security Considerations

### Input Validation
- Strict type checking and range validation
- SQL injection prevention through parameterized queries
- No direct file system access from user input

### Error Handling
- Generic error messages to prevent information disclosure
- Detailed errors logged server-side only
- No sensitive data in API responses

### CORS Configuration
- Currently configured for local development
- Update for production domains

### Recommended Production Security
- HTTPS enforcement
- API rate limiting
- Request size limits
- Authentication/authorization
- Input sanitization
- Security headers

## 📝 Changelog

### Version 1.0.0 (Current)
- Initial release with infielder and outfielder support
- Two-stage hierarchical prediction pipeline
- Comprehensive input validation
- Detailed error handling and logging
- Performance monitoring and metrics

### Upcoming Features
- Catcher position support
- Pitcher position support  
- Batch prediction endpoints
- Model confidence intervals
- A/B testing framework
- Real-time model updates

## 🤝 Support

For technical issues or questions:
- Check the `/health` endpoints for pipeline status
- Review logs for detailed error information
- Validate input data against schema requirements
- Test with the `/example` endpoints for reference data

## 📚 Additional Resources

- [Router Documentation](backend/ml/router/README.md)
- [Model Training Documentation](backend/ml/train/README.md)
- [Testing Guide](tests/README.md)
- [Deployment Guide](docs/deployment.md)