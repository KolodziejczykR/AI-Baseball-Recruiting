# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI/ML-assisted baseball recruitment platform that predicts college level classifications (Non D1, Non P4 D1, Power 4 D1) for baseball players based on their statistics and position. The system supports three positions so far: infielders, outfielders, and catchers. These are all within the category of hitters, with pitchers coming soon.

Do not run files at the end to test the results unless specifically asked too, as it wastes tokens. Just tell me what file to run if you need to.

## Development Commands

### Backend (FastAPI)
```bash
# Start backend server
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# Install dependencies
pip install -r requirements.txt

# Run tests
python3 -m pytest tests/ -v

# Run specific test files
python3 -m pytest tests/test_api.py -v
python3 -m pytest tests/test_infield_model_pipeline.py -v
python3 -m pytest tests/test_outfielder_model_pipeline.py -v
```

### Frontend (React TypeScript)
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test
```

## Architecture

### Directory Structure
- `backend/` - FastAPI application with ML models and API endpoints
- `frontend/` - React TypeScript application
- `tests/` - Comprehensive test suite for API and ML pipelines

### Backend Architecture
The backend uses a two-stage ML prediction system:

1. **Stage 1**: Predict D1 vs Non-D1
2. **Stage 2**: If D1 predicted, predict Power 4 D1 vs Non-Power 4 D1

#### ML Pipeline Structure
```
backend/ml/
├── pipeline/           # Core prediction logic
│   ├── infielder_pipeline.py
│   ├── outfielder_pipeline.py
│   └── catcher_pipeline.py
├── router/            # FastAPI route handlers
│   ├── infielder_router.py
│   ├── outfielder_router.py
│   └── catcher_router.py
├── models/            # Trained ML models (.pkl files)
├── train/             # Model training scripts
└── ml_router.py       # Router aggregation
```

#### Position-Specific Models
- **Infielders**: LightGBM (D1) + CatBoost (Power 4)
- **Outfielders**: XGBoost (both stages)  
- **Catchers**: XGBoost (both stages)

### API Endpoints
Each position has consistent endpoints:
- `POST /{position}/predict` - Make predictions
- `GET /{position}/features` - Get feature information
- `GET /{position}/health` - Health check
- `GET /{position}/example` - Get example input

### Key Features
**Common Features**: height, weight, exit_velo_max, hand_speed_max, bat_speed_max, sixty_time, throwing_hand, hitting_handedness, player_region, primary_position

**Position-Specific Features**:
- **Infielders**: `inf_velo` (infield velocity)
- **Outfielders**: `of_velo` (outfield velocity)
- **Catchers**: `c_velo` (catcher velocity), `pop_time`

## Testing

The project has comprehensive tests covering both ML pipelines and API endpoints:

- **Total Tests**: ~51 tests
- **Pipeline Tests**: Input validation, missing value handling, edge cases, different player types
- **API Tests**: Endpoint validation, error handling, request/response schemas

### Running Tests
Always run tests from project root directory. Use VS Code's Testing sidebar for best experience, or run from command line with pytest.

## Model Files
Ensure all required model files exist in `backend/ml/models/` before running the application. The system includes trained models, scalers, and label encoders for each position and prediction stage.

## Development Notes
- Backend runs on port 8000, frontend on port 3000
- CORS is configured for local development
- All pipelines include comprehensive error handling and input validation
- Missing values are handled intelligently with default values
- Model predictions are normalized to ensure probabilities sum to 1.0