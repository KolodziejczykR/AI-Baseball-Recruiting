from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os
import sys

# Use absolute path for models directory
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pipeline.outfielder_pipeline import OutfielderPredictionPipeline

router = APIRouter()

# Initialize the pipeline
try:
    pipeline = OutfielderPredictionPipeline(models_dir=models_dir)
except Exception as e:
    print(f"Failed to initialize outfielder pipeline: {e}")
    pipeline = None

class OutfielderInput(BaseModel):
    """Input model for outfielder statistics"""
    
    # Numerical features
    height: Optional[float] = Field(None, description="Player height in inches")
    weight: Optional[float] = Field(None, description="Player weight in pounds")
    hand_speed_max: Optional[float] = Field(None, description="Maximum hand speed (mph)")
    bat_speed_max: Optional[float] = Field(None, description="Maximum bat speed (mph)")
    rot_acc_max: Optional[float] = Field(None, description="Maximum rotational acceleration")
    sixty_time: Optional[float] = Field(None, description="60-yard dash time (seconds)")
    thirty_time: Optional[float] = Field(None, description="30-yard dash time (seconds)")
    ten_yard_time: Optional[float] = Field(None, description="10-yard dash time (seconds)")
    run_speed_max: Optional[float] = Field(None, description="Maximum running speed (mph)")
    exit_velo_max: Optional[float] = Field(None, description="Maximum exit velocity (mph)")
    exit_velo_avg: Optional[float] = Field(None, description="Average exit velocity (mph)")
    distance_max: Optional[float] = Field(None, description="Maximum hit distance (feet)")
    sweet_spot_p: Optional[float] = Field(None, description="Sweet spot percentage (0-1)")
    of_velo: Optional[float] = Field(None, description="Outfield velocity (mph)")
    number_of_missing: Optional[float] = Field(None, description="Number of missing values in player data")
    
    # Categorical features
    throwing_hand: Optional[str] = Field(None, description="Throwing hand (L/R)")
    hitting_handedness: Optional[str] = Field(None, description="Hitting handedness (L/R/S)")
    player_region: Optional[str] = Field(None, description="Player region")
    primary_position: Optional[str] = Field(None, description="Primary position (OF, CF, LF, RF)")

class PredictionResponse(BaseModel):
    """Response model for outfielder predictions"""
    prediction: str
    probabilities: Dict[str, float]
    confidence: float
    stage: str
    error: Optional[str] = None

@router.post("/predict", response_model=PredictionResponse)
async def predict_outfielder(input_data: OutfielderInput) -> Dict[str, Any]:
    """
    Predict outfielder college level using the two-stage XGBoost pipeline.
    
    The pipeline works as follows:
    1. First stage: Predict D1 vs Non-D1
    2. Second stage: If D1, predict Power 4 D1 vs Non-Power 4 D1
    
    Returns probabilities for all categories: Non D1, D1, Power 4 D1, Non P4 D1
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Prediction pipeline not available")
    
    # Convert Pydantic model to dictionary
    input_dict = input_data.model_dump(exclude_none=True)
    
    # Run prediction
    result = pipeline.predict(input_dict)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@router.get("/features")
async def get_required_features() -> Dict[str, Any]:
    """Get information about required features for prediction"""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Prediction pipeline not available")
    
    return {
        "required_features": pipeline.get_required_features(),
        "feature_info": pipeline.get_feature_info()
    }

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy" if pipeline is not None else "unhealthy",
        "pipeline_loaded": pipeline is not None
    }

# Example usage endpoint
@router.get("/example")
async def get_example_input() -> Dict[str, Any]:
    """Get an example of valid input data"""
    return {
        "example_input": {
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
            "of_velo": 78.0,
            "player_state": "CA",
            "throwing_hand": "R",
            "hitting_handedness": "R",
            "player_region": "West",
            "primary_position": "OF"
        },
        "description": "This is an example of a high-performing outfielder's statistics"
    } 