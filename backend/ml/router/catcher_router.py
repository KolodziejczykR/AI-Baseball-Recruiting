from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os
import logging
import sys

# Use absolute path for models directory
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pipeline.catcher_pipeline import CatcherPredictionPipeline
from backend.utils.player_types import PlayerCatcher

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize the pipeline
try:
    pipeline = CatcherPredictionPipeline()
    print("Catcher pipeline initialized successfully")
except Exception as e:
    print(f"Failed to initialize catcher pipeline: {e}")
    pipeline = None

class CatcherInput(BaseModel):
    """Input model for catcher statistics"""
    # Numerical features
    height: int = Field(..., ge=60, le=84, description="Player height in inches")
    weight: int = Field(..., ge=120, le=300, description="Player weight in pounds")
    sixty_time: float = Field(..., ge=5.0, le=10.0, description="60-yard dash time in seconds")
    exit_velo_max: float = Field(..., ge=50, le=130, description="Maximum exit velocity in mph")
    c_velo: float = Field(..., ge=50, le=100, description="Catcher velocity in mph")
    pop_time: float = Field(..., ge=1.5, le=4.0, description="Pop time (seconds)")
    
    # Required categorical features
    primary_position: str = Field(..., description="Primary position ('C')")
    hitting_handedness: str = Field(..., description="Hitting handedness (R, L, S)")
    throwing_hand: str = Field(..., description="Throwing hand (L, R)")
    player_region: str = Field(..., description="Player region (Midwest, Northeast, South, West)")

@router.post("/predict")
async def predict_catcher(input_data: CatcherInput) -> Dict[str, Any]:
    """
    Predict catcher college level using the two-stage XGBoost pipeline.
    
    The pipeline works as follows:
    1. First stage: Predict D1 vs Non-D1
    2. Second stage: If D1, predict Power 4 D1 vs Non-Power 4 D1
    
    Returns probabilities for all categories: Non D1, D1, Power 4 D1, Non P4 D1
    """
    if pipeline is None:
        logger.error("Prediction pipeline not available")
        raise HTTPException(status_code=500, detail="Prediction pipeline not available")
    
    try:
        # Convert validated input to dictionary
        input_dict = input_data.model_dump(exclude_none=True)
        logger.info(f"Processing catcher prediction for position: {input_dict['primary_position']}")
        
        # test git 

        # Create PlayerCatcher object
        player = PlayerCatcher(
            height=input_dict.get('height'),
            weight=input_dict.get('weight'),
            primary_position=input_dict.get('primary_position', 'C'),
            hitting_handedness=input_dict.get('hitting_handedness'),
            throwing_hand=input_dict.get('throwing_hand'),
            region=input_dict.get('player_region'),
            exit_velo_max=input_dict.get('exit_velo_max'),
            c_velo=input_dict.get('c_velo'),
            pop_time=input_dict.get('pop_time'),
            sixty_time=input_dict.get('sixty_time')
        )
        
        # Run prediction
        result = pipeline.predict(player)    
        
        logger.info(f"Prediction successful: {result.get_final_prediction()}")
        return result.get_api_response()
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Input validation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/features")
async def get_required_features() -> Dict[str, Any]:
    """Get information about required features for prediction using player object"""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Prediction pipeline not available")
    
    # Create a dummy player to get feature information
    dummy_player = PlayerCatcher(
        height=72, weight=180, primary_position="C", 
        hitting_handedness="R", throwing_hand="R", region="West",
        exit_velo_max=85.0, c_velo=75.0, pop_time=2.0, sixty_time=7.0
    )
    
    return {
        "required_features": dummy_player.get_player_features()
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
            "c_velo": 78.0,
            "pop_time": 1.9,
            "player_state": "CA",
            "throwing_hand": "R",
            "hitting_handedness": "R",
            "player_region": "West",
            "primary_position": "C"
        },
        "description": "This is an example of a high-performing catcher's statistics"
    } 