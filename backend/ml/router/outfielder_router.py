from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional
import os
import sys
import logging

# Use absolute path for models directory
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pipeline.outfielder_pipeline import OutfielderPredictionPipeline
from backend.utils.player_types import PlayerOutfielder

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize the pipeline
try:
    pipeline = OutfielderPredictionPipeline()
    logger.info("Outfielder pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize outfielder pipeline: {e}")
    pipeline = None

class OutfielderInput(BaseModel):
    """Input validation model for outfielder predictions"""
    
    # Required numerical features
    height: int = Field(..., ge=60, le=84, description="Player height in inches")
    weight: int = Field(..., ge=120, le=300, description="Player weight in pounds")
    sixty_time: float = Field(..., ge=5.0, le=10.0, description="60-yard dash time in seconds")
    exit_velo_max: float = Field(..., ge=50, le=130, description="Maximum exit velocity in mph")
    of_velo: float = Field(..., ge=50, le=110, description="Outfield velocity in mph")
    
    # Required categorical features
    primary_position: str = Field(..., description="Primary position ('OF')")
    hitting_handedness: str = Field(..., description="Hitting handedness (R, L, S)")
    throwing_hand: str = Field(..., description="Throwing hand (L, R)")
    player_region: str = Field(..., description="Player region (Midwest, Northeast, South, West)")

@router.post("/predict", responses={
    400: {"description": "Validation error or prediction failed"},
    422: {"description": "Input validation error"},
    500: {"description": "Internal server error"}
})
async def predict_outfielder(input_data: OutfielderInput) -> Dict[str, Any]:
    """
    Predict outfielder college level using the two-stage hierarchical pipeline.
    
    The pipeline works as follows:
    1. First stage: Predict D1 vs Non-D1 using XGBoost ensemble
    2. Second stage: If D1, predict Power 4 D1 vs Non-Power 4 D1 using XGBoost ensemble
    
    Returns probabilities for all categories: Non D1, Non-P4 D1, Power 4 D1
    """
    if pipeline is None:
        logger.error("Prediction pipeline not available")
        raise HTTPException(status_code=500, detail="Prediction pipeline not available")
    
    try:
        # Convert validated input to dictionary
        input_dict = input_data.model_dump(exclude_none=True)
        logger.info(f"Processing outfielder prediction for position: {input_dict['primary_position']}")
        
        # Create PlayerOutfielder object
        player = PlayerOutfielder(
            height=input_dict['height'],
            weight=input_dict['weight'],
            primary_position=input_dict['primary_position'],
            hitting_handedness=input_dict['hitting_handedness'],
            throwing_hand=input_dict['throwing_hand'],
            region=input_dict['player_region'],
            exit_velo_max=input_dict['exit_velo_max'],
            of_velo=input_dict['of_velo'],
            sixty_time=input_dict['sixty_time']
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
    dummy_player = PlayerOutfielder(
        height=72, weight=180, primary_position="OF", 
        hitting_handedness="R", throwing_hand="R", region="West",
        exit_velo_max=85.0, of_velo=80.0, sixty_time=7.0
    )
    
    return {
        "required_features": dummy_player.get_player_features(),
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
            "sixty_time": 6.8,
            "exit_velo_max": 88.0,
            "of_velo": 78.0,
            "throwing_hand": "R",
            "hitting_handedness": "R",
            "player_region": "West",
            "primary_position": "OF"
        },
        "description": "This is an example of a high-performing outfielder's statistics"
    } 