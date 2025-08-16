from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional
import os
import sys
import logging

# Use absolute path for models directory
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pipeline.infielder_pipeline import InfielderPredictionPipeline
from backend.utils.player_types import PlayerInfielder

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize the pipeline
try:
    pipeline = InfielderPredictionPipeline()
    logger.info("Infielder pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize infielder pipeline: {e}")
    pipeline = None

class InfielderInput(BaseModel):
    """Input validation model for infielder predictions"""
    
    # Required numerical features
    height: int = Field(..., ge=60, le=84, description="Player height in inches")
    weight: int = Field(..., ge=120, le=300, description="Player weight in pounds")
    sixty_time: float = Field(..., ge=5.0, le=10.0, description="60-yard dash time in seconds")
    exit_velo_max: float = Field(..., ge=50, le=130, description="Maximum exit velocity in mph")
    inf_velo: float = Field(..., ge=50, le=100, description="Infield velocity in mph")
    
    # Required categorical features
    primary_position: str = Field(..., description="Primary position")
    hitting_handedness: str = Field(..., description="Hitting handedness")
    throwing_hand: str = Field(..., description="Throwing hand")
    player_region: str = Field(..., description="Player region")
    
    @field_validator('primary_position')
    @classmethod
    def validate_position(cls, v):
        valid_positions = ['SS', '2B', '3B', '1B']
        if v not in valid_positions:
            raise ValueError(f"Position must be one of: {valid_positions}")
        return v
    
    @field_validator('hitting_handedness', 'throwing_hand')
    @classmethod
    def validate_handedness(cls, v):
        valid_hands = ['R', 'L', 'S']
        if v not in valid_hands:
            raise ValueError(f"Hand must be one of: {valid_hands}")
        return v
    
    @field_validator('player_region')
    @classmethod
    def validate_region(cls, v):
        valid_regions = ['West', 'South', 'Northeast', 'Midwest']
        if v not in valid_regions:
            raise ValueError(f"Region must be one of: {valid_regions}")
        return v

class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str
    error_type: str
    details: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    """Response model for infielder predictions"""
    final_prediction: str
    final_category: int
    d1_probability: float
    p4_probability: Optional[float]
    probabilities: Dict[str, float]
    confidence: str
    model_chain: str
    d1_details: Optional[Dict[str, Any]] = None
    p4_details: Optional[Dict[str, Any]] = None
    player_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@router.post("/predict", response_model=PredictionResponse, responses={
    400: {"description": "Validation error or prediction failed"},
    422: {"description": "Input validation error"},
    500: {"description": "Internal server error"}
})
async def predict_infielder(input_data: InfielderInput) -> Dict[str, Any]:
    """
    Predict infielder college level using the two-stage hierarchical pipeline.
    
    The pipeline works as follows:
    1. First stage: Predict D1 vs Non-D1
    2. Second stage: If D1, predict Power 4 D1 vs Non-Power 4 D1
    
    Returns probabilities for all categories: Non D1, Non-P4 D1, Power 4 D1
    """
    if pipeline is None:
        logger.error("Prediction pipeline not available")
        raise HTTPException(status_code=500, detail="Prediction pipeline not available")
    
    try:
        # Convert validated input to dictionary
        input_dict = input_data.model_dump(exclude_none=True)
        logger.info(f"Processing infielder prediction for position: {input_dict['primary_position']}")
        
        # Create PlayerInfielder object
        player = PlayerInfielder(
            height=input_dict['height'],
            weight=input_dict['weight'],
            primary_position=input_dict['primary_position'],
            hitting_handedness=input_dict['hitting_handedness'],
            throwing_hand=input_dict['throwing_hand'],
            region=input_dict['player_region'],
            exit_velo_max=input_dict['exit_velo_max'],
            inf_velo=input_dict['inf_velo'],
            sixty_time=input_dict['sixty_time']
        )
        
        # Run prediction
        result = pipeline.predict(player)
        
        if "error" in result:
            logger.error(f"Prediction failed: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"Prediction successful: {result['final_prediction']}")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Input validation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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
            "sixty_time": 6.8,
            "exit_velo_max": 88.0,
            "inf_velo": 78.0,
            "throwing_hand": "R",
            "hitting_handedness": "R",
            "player_region": "West",
            "primary_position": "SS"
        },
        "description": "This is an example of a high-performing infielder's statistics"
    } 