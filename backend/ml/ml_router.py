from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
from backend.ml.model_inference import predict_hitter, predict_pitcher

# Import the position-specific routers
from .infielder_router import router as infielder_router
from .outfielder_router import router as outfielder_router

router = APIRouter()

# Include the position-specific routers
router.include_router(infielder_router, prefix="/infielder", tags=["infielder"])
router.include_router(outfielder_router, prefix="/outfielder", tags=["outfielder"])

class HitterInput(BaseModel):
    # Define your hitter features here, e.g.:
    feature1: float
    feature2: float
    # Add more features as needed

class PitcherInput(BaseModel):
    # Define your pitcher features here, e.g.:
    feature1: float
    feature2: float
    # Add more features as needed

@router.post("/hitter")
def predict_hitter_endpoint(input_data: HitterInput) -> Dict[str, Any]:
    result = predict_hitter(input_data.model_dump())
    return result

@router.post("/pitcher")
def predict_pitcher_endpoint(input_data: PitcherInput) -> Dict[str, Any]:
    result = predict_pitcher(input_data.model_dump())
    return result 