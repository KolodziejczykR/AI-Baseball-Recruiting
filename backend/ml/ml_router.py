from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
from backend.ml.model_inference import predict_hitter, predict_pitcher

router = APIRouter()

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