"""
Prediction classes for ML pipeline returns
"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class P4PredictionResult:
    """
    Prediction result class for P4 classification
    Designed for LLM school selection use cases
    """
    p4_probability: float
    p4_prediction: bool  
    confidence: str    
    is_elite_p4: bool         
    model_version: str
    elite_indicators: Optional[List[str]] = None  # Why they're elite
    
    def __post_init__(self):
        """Validate probability bounds"""
        if not 0.0 <= self.p4_probability <= 1.0:
            raise ValueError(f"P4 probability must be between 0 and 1, got {self.p4_probability}")
    
@dataclass
class D1PredictionResult:
    """
    Prediction result class for D1 classification
    Designed for LLM school selection use cases and ML pipeline
    """
    d1_probability: float
    d1_prediction: bool
    confidence: str
    model_version: str

    def __post_init__(self):
        """Validate probability bounds"""
        if not 0.0 <= self.d1_probability <= 1.0:
            raise ValueError(f"D1 probability must be between 0 and 1, got {self.d1_probability}")