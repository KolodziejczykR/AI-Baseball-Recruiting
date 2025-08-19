"""
Prediction classes for ML pipeline returns
"""
from dataclasses import dataclass
from typing import List, Optional

from backend.utils.player_types import PlayerType

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
        
@dataclass
class MLPipelineResults:
    player: PlayerType
    d1_results: D1PredictionResult

    p4_results: Optional[P4PredictionResult] = None  # P4PredictionResult only if it passes D1

    def get_final_prediction(self) -> str:
        """
        Determine the final prediction given the results of both stages
        
        Returns:
            str: One of "Power 4 D1", "Non-P4 D1", or "Non-D1"
        """
        if self.p4_results and self.p4_results.p4_prediction:
            return "Power 4 D1"
        elif self.d1_results.d1_prediction:
            return "Non-P4 D1"
        else:
            return "Non-D1"

    def get_pipeline_confidence(self) -> str:
        """
        Return a string indicating the confidence levels of both D1 and P4 models,
        or just the D1 model if the P4 model was not used.

        Returns:
            str: A string describing the confidence levels of the models
        """
        if self.p4_results:
            return f"D1 Model Confidence: {self.d1_results.confidence}, P4 Model Confidence: {self.p4_results.confidence}"
        else:
            return {f"D1 Model Confidence: {self.d1_results.confidence}"}

    def get_player_type(self) -> str:
        """Gets the player type, a str"""
        return self.player.get_player_type()

    def get_player_info(self) -> dict:
        """
        Gets the player info as a dictionary
        
        Returns:
            dict: The player info as a dictionary
        """
        return self.player.get_player_info()

    def get_models_used(self) -> List[str]:
        """
        Gets a list of the models used to make the prediction.
        Returns a list with two elements: the D1 model version and the P4 model version.
        If the P4 model was not used, the second element will be None.

        Returns:
            List[str]: A list of the models used to make the prediction.
        """
        return [self.d1_results.model_version, self.p4_results.model_version if self.p4_results else None]