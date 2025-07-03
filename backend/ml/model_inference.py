import os
from typing import Dict
import joblib

# Paths to model files
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
HITTER_MODEL_PATH = os.path.join(DATA_DIR, 'hitter_model.pkl')
PITCHER_MODEL_PATH = os.path.join(DATA_DIR, 'pitcher_model.pkl')

# Load models
try:
    hitter_model = joblib.load(HITTER_MODEL_PATH)
except Exception as e:
    hitter_model = None
    print(f"Failed to load hitter model: {e}")

try:
    pitcher_model = joblib.load(PITCHER_MODEL_PATH)
except Exception as e:
    pitcher_model = None
    print(f"Failed to load pitcher model: {e}")

def predict_hitter(input_data: Dict) -> Dict:
    """
    Run inference using the hitter model.
    Args:
        input_data (dict): Dictionary of features for the hitter model.
    Returns:
        dict: {"class": predicted_class, "probability": probability}
    """
    if hitter_model is None:
        return {"error": "Hitter model not loaded."}
    try:
        X = [list(input_data.values())]
        pred_class = hitter_model.predict(X)[0]
        if hasattr(hitter_model, 'predict_proba'):
            proba = max(hitter_model.predict_proba(X)[0])
        else:
            proba = None
        return {"class": pred_class, "probability": proba}
    except Exception as e:
        return {"error": str(e)}

def predict_pitcher(input_data: Dict) -> Dict:
    """
    Run inference using the pitcher model.
    Args:
        input_data (dict): Dictionary of features for the pitcher model.
    Returns:
        dict: {"class": predicted_class, "probability": probability}
    """
    if pitcher_model is None:
        return {"error": "Pitcher model not loaded."}
    try:
        X = [list(input_data.values())]
        pred_class = pitcher_model.predict(X)[0]
        if hasattr(pitcher_model, 'predict_proba'):
            proba = max(pitcher_model.predict_proba(X)[0])
        else:
            proba = None
        return {"class": pred_class, "probability": proba}
    except Exception as e:
        return {"error": str(e)} 