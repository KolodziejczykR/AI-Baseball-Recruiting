"""
P4 Outfielder Prediction Pipeline - Clean Production Version
Performance: 73.3% accuracy, 69.3% P4 recall
"""

import pandas as pd
import numpy as np
import joblib
import json
import sys
import os

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.utils.prediction_types import P4PredictionResult
from backend.utils.elite_weighting_constants import (
    ELITE_EXIT_VELO_MAX, ELITE_OF_VELO, ELITE_SIXTY_TIME_OF, ELITE_HEIGHT_MIN
)

def predict_outfielder_p4_probability(player_data: dict, models_dir: str, d1_probability: float) -> P4PredictionResult:
    """
    Clean P4 probability prediction for outfielders using trained ensemble
    
    Args:
        player_data (dict): Player statistics
        models_dir (str): Path to model files  
        d1_probability (float): D1 probability from previous stage
    
    Returns:
        P4PredictionResult: Structured prediction result
    """
    
    # Load trained models and configuration
    xgb_model = joblib.load(f'{models_dir}/xgb_model.pkl')
    lgb_model = joblib.load(f'{models_dir}/lgb_model.pkl')
    mlp_model = joblib.load(f'{models_dir}/mlp_model.pkl')
    svm_model = joblib.load(f'{models_dir}/svm_model.pkl')
    elite_model = joblib.load(f'{models_dir}/elite_model.pkl')
    scaler = joblib.load(f'{models_dir}/scaler.pkl')
    
    with open(f'{models_dir}/model_config.json', 'r') as f:
        config = json.load(f)
    
    with open(f'{models_dir}/feature_metadata.json', 'r') as f:
        feature_metadata = json.load(f)
    
    # Feature engineering (same as training)
    df = pd.DataFrame([player_data])
    
    # Basic engineered features
    df['power_speed'] = df['exit_velo_max'] * (7.6 - df['sixty_time'])
    df['of_velo_sixty_ratio'] = df['of_velo'] / df['sixty_time']
    df['height_weight'] = df['height'] * df['weight']
    df['power_per_pound'] = df['exit_velo_max'] / df['weight']
    df['exit_to_sixty_ratio'] = df['exit_velo_max'] / df['sixty_time']
    df['speed_size_efficiency'] = df['height'] / df['sixty_time']
    df['athletic_index'] = (df['exit_velo_max'] + df['of_velo'] + (100 - df['sixty_time']*10)) / 3
    df['power_speed_index'] = df['power_speed'] / df['athletic_index']
    df['exit_velo_body'] = df['exit_velo_max'] / df['height_weight']
    
    # Elite indicators
    df['elite_exit_velo'] = (df['exit_velo_max'] >= ELITE_EXIT_VELO_MAX).astype(int)
    df['elite_of_velo'] = (df['of_velo'] >= ELITE_OF_VELO).astype(int)
    df['elite_speed'] = (df['sixty_time'] <= ELITE_SIXTY_TIME_OF).astype(int)
    df['elite_size'] = (df['height'] >= ELITE_HEIGHT_MIN).astype(int)
    
    df['multi_tool_count'] = (df['elite_exit_velo'] + df['elite_of_velo'] + 
                             df['elite_speed'] + df['elite_size'])
    
    # Categorical encoding (ensure all expected categories exist)
    # Player region
    for region in ['Midwest', 'Northeast', 'South', 'West']:
        df[f'player_region_{region}'] = (df['player_region'] == region).astype(int)
    
    # Hitting handedness
    df['hitting_handedness_L'] = (df['hitting_handedness'] == 'L').astype(int)
    df['hitting_handedness_R'] = (df['hitting_handedness'] == 'R').astype(int)
    
    # Drop original categorical columns
    df = df.drop(['player_region', 'hitting_handedness'], axis=1)
    
    # D1-based features (required by model)
    df['d1_probability'] = d1_probability
    df['d1_prob_size'] = df['d1_probability'] * df['height_weight']
    df['d1_size_speed'] = df['d1_probability'] * df['speed_size_efficiency']
    df['d1_squared'] = df['d1_probability'] ** 2
    df['d1_athletic_index'] = df['d1_probability'] * df['athletic_index']
    df['d1_exit_velo'] = df['d1_probability'] * df['exit_velo_max']
    df['d1_power_per_pound'] = df['d1_probability'] * df['power_per_pound']
    df['d1_speed_size'] = df['d1_probability'] * df['speed_size_efficiency']
    
    # Additional required features
    df['p4_region_bonus'] = 0
    if 'player_region_South' in df.columns:
        df['p4_region_bonus'] += df['player_region_South'] * 0.15
    if 'player_region_West' in df.columns:
        df['p4_region_bonus'] += df['player_region_West'] * 0.12

    high_d1_mask = df['d1_probability'] >= 0.6
    df['p4_among_high_d1'] = 0.0
    if high_d1_mask.any():
        # For high D1 players, normalize their D1 probability by the overall P4/D1 rate (0.325)
        df.loc[high_d1_mask, 'p4_among_high_d1'] = df.loc[high_d1_mask, 'd1_probability'] / 0.325
        
    df['tool_count'] = df['multi_tool_count']
    df['athletic_index_v2'] = df['athletic_index'] * (1 + df['tool_count'] * 0.1)
    df['d1_composite_score'] = (df['exit_velo_max']/100 + df['of_velo']/100 + 
                               (7-df['sixty_time']) + df['athletic_index_v2']/100) / 4
    
    # Ensure all required features exist and are in correct order
    missing_features = []
    for feature in feature_metadata['features']:
        if feature not in df.columns:
            missing_features.append(feature)
    
    if missing_features:
        raise ValueError(f"Missing required features for model prediction: {missing_features}")
    
    X = df[feature_metadata['features']].fillna(0)
    X_scaled = scaler.transform(X)
    
    # Elite detection (for threshold selection)
    elite_prob = elite_model.predict_proba(X)[0, 1]
    is_elite = elite_prob >= 0.5
    
    # Get ensemble predictions using trained models
    xgb_prob = xgb_model.predict_proba(X.values)[0, 1]
    lgb_prob = lgb_model.predict_proba(X)[0, 1]
    mlp_prob = mlp_model.predict_proba(X_scaled)[0, 1]
    svm_prob = svm_model.predict_proba(X_scaled)[0, 1]
    
    # Apply trained ensemble weights (no dynamic adjustment)
    weights = config['ensemble_weights']
    ensemble_prob = (xgb_prob * weights['xgb'] + 
                    lgb_prob * weights['lgb'] + 
                    mlp_prob * weights['mlp'] + 
                    svm_prob * weights['svm'])
    
    # Apply trained thresholds
    threshold = config['thresholds']['elite_threshold'] if is_elite else config['thresholds']['non_elite_threshold']
    p4_prediction = ensemble_prob >= threshold
    
    # Combined confidence: ensemble agreement + boundary distance (adjusted for diverse models)
    individual_probs = [xgb_prob, lgb_prob, mlp_prob, svm_prob]
    
    # Agreement: More lenient for diverse model types (tree vs neural vs SVM)
    agreement_score = max(0, 1 - np.std(individual_probs) * 2.5)
    
    # Boundary distance: Far from 0.5 = high confidence  
    boundary_confidence = 2 * abs(ensemble_prob - 0.5)
    
    # Combined confidence mean
    combined_confidence = (agreement_score + boundary_confidence) / 2
    
    # More realistic thresholds for diverse ensemble
    if combined_confidence > 0.6:
        confidence = 'High'
    elif combined_confidence > 0.3:
        confidence = 'Medium'
    else:
        confidence = 'Low'
    
    # Elite P4 detection for result
    elite_indicators = []
    
    if player_data.get('exit_velo_max') >= ELITE_EXIT_VELO_MAX:
        elite_indicators.append(f"Elite exit velocity: {player_data['exit_velo_max']} mph")
    
    if player_data.get('of_velo') >= ELITE_OF_VELO:
        elite_indicators.append(f"Elite outfield velocity: {player_data['of_velo']} mph")
    
    if player_data.get('sixty_time') <= ELITE_SIXTY_TIME_OF:
        elite_indicators.append(f"Elite speed: {player_data['sixty_time']} seconds")
    
    if player_data.get('height') >= ELITE_HEIGHT_MIN:
        elite_indicators.append(f"Elite height: {player_data['height']} inches")
    
    
    return P4PredictionResult(
        p4_probability=float(ensemble_prob),
        p4_prediction=bool(p4_prediction),
        confidence=confidence,
        is_elite_p4=bool(is_elite),
        elite_indicators=elite_indicators if elite_indicators else None,
        model_version=config['model_version']
    )

if __name__ == "__main__":
    # Example usage
    test_player = {
        'height': 74.0,
        'weight': 190.0,
        'sixty_time': 6.5,
        'exit_velo_max': 98.0,
        'of_velo': 88.0,
        'player_region': 'South',
        'throwing_hand': 'Right',
        'hitting_handedness': 'Right'
    }
    
    models_dir = os.path.dirname(__file__)
    result = predict_outfielder_p4_probability(test_player, models_dir, d1_probability=0.6)

    print(f"P4 Probability: {result.p4_probability:.1%}")
    print(f"P4 Prediction: {result.p4_prediction}")
    print(f"Confidence: {result.confidence}")
