#!/usr/bin/env python3
"""
P4 Infielder Prediction Pipeline - Clean Production Version
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.utils.prediction_types import P4PredictionResult
from backend.utils.elite_weighting_constants import (
    ELITE_EXIT_VELO_MAX, ELITE_INF_VELO, ELITE_SIXTY_TIME_INF, ELITE_HEIGHT_MIN
)

def predict_infielder_p4_probability(player_data: dict, models_dir: str, d1_probability: float) -> P4PredictionResult:
    """
    Clean P4 probability prediction for infielders using trained ensemble
    
    Args:
        player_data (dict): Player statistics
        models_dir (str): Path to model files  
        d1_probability (float): D1 probability from previous stage
    
    Returns:
        P4PredictionResult: Structured prediction result
    """
    
    # Load trained models and configuration
    xgb_model = joblib.load(f'{models_dir}/p4_ensemble_xgboost_model.pkl')
    cb_model = joblib.load(f'{models_dir}/p4_ensemble_catboost_model.pkl')
    lgb_model = joblib.load(f'{models_dir}/p4_ensemble_lightgbm_model.pkl')
    svm_model = joblib.load(f'{models_dir}/p4_ensemble_svm_model.pkl')
    elite_model = joblib.load(f'{models_dir}/elite_detection_xgboost_model.pkl')
    scaler = joblib.load(f'{models_dir}/feature_scaler_for_svm.pkl')
    
    # Load config and metadata
    config_metadata = joblib.load(f'{models_dir}/model_config_and_metadata.pkl')
    
    # Feature engineering (same as training)
    df = pd.DataFrame([player_data])
    
    # Categorical encodings
    for pos in ['2B', '3B', 'SS']:
        df[f'primary_position_{pos}'] = (df['primary_position'] == pos).astype(int)
    
    df['throwing_hand_R'] = (df['throwing_hand'] == 'R').astype(int)
    df['hitting_handedness_R'] = (df['hitting_handedness'] == 'R').astype(int)
    df['hitting_handedness_S'] = (df['hitting_handedness'] == 'S').astype(int)
    
    for region in ['Northeast', 'South', 'West']:
        df[f'player_region_{region}'] = (df['player_region'] == region).astype(int)
    
    # Drop original categorical columns
    df = df.drop(['primary_position', 'throwing_hand', 'hitting_handedness', 'player_region'], axis=1)
    
    # Basic engineered features
    df['velo_by_inf'] = df['exit_velo_max'] / df['inf_velo']
    df['power_speed'] = df['exit_velo_max'] / df['sixty_time']
    df['sixty_inv'] = 1 / df['sixty_time']
    df['height_weight'] = df['height'] * df['weight']
    
    # Percentile features (using fixed values for single prediction)
    df['exit_velo_max_percentile'] = 50.0  # Default percentile
    df['inf_velo_percentile'] = 50.0
    df['sixty_time_percentile'] = 50.0
    df['height_percentile'] = 50.0
    df['weight_percentile'] = 50.0
    df['power_speed_percentile'] = 50.0
    
    # Additional engineered features
    df['power_per_pound'] = df['exit_velo_max'] / df['weight']
    df['exit_to_sixty_ratio'] = df['exit_velo_max'] / df['sixty_time']
    df['speed_size_efficiency'] = (df['height'] * df['weight']) / (df['sixty_time'] ** 2)
    df['athletic_index'] = (df['power_speed'] * df['height'] * df['weight']) / df['sixty_time']
    df['power_speed_index'] = df['exit_velo_max'] * (1 / df['sixty_time'])
    
    # Elite binary features (using fixed thresholds)
    df['elite_exit_velo'] = (df['exit_velo_max'] >= ELITE_EXIT_VELO_MAX).astype(int)
    df['elite_inf_velo'] = (df['inf_velo'] >= ELITE_INF_VELO).astype(int)
    df['elite_speed'] = (df['sixty_time'] <= ELITE_SIXTY_TIME_INF).astype(int)
    df['elite_size'] = (df['height'] >= ELITE_HEIGHT_MIN).astype(int)
    df['multi_tool_count'] = (df['elite_exit_velo'] + df['elite_inf_velo'] + 
                             df['elite_speed'] + df['elite_size'])
    
    # Scaled features (using approximate scaling)
    df['exit_velo_scaled'] = np.clip((df['exit_velo_max'] - 75) / (105 - 75) * 100, 0, 100)
    df['speed_scaled'] = np.clip((1 - (df['sixty_time'] - 6.0) / (8.5 - 6.0)) * 100, 0, 100)
    df['arm_scaled'] = np.clip((df['inf_velo'] - 70) / (95 - 70) * 100, 0, 100)
    
    # D1-based features (required by model)
    df['d1_ensemble_prob'] = d1_probability
    df['d1_confidence_high'] = 1 if d1_probability > 0.7 else 0
    df['d1_confidence_medium'] = 1 if 0.4 <= d1_probability <= 0.7 else 0
    df['d1_prob_squared'] = df['d1_ensemble_prob'] ** 2
    
    # Power 4 region indicator
    df['power4_region'] = ((df.get('player_region_South', 0) == 1) | 
                          (df.get('player_region_West', 0) == 1)).astype(int)
    
    # Ensure all required features exist and are in correct order
    missing_elite_features = []
    for feature in config_metadata['elite_features']:
        if feature not in df.columns:
            missing_elite_features.append(feature)
    
    missing_p4_features = []
    for feature in config_metadata['p4_features']:
        if feature not in df.columns:
            missing_p4_features.append(feature)
    
    if missing_elite_features:
        raise ValueError(f"Missing required elite features for model prediction: {missing_elite_features}")
    
    if missing_p4_features:
        raise ValueError(f"Missing required P4 features for model prediction: {missing_p4_features}")
    
    # Select features
    elite_feats = df[config_metadata['elite_features']].fillna(0)
    p4_feats = df[config_metadata['p4_features']].fillna(0)
    
    # Clean data
    elite_feats = elite_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
    p4_feats = p4_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features for SVM
    p4_feats_scaled = scaler.transform(p4_feats)
    
    # Elite detection (for threshold selection)
    elite_prob = elite_model.predict_proba(elite_feats)[0, 1]
    is_elite = elite_prob >= (float(config_metadata['elite_threshold']) / 100) # elite_threshold is 47.19, rather than 0.4719
    
    # Get ensemble predictions using trained models
    xgb_prob = xgb_model.predict_proba(p4_feats)[0, 1]
    cb_prob = cb_model.predict_proba(p4_feats)[0, 1]
    lgb_prob = lgb_model.predict_proba(p4_feats)[0, 1]
    svm_prob = svm_model.predict_proba(p4_feats_scaled)[0, 1]
    
    # Apply trained ensemble weights (no dynamic adjustment)
    weights = config_metadata['ensemble_weights']
    ensemble_prob = (xgb_prob * weights['xgboost'] + 
                    cb_prob * weights['catboost'] + 
                    lgb_prob * weights['lightgbm'] + 
                    svm_prob * weights['svm'])
    
    # Apply trained threshold
    threshold = config_metadata['optimal_threshold']
    p4_prediction = ensemble_prob >= threshold
    
    # Combined confidence: ensemble agreement + boundary distance
    individual_probs = [xgb_prob, cb_prob, lgb_prob, svm_prob]
    
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
    
    if player_data.get('inf_velo') >= ELITE_INF_VELO:
        elite_indicators.append(f"Elite infield velocity: {player_data['inf_velo']} mph")
    
    if player_data.get('sixty_time') <= ELITE_SIXTY_TIME_INF:
        elite_indicators.append(f"Elite speed: {player_data['sixty_time']} seconds")
    
    if player_data.get('height') >= ELITE_HEIGHT_MIN:
        elite_indicators.append(f"Elite height: {player_data['height']} inches")
        
    return P4PredictionResult(
        p4_probability=float(ensemble_prob),
        p4_prediction=bool(p4_prediction),
        confidence=confidence,
        is_elite_p4=bool(is_elite),
        elite_indicators=elite_indicators if elite_indicators else None,
        model_version="infielder_p4_08072025"
    )


if __name__ == "__main__":
    # Example usage
    test_player = {
        'height': 74.0,
        'weight': 190.0,
        'sixty_time': 6.9,
        'exit_velo_max': 92.0,
        'inf_velo': 82.0,
        'player_region': 'South',
        'throwing_hand': 'Right',
        'hitting_handedness': 'Right',
        'primary_position': 'SS'
    }
    
    models_dir = os.path.dirname(__file__)
    result = predict_infielder_p4_probability(test_player, models_dir, d1_probability=0.6)
    
    print(f"P4 Prediction: {result.p4_prediction} ({result.p4_probability:.1%})")
    print(f"Confidence: {result.confidence}")
    print(f"Elite Status: {result.is_elite_p4}")
    print(f"Elite Indicators: {result.elite_indicators}")