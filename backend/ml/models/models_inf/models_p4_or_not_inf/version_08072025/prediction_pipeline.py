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

def calculate_percentile_from_quantiles(value, quantiles, lower_is_better=False):
    """
    Calculate percentile of a value using pre-computed quantiles from training data
    
    Args:
        value: The value to calculate percentile for
        quantiles: List of quantiles from training data (every 5%: 0%, 5%, 10%, ..., 100%)
        lower_is_better: If True, invert percentile (for metrics like sixty_time)
    
    Returns:
        Percentile value (0-100)
    """
    # Handle edge cases
    if pd.isna(value) or value is None:
        return 50.0  # Default to median
    
    # Find which quantile bin the value falls into
    percentile = 0
    for i, q_val in enumerate(quantiles):
        if value <= q_val:
            percentile = i * 5  # Since quantiles are every 5%
            break
    else:
        percentile = 100  # Value is above all quantiles
    
    # Invert if lower is better
    if lower_is_better:
        percentile = 100 - percentile
    
    return min(100, max(0, percentile))

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
    
    # Training data quantiles for percentile calculation
    exit_velo_max_quantiles = [61.0, 80.5, 83.0, 85.0, 86.3, 87.2, 88.2, 89.0, 89.9, 90.5, 91.2, 92.0, 92.8, 93.5, 94.2, 95.1, 96.1, 97.3, 98.9, 100.8, 110.7]
    inf_velo_quantiles = [53.0, 70.0, 73.0, 75.0, 76.0, 77.0, 78.0, 78.0, 79.0, 80.0, 81.0, 81.0, 82.0, 83.0, 84.0, 84.0, 85.0, 86.0, 87.0, 89.0, 99.0]
    sixty_time_quantiles = [6.0, 6.73, 6.83, 6.9, 6.94, 7.0, 7.05, 7.09, 7.13, 7.18, 7.21, 7.26, 7.3, 7.36, 7.41, 7.47, 7.56, 7.64, 7.75, 7.94, 9.61]
    height_quantiles = [62.0, 68.0, 69.0, 69.0, 70.0, 70.0, 70.0, 71.0, 71.0, 71.0, 72.0, 72.0, 72.0, 73.0, 73.0, 73.0, 74.0, 74.0, 75.0, 75.0, 80.0]
    weight_quantiles = [110.0, 145.0, 155.0, 160.0, 160.0, 165.0, 168.0, 170.0, 175.0, 175.0, 180.0, 180.0, 185.0, 185.0, 190.0, 192.2, 196.42000000000024, 205.0, 210.0, 220.0, 296.0]
    power_speed_quantiles = [6.907894736842106, 10.51981666404049, 10.987698814011218, 11.3463080337328, 11.61017324306378, 11.84110970996217, 12.021499242013904, 12.179052730444118, 12.333782079291819, 12.485938831550042, 12.627551020408164, 12.780271169418116, 12.934873394737236, 13.085682228010418, 13.256082032035387, 13.417218543046358, 13.596039066739014, 13.81429096645807, 14.038563210681229, 14.4363939404699, 16.875]
    
    # Calculate actual percentiles using training data quantiles
    df['exit_velo_max_percentile'] = calculate_percentile_from_quantiles(df['exit_velo_max'].iloc[0], exit_velo_max_quantiles, lower_is_better=False)
    df['inf_velo_percentile'] = calculate_percentile_from_quantiles(df['inf_velo'].iloc[0], inf_velo_quantiles, lower_is_better=False)
    df['sixty_time_percentile'] = calculate_percentile_from_quantiles(df['sixty_time'].iloc[0], sixty_time_quantiles, lower_is_better=True)
    df['height_percentile'] = calculate_percentile_from_quantiles(df['height'].iloc[0], height_quantiles, lower_is_better=False)
    df['weight_percentile'] = calculate_percentile_from_quantiles(df['weight'].iloc[0], weight_quantiles, lower_is_better=False)
    df['power_speed_percentile'] = calculate_percentile_from_quantiles(df['power_speed'].iloc[0], power_speed_quantiles, lower_is_better=False)
    
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
        is_elite=bool(is_elite),
        elite_indicators=elite_indicators if elite_indicators else None,
        model_version="infielder_p4_08072025"
    )


if __name__ == "__main__":
    # Example usage
    test_player = {
        'height': 75.0,
        'weight': 215.0,
        'sixty_time': 6.7,
        'exit_velo_max': 102.0,
        'inf_velo': 91.0,
        'player_region': 'South',
        'throwing_hand': 'Right',
        'hitting_handedness': 'Right',
        'primary_position': 'SS'
    }
    
    models_dir = os.path.dirname(__file__)
    result = predict_infielder_p4_probability(test_player, models_dir, d1_probability=0.6)
    
    print(f"P4 Prediction: {result.p4_prediction} ({result.p4_probability:.1%})")
    print(f"Confidence: {result.confidence}")
    print(f"Elite Status: {result.is_elite}")
    print(f"Elite Indicators: {result.elite_indicators}")