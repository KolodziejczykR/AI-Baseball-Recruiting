#!/usr/bin/env python3
"""
P4 Infielder Prediction Pipeline - Production Version
Generated: 2025-08-07
Performance: Hierarchical ensemble with elite detection
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

from backend.utils.elite_weighting_constants import (
    ELITE_EXIT_VELO_MAX, ELITE_INF_VELO, ELITE_SIXTY_TIME_INF, ELITE_HEIGHT_MIN
)

def predict_infielder_p4_probability(player_data, models_dir):
    """
    Predict P4 college probability for infielder
    
    Args:
        player_data (dict): Player statistics
        {
            'height': float,          # inches
            'weight': float,          # pounds  
            'sixty_time': float,      # seconds
            'exit_velo_max': float,   # mph
            'inf_velo': float,        # mph (infield velocity)
            'player_region': str,     # Geographic region
            'throwing_hand': str,     # 'Left' or 'Right'
            'hitting_handedness': str,# 'Left', 'Right', or 'Switch'
            'primary_position': str   # 'SS', '2B', '3B', '1B'
        }
        models_dir (str): Path to model files
    
    Returns:
        dict: Prediction results
    """
    
    # Load models and config
    xgb_model = joblib.load(f'{models_dir}/p4_ensemble_xgboost_model.pkl')
    cb_model = joblib.load(f'{models_dir}/p4_ensemble_catboost_model.pkl')
    lgb_model = joblib.load(f'{models_dir}/p4_ensemble_lightgbm_model.pkl')
    svm_model = joblib.load(f'{models_dir}/p4_ensemble_svm_model.pkl')
    elite_model = joblib.load(f'{models_dir}/elite_detection_xgboost_model.pkl')
    scaler = joblib.load(f'{models_dir}/feature_scaler_for_svm.pkl')
    
    # Load config and metadata
    config_metadata = joblib.load(f'{models_dir}/model_config_and_metadata.pkl')
    
    # Convert to DataFrame and engineer features
    df = pd.DataFrame([player_data])
    
    # Create categorical encodings with all expected categories
    # Primary position encoding
    for pos in ['2B', '3B', 'SS']:  # All infield positions in P4 model
        df[f'primary_position_{pos}'] = (df['primary_position'] == pos).astype(int)
    
    # Throwing hand encoding  
    df['throwing_hand_R'] = (df['throwing_hand'] == 'R').astype(int)
    
    # Hitting handedness encoding
    df['hitting_handedness_R'] = (df['hitting_handedness'] == 'R').astype(int)
    df['hitting_handedness_S'] = (df['hitting_handedness'] == 'S').astype(int)
    
    # Player region encoding
    for region in ['Northeast', 'South', 'West']:
        df[f'player_region_{region}'] = (df['player_region'] == region).astype(int)
    
    # Drop original categorical columns
    df = df.drop(['primary_position', 'throwing_hand', 'hitting_handedness', 'player_region'], axis=1)
    
    # Feature engineering (same as training pipeline)
    df['velo_by_inf'] = df['exit_velo_max'] / df['inf_velo']
    df['power_speed'] = df['exit_velo_max'] / df['sixty_time']
    df['sixty_inv'] = 1 / df['sixty_time']
    df['height_weight'] = df['height'] * df['weight']
    
    # Add percentile features
    percentile_features = ['exit_velo_max', 'inf_velo', 'sixty_time', 'height', 'weight', 'power_speed']
    for col in percentile_features:
        if col in df.columns:
            if col == 'sixty_time':
                df[f'{col}_percentile'] = (1 - df[col].rank(pct=True)) * 100
            else:
                df[f'{col}_percentile'] = df[col].rank(pct=True) * 100
    
    # Add remaining engineered features
    df['power_per_pound'] = df['exit_velo_max'] / df['weight']
    df['exit_to_sixty_ratio'] = df['exit_velo_max'] / df['sixty_time']
    df['speed_size_efficiency'] = (df['height'] * df['weight']) / (df['sixty_time'] ** 2)
    df['athletic_index'] = (df['power_speed'] * df['height'] * df['weight']) / df['sixty_time']
    df['power_speed_index'] = df['exit_velo_max'] * (1 / df['sixty_time'])
    
    # Elite binary features
    df['elite_exit_velo'] = (df['exit_velo_max'] >= df['exit_velo_max'].quantile(0.75)).astype(int)
    df['elite_inf_velo'] = (df['inf_velo'] >= df['inf_velo'].quantile(0.75)).astype(int)
    df['elite_speed'] = (df['sixty_time'] <= df['sixty_time'].quantile(0.25)).astype(int)
    df['elite_size'] = ((df['height'] >= df['height'].quantile(0.6)) & 
                       (df['weight'] >= df['weight'].quantile(0.6))).astype(int)
    df['multi_tool_count'] = (df['elite_exit_velo'] + df['elite_inf_velo'] + 
                             df['elite_speed'] + df['elite_size'])
    
    # Scale features
    df['exit_velo_scaled'] = (df['exit_velo_max'] - df['exit_velo_max'].min()) / (df['exit_velo_max'].max() - df['exit_velo_max'].min()) * 100
    df['speed_scaled'] = (1 - (df['sixty_time'] - df['sixty_time'].min()) / (df['sixty_time'].max() - df['sixty_time'].min())) * 100
    df['arm_scaled'] = (df['inf_velo'] - df['inf_velo'].min()) / (df['inf_velo'].max() - df['inf_velo'].min()) * 100
    
    # Add D1 ensemble probability (placeholder - would come from D1 stage in practice)
    df['d1_ensemble_prob'] = 0.7  # Placeholder value
    df['d1_confidence_high'] = 1
    df['d1_confidence_medium'] = 0
    df['d1_prob_squared'] = df['d1_ensemble_prob'] ** 2
    
    # Power 4 region indicator
    df['power4_region'] = ((df.get('player_region_South', 0) == 1) | 
                          (df.get('player_region_West', 0) == 1)).astype(int)
    
    # Ensure all required features are present
    for feature in config_metadata['elite_features']:
        if feature not in df.columns:
            df[feature] = 0
    
    for feature in config_metadata['p4_features']:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select features
    elite_feats = df[config_metadata['elite_features']]
    p4_feats = df[config_metadata['p4_features']]
    
    # Clean data
    elite_feats = elite_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
    p4_feats = p4_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features for SVM
    p4_feats_scaled = scaler.transform(p4_feats)
    
    # Get elite detection probability
    elite_prob = elite_model.predict_proba(elite_feats)[0, 1]
    is_elite = elite_prob >= config_metadata['elite_threshold']
    
    # Get ensemble P4 probabilities
    xgb_prob = xgb_model.predict_proba(p4_feats)[0, 1]
    cb_prob = cb_model.predict_proba(p4_feats)[0, 1]
    lgb_prob = lgb_model.predict_proba(p4_feats)[0, 1]
    svm_prob = svm_model.predict_proba(p4_feats_scaled)[0, 1]
    
    # ELITE P4 PLAYER DETECTION AND ADAPTIVE ENSEMBLE
    # Check for elite P4 characteristics using imported constants
    elite_p4_thresholds = {
        'exit_velo_max': ELITE_EXIT_VELO_MAX,
        'inf_velo': ELITE_INF_VELO,
        'sixty_time_max': ELITE_SIXTY_TIME_INF,
        'height_min': ELITE_HEIGHT_MIN
    }
    
    elite_p4_score = 0
    elite_p4_indicators = []
    
    if player_data.get('exit_velo_max', 0) >= elite_p4_thresholds['exit_velo_max']:
        elite_p4_score += 2
        elite_p4_indicators.append(f"Elite P4 exit velocity: {player_data['exit_velo_max']} mph")
    
    if player_data.get('inf_velo', 0) >= elite_p4_thresholds['inf_velo']:
        elite_p4_score += 2
        elite_p4_indicators.append(f"Elite P4 infield velocity: {player_data['inf_velo']} mph")
    
    if player_data.get('sixty_time', 10) <= elite_p4_thresholds['sixty_time_max']:
        elite_p4_score += 2
        elite_p4_indicators.append(f"Elite P4 speed: {player_data['sixty_time']} seconds")
    
    if player_data.get('height', 0) >= elite_p4_thresholds['height_min']:
        elite_p4_score += 1
        elite_p4_indicators.append(f"Elite P4 height: {player_data['height']} inches")
    
    # Count feature outliers in P4 features
    extreme_p4_features = np.sum(np.abs(p4_feats_scaled) > 3.0)
    p4_outlier_ratio = extreme_p4_features / len(p4_feats_scaled.flatten()) if len(p4_feats_scaled.flatten()) > 0 else 0
    
    # Elite P4 classification
    is_elite_p4 = elite_p4_score >= 4
    is_super_elite_p4 = elite_p4_score >= 6
    
    # OPTION 1: CONFIDENCE-BASED WEIGHT REDISTRIBUTION
    weights = config_metadata['ensemble_weights']
    base_weights = [weights['xgboost'], weights['catboost'], weights['lightgbm'], weights['svm']]
    individual_p4_probs = [xgb_prob, cb_prob, lgb_prob, svm_prob]
    
    # Calculate confidence scores based on how far predictions are from 0.5 (uncertainty)
    confidence_scores = [2 * abs(prob - 0.5) for prob in individual_p4_probs]  # 0-1 scale
    
    # Apply confidence multipliers
    confidence_multipliers = []
    for conf_score in confidence_scores:
        if conf_score >= 0.6:  # High confidence (80%+ or 20%-)
            multiplier = 1.5
        elif conf_score >= 0.2:  # Medium confidence (60-80% or 20-40%)
            multiplier = 1.0
        else:  # Low confidence (40-60%)
            multiplier = 0.4
        confidence_multipliers.append(multiplier)
    
    # Apply confidence multipliers to base weights
    confidence_adjusted_weights = [base_weight * mult for base_weight, mult in zip(base_weights, confidence_multipliers)]
    
    # Normalize weights to sum to 1.0
    weight_sum = sum(confidence_adjusted_weights)
    confidence_adjusted_weights = [w / weight_sum for w in confidence_adjusted_weights]
    
    # Adaptive P4 ensemble weighting (keeping elite detection but adding confidence weighting)
    if is_super_elite_p4 and (xgb_prob > 0.7 or lgb_prob > 0.7):
        # Super elite P4 + confident tree models = boost tree models, reduce failing catboost/svm
        if cb_prob < 0.1:  # CatBoost is failing
            adjusted_p4_weights = [0.45, 0.05, 0.45, 0.05]  # Boost XGB+LGB, reduce CB+SVM
            strategy_p4 = 'super_elite_p4_tree_dominant'
        else:
            adjusted_p4_weights = [0.4, 0.2, 0.3, 0.1]
            strategy_p4 = 'super_elite_p4_tree_dominant'
        
    elif is_elite_p4 and (xgb_prob > 0.6 or lgb_prob > 0.6):
        # Elite P4 + decent tree models = moderate boost
        adjusted_p4_weights = [0.35, 0.25, 0.25, 0.15]
        strategy_p4 = 'elite_p4_tree_boosted'
        
    elif p4_outlier_ratio > 0.3:
        # High P4 outliers = reduce catboost weight if it's very low
        if cb_prob < 0.1:
            adjusted_p4_weights = [0.4, 0.1, 0.3, 0.2]
        else:
            adjusted_p4_weights = base_weights
        strategy_p4 = 'p4_outlier_adjusted'
        
    else:
        # Use confidence-based weighting as the standard approach
        adjusted_p4_weights = confidence_adjusted_weights
        strategy_p4 = 'confidence_based_ensemble'
    
    # Calculate P4 ensemble probability with adaptive weights
    ensemble_prob = sum(prob * weight for prob, weight in zip(individual_p4_probs, adjusted_p4_weights))
    
    # Enhanced P4 threshold logic for elite players
    threshold = config_metadata['optimal_threshold']  # Default 78%
    if is_super_elite_p4 and ensemble_prob > 0.5:
        # Lower threshold for super elite players
        adjusted_threshold = 0.5
        threshold_reason = 'lowered_for_super_elite_p4'
    elif is_elite_p4 and ensemble_prob > 0.55:
        # Slightly lower threshold for elite players  
        adjusted_threshold = 0.6
        threshold_reason = 'lowered_for_elite_p4'
    else:
        adjusted_threshold = threshold
        threshold_reason = 'standard_threshold'
    
    # Apply adaptive threshold
    p4_prediction = 1 if ensemble_prob >= adjusted_threshold else 0
    
    # Enhanced confidence calculation for elite players
    if is_super_elite_p4 and ensemble_prob > 0.6:
        confidence = 'High'
    elif ensemble_prob > 0.7 or ensemble_prob < 0.3:
        confidence = 'High'
    elif ensemble_prob > 0.6 or ensemble_prob < 0.4:
        confidence = 'Medium'
    else:
        confidence = 'Low'

    return {
        'p4_probability': float(ensemble_prob),
        'p4_prediction': int(p4_prediction),
        'confidence': confidence,
        'is_elite_candidate': bool(is_elite),  # Original elite detection
        'elite_probability': float(elite_prob),
        'elite_p4_detection': {
            'is_elite_p4': is_elite_p4,
            'is_super_elite_p4': is_super_elite_p4,
            'elite_p4_score': elite_p4_score,
            'elite_p4_indicators': elite_p4_indicators,
            'strategy_used': strategy_p4
        },
        'p4_outlier_info': {
            'extreme_features': int(extreme_p4_features),
            'outlier_ratio': float(p4_outlier_ratio)
        },
        'threshold_info': {
            'original_threshold': float(threshold),
            'adjusted_threshold': float(adjusted_threshold),
            'threshold_reason': threshold_reason
        },
        'ensemble_weights': {
            'original': dict(zip(['xgboost', 'catboost', 'lightgbm', 'svm'], base_weights)),
            'confidence_adjusted': dict(zip(['xgboost', 'catboost', 'lightgbm', 'svm'], confidence_adjusted_weights)),
            'final_adjusted': dict(zip(['xgboost', 'catboost', 'lightgbm', 'svm'], adjusted_p4_weights))
        },
        'confidence_analysis': {
            'confidence_scores': dict(zip(['xgboost', 'catboost', 'lightgbm', 'svm'], [float(x) for x in confidence_scores])),
            'confidence_multipliers': dict(zip(['xgboost', 'catboost', 'lightgbm', 'svm'], confidence_multipliers))
        },
        'model_components': {
            'xgb_prob': float(xgb_prob),
            'cb_prob': float(cb_prob), 
            'lgb_prob': float(lgb_prob),
            'svm_prob': float(svm_prob)
        },
        'model_version': 'infielder_p4_v2_elite_adaptive'
    }

if __name__ == "__main__":
    # Example usage
    test_player = {
        'height': 74.0,
        'weight': 190.0,
        'sixty_time': 6.8,
        'exit_velo_max': 92.0,
        'inf_velo': 82.0,
        'player_region': 'South',
        'throwing_hand': 'Right',
        'hitting_handedness': 'Right',
        'primary_position': 'SS'
    }
    
    result = predict_infielder_p4_probability(test_player)
    print(f"P4 Probability: {result['p4_probability']:.1%}")
    print(f"P4 Prediction: {result['p4_prediction']}")
    print(f"Confidence: {result['confidence']}")