#!/usr/bin/env python3
"""
P4 Outfielder Prediction Pipeline - Production Version
Generated: 2025-08-07T10:50:39.926729
Performance: 73.3% accuracy, 69.3% P4 recall
"""

import pandas as pd
import numpy as np
import joblib
import json
import sys
import os
from sklearn.preprocessing import StandardScaler

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.utils.elite_weighting_constants import (
    ELITE_EXIT_VELO_MAX, ELITE_OF_VELO, ELITE_SIXTY_TIME_OF, ELITE_HEIGHT_MIN
)

def predict_outfielder_p4_probability(player_data, models_dir='/Users/ryankolodziejczyk/Documents/AI Baseball Recruitment/code/backend/ml/models/models_of/models_p4_or_not_of/v4_20250807_105039', d1_probability=None):
    """
    Predict P4 college probability for outfielder
    
    Args:
        player_data (dict): Player statistics
        {
            'height': float,          # inches
            'weight': float,          # pounds  
            'sixty_time': float,      # seconds
            'exit_velo_max': float,   # mph
            'of_velo': float,         # mph (outfield velocity)
            'player_region': str,     # Geographic region
            'throwing_hand': str,     # 'Left' or 'Right'
            'hitting_handedness': str # 'Left', 'Right', or 'Switch'
        }
        models_dir (str): Path to model files
    
    Returns:
        dict: Prediction results
    """
    
    # Load models and config
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
    
    # Convert to DataFrame and engineer features
    df = pd.DataFrame([player_data])
    
    # Basic feature engineering (same as training)
    df['power_speed'] = df['exit_velo_max'] * (7.6 - df['sixty_time'])
    df['of_velo_sixty_ratio'] = df['of_velo'] / df['sixty_time']
    df['height_weight'] = df['height'] * df['weight']
    df['power_per_pound'] = df['exit_velo_max'] / df['weight']
    df['exit_to_sixty_ratio'] = df['exit_velo_max'] / df['sixty_time']
    df['speed_size_efficiency'] = df['height'] / df['sixty_time']
    df['athletic_index'] = (df['exit_velo_max'] + df['of_velo'] + (100 - df['sixty_time']*10)) / 3
    df['power_speed_index'] = df['power_speed'] / df['athletic_index']
    df['exit_velo_body'] = df['exit_velo_max'] / df['height_weight']
    
    # Elite indicators using proper constants
    df['elite_exit_velo'] = (df['exit_velo_max'] >= ELITE_EXIT_VELO_MAX).astype(int)
    df['elite_of_velo'] = (df['of_velo'] >= ELITE_OF_VELO).astype(int)
    df['elite_speed'] = (df['sixty_time'] <= ELITE_SIXTY_TIME_OF).astype(int)
    df['elite_size'] = (df['height'] >= ELITE_HEIGHT_MIN).astype(int)
    
    df['multi_tool_count'] = (df['elite_exit_velo'] + df['elite_of_velo'] + 
                             df['elite_speed'] + df['elite_size'])
    df['is_multi_tool'] = (df['multi_tool_count'] >= 2).astype(int)
    
    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=['player_region', 'throwing_hand', 'hitting_handedness'], drop_first=False)
    
    # Use actual D1 probability from D1 stage, or fallback to placeholder
    if d1_probability is not None:
        df['d1_probability'] = d1_probability
    else:
        df['d1_probability'] = 0.7  # Fallback placeholder
    
    # D1-based features
    df['d1_prob_size'] = df['d1_probability'] * df['height_weight']
    df['d1_size_speed'] = df['d1_probability'] * df['speed_size_efficiency']
    df['d1_squared'] = df['d1_probability'] ** 2
    df['d1_athletic_index'] = df['d1_probability'] * df['athletic_index']
    df['d1_exit_velo'] = df['d1_probability'] * df['exit_velo_max']
    df['d1_power_per_pound'] = df['d1_probability'] * df['power_per_pound']
    df['d1_speed_size'] = df['d1_probability'] * df['speed_size_efficiency']
    
    # Additional required features
    df['p4_region_bonus'] = 0  # Calculate based on region if needed
    df['exit_velo_body'] = df['exit_velo_max'] / df['height_weight']
    df['p4_among_high_d1'] = 0.0  # Placeholder
    df['tool_count'] = df['multi_tool_count']  # Alias
    df['athletic_index_v2'] = df['athletic_index'] * (1 + df['tool_count'] * 0.1)
    df['d1_composite_score'] = (df['exit_velo_max']/100 + df['of_velo']/100 + 
                               (7-df['sixty_time']) + df['athletic_index_v2']/100) / 4
    
    # Ensure all required features exist
    for feature in feature_metadata['features']:
        if feature not in df.columns:
            df[feature] = 0  # Default value
    
    # Select features in correct order
    X = df[feature_metadata['features']].fillna(0)
    X_scaled = scaler.transform(X)
    
    # Elite detection
    elite_prob = elite_model.predict_proba(X)[0, 1]
    is_elite = elite_prob >= 0.5
    
    # Get ensemble predictions
    xgb_prob = xgb_model.predict_proba(X.values)[0, 1]
    lgb_prob = lgb_model.predict_proba(X)[0, 1]
    mlp_prob = mlp_model.predict_proba(X_scaled)[0, 1]
    svm_prob = svm_model.predict_proba(X_scaled)[0, 1]
    
    # ELITE P4 OUTFIELDER DETECTION AND ADAPTIVE ENSEMBLE
    # Check for elite P4 outfielder characteristics using imported constants
    elite_p4_of_thresholds = {
        'exit_velo_max': ELITE_EXIT_VELO_MAX,
        'of_velo': ELITE_OF_VELO,
        'sixty_time_max': ELITE_SIXTY_TIME_OF,
        'height_min': ELITE_HEIGHT_MIN
    }
    
    elite_p4_of_score = 0
    elite_p4_of_indicators = []
    
    if player_data.get('exit_velo_max', 0) >= elite_p4_of_thresholds['exit_velo_max']:
        elite_p4_of_score += 2
        elite_p4_of_indicators.append(f"Elite P4 OF exit velocity: {player_data['exit_velo_max']} mph")
    
    if player_data.get('of_velo', 0) >= elite_p4_of_thresholds['of_velo']:
        elite_p4_of_score += 2
        elite_p4_of_indicators.append(f"Elite P4 OF outfield velocity: {player_data['of_velo']} mph")
    
    if player_data.get('sixty_time', 10) <= elite_p4_of_thresholds['sixty_time_max']:
        elite_p4_of_score += 2
        elite_p4_of_indicators.append(f"Elite P4 OF speed: {player_data['sixty_time']} seconds")
    
    if player_data.get('height', 0) >= elite_p4_of_thresholds['height_min']:
        elite_p4_of_score += 1
        elite_p4_of_indicators.append(f"Elite P4 OF height: {player_data['height']} inches")
    
    # Count feature outliers in P4 OF features
    extreme_p4_of_features = np.sum(np.abs(X_scaled) > 3.0)
    p4_of_outlier_ratio = extreme_p4_of_features / len(X_scaled.flatten()) if len(X_scaled.flatten()) > 0 else 0
    
    # Elite P4 OF classification
    is_elite_p4_of = elite_p4_of_score >= 4
    is_super_elite_p4_of = elite_p4_of_score >= 6
    
    # CONFIDENCE-BASED WEIGHT REDISTRIBUTION FOR P4 OUTFIELDERS
    weights = config['ensemble_weights']
    base_of_weights = [weights['xgb'], weights['lgb'], weights['mlp'], weights['svm']]
    individual_p4_of_probs = [xgb_prob, lgb_prob, mlp_prob, svm_prob]
    
    # Calculate confidence scores based on how far predictions are from 0.5 (uncertainty)
    p4_of_confidence_scores = [2 * abs(prob - 0.5) for prob in individual_p4_of_probs]  # 0-1 scale
    
    # Apply confidence multipliers
    p4_of_confidence_multipliers = []
    for conf_score in p4_of_confidence_scores:
        if conf_score >= 0.6:  # High confidence (80%+ or 20%-)
            multiplier = 1.5
        elif conf_score >= 0.2:  # Medium confidence (60-80% or 20-40%)
            multiplier = 1.0
        else:  # Low confidence (40-60%)
            multiplier = 0.4
        p4_of_confidence_multipliers.append(multiplier)
    
    # Apply confidence multipliers to base weights
    p4_of_confidence_adjusted_weights = [base_weight * mult for base_weight, mult in zip(base_of_weights, p4_of_confidence_multipliers)]
    
    # Normalize weights to sum to 1.0
    p4_of_weight_sum = sum(p4_of_confidence_adjusted_weights)
    p4_of_confidence_adjusted_weights = [w / p4_of_weight_sum for w in p4_of_confidence_adjusted_weights]
    
    # Adaptive P4 OF ensemble weighting (keeping elite detection but adding confidence weighting)
    if is_super_elite_p4_of and (mlp_prob > 0.2 or svm_prob > 0.3):
        # Super elite P4 OF + confident neural/SVM models = boost them since tree models struggle with elite OF
        if mlp_prob > svm_prob:
            adjusted_p4_of_weights = [0.1, 0.1, 0.7, 0.1]  # Boost MLP heavily
        else:
            adjusted_p4_of_weights = [0.1, 0.1, 0.3, 0.5]  # Boost SVM heavily  
        strategy_p4_of = 'super_elite_p4_of_neural_dominant'
        
    elif is_elite_p4_of and (mlp_prob > 0.15 or svm_prob > 0.25):
        # Elite P4 OF + decent neural/SVM models = moderate boost
        adjusted_p4_of_weights = [0.15, 0.15, 0.4, 0.3]
        strategy_p4_of = 'elite_p4_of_neural_boosted'
        
    elif p4_of_outlier_ratio > 0.3:
        # High P4 OF outliers = reduce poorly performing models
        if mlp_prob < 0.1:
            adjusted_p4_of_weights = [0.4, 0.3, 0.1, 0.2]
        else:
            adjusted_p4_of_weights = base_of_weights
        strategy_p4_of = 'p4_of_outlier_adjusted'
        
    else:
        # Use confidence-based weighting as the standard approach
        adjusted_p4_of_weights = p4_of_confidence_adjusted_weights
        strategy_p4_of = 'confidence_based_p4_of_ensemble'
    
    # Calculate P4 OF ensemble probability with adaptive weights
    ensemble_prob = sum(prob * weight for prob, weight in zip(individual_p4_of_probs, adjusted_p4_of_weights))
    
    # Enhanced P4 OF threshold logic for elite players
    elite_thresh = config['thresholds']['elite_threshold']
    non_elite_thresh = config['thresholds']['non_elite_threshold']
    
    if is_super_elite_p4_of and ensemble_prob > 0.25:
        # Lower threshold significantly for super elite P4 OF players (OF models seem to predict lower than INF)
        adjusted_threshold = 0.3
        threshold_reason = 'lowered_for_super_elite_p4_of'
    elif is_elite_p4_of and ensemble_prob > 0.2:
        # Lower threshold for elite P4 OF players  
        adjusted_threshold = 0.35
        threshold_reason = 'lowered_for_elite_p4_of'
    else:
        adjusted_threshold = elite_thresh if is_elite else non_elite_thresh
        threshold_reason = 'standard_threshold'
    
    # Apply adaptive threshold
    p4_prediction = 1 if ensemble_prob >= adjusted_threshold else 0
    
    # Enhanced confidence calculation for elite P4 OF players
    if is_super_elite_p4_of and ensemble_prob > 0.6:
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
        'elite_p4_of_detection': {
            'is_elite_p4_of': is_elite_p4_of,
            'is_super_elite_p4_of': is_super_elite_p4_of,
            'elite_p4_of_score': elite_p4_of_score,
            'elite_p4_of_indicators': elite_p4_of_indicators,
            'strategy_used': strategy_p4_of
        },
        'p4_of_outlier_info': {
            'extreme_features': int(extreme_p4_of_features),
            'outlier_ratio': float(p4_of_outlier_ratio)
        },
        'threshold_info': {
            'original_threshold': float(elite_thresh if is_elite else non_elite_thresh),
            'adjusted_threshold': float(adjusted_threshold),
            'threshold_reason': threshold_reason
        },
        'ensemble_weights': {
            'original': dict(zip(['xgb', 'lgb', 'mlp', 'svm'], base_of_weights)),
            'confidence_adjusted': dict(zip(['xgb', 'lgb', 'mlp', 'svm'], p4_of_confidence_adjusted_weights)),
            'final_adjusted': dict(zip(['xgb', 'lgb', 'mlp', 'svm'], adjusted_p4_of_weights))
        },
        'confidence_analysis': {
            'confidence_scores': dict(zip(['xgb', 'lgb', 'mlp', 'svm'], [float(x) for x in p4_of_confidence_scores])),
            'confidence_multipliers': dict(zip(['xgb', 'lgb', 'mlp', 'svm'], p4_of_confidence_multipliers))
        },
        'model_components': {
            'xgb_prob': float(xgb_prob),
            'lgb_prob': float(lgb_prob), 
            'mlp_prob': float(mlp_prob),
            'svm_prob': float(svm_prob)
        },
        'model_version': config['model_version'] + '_elite_adaptive'
    }

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
    
    result = predict_outfielder_p4_probability(test_player)
    print(f"P4 Probability: {result['p4_probability']:.1%}")
    print(f"P4 Prediction: {result['p4_prediction']}")
    print(f"Confidence: {result['confidence']}")
