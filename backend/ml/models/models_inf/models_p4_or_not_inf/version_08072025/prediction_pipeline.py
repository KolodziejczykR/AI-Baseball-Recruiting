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
    
    # Calculate ensemble probability
    weights = config_metadata['ensemble_weights']
    ensemble_prob = (xgb_prob * weights['xgboost'] + 
                    cb_prob * weights['catboost'] + 
                    lgb_prob * weights['lightgbm'] + 
                    svm_prob * weights['svm'])
    
    # Apply hierarchical thresholds
    threshold = config_metadata['optimal_threshold']
    p4_prediction = 1 if ensemble_prob >= threshold else 0
    
    return {
        'p4_probability': float(ensemble_prob),
        'p4_prediction': int(p4_prediction),
        'confidence': 'High' if abs(ensemble_prob - 0.5) > 0.3 else 'Medium' if abs(ensemble_prob - 0.5) > 0.15 else 'Low',
        'is_elite_candidate': bool(is_elite),
        'elite_probability': float(elite_prob),
        'threshold_used': float(threshold),
        'model_components': {
            'xgb_prob': float(xgb_prob),
            'cb_prob': float(cb_prob), 
            'lgb_prob': float(lgb_prob),
            'svm_prob': float(svm_prob)
        },
        'model_version': 'infielder_p4_v1'
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