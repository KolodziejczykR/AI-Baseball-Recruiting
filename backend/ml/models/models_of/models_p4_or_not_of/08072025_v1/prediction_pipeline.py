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
from sklearn.preprocessing import StandardScaler

def predict_outfielder_p4_probability(player_data, models_dir='/Users/ryankolodziejczyk/Documents/AI Baseball Recruitment/code/backend/ml/models/models_of/models_p4_or_not_of/v4_20250807_105039'):
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
    
    # Elite indicators
    df['elite_exit_velo'] = (df['exit_velo_max'] >= 95).astype(int)  # Use fixed thresholds for production
    df['elite_of_velo'] = (df['of_velo'] >= 85).astype(int)
    df['elite_speed'] = (df['sixty_time'] <= 6.8).astype(int)
    df['elite_size'] = (df['height'] >= 72).astype(int)
    
    df['multi_tool_count'] = (df['elite_exit_velo'] + df['elite_of_velo'] + 
                             df['elite_speed'] + df['elite_size'])
    df['is_multi_tool'] = (df['multi_tool_count'] >= 2).astype(int)
    
    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=['player_region', 'throwing_hand', 'hitting_handedness'], drop_first=False)
    
    # NOTE: D1 probability features would need to be generated from D1 model
    # For now, using placeholder values - integrate with actual D1 model in production
    df['d1_probability'] = 0.7  # Placeholder - replace with actual D1 model prediction
    
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
    
    # Weighted ensemble
    weights = config['ensemble_weights']
    ensemble_prob = (xgb_prob * weights['xgb'] + 
                    lgb_prob * weights['lgb'] + 
                    mlp_prob * weights['mlp'] + 
                    svm_prob * weights['svm'])
    
    # Apply hierarchical thresholds
    elite_thresh = config['thresholds']['elite_threshold']
    non_elite_thresh = config['thresholds']['non_elite_threshold']
    
    threshold = elite_thresh if is_elite else non_elite_thresh
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
            'lgb_prob': float(lgb_prob), 
            'mlp_prob': float(mlp_prob),
            'svm_prob': float(svm_prob)
        },
        'model_version': config['model_version']
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
