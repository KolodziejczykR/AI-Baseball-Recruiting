#!/usr/bin/env python3
"""
D1 Outfielder Prediction Pipeline - Clean Production Version
Performance: 74.9% accuracy, 55.4% D1 recall
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

from backend.utils.prediction_types import D1PredictionResult

def predict_outfielder_d1_probability(player_data: dict, models_dir: str) -> D1PredictionResult:
    """
    Clean D1 probability prediction for outfielders using trained ensemble
    
    Args:
        player_data (dict): Player statistics
        models_dir (str): Path to model files
    
    Returns:
        D1PredictionResult: Structured prediction result
    """
    
    # Load trained models and configuration
    elite_model = joblib.load(f"{models_dir}/elite_model.pkl")
    xgb_model = joblib.load(f"{models_dir}/xgb_full_model.pkl")
    dnn_model = joblib.load(f"{models_dir}/dnn_full_model.pkl")
    lgb_model = joblib.load(f"{models_dir}/lgb_full_model.pkl")
    svm_model = joblib.load(f"{models_dir}/svm_full_model.pkl")
    scaler = joblib.load(f"{models_dir}/scaler_full.pkl")
    
    with open(f"{models_dir}/model_config.json", 'r') as f:
        config = json.load(f)
    
    with open(f"{models_dir}/feature_metadata.json", 'r') as f:
        feature_meta = json.load(f)
    
    # Feature engineering (same as training)
    df = pd.DataFrame([player_data])
    
    df['throwing_hand_R'] = (df['throwing_hand'] == 'R').astype(int)
    df['hitting_handedness_R'] = (df['hitting_handedness'] == 'R').astype(int)
    df['hitting_handedness_S'] = (df['hitting_handedness'] == 'S').astype(int)
    
    for region in ['Northeast', 'South', 'West']:
        df[f'player_region_{region}'] = (df['player_region'] == region).astype(int)
    
    # Drop original categorical columns
    df = df.drop(['throwing_hand', 'hitting_handedness', 'player_region'], axis=1)

    # Basic engineered features
    df['power_speed'] = df['exit_velo_max'] / df['sixty_time']
    df['of_velo_sixty_ratio'] = df['of_velo'] / df['sixty_time']
    df['height_weight'] = df['height'] * df['weight']
    
    # Percentile features (using fixed values for single prediction)
    df['exit_velo_max_percentile'] = 50.0  # Default percentile
    df['of_velo_percentile'] = 50.0
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
    
    # Elite binary features (values taken from trained model cutoffs via large data)
    df['elite_exit_velo'] = (df['exit_velo_max'] >= 96).astype(int)
    df['elite_of_velo'] = (df['of_velo'] >= 87).astype(int)
    df['elite_speed'] = (df['sixty_time'] <= 6.82).astype(float)
    df['elite_size'] = (df['height'] >= 72).astype(int)
    df['multi_tool_count'] = (df['elite_exit_velo'] + df['elite_of_velo'] + 
                             df['elite_speed'] + df['elite_size'])
    
    # Scaled features (using approximate scaling)
    df['exit_velo_scaled'] = np.clip((df['exit_velo_max'] - 75) / (105 - 75) * 100, 0, 100)
    df['speed_scaled'] = np.clip((1 - (df['sixty_time'] - 6.0) / (8.5 - 6.0)) * 100, 0, 100)
    df['arm_scaled'] = np.clip((df['of_velo'] - 70) / (95 - 70) * 100, 0, 100)
    
    # Additional D1 features from training
    df['d1_region_advantage'] = 0
    if 'player_region_South' in df.columns:
        df['d1_region_advantage'] += df['player_region_South'] * 0.1
    if 'player_region_West' in df.columns:
        df['d1_region_advantage'] += df['player_region_West'] * 0.08
    
    df['of_arm_strength'] = df['of_velo']
    df['of_arm_plus'] = (df['of_velo'] - 80) / 5.0  # Plus scale approximation
    df['exit_velo_elite'] = df['elite_exit_velo']
    df['speed_elite'] = df['elite_speed']
    
    # D1 threshold features
    df['d1_exit_velo_threshold'] = (df['exit_velo_max'] >= 90).astype(int)
    df['d1_arm_threshold'] = (df['of_velo'] >= 85).astype(int)
    df['d1_speed_threshold'] = (df['sixty_time'] <= 6.8).astype(int)
    df['d1_size_threshold'] = (df['height'] >= 72).astype(int)
    
    # Tool counting features
    df['tool_count'] = df['multi_tool_count']
    df['is_multi_tool'] = (df['tool_count'] >= 2).astype(int)
    df['athletic_index_v2'] = df['athletic_index'] * (1 + df['tool_count'] * 0.1)
    df['tools_athlete'] = df['tool_count'] * df['athletic_index_v2']
    
    # Composite D1 score
    df['d1_composite_score'] = (
        df['exit_velo_max']/100 + 
        df['of_velo']/100 + 
        (7-df['sixty_time']) + 
        df['athletic_index_v2']/100
    ) / 4
    
    # Ensure all required features exist and are in correct order
    missing_elite_features = []
    for feature in feature_meta['elite_features']:
        if feature not in df.columns:
            missing_elite_features.append(feature)
    
    missing_d1_features = []
    for feature in feature_meta['all_features']:
        if feature not in df.columns:
            missing_d1_features.append(feature)
    
    if missing_elite_features:
        raise ValueError(f"Missing required elite features for model prediction: {missing_elite_features}")
    
    if missing_d1_features:
        raise ValueError(f"Missing required D1 features for model prediction: {missing_d1_features}")
    
    # Select features
    elite_feats = df[feature_meta['elite_features']].fillna(0)
    d1_feats = df[feature_meta['all_features']].fillna(0)
    
    # Clean data
    elite_feats = elite_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
    d1_feats = d1_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features for neural network and SVM
    d1_feats_scaled = scaler.transform(d1_feats)
    
    # Elite detection (for hierarchical combination)
    elite_prob = elite_model.predict_proba(elite_feats)[0, 1]
    is_elite = elite_prob >= config['elite_threshold']
    
    # Get ensemble D1 predictions using trained models
    xgb_prob = xgb_model.predict_proba(d1_feats)[0, 1]
    dnn_prob = dnn_model.predict_proba(d1_feats_scaled)[0, 1]
    lgb_prob = lgb_model.predict_proba(d1_feats)[0, 1]
    svm_prob = svm_model.predict_proba(d1_feats_scaled)[0, 1]
    
    # Apply trained ensemble weights (no dynamic adjustment)
    weights = config['ensemble_weights']
    ensemble_prob = (xgb_prob * weights['XGB'] + 
                    dnn_prob * weights['DNN'] + 
                    lgb_prob * weights['LGB'] + 
                    svm_prob * weights['SVM'])
    
    # Apply hierarchical combination using trained weights
    hierarchical_weights = config['hierarchical_weights']
    final_prob = (elite_prob * hierarchical_weights['elite_weight'] + 
                  ensemble_prob * hierarchical_weights['ensemble_weight'])
    
    # Apply trained threshold
    threshold = config['optimal_prediction_threshold']
    d1_prediction = final_prob >= threshold
    
    # Combined confidence: ensemble agreement + boundary distance
    individual_probs = [xgb_prob, dnn_prob, lgb_prob, svm_prob]
    
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
    
    return D1PredictionResult(
        d1_probability=float(final_prob),
        d1_prediction=bool(d1_prediction),
        confidence=confidence,
        model_version=config.get('model_version', 'outfielder_d1_08072025')
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
    result = predict_outfielder_d1_probability(test_player, models_dir)
    
    print(f"D1 Prediction: {result.d1_prediction} ({result.d1_probability:.1%})")
    print(f"Confidence: {result.confidence}")
    print(f"Model Version: {result.model_version}")
