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
    
    # Training data quantiles for percentile calculation
    exit_velo_max_quantiles = [61.0, 81.325, 84.0, 86.0, 87.0, 88.2, 89.1, 90.0, 90.7, 91.4, 92.1, 92.9, 93.6, 94.3, 95.0, 95.9, 96.8, 97.8, 99.1, 101.2, 121.7]
    of_velo_quantiles = [51.0, 73.0, 76.0, 77.0, 79.0, 80.0, 81.0, 81.0, 82.0, 83.0, 83.0, 84.0, 85.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 92.0, 101.0]
    sixty_time_quantiles = [3.94, 6.57, 6.65, 6.72, 6.77, 6.82, 6.86, 6.9, 6.94, 6.98, 7.01, 7.06, 7.1, 7.14, 7.19, 7.23580539226532, 7.292672157287598, 7.36, 7.46, 7.5947386837005615, 9.17]
    height_quantiles = [62.0, 68.0, 69.0, 69.0, 70.0, 70.0, 70.0, 71.0, 71.0, 71.0, 72.0, 72.0, 72.0, 72.0, 73.0, 73.0, 74.0, 74.0, 75.0, 75.0, 83.0]
    weight_quantiles = [110.0, 150.0, 155.0, 160.0, 163.0, 165.0, 170.0, 170.0, 172.8, 175.0, 175.0, 180.0, 180.0, 185.0, 185.0, 190.0, 190.0, 195.0, 200.0, 206.325, 255.0]
    power_speed_quantiles = [7.625, 11.040981092730975, 11.522471728071029, 11.842382855873863, 12.10081362660295, 12.328530236892538, 12.522441445882583, 12.678821879382891, 12.841068917018283, 12.98642765310893, 13.143878448089971, 13.293593835568943, 13.4375, 13.582675748926679, 13.732455929469667, 13.904494382022472, 14.08284023668639, 14.298610951406296, 14.600150297580598, 15.029761904761905, 21.065989847715738]
    
    # Calculate actual percentiles using training data quantiles
    df['exit_velo_max_percentile'] = calculate_percentile_from_quantiles(df['exit_velo_max'].iloc[0], exit_velo_max_quantiles, lower_is_better=False)
    df['of_velo_percentile'] = calculate_percentile_from_quantiles(df['of_velo'].iloc[0], of_velo_quantiles, lower_is_better=False)
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
