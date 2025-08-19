"""
D1 Catcher Prediction Pipeline - Clean Production Version
Performance: Meta-learner ensemble with LightGBM + DNN base models
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.utils.prediction_types import D1PredictionResult

def predict_catcher_d1_probability(player_data: dict, models_dir: str) -> D1PredictionResult:
    """
    Clean D1 probability prediction for catchers using trained ensemble
    
    Args:
        player_data (dict): Player statistics
        models_dir (str): Path to model files
    
    Returns:
        D1PredictionResult: Structured prediction result
    """
    
    # Load trained models and configuration
    lgb_model = joblib.load(f"{models_dir}/lightgbm_model.pkl")
    dnn_model = joblib.load(f"{models_dir}/dnn_model.pkl")
    meta_learner = joblib.load(f"{models_dir}/meta_learner.pkl")
    dnn_scaler = joblib.load(f"{models_dir}/dnn_scaler.pkl")
    
    # Load metadata
    metadata = joblib.load(f"{models_dir}/model_metadata.pkl")
    
    # Feature engineering (same as training)
    df = pd.DataFrame([player_data])
    
    # Categorical encodings (ensure all expected categories exist)
    # Player region encoding
    for region in ['Northeast', 'South', 'West']:
        df[f'player_region_{region}'] = (df['player_region'] == region).astype(int)
    
    # Hitting handedness encoding
    df['hitting_handedness_R'] = (df['hitting_handedness'] == 'R').astype(int)
    
    # Drop original categorical columns
    df = df.drop(['hitting_handedness', 'player_region'], axis=1)
    
    # Basic engineered features
    df['c_velo_sixty_ratio'] = df['c_velo'] / df['sixty_time']
    df['height_weight'] = df['height'] * df['weight']
    df['pop_time_c_velo_ratio'] = df['pop_time'] / df['c_velo']
    
    # Percentile features (using fixed values for single prediction)
    df['c_velo_percentile'] = 50.0  # Default percentile
    df['pop_time_percentile'] = 50.0
    df['catcher_defensive_percentile'] = 50.0
    df['catcher_offensive_percentile'] = 50.0
    df['catcher_overall_percentile'] = 50.0
    
    # Additional engineered features
    df['power_per_pound'] = df['exit_velo_max'] / df['weight']
    df['athletic_index'] = (df['exit_velo_max'] + df['c_velo'] + (100/df['sixty_time'])) / 3
    df['arm_strength_per_pound'] = df['c_velo'] / df['weight']
    df['defensive_power_combo'] = df['c_velo'] * df['exit_velo_max'] / df['pop_time']
    
    # Regional and tool features
    df['d1_region_advantage'] = 0.1 if df.get('player_region_South', [0])[0] == 1 else 0.0
    df['tool_count'] = 2  # Default for catchers
    df['athletic_index_v2'] = df['athletic_index'] * 1.1  # Enhanced version
    df['tools_athlete'] = df['tool_count'] * df['athletic_index_v2'] / 100
    df['d1_composite_score'] = (df['exit_velo_max']/100 + df['c_velo']/100 + 
                               (3-df['pop_time']) + df['athletic_index_v2']/100) / 4
    df['power_defense_balance'] = df['exit_velo_max'] * df['c_velo'] / (df['pop_time'] * 1000)
    df['athleticism_defense'] = df['athletic_index_v2'] * df['c_velo'] / df['pop_time']
    
    # Advanced composite features
    df['arm_athleticism_correlation'] = df['c_velo'] * df['athletic_index_v2'] / 100
    df['defensive_consistency'] = df['pop_time_percentile'] * df['c_velo_percentile'] * df['catcher_defensive_percentile']
    df['power_speed_size_ratio'] = df['exit_velo_max'] / (df['sixty_time'] * df['weight'] / 100)
    df['pop_efficiency'] = df['c_velo'] / (df['pop_time'] ** 2)
    df['region_athletic_adjustment'] = df['athletic_index_v2'] * (1 + df['d1_region_advantage'])
    df['region_athletic_adjustment_exp'] = np.exp(df['region_athletic_adjustment'] / 100)
    df['tool_synergy'] = df['tools_athlete'] * df['tool_count']
    df['athletic_ceiling'] = df['athletic_index_v2'] ** 1.5
    df['arm_athleticism_correlation_x_region_athletic_adjustment'] = (df['arm_athleticism_correlation'] * 
                                                                     df['region_athletic_adjustment'])
    df['tool_count_x_athletic_index'] = df['tool_count'] * df['athletic_index_v2']
    
    # Ensure all required features exist and are in correct order
    missing_features = []
    for feature in metadata['feature_columns']:
        if feature not in df.columns:
            missing_features.append(feature)
    
    if missing_features:
        raise ValueError(f"Missing required features for model prediction: {missing_features}")
    
    # Select features in correct order
    df_features = df[metadata['feature_columns']].fillna(0)
    
    # Clean data
    df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features for DNN
    df_scaled_dnn = dnn_scaler.transform(df_features)
    
    # Get predictions from base models
    lgb_prob = lgb_model.predict_proba(df_features)[0, 1]
    dnn_prob = dnn_model.predict_proba(df_scaled_dnn)[0, 1]
    
    # Create meta-features for meta-learner (same as training)
    meta_features = np.array([[
        lgb_prob,  # LGB prediction
        dnn_prob,  # DNN prediction
        lgb_prob * dnn_prob,  # LGB × DNN interaction
        (lgb_prob + dnn_prob) / 2,  # Average prediction
        ((lgb_prob + dnn_prob) / 2) ** 2,  # Average prediction squared
        abs(lgb_prob - dnn_prob),  # Prediction uncertainty
        max(lgb_prob, dnn_prob),  # Max confidence
        min(lgb_prob, dnn_prob),  # Min confidence
        abs(lgb_prob - dnn_prob),  # Model disagreement
        lgb_prob ** 2,  # LGB squared
        lgb_prob ** 3,  # LGB cubed
        lgb_prob ** 4,  # LGB fourth power
        np.exp(lgb_prob)  # LGB exponential
    ]])
    
    # Get final prediction from meta-learner
    final_prob = meta_learner.predict_proba(meta_features)[0, 1]
    
    # Apply trained threshold
    threshold = metadata['optimal_threshold']
    d1_prediction = final_prob >= threshold
    
    # Combined confidence: ensemble agreement + boundary distance (adjusted for 2-model ensemble)
    individual_probs = [lgb_prob, dnn_prob]
    
    # Agreement: More lenient for 2-model ensemble
    agreement_score = max(0, 1 - np.std(individual_probs) * 2.0)  # Reduced multiplier
    
    # Boundary distance: Far from 0.5 = high confidence  
    boundary_confidence = 2 * abs(final_prob - 0.5)
    
    # Combined confidence mean
    combined_confidence = (agreement_score + boundary_confidence) / 2
    
    # More realistic thresholds for 2-model meta-learner
    if combined_confidence > 0.7:
        confidence = 'High'
    elif combined_confidence > 0.4:
        confidence = 'Medium'
    else:
        confidence = 'Low'
    
    return D1PredictionResult(
        d1_probability=float(final_prob),
        d1_prediction=bool(d1_prediction),
        confidence=confidence,
        model_version=metadata.get('model_type', 'catcher_d1_meta_learner_08182025')
    )


if __name__ == "__main__":
    # Example usage
    test_player = {
        'height': 72.0,
        'weight': 190.0,
        'sixty_time': 7.2,
        'c_velo': 78.0,
        'pop_time': 2.0,
        'exit_velo_max': 92.0,
        'hitting_handedness': 'R',
        'player_region': 'South'
    }
    
    models_dir = os.path.dirname(__file__)
    result = predict_catcher_d1_probability(test_player, models_dir)
    
    print(f"D1 Prediction: {result.d1_prediction} ({result.d1_probability:.1%})")
    print(f"Confidence: {result.confidence}")
    print(f"Model Version: {result.model_version}")