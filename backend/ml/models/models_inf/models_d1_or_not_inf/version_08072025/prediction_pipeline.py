#!/usr/bin/env python3
"""
D1 Infielder Prediction Pipeline - Clean Production Version
Performance: Ensemble model with weighted soft voting
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

def predict_infielder_d1_probability(player_data: dict, models_dir: str) -> D1PredictionResult:
    """
    Clean D1 probability prediction for infielders using trained ensemble
    
    Args:
        player_data (dict): Player statistics
        models_dir (str): Path to model files
    
    Returns:
        D1PredictionResult: Structured prediction result
    """
    
    # Load trained models and configuration
    xgb_model = joblib.load(f"{models_dir}/xgboost_model.pkl")
    lgb_model = joblib.load(f"{models_dir}/lightgbm_model.pkl")
    cb_model = joblib.load(f"{models_dir}/catboost_model.pkl")
    svm_model = joblib.load(f"{models_dir}/svm_model.pkl")
    scaler = joblib.load(f"{models_dir}/ensemble_scaler.pkl")
    
    # Load ensemble metadata
    ensemble_metadata = joblib.load(f"{models_dir}/ensemble_metadata.pkl")
    
    # Feature engineering (same as training)
    df = pd.DataFrame([player_data])
    
    # Categorical encodings (ensure all expected categories exist)
    # Primary position encoding
    for pos in ['2B', 'SS']:  # Only these positions are in the model
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
    df['exit_and_inf_velo_ss'] = ((df['primary_position_SS'] == 1) & 
                                  (df['exit_velo_max'] > 90) & 
                                  (df['inf_velo'] > 80)).astype(int)
    df['west_coast_ss'] = ((df['primary_position_SS'] == 1) & 
                          (df['player_region_West'] == 1)).astype(int)
    df['all_around_ss'] = ((df['primary_position_SS'] == 1) & 
                          (df['exit_velo_max'] > 88) & 
                          (df['inf_velo'] > 78) & 
                          (df['sixty_time'] < 7.0)).astype(int)
    df['inf_velo_x_velo_by_inf'] = df['inf_velo'] * df['velo_by_inf']
    df['inf_velo_sq'] = df['inf_velo'] ** 2
    df['velo_by_inf_sq'] = df['velo_by_inf'] ** 2
    df['inf_velo_x_velo_by_inf_sq'] = df['inf_velo'] * (df['velo_by_inf'] ** 2)
    df['inf_velo_x_velo_by_inf_cubed'] = df['inf_velo'] * (df['velo_by_inf'] ** 3)
    df['exit_inf_velo_inv'] = 1 / (df['exit_velo_max'] + df['inf_velo'])
    df['inf_velo_sixty_ratio'] = df['inf_velo'] / df['sixty_time']
    df['inf_velo_sixty_ratio_sq'] = df['inf_velo_sixty_ratio'] ** 2
    
    # Ensure all required features exist and are in correct order
    missing_features = []
    for feature in ensemble_metadata['feature_columns']:
        if feature not in df.columns:
            missing_features.append(feature)
    
    if missing_features:
        raise ValueError(f"Missing required features for model prediction: {missing_features}")
    
    # Select features in correct order
    df_features = df[ensemble_metadata['feature_columns']].fillna(0)
    
    # Clean data
    df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features
    df_scaled = scaler.transform(df_features)
    
    # Get ensemble D1 predictions using trained models
    xgb_prob = xgb_model.predict_proba(df_scaled)[0, 1]
    lgb_prob = lgb_model.predict_proba(df_scaled)[0, 1]  
    cb_prob = cb_model.predict_proba(df_scaled)[0, 1]
    svm_prob = svm_model.predict_proba(df_scaled)[0, 1]
    
    # Apply trained ensemble weights (no dynamic adjustment)
    weights = ensemble_metadata['weights']
    ensemble_prob = (xgb_prob * weights[0] + 
                    lgb_prob * weights[1] + 
                    cb_prob * weights[2] + 
                    svm_prob * weights[3])
    
    # Apply trained threshold (soft voting uses 0.5)
    threshold = 0.5
    d1_prediction = ensemble_prob >= threshold
    
    # Combined confidence: ensemble agreement + boundary distance (adjusted for diverse models)
    individual_probs = [xgb_prob, lgb_prob, cb_prob, svm_prob]
    
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
        d1_probability=float(ensemble_prob),
        d1_prediction=bool(d1_prediction),
        confidence=confidence,
        model_version=ensemble_metadata.get('model_type', 'infielder_d1_ensemble_08072025')
    )

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
    
    models_dir = os.path.dirname(__file__)
    result = predict_infielder_d1_probability(test_player, models_dir)
    
    print(f"D1 Prediction: {result.d1_prediction} ({result.d1_probability:.1%})")
    print(f"Confidence: {result.confidence}")
    print(f"Model Version: {result.model_version}")