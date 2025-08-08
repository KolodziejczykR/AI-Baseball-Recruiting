import pandas as pd
import numpy as np
import joblib
import os

def predict_infielder_d1_probability(player_data, models_dir):
    """
    Predict D1 probability for an infielder using ensemble model.
    
    Args:
        player_data (dict): Dictionary containing player features
        models_dir (str): Path to the saved models directory
    
    Returns:
        dict: Prediction results including probability and components
    """
    
    # Load ensemble metadata
    ensemble_metadata = joblib.load(f"{models_dir}/ensemble_metadata.pkl")
    
    # Load individual models
    xgb_model = joblib.load(f"{models_dir}/xgboost_model.pkl")
    lgb_model = joblib.load(f"{models_dir}/lightgbm_model.pkl")
    cb_model = joblib.load(f"{models_dir}/catboost_model.pkl")
    svm_model = joblib.load(f"{models_dir}/svm_model.pkl")
    
    # Load scaler
    scaler = joblib.load(f"{models_dir}/ensemble_scaler.pkl")
    
    # Convert player data to DataFrame
    df = pd.DataFrame([player_data])
    
    # Create categorical encodings with all expected categories
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
    
    # Ensure all required features are present
    for feature in ensemble_metadata['feature_columns']:
        if feature not in df.columns:
            df[feature] = 0  # Default value for missing features
    
    # Select features in correct order
    df_features = df[ensemble_metadata['feature_columns']]
    
    # Clean data
    df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features
    df_scaled = scaler.transform(df_features)
    
    # Get predictions from each model
    xgb_proba = xgb_model.predict_proba(df_scaled)[0, 1]
    lgb_proba = lgb_model.predict_proba(df_scaled)[0, 1]  
    cb_proba = cb_model.predict_proba(df_scaled)[0, 1]
    svm_proba = svm_model.predict_proba(df_scaled)[0, 1]
    
    # Calculate weighted ensemble probability
    weights = ensemble_metadata['weights']
    ensemble_prob = (xgb_proba * weights[0] + 
                    lgb_proba * weights[1] + 
                    cb_proba * weights[2] + 
                    svm_proba * weights[3])
    
    # Final prediction (ensemble uses soft voting, so threshold is 0.5)
    prediction = ensemble_prob >= 0.5
    
    return {
        'player_id': player_data.get('player_id', 'unknown'),
        'd1_probability': float(ensemble_prob),
        'd1_prediction': bool(prediction),
        'confidence_level': 'high' if abs(ensemble_prob - 0.5) > 0.3 else 'medium' if abs(ensemble_prob - 0.5) > 0.15 else 'low',
        'components': {
            'individual_models': {
                'xgboost': float(xgb_proba),
                'lightgbm': float(lgb_proba),
                'catboost': float(cb_proba),
                'svm': float(svm_proba)
            }
        },
        'threshold_used': 0.5,
        'model_version': ensemble_metadata.get('model_type', 'ensemble_v1')
    }