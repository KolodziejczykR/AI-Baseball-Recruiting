import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def predict_catcher_d1_probability(player_data, models_dir):
    """
    Predict D1 probability for a catcher using meta-learner ensemble model.
    
    Args:
        player_data (dict): Dictionary containing player features
        models_dir (str): Path to the saved models directory
    
    Returns:
        dict: Prediction results including probability and components
    """
    
    # Load model metadata
    metadata = joblib.load(f"{models_dir}/model_metadata.pkl")
    
    # Load base models
    lgb_model = joblib.load(f"{models_dir}/lightgbm_model.pkl")
    dnn_model = joblib.load(f"{models_dir}/dnn_model.pkl")
    
    # Load meta-learner if available
    meta_learner = None
    if metadata['ensemble_type'] == 'meta_learner':
        meta_learner = joblib.load(f"{models_dir}/meta_learner.pkl")
    
    # Load scalers if they exist
    dnn_scaler = None
    if os.path.exists(f"{models_dir}/dnn_scaler.pkl"):
        dnn_scaler = joblib.load(f"{models_dir}/dnn_scaler.pkl")
    
    # Convert player data to DataFrame
    df = pd.DataFrame([player_data])
    
    # Create categorical encodings with all expected categories
    # Throwing hand encoding  
    df['throwing_hand_L'] = (df['throwing_hand'] == 'L').astype(int)
    
    # Hitting handedness encoding
    df['hitting_handedness_L'] = (df['hitting_handedness'] == 'L').astype(int)
    df['hitting_handedness_R'] = (df['hitting_handedness'] == 'R').astype(int)
    
    # Player region encoding
    for region in ['Midwest', 'Northeast', 'South', 'West']:
        df[f'player_region_{region}'] = (df['player_region'] == region).astype(int)
    
    # Drop original categorical columns
    df = df.drop(['throwing_hand', 'hitting_handedness', 'player_region'], axis=1)
    
    # Feature engineering (catcher-specific features based on training pipeline)
    # Height and weight percentiles
    df['height_percentile'] = 50  # Default values - would need population stats for accurate percentiles
    df['weight_percentile'] = 50
    df['c_velo_percentile'] = 50
    df['pop_time_percentile'] = 50
    df['catcher_defensive_percentile'] = 50
    
    # Athletic indices
    df['athletic_index_v2'] = (df['exit_velo_max'] * 0.3 + 
                              (1/df['sixty_time']) * 100 * 0.2 + 
                              df['c_velo'] * 0.25 + 
                              (1/df['pop_time']) * 0.15 + 
                              df['height'] * 0.1)
    
    # Multi-tool features
    df['multi_tool_count'] = 2  # Default
    df['tools_athlete'] = 1    # Default
    df['complete_catcher'] = 1 # Default
    
    # Velocity ratios
    df['exit_velo_over_c_velo'] = df['exit_velo_max'] / df['c_velo']
    df['pop_c_velo_ratio'] = df['pop_time'] / df['c_velo']
    df['arm_athleticism_correlation'] = df['c_velo'] * df['athletic_index_v2'] / 100
    
    # Scaled features (min-max scaling simulation)
    df['exit_velo_scaled'] = np.clip((df['exit_velo_max'] - 70) / (105 - 70) * 100, 0, 100)
    df['speed_scaled'] = np.clip((1 - (df['sixty_time'] - 6.0) / (8.5 - 6.0)) * 100, 0, 100)
    df['arm_scaled'] = np.clip((df['c_velo'] - 60) / (85 - 60) * 100, 0, 100)
    df['pop_time_scaled'] = np.clip((1 - (df['pop_time'] - 1.8) / (2.5 - 1.8)) * 100, 0, 100)
    
    # Regional advantages
    df['d1_region_advantage'] = 0.5  # Default
    
    # Elite composite score
    df['elite_composite_score'] = (
        df['exit_velo_scaled'] * 0.25 +
        df['speed_scaled'] * 0.20 +
        df['arm_scaled'] * 0.30 +
        df['pop_time_scaled'] * 0.15 +
        df['height_percentile'] * 0.10
    )
    
    # Additional engineered features
    df['defensive_consistency'] = df['pop_time_percentile'] * df['c_velo_percentile'] * df['catcher_defensive_percentile']
    df['power_speed_size_ratio'] = (df['exit_velo_max'] * df['sixty_time']) / df['weight']
    df['pop_efficiency'] = df['c_velo'] / (df['pop_time'] ** 2)
    df['region_athletic_adjustment'] = df['athletic_index_v2'] * df['d1_region_advantage']
    df['tool_synergy'] = df['tools_athlete'] * df['multi_tool_count'] * df['complete_catcher']
    df['athletic_ceiling'] = df['athletic_index_v2'] ** 2 * df['height_percentile']
    df['arm_athleticism_correlation_x_region_athletic_adjustment'] = df['arm_athleticism_correlation'] * df['region_athletic_adjustment']
    df['tool_count_x_athletic_index'] = df['multi_tool_count'] * df['athletic_index_v2']
    df['tool_athletic_log'] = np.log1p(df['tool_count_x_athletic_index'])
    
    # Ensure all required features are present
    for feature in metadata['feature_columns']:
        if feature not in df.columns:
            df[feature] = 0  # Default value for missing features
    
    # Select features in correct order
    df_features = df[metadata['feature_columns']]
    
    # Clean data
    df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features for DNN if scaler exists
    if dnn_scaler is not None:
        df_scaled_dnn = dnn_scaler.transform(df_features)
    else:
        df_scaled_dnn = df_features.values
    
    # Get predictions from base models
    lgb_proba = lgb_model.predict_proba(df_features)[0, 1]
    dnn_proba = dnn_model.predict_proba(df_scaled_dnn)[0, 1]
    
    # Calculate final prediction based on ensemble type
    if metadata['ensemble_type'] == 'meta_learner' and meta_learner is not None:
        # Create enhanced meta-features
        meta_features = np.array([[
            lgb_proba,  # LGB pred
            dnn_proba,  # DNN pred
            lgb_proba * dnn_proba,  # LGB * DNN interaction
            (lgb_proba + dnn_proba) / 2,  # Average prediction
            ((lgb_proba + dnn_proba) / 2) ** 2,  # Average prediction squared
            abs(lgb_proba - dnn_proba),  # Prediction uncertainty (disagreement)
            max(lgb_proba, dnn_proba),  # Max confidence
            min(lgb_proba, dnn_proba),  # Min confidence
            abs(lgb_proba - dnn_proba),  # Model disagreement
            lgb_proba ** 2,  # LGB^2
            lgb_proba ** 3,  # LGB^3
            lgb_proba ** 4,  # LGB^4
            np.exp(lgb_proba)  # e^LGB
        ]])
        
        ensemble_prob = meta_learner.predict_proba(meta_features)[0, 1]
        strategy = 'meta_learner'
    else:
        # Traditional weighted ensemble
        ensemble_weights = metadata.get('ensemble_weights', {'LGB': 0.6, 'DNN': 0.4})
        ensemble_prob = (lgb_proba * ensemble_weights['LGB'] + 
                        dnn_proba * ensemble_weights['DNN'])
        strategy = 'weighted_ensemble'
    
    # Apply optimal threshold
    optimal_threshold = metadata.get('optimal_threshold', 0.5)
    prediction = ensemble_prob >= optimal_threshold
    
    # Determine confidence level
    if ensemble_prob >= 0.8 or ensemble_prob <= 0.2:
        confidence = "high"
    elif ensemble_prob >= 0.65 or ensemble_prob <= 0.35:
        confidence = "medium"
    else:
        confidence = "low"
    
    return {
        'player_id': player_data.get('player_id', 'unknown'),
        'd1_probability': float(ensemble_prob),
        'd1_prediction': bool(prediction),
        'confidence_level': confidence,
        'ensemble_strategy': strategy,
        'threshold_used': optimal_threshold,
        'components': {
            'individual_models': {
                'lightgbm': float(lgb_proba),
                'dnn': float(dnn_proba)
            }
        },
        'model_version': metadata.get('model_type', 'catcher_d1_meta_learner'),
        'model_metadata': {
            'test_accuracy': metadata.get('test_accuracy', 'unknown'),
            'training_date': metadata.get('training_date', 'unknown'),
            'ensemble_type': metadata.get('ensemble_type', 'unknown')
        }
    }

def predict_catcher_d1_batch(players_df, models_dir):
    """
    Predict D1 probability for multiple catchers.
    
    Args:
        players_df (pd.DataFrame): DataFrame containing player features
        models_dir (str): Path to the saved models directory
    
    Returns:
        pd.DataFrame: DataFrame with predictions added
    """
    results = []
    for _, player in players_df.iterrows():
        player_dict = player.to_dict()
        result = predict_catcher_d1_probability(player_dict, models_dir)
        results.append(result)
    
    # Add results to dataframe
    results_df = players_df.copy()
    results_df['d1_probability'] = [r['d1_probability'] for r in results]
    results_df['d1_prediction'] = [r['d1_prediction'] for r in results]
    results_df['confidence_level'] = [r['confidence_level'] for r in results]
    results_df['d1_prediction_label'] = results_df['d1_prediction'].map({True: 'D1', False: 'Non-D1'})
    
    return results_df

# Example usage
if __name__ == "__main__":
    # Example player data
    example_player = {
        'player_id': 'example_catcher_001',
        'height': 72,
        'weight': 190,
        'sixty_time': 7.2,
        'c_velo': 78,
        'pop_time': 2.0,
        'exit_velo_max': 92,
        'throwing_hand': 'R',
        'hitting_handedness': 'R',
        'player_region': 'South'
    }
    
    models_directory = os.path.dirname(__file__)
    result = predict_catcher_d1_probability(example_player, models_directory)
    
    print("🏟️ Catcher D1 Prediction Results:")
    print(f"Player ID: {result['player_id']}")
    print(f"D1 Prediction: {result['d1_prediction_label']}")
    print(f"D1 Probability: {result['d1_probability']:.3f}")
    print(f"Confidence: {result['confidence_level']}")
    print(f"Strategy: {result['ensemble_strategy']}")
    print(f"LightGBM: {result['components']['individual_models']['lightgbm']:.3f}")
    print(f"DNN: {result['components']['individual_models']['dnn']:.3f}")