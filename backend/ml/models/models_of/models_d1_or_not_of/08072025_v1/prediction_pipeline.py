
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler

def predict_outfielder_d1_probability(player_data, models_dir):
    """
    Predict D1 probability for an outfielder using hierarchical ensemble model.
    
    Args:
        player_data (dict): Dictionary containing player features
        models_dir (str): Path to the saved models directory
    
    Returns:
        dict: Prediction results including probability and components
    """
    
    # Load model configuration
    with open(f"{models_dir}/model_config.json", 'r') as f:
        config = json.load(f)
    
    # Load feature metadata
    with open(f"{models_dir}/feature_metadata.json", 'r') as f:
        feature_meta = json.load(f)
    
    # Load models
    elite_model = joblib.load(f"{models_dir}/elite_model.pkl")
    xgb_full = joblib.load(f"{models_dir}/xgb_full_model.pkl")
    dnn_full = joblib.load(f"{models_dir}/dnn_full_model.pkl")
    lgb_full = joblib.load(f"{models_dir}/lgb_full_model.pkl")
    svm_full = joblib.load(f"{models_dir}/svm_full_model.pkl")
    
    # Load scalers
    scaler_full = joblib.load(f"{models_dir}/scaler_full.pkl")
    
    # Convert player data to DataFrame
    df = pd.DataFrame([player_data])
    
    # Create categorical encodings (same as training)
    df = pd.get_dummies(df, columns=['player_region', 'throwing_hand', 'hitting_handedness'], 
                       prefix_sep='_', drop_first=True)
    
    # Feature engineering (same as training pipeline)
    df['power_speed'] = df['exit_velo_max'] / df['sixty_time']
    df['of_velo_sixty_ratio'] = df['of_velo'] / df['sixty_time']
    df['height_weight'] = df['height'] * df['weight']
    
    # Add percentile features
    percentile_features = ['exit_velo_max', 'of_velo', 'sixty_time', 'height', 'weight', 'power_speed']
    for col in percentile_features:
        if col in df.columns:
            if col == 'sixty_time':
                df[f'{col}_percentile'] = (1 - df[col].rank(pct=True)) * 100
            else:
                df[f'{col}_percentile'] = df[col].rank(pct=True) * 100
    
    # Add all other engineered features (abbreviated for space)
    df['power_per_pound'] = df['exit_velo_max'] / df['weight']
    df['exit_to_sixty_ratio'] = df['exit_velo_max'] / df['sixty_time']
    df['speed_size_efficiency'] = (df['height'] * df['weight']) / (df['sixty_time'] ** 2)
    df['athletic_index'] = (df['power_speed'] * df['height'] * df['weight']) / df['sixty_time']
    df['power_speed_index'] = df['exit_velo_max'] * (1 / df['sixty_time'])
    
    # Elite binary features
    df['elite_exit_velo'] = (df['exit_velo_max'] >= df['exit_velo_max'].quantile(0.75)).astype(int)
    df['elite_of_velo'] = (df['of_velo'] >= df['of_velo'].quantile(0.75)).astype(int)
    df['elite_speed'] = (df['sixty_time'] <= df['sixty_time'].quantile(0.25)).astype(int)
    df['elite_size'] = ((df['height'] >= df['height'].quantile(0.6)) & 
                       (df['weight'] >= df['weight'].quantile(0.6))).astype(int)
    df['multi_tool_count'] = (df['elite_exit_velo'] + df['elite_of_velo'] + 
                             df['elite_speed'] + df['elite_size'])
    
    # Scale features
    df['exit_velo_scaled'] = (df['exit_velo_max'] - df['exit_velo_max'].min()) / (df['exit_velo_max'].max() - df['exit_velo_max'].min()) * 100
    df['speed_scaled'] = (1 - (df['sixty_time'] - df['sixty_time'].min()) / (df['sixty_time'].max() - df['sixty_time'].min())) * 100
    df['arm_scaled'] = (df['of_velo'] - df['of_velo'].min()) / (df['of_velo'].max() - df['of_velo'].min()) * 100
    
    # Elite composite score
    df['elite_composite_score'] = (
        df['exit_velo_scaled'] * 0.30 +
        df['speed_scaled'] * 0.25 +
        df['arm_scaled'] * 0.25 +
        df['height_percentile'] * 0.10 +
        df['power_speed'] * 0.10
    )
    
    # Add remaining features to match training set
    # ... (add all other features from training pipeline)
    
    # Ensure all required features are present
    for feature in feature_meta['all_features']:
        if feature not in df.columns:
            df[feature] = 0  # Default value for missing features
    
    # Select features in correct order
    elite_features_subset = [col for col in feature_meta['elite_features'] if col in df.columns]
    elite_feats = df[elite_features_subset]
    d1_feats = df[feature_meta['all_features']]
    
    # Clean data
    elite_feats = elite_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
    d1_feats = d1_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features for neural network and SVM
    d1_feats_scaled = scaler_full.transform(d1_feats)
    
    # Get probabilities
    elite_prob = elite_model.predict_proba(elite_feats)[0, 1]
    
    # Get ensemble D1 probabilities
    xgb_proba = xgb_full.predict_proba(d1_feats)[0, 1]
    dnn_proba = dnn_full.predict_proba(d1_feats_scaled)[0, 1]
    lgb_proba = lgb_full.predict_proba(d1_feats)[0, 1]
    svm_proba = svm_full.predict_proba(d1_feats_scaled)[0, 1]
    
    # Calculate ensemble probability
    ensemble_weights = config['ensemble_weights']
    d1_prob = (xgb_proba * ensemble_weights['XGB'] + 
               dnn_proba * ensemble_weights['DNN'] + 
               lgb_proba * ensemble_weights['LGB'] + 
               svm_proba * ensemble_weights['SVM'])
    
    # Hierarchical combination
    hierarchical_prob = (elite_prob * 0.4) + (d1_prob * 0.6)
    
    # Final prediction
    prediction = hierarchical_prob >= config['optimal_prediction_threshold']
    
    return {
        'player_id': player_data.get('player_id', 'unknown'),
        'd1_probability': float(hierarchical_prob),
        'd1_prediction': bool(prediction),
        'confidence_level': 'high' if abs(hierarchical_prob - 0.5) > 0.3 else 'medium' if abs(hierarchical_prob - 0.5) > 0.15 else 'low',
        'components': {
            'elite_probability': float(elite_prob),
            'ensemble_probability': float(d1_prob),
            'individual_models': {
                'xgboost': float(xgb_proba),
                'neural_network': float(dnn_proba),
                'lightgbm': float(lgb_proba),
                'svm': float(svm_proba)
            }
        },
        'threshold_used': float(config['optimal_prediction_threshold']),
        'model_version': config['model_version']
    }

# Example usage:
# result = predict_outfielder_d1_probability(
#     player_data={
#         'height': 73,
#         'weight': 190,
#         'sixty_time': 6.8,
#         'exit_velo_max': 95,
#         'of_velo': 85,
#         'player_region': 'West',
#         'throwing_hand': 'Right',
#         'hitting_handedness': 'Right'
#     },
#     models_dir='/path/to/models/directory'
# )
