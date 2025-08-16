
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
    
    # ELITE OUTFIELDER DETECTION AND ADAPTIVE ENSEMBLE
    # Check for elite outfielder characteristics using imported constants
    elite_of_thresholds = {
        'exit_velo_max': ELITE_EXIT_VELO_MAX,
        'of_velo': ELITE_OF_VELO,
        'sixty_time_max': ELITE_SIXTY_TIME_OF,
        'height_min': ELITE_HEIGHT_MIN
    }
    
    elite_of_score = 0
    elite_of_indicators = []
    
    if player_data.get('exit_velo_max', 0) >= elite_of_thresholds['exit_velo_max']:
        elite_of_score += 2
        elite_of_indicators.append(f"Elite OF exit velocity: {player_data['exit_velo_max']} mph")
    
    if player_data.get('of_velo', 0) >= elite_of_thresholds['of_velo']:
        elite_of_score += 2
        elite_of_indicators.append(f"Elite outfield velocity: {player_data['of_velo']} mph")
    
    if player_data.get('sixty_time', 10) <= elite_of_thresholds['sixty_time_max']:
        elite_of_score += 2
        elite_of_indicators.append(f"Elite OF speed: {player_data['sixty_time']} seconds")
    
    if player_data.get('height', 0) >= elite_of_thresholds['height_min']:
        elite_of_score += 1
        elite_of_indicators.append(f"Elite OF height: {player_data['height']} inches")
    
    # Count feature outliers
    extreme_of_features = np.sum(np.abs(d1_feats_scaled) > 3.0)
    of_outlier_ratio = extreme_of_features / len(d1_feats_scaled.flatten()) if len(d1_feats_scaled.flatten()) > 0 else 0
    
    # Elite OF classification
    is_elite_of = elite_of_score >= 4
    is_super_elite_of = elite_of_score >= 6
    
    # Calculate ensemble probability with base weights
    ensemble_weights = config['ensemble_weights']
    base_d1_prob = (xgb_proba * ensemble_weights['XGB'] + 
                    dnn_proba * ensemble_weights['DNN'] + 
                    lgb_proba * ensemble_weights['LGB'] + 
                    svm_proba * ensemble_weights['SVM'])
    
    # CONFIDENCE-BASED WEIGHT REDISTRIBUTION FOR OUTFIELDERS
    individual_of_probs = [xgb_proba, dnn_proba, lgb_proba, svm_proba]
    base_of_weights = [ensemble_weights['XGB'], ensemble_weights['DNN'], ensemble_weights['LGB'], ensemble_weights['SVM']]
    
    # Calculate confidence scores based on how far predictions are from 0.5 (uncertainty)
    of_confidence_scores = [2 * abs(prob - 0.5) for prob in individual_of_probs]  # 0-1 scale
    
    # Apply confidence multipliers
    of_confidence_multipliers = []
    for conf_score in of_confidence_scores:
        if conf_score >= 0.6:  # High confidence (80%+ or 20%-)
            multiplier = 1.5
        elif conf_score >= 0.2:  # Medium confidence (60-80% or 20-40%)
            multiplier = 1.0
        else:  # Low confidence (40-60%)
            multiplier = 0.4
        of_confidence_multipliers.append(multiplier)
    
    # Apply confidence multipliers to base weights
    of_confidence_adjusted_weights = [base_weight * mult for base_weight, mult in zip(base_of_weights, of_confidence_multipliers)]
    
    # Normalize weights to sum to 1.0
    of_weight_sum = sum(of_confidence_adjusted_weights)
    of_confidence_adjusted_weights = [w / of_weight_sum for w in of_confidence_adjusted_weights]
    
    # Adaptive ensemble weighting for outfielders (keeping elite detection but adding confidence weighting)
    if is_super_elite_of and max(xgb_proba, lgb_proba) > 0.8:
        # Super elite OF + very confident models = boost best performing models
        if xgb_proba > lgb_proba:
            adjusted_of_weights = [0.5, 0.15, 0.25, 0.1]  # Boost XGBoost
        else:
            adjusted_of_weights = [0.25, 0.15, 0.5, 0.1]  # Boost LightGBM
        strategy_of = 'super_elite_of_tree_dominant'
        
    elif is_elite_of and max(xgb_proba, lgb_proba) > 0.7:
        # Elite OF + confident models = moderate boost
        adjusted_of_weights = [0.35, 0.2, 0.35, 0.1]
        strategy_of = 'elite_of_tree_boosted'
        
    elif of_outlier_ratio > 0.3:
        # High outliers = balanced approach
        adjusted_of_weights = [0.3, 0.2, 0.3, 0.2]
        strategy_of = 'of_outlier_balanced'
        
    else:
        # Use confidence-based weighting as the standard approach
        adjusted_of_weights = of_confidence_adjusted_weights
        strategy_of = 'confidence_based_of_ensemble'
    
    # Calculate adaptive ensemble probability
    adaptive_d1_prob = sum(prob * weight for prob, weight in zip(individual_of_probs, adjusted_of_weights))
    
    # Enhanced hierarchical combination for elite players
    if is_super_elite_of:
        # For super elite, weight ensemble more heavily than elite model
        hierarchical_prob = (elite_prob * 0.25) + (adaptive_d1_prob * 0.75)
    elif is_elite_of:
        # For elite, balanced combination
        hierarchical_prob = (elite_prob * 0.3) + (adaptive_d1_prob * 0.7)
    else:
        # Standard hierarchical combination
        hierarchical_prob = (elite_prob * 0.4) + (adaptive_d1_prob * 0.6)
    
    # Enhanced confidence calculation
    if is_super_elite_of and hierarchical_prob > 0.6:
        confidence = 'high'
    elif hierarchical_prob > 0.7 or hierarchical_prob < 0.3:
        confidence = 'high'
    elif hierarchical_prob > 0.6 or hierarchical_prob < 0.4:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    # Final prediction
    prediction = hierarchical_prob >= config['optimal_prediction_threshold']
    
    return {
        'player_id': player_data.get('player_id', 'unknown'),
        'd1_probability': float(hierarchical_prob),
        'd1_prediction': bool(prediction),
        'confidence_level': confidence,
        'elite_of_detection': {
            'is_elite_of': is_elite_of,
            'is_super_elite_of': is_super_elite_of,
            'elite_of_score': elite_of_score,
            'elite_of_indicators': elite_of_indicators,
            'strategy_used': strategy_of
        },
        'of_outlier_info': {
            'extreme_features': int(extreme_of_features),
            'outlier_ratio': float(of_outlier_ratio)
        },
        'ensemble_weights': {
            'original': dict(zip(['XGB', 'DNN', 'LGB', 'SVM'], base_of_weights)),
            'confidence_adjusted': dict(zip(['XGB', 'DNN', 'LGB', 'SVM'], of_confidence_adjusted_weights)),
            'final_adjusted': dict(zip(['XGB', 'DNN', 'LGB', 'SVM'], adjusted_of_weights))
        },
        'confidence_analysis': {
            'confidence_scores': dict(zip(['XGB', 'DNN', 'LGB', 'SVM'], [float(x) for x in of_confidence_scores])),
            'confidence_multipliers': dict(zip(['XGB', 'DNN', 'LGB', 'SVM'], of_confidence_multipliers))
        },
        'components': {
            'elite_probability': float(elite_prob),
            'ensemble_probability': float(base_d1_prob),
            'adaptive_ensemble_probability': float(adaptive_d1_prob),
            'individual_models': {
                'xgboost': float(xgb_proba),
                'neural_network': float(dnn_proba),
                'lightgbm': float(lgb_proba),
                'svm': float(svm_proba)
            }
        },
        'threshold_used': float(config['optimal_prediction_threshold']),
        'model_version': config.get('model_version', 'outfielder_d1_v2_elite_adaptive')
    }
