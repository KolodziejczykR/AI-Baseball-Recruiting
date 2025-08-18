import pandas as pd
import numpy as np
import joblib
import sys
import os

from backend.utils.player_types import PlayerInfielder

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.utils.elite_weighting_constants import (
    ELITE_EXIT_VELO_MAX, ELITE_INF_VELO, ELITE_SIXTY_TIME_INF, ELITE_HEIGHT_MIN
)

def predict_infielder_d1_probability(player_data: PlayerInfielder, models_dir: str) -> dict:
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
    
    # ELITE PLAYER DETECTION AND ADAPTIVE ENSEMBLE
    # Check for elite characteristics using imported constants
    elite_thresholds = {
        'exit_velo_max': ELITE_EXIT_VELO_MAX,
        'inf_velo': ELITE_INF_VELO,
        'sixty_time_max': ELITE_SIXTY_TIME_INF, 
        'height_min': ELITE_HEIGHT_MIN
    }
    
    elite_score = 0
    elite_indicators = []
    
    if player_data.get('exit_velo_max', 0) >= elite_thresholds['exit_velo_max']:
        elite_score += 2
        elite_indicators.append(f"Elite exit velocity: {player_data['exit_velo_max']} mph")
    
    if player_data.get('inf_velo', 0) >= elite_thresholds['inf_velo']:
        elite_score += 2  
        elite_indicators.append(f"Elite infield velocity: {player_data['inf_velo']} mph")
    
    if player_data.get('sixty_time', 10) <= elite_thresholds['sixty_time_max']:
        elite_score += 2
        elite_indicators.append(f"Elite speed: {player_data['sixty_time']} seconds")
    
    if player_data.get('height', 0) >= elite_thresholds['height_min']:
        elite_score += 1
        elite_indicators.append(f"Elite height: {player_data['height']} inches")
    
    # Count feature outliers  
    extreme_features = np.sum(np.abs(df_scaled) > 3.0)
    outlier_ratio = extreme_features / len(df_scaled.flatten())
    
    # CONFIDENCE-BASED WEIGHT REDISTRIBUTION
    is_elite = elite_score >= 4
    is_super_elite = elite_score >= 6
    weights = ensemble_metadata['weights']
    individual_d1_probs = [xgb_proba, lgb_proba, cb_proba, svm_proba]
    
    # Calculate confidence scores based on how far predictions are from 0.5 (uncertainty)
    confidence_scores = [2 * abs(prob - 0.5) for prob in individual_d1_probs]  # 0-1 scale
    
    # Apply confidence multipliers
    confidence_multipliers = []
    for conf_score in confidence_scores:
        if conf_score >= 0.6:  # High confidence (80%+ or 20%-)
            multiplier = 1.5
        elif conf_score >= 0.2:  # Medium confidence (60-80% or 20-40%)
            multiplier = 1.0
        else:  # Low confidence (40-60%)
            multiplier = 0.4
        confidence_multipliers.append(multiplier)
    
    # Apply confidence multipliers to base weights
    confidence_adjusted_weights = [base_weight * mult for base_weight, mult in zip(weights, confidence_multipliers)]
    
    # Normalize weights to sum to 1.0
    weight_sum = sum(confidence_adjusted_weights)
    confidence_adjusted_weights = [w / weight_sum for w in confidence_adjusted_weights]
    
    # Adaptive ensemble weighting (keeping elite detection but adding confidence weighting)
    if is_super_elite and svm_proba > 0.9:
        # Super elite + very confident SVM = SVM dominant
        adjusted_weights = [0.1, 0.1, 0.1, 0.7]
        strategy = 'super_elite_svm_dominant'
    elif is_elite and svm_proba > 0.8:
        # Elite + confident SVM = boost SVM
        adjusted_weights = [0.15, 0.15, 0.15, 0.55]  
        strategy = 'elite_svm_boosted'
    elif outlier_ratio > 0.3:
        # High outliers = moderate SVM boost
        adjusted_weights = [0.2, 0.2, 0.2, 0.4]
        strategy = 'outlier_svm_boosted'
    else:
        # Use confidence-based weighting as the standard approach
        adjusted_weights = confidence_adjusted_weights
        strategy = 'confidence_based_ensemble'
    
    # Calculate ensemble probability with adaptive weights
    ensemble_prob = (xgb_proba * adjusted_weights[0] + 
                    lgb_proba * adjusted_weights[1] + 
                    cb_proba * adjusted_weights[2] + 
                    svm_proba * adjusted_weights[3])
    
    # Enhanced confidence calculation
    if svm_proba > 0.9 and is_elite:
        confidence = "high"
    elif ensemble_prob > 0.7 or ensemble_prob < 0.3:
        confidence = "high" 
    elif ensemble_prob > 0.6 or ensemble_prob < 0.4:
        confidence = "medium"
    else:
        confidence = "low"
    
    # Final prediction (ensemble uses soft voting, so threshold is 0.5)
    prediction = ensemble_prob >= 0.5
    
    return {
        'player_id': player_data.get('player_id', 'unknown'),
        'd1_probability': float(ensemble_prob),
        'd1_prediction': bool(prediction),
        'confidence_level': confidence,
        'elite_detection': {
            'is_elite': is_elite,
            'is_super_elite': is_super_elite,
            'elite_score': elite_score,
            'elite_indicators': elite_indicators,
            'strategy_used': strategy
        },
        'outlier_info': {
            'extreme_features': int(extreme_features),
            'outlier_ratio': float(outlier_ratio)
        },
        'ensemble_weights': {
            'original': weights,
            'confidence_adjusted': confidence_adjusted_weights,
            'final_adjusted': adjusted_weights
        },
        'confidence_analysis': {
            'confidence_scores': dict(zip(['xgboost', 'lightgbm', 'catboost', 'svm'], [float(x) for x in confidence_scores])),
            'confidence_multipliers': dict(zip(['xgboost', 'lightgbm', 'catboost', 'svm'], confidence_multipliers))
        },
        'components': {
            'individual_models': {
                'xgboost': float(xgb_proba),
                'lightgbm': float(lgb_proba),
                'catboost': float(cb_proba),
                'svm': float(svm_proba)
            }
        },
        'threshold_used': 0.5,
        'model_version': ensemble_metadata.get('model_type', 'VotingClassifier_WeightedSoft')
    }