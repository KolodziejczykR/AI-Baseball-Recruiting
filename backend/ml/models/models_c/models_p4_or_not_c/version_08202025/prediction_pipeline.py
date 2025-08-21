"""
P4 Catcher Prediction Pipeline - Clean Production Version
Performance: 66.3% accuracy, 36.8% P4 recall
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

from backend.utils.prediction_types import P4PredictionResult
from backend.utils.elite_weighting_constants import (
    ELITE_EXIT_VELO_MAX, ELITE_C_VELO, ELITE_SIXTY_TIME_C, ELITE_HEIGHT_MIN, ELITE_POP_TIME
)

def predict_catcher_p4_probability(player_data: dict, models_dir: str, d1_probability: float) -> P4PredictionResult:
    """
    Clean P4 probability prediction for catchers using trained ensemble
    
    Args:
        player_data (dict): Player statistics
        models_dir (str): Path to model files  
        d1_probability (float): D1 probability from previous stage
    
    Returns:
        P4PredictionResult: Structured prediction result
    """
    
    # Load trained models and configuration
    lgb_model = joblib.load(f'{models_dir}/lightgbm_model.pkl')
    xgb_model = joblib.load(f'{models_dir}/xgboost_model.pkl')
    mlp_model = joblib.load(f'{models_dir}/mlp_model.pkl')
    svm_model = joblib.load(f'{models_dir}/svm_model.pkl')
    scaler = joblib.load(f'{models_dir}/feature_scaler.pkl')
    
    # Load metadata
    metadata = joblib.load(f'{models_dir}/model_metadata.pkl')
    
    # Feature engineering (same as training)
    df = pd.DataFrame([player_data])
    
    # Categorical encodings (ensure all expected categories exist)
    # Player region encoding (drop first = True, so only keep these)
    for region in ['Northeast', 'South', 'West']:
        df[f'player_region_{region}'] = (df['player_region'] == region).astype(int)
    
    # Throwing hand encoding (drop first = True, so only keep Right)
    df['throwing_hand_R'] = (df['throwing_hand'] == 'R').astype(int)
    
    # Hitting handedness encoding (drop first = True, so only keep R and S)
    df['hitting_handedness_R'] = (df['hitting_handedness'] == 'R').astype(int)
    df['hitting_handedness_S'] = (df['hitting_handedness'] == 'S').astype(int)
    
    # Drop original categorical columns
    df = df.drop(['player_region', 'throwing_hand', 'hitting_handedness'], axis=1)
    
    # ONLY REQUIRED FEATURES - CORE METRICS
    df['power_speed'] = df['exit_velo_max'] / df['sixty_time']
    df['c_velo_sixty_ratio'] = df['c_velo'] / df['sixty_time']
    df['height_weight'] = df['height'] * df['weight']
    df['bmi'] = df['weight'] / ((df['height'] / 12) ** 2)
    df['power_per_pound'] = df['exit_velo_max'] / df['weight']
    df['arm_per_pound'] = df['c_velo'] / df['weight']
    df['speed_size_efficiency'] = (df['height'] * df['weight']) / (df['sixty_time'] ** 2)
    df['size_adjusted_power'] = df['exit_velo_max'] / (df['height'] / 72) / (df['weight'] / 180)
    
    # DEFENSIVE METRICS (only ones in the list)
    df['pop_time_c_velo_ratio'] = df['pop_time'] / df['c_velo'] * 100
    df['defensive_efficiency'] = df['c_velo'] / df['pop_time']
    df['framing_potential'] = df['height'] * (1 / df['pop_time'])
    
    # BMI INTERACTIONS
    df['bmi_swing_power'] = df['bmi'] * df['exit_velo_max']
    df['speed_size_eff_x_bmi'] = df['speed_size_efficiency'] * df['bmi']
    df['c_velo_sixty_ration_x_bmi'] = df['c_velo_sixty_ratio'] * df['bmi']
    df['bmi_swing_correlations'] = df['size_adjusted_power'] * df['power_per_pound'] * df['bmi_swing_power']
    
    # PERCENTILE CALCULATION USING TRAINING DATA QUANTILES
    def calculate_percentile_from_quantiles(value, quantiles, lower_is_better=False):
        # Find position in quantiles array
        for i, q_val in enumerate(quantiles):
            if value <= q_val:
                percentile = i * 5  # Since quantiles are every 5%
                break
        else:
            percentile = 100
        
        if lower_is_better:
            percentile = 100 - percentile
        
        return min(100, max(0, percentile))
    
    # Training data quantiles (from balanced_accuracy_v4.py output)
    exit_velo_max_quantiles = [77.0, 86.0, 88.19, 90.0, 91.0, 91.9, 92.5, 93.05, 93.7, 94.3, 94.9, 95.6, 96.3, 96.7, 97.3, 98.0, 98.8, 99.7, 100.8, 102.35, 110.8]
    c_velo_quantiles = [67.0, 73.0, 74.0, 75.0, 75.0, 76.0, 76.0, 77.0, 78.0, 78.0, 78.0, 79.0, 79.0, 80.0, 80.0, 81.0, 81.0, 82.0, 83.0, 84.0, 92.0]
    pop_time_quantiles = [1.7, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.1, 2.1, 2.1, 2.5]
    sixty_time_quantiles = [3.93, 6.755, 6.85, 6.937, 6.99, 7.03, 7.07, 7.11, 7.14, 7.18, 7.21, 7.25, 7.29, 7.33, 7.38, 7.42, 7.47, 7.54, 7.64, 7.785, 9.9]
    height_quantiles = [66.0, 69.0, 70.0, 70.0, 70.0, 71.0, 71.0, 71.0, 72.0, 72.0, 72.0, 72.0, 73.0, 73.0, 73.0, 73.0, 74.0, 74.0, 75.0, 75.0, 84.0]
    weight_quantiles = [150.0, 165.0, 170.0, 175.0, 180.0, 180.0, 185.0, 185.0, 186.0, 190.0, 190.0, 193.0, 195.0, 196.1, 200.0, 201.3, 205.0, 210.0, 214.0, 220.0, 256.6]
    
    # Calculate percentiles for each metric
    df['exit_velo_max_percentile'] = calculate_percentile_from_quantiles(df['exit_velo_max'].iloc[0], exit_velo_max_quantiles, lower_is_better=False)
    df['c_velo_percentile'] = calculate_percentile_from_quantiles(df['c_velo'].iloc[0], c_velo_quantiles, lower_is_better=False)
    df['pop_time_percentile'] = calculate_percentile_from_quantiles(df['pop_time'].iloc[0], pop_time_quantiles, lower_is_better=True)
    df['sixty_time_percentile'] = calculate_percentile_from_quantiles(df['sixty_time'].iloc[0], sixty_time_quantiles, lower_is_better=True)
    df['height_percentile'] = calculate_percentile_from_quantiles(df['height'].iloc[0], height_quantiles, lower_is_better=False)
    df['weight_percentile'] = calculate_percentile_from_quantiles(df['weight'].iloc[0], weight_quantiles, lower_is_better=False)

    # COMPOSITES (need training data for proper calculation) 
    df['offensive_composite'] = (
        df['exit_velo_max_percentile'] * 0.4 +
        df['sixty_time_percentile'] * 0.3 +
        df['height_percentile'] * 0.2 +
        df['weight_percentile'] * 0.1
    ) 

    df['defensive_composite'] = (
        df['c_velo_percentile'] * 0.5 +
        df['pop_time_percentile'] * 0.3 +
        df['height_percentile'] * 0.2
    )

    df['overall_composite'] = (
        df['offensive_composite'] * 0.6 +
        df['defensive_composite'] * 0.4
    )

    df['athletic_index'] = (
        df['exit_velo_max_percentile'] * 0.25 +
        df['c_velo_percentile'] * 0.30 + 
        df['pop_time_percentile'] * 0.20 +
        df['sixty_time_percentile'] * 0.15 +
        df['height_percentile'] * 0.05 +
        df['weight_percentile'] * 0.05
    )    

    # ELITE INDICATORS
    df['p4_exit_threshold'] = (df['exit_velo_max'] >= 98.0).astype(int)
    df['p4_arm_threshold'] = (df['c_velo'] >= 78.0).astype(int)
    df['p4_pop_threshold'] = (df['pop_time'] <= 1.95).astype(int)
    df['p4_speed_threshold'] = (df['sixty_time'] <= 7.0).astype(int)
    df['p4_size_threshold'] = ((df['height'] >= 72) & (df['weight'] >= 190)).astype(int)

    # 4. TOOL COMBINATIONS
    df['elite_combo_score'] = (df['p4_exit_threshold'] + df['p4_arm_threshold'] + 
                            df['p4_pop_threshold'] + df['p4_speed_threshold'] + 
                            df['p4_size_threshold'])
    
    # ELITE INDICATORS (using training quantile thresholds)
    df['exit_velo_elite'] = (df['exit_velo_max'] >= 98.0).astype(int)  # 75th percentile
    df['c_arm_strength'] = (df['c_velo'] >= 81.0).astype(int)  # 75th percentile  
    df['pop_time_elite'] = (df['pop_time'] <= 1.90).astype(int)  # 25th percentile
    df['speed_elite'] = (df['sixty_time'] <= 7.03).astype(int)  # 25th percentile
    df['elite_size'] = ((df['height'] >= 73) & (df['weight'] >= 195)).astype(int)  # Fixed thresholds from training

    df['tool_count'] = (df['exit_velo_elite'] + df['c_arm_strength'] + 
                    df['pop_time_elite'] + df['speed_elite'] + df['elite_size'])

    df['tools_athlete'] = df['tool_count'] * df['athletic_index']    

    # D1-BASED FEATURES (required by model) - MUST BE BEFORE ELITE MODEL
    df['d1_probability'] = d1_probability

    # ELITE PROBABILITY (load from elite model)
    elite_model = joblib.load(f'{models_dir}/elite_model.pkl')
    elite_features = ['exit_velo_max', 'c_velo', 'sixty_time', 'height', 'weight', 
                     'overall_composite', 'athletic_index', 'd1_probability']
    X_for_elite = df[elite_features].fillna(0)
    df['elite_probability'] = elite_model.predict_proba(X_for_elite)[:, 1]
    
    # D1 INTERACTIONS (only ones in feature list)
    df['d1_bmi_swing_power'] = df['d1_probability'] * df['bmi_swing_power']
    df['d1_bmi_swing_correlations'] = df['d1_probability'] * df['bmi_swing_correlations'] 
    df['d1_power_per_pound'] = df['d1_probability'] * df['power_per_pound']
    df['d1_arm_per_pound'] = df['d1_probability'] * df['arm_per_pound']
    df['d1_speed_size_efficiency'] = df['d1_probability'] * df['speed_size_efficiency']
    df['d1_c_velo_sixty_ratio'] = df['d1_probability'] * df['c_velo_sixty_ratio']
    df['d1_overall_composite'] = df['d1_probability'] * df['overall_composite']
    df['d1_weighted_exit_velo'] = df['d1_probability'] * df['exit_velo_max'] / 100
    df['d1_weighted_c_velo'] = df['d1_probability'] * df['c_velo'] / 100
    df['d1_weighted_speed'] = df['d1_probability'] * (8.0 - df['sixty_time'])
    df['d1_p4_power_boost'] = df['d1_probability'] * df['exit_velo_max'] * (df['height'] / 72)
    df['d1_p4_arm_boost'] = df['d1_probability'] * df['c_velo'] * (df['weight'] / 200) 
    df['d1_p4_athleticism'] = df['d1_probability'] * df['athletic_index'] / 100
    df['d1_confidence'] = np.abs(df['d1_probability'] - 0.5) * 2
    
    # D1 ELITE FEATURES (only ones in feature list)
    df['d1_elite_gap'] = np.abs(df['d1_probability'] - df['elite_probability'])
    df['d1_elite_max'] = np.maximum(df['d1_probability'], df['elite_probability'])
    df['d1_elite_min'] = np.minimum(df['d1_probability'], df['elite_probability'])
    df['d1_elite_synergy'] = df['d1_probability'] * df['elite_probability']

    # Ensure all required features exist and are in correct order
    missing_features = []
    for feature in metadata['feature_columns']:
        if feature not in df.columns:
            missing_features.append(feature)
    
    if missing_features:
        raise ValueError(f"Missing required features for model prediction: {missing_features}")
    
    # Select features in correct order and handle missing
    X = df[metadata['feature_columns']].fillna(0)
    
    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features for models that need it
    X_scaled = scaler.transform(X)
    
    # Get ensemble predictions using trained models
    lgb_prob = lgb_model.predict_proba(X)[0, 1]
    xgb_prob = xgb_model.predict_proba(X)[0, 1]
    mlp_prob = mlp_model.predict_proba(X_scaled)[0, 1]
    svm_prob = svm_model.predict_proba(X_scaled)[0, 1]
    
    # Apply trained ensemble weights (squared for amplified differences)
    weights = metadata['ensemble_weights']
    ensemble_prob = (lgb_prob * weights['lgb'] + 
                    xgb_prob * weights['xgb'] + 
                    mlp_prob * weights['mlp'] + 
                    svm_prob * weights['svm'])
    
    # Apply trained threshold
    threshold = metadata['optimal_threshold']
    p4_prediction = ensemble_prob >= threshold
    
    # Combined confidence: ensemble agreement + boundary distance (adjusted for 4-model ensemble)
    individual_probs = [lgb_prob, xgb_prob, mlp_prob, svm_prob]
    
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
    
# Elite P4 detection for result
    elite_indicators = []
    
    if player_data.get('exit_velo_max') >= ELITE_EXIT_VELO_MAX:
        elite_indicators.append(f"Elite exit velocity: {player_data['exit_velo_max']} mph")
    
    if player_data.get('c_velo') >= ELITE_C_VELO:
        elite_indicators.append(f"Elite outfield velocity: {player_data['c_velo']} mph")
    
    if player_data.get('sixty_time') <= ELITE_SIXTY_TIME_C:
        elite_indicators.append(f"Elite speed: {player_data['sixty_time']} seconds")

    if player_data.get('pop_time') <= ELITE_POP_TIME:
        elite_indicators.append(f"Elite pop time: {player_data['pop_time']} seconds")
    
    if player_data.get('height') >= ELITE_HEIGHT_MIN:
        elite_indicators.append(f"Elite height: {player_data['height']} inches")
    
    # Determine if elite P4 (multiple elite indicators)
    is_elite_p4 = len(elite_indicators) >= 2
    
    return P4PredictionResult(
        p4_probability=float(ensemble_prob),
        p4_prediction=bool(p4_prediction),
        confidence=confidence,
        is_elite=bool(is_elite_p4),
        elite_indicators=elite_indicators if elite_indicators else None,
        model_version=metadata.get('model_type', 'catcher_p4_ensemble_08202025')
    )


if __name__ == "__main__":
    # Example usage
    test_player = {
        'height': 74.0,
        'weight': 210.0,
        'sixty_time': 6.9,
        'exit_velo_max': 99.0,
        'c_velo': 82.0,
        'pop_time': 1.78,
        'player_region': 'South',
        'throwing_hand': 'R',
        'hitting_handedness': 'L'
    }
    
    models_dir = os.path.dirname(__file__)
    result = predict_catcher_p4_probability(test_player, models_dir, d1_probability=0.75)
    
    print(f"P4 Probability: {result.p4_probability:.1%}")
    print(f"P4 Prediction: {result.p4_prediction}")
    print(f"Confidence: {result.confidence}")
    print(f"Elite P4: {result.is_elite}")
    if result.elite_indicators:
        print(f"Elite Indicators: {result.elite_indicators}")