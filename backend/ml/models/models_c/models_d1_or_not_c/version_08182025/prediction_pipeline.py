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

def calculate_percentile_from_quantiles(value, quantiles, lower_is_better=False):
    """
    Calculate percentile of a value using pre-computed quantiles from training data
    
    Args:
        value: The value to calculate percentile for
        quantiles: List of quantiles from training data (every 5%: 0%, 5%, 10%, ..., 100%)
        lower_is_better: If True, invert percentile (for metrics like sixty_time, pop_time)
    
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
    
    # Training data quantiles for percentile calculation
    c_velo_quantiles = [54.0, 68.0, 70.0, 71.0, 72.0, 73.0, 73.0, 74.0, 75.0, 75.0, 76.0, 76.0, 77.0, 77.0, 78.0, 78.0, 79.0, 80.0, 81.0, 82.0, 92.0]
    pop_time_quantiles = [1.6, 1.9, 1.9, 1.9, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.1, 2.1, 2.1, 2.1, 2.1, 2.2, 2.2, 2.2, 2.3, 3.0]
    catcher_defensive_quantiles = [0.011385199240986717, 7.975332068311193, 13.810246679316887, 18.222011385199238, 22.206831119544596, 29.903225806451616, 34.89563567362429, 38.35863377609108, 42.5370018975332, 46.35483870967742, 52.189753320683124, 52.98861480075901, 58.98861480075901, 64.82352941176471, 65.85768500948767, 70.03795066413662, 75.08538899430741, 80.29981024667931, 84.84250474383302, 91.66223908918406, 99.76470588235294]
    catcher_offensive_quantiles = [0.08538899430739892, 9.758965844402274, 16.168595825426948, 21.349146110056925, 25.98121442125237, 30.192836812144208, 34.35645161290322, 38.255407969639464, 42.50740037950664, 46.29943074003795, 50.063092979127134, 53.93704933586338, 57.85426944971537, 61.72333965844402, 65.59525616698292, 69.5220588235294, 74.44554079696394, 78.92803605313092, 83.71622390891841, 89.82689753320682, 99.66888045540796]
    catcher_overall_quantiles = [0.046299810246678806, 16.30904174573055, 22.542960151802657, 27.095607210626188, 30.99872865275142, 34.394271821631875, 37.684217267552185, 41.1058372865275, 44.437846299810246, 47.41603889943074, 50.301707779886144, 53.56205407969639, 56.20286527514231, 59.56032732447817, 62.78022296015179, 65.8533088235294, 68.93480075901329, 72.77673387096773, 76.6403036053131, 82.09949715370021, 98.01503795066414]
    exit_velo_max_quantiles = [47.9, 81.0, 83.8, 85.3, 86.5, 87.6, 88.5, 89.4, 90.1, 90.9, 91.7, 92.3, 93.1, 93.8, 94.6, 95.4, 96.3, 97.2, 98.4, 100.2, 129.4]
    sixty_time_quantiles = [4.030349999999995, 6.86, 6.97, 7.04, 7.09, 7.14, 7.19, 7.237681423187255, 7.28, 7.32, 7.36, 7.409153294563295, 7.45, 7.49, 7.54, 7.6, 7.67, 7.75, 7.86, 8.03, 9.9]
    height_quantiles = [60.0, 68.0, 69.0, 69.0, 70.0, 70.0, 70.0, 71.0, 71.0, 71.0, 72.0, 72.0, 72.0, 72.0, 72.0, 73.0, 73.0, 74.0, 74.0, 75.0, 84.0]
    weight_quantiles = [104.0, 155.6, 163.98, 165.0, 170.0, 175.0, 175.0, 180.0, 180.0, 185.0, 185.0, 186.5, 190.0, 190.0, 195.0, 195.075, 200.0, 205.0, 210.0, 215.0, 282.2]

    # Calculate actual percentiles using training data quantiles
    df['c_velo_percentile'] = calculate_percentile_from_quantiles(df['c_velo'].iloc[0], c_velo_quantiles, lower_is_better=False)
    df['pop_time_percentile'] = calculate_percentile_from_quantiles(df['pop_time'].iloc[0], pop_time_quantiles, lower_is_better=True)
    
    # Calculate composite percentiles using the same formulas as training
    # Catcher defensive percentile: (c_velo_percentile * 0.6) + (pop_time_percentile * 0.4)
    calculated_defensive = (df['c_velo_percentile'] * 0.6) + (df['pop_time_percentile'] * 0.4)
    df['catcher_defensive_percentile'] = calculate_percentile_from_quantiles(calculated_defensive.iloc[0], catcher_defensive_quantiles, lower_is_better=False)
    
    # Calculate individual percentiles needed for composite calculations
    exit_velo_max_percentile = calculate_percentile_from_quantiles(df['exit_velo_max'].iloc[0], exit_velo_max_quantiles, lower_is_better=False)
    sixty_time_percentile = calculate_percentile_from_quantiles(df['sixty_time'].iloc[0], sixty_time_quantiles, lower_is_better=True)
    height_percentile = calculate_percentile_from_quantiles(df['height'].iloc[0], height_quantiles, lower_is_better=False)
    weight_percentile = calculate_percentile_from_quantiles(df['weight'].iloc[0], weight_quantiles, lower_is_better=False)
    
    # Catcher offensive percentile: (exit_velo_max_percentile * 0.7) + (sixty_time_percentile * 0.3)
    calculated_offensive = (exit_velo_max_percentile * 0.7) + (sixty_time_percentile * 0.3)
    df['catcher_offensive_percentile'] = calculate_percentile_from_quantiles(calculated_offensive, catcher_offensive_quantiles, lower_is_better=False)
    
    # Catcher overall percentile: (defensive * 0.4) + (offensive * 0.35) + (height * 0.15) + (weight * 0.10)
    calculated_overall = (df['catcher_defensive_percentile'] * 0.4) + (df['catcher_offensive_percentile'] * 0.35) + (height_percentile * 0.15) + (weight_percentile * 0.10)
    df['catcher_overall_percentile'] = calculate_percentile_from_quantiles(calculated_overall.iloc[0], catcher_overall_quantiles, lower_is_better=False)
    
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
    
    # Apply trained threshold (0.61 threshold to send more rated players to D1s)
    threshold = metadata['optimal_threshold'] - 0.15    # original threshold was 0.76, extremely high
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