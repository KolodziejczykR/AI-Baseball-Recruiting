
# HIERARCHICAL ENSEMBLE P4 PREDICTION PIPELINE
# Use this code structure in production

import joblib
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

def predict_p4_hierarchical_ensemble_production(player_data):
    """
    Production prediction function for P4 classification
    
    Args:
        player_data: DataFrame with engineered features
    
    Returns:
        dict with elite_prob, p4_prob, hierarchical_prob, prediction
    """
    
    # Load models (do this once at startup in production)
    elite_model = joblib.load('elite_detection_xgboost_model.pkl')
    xgb_model = joblib.load('p4_ensemble_xgboost_model.pkl')
    cat_model = joblib.load('p4_ensemble_catboost_model.pkl')
    lgb_model = joblib.load('p4_ensemble_lightgbm_model.pkl')
    svm_model = joblib.load('p4_ensemble_svm_model.pkl')
    scaler = joblib.load('feature_scaler_for_svm.pkl')
    
    # Load config
    with open('model_config_and_metadata.pkl', 'rb') as f:
        config = pickle.load(f)
    
    # Extract features
    elite_feats = player_data[config['elite_features']]
    p4_feats = player_data[config['p4_features']]
    p4_feats_scaled = scaler.transform(p4_feats)
    
    # Get elite probability
    elite_prob = elite_model.predict_proba(elite_feats)[:, 1]
    
    # Get ensemble P4 probabilities
    xgb_proba = xgb_model.predict_proba(p4_feats)[:, 1]
    cat_proba = cat_model.predict_proba(p4_feats)[:, 1]
    lgb_proba = lgb_model.predict_proba(p4_feats)[:, 1]
    svm_proba = svm_model.predict_proba(p4_feats_scaled)[:, 1]
    
    # Weighted ensemble
    p4_prob = (xgb_proba * 0.3 + cat_proba * 0.3 + 
               lgb_proba * 0.2 + svm_proba * 0.2)
    
    # Hierarchical combination
    hierarchical_prob = (elite_prob * 0.6) + (p4_prob * 0.4)
    
    # Final prediction
    prediction = (hierarchical_prob >= config['optimal_threshold']).astype(int)
    
    return {
        'elite_probability': elite_prob[0],
        'p4_probability': p4_prob[0], 
        'hierarchical_probability': hierarchical_prob[0],
        'p4_prediction': int(prediction[0]),
        'confidence': 'high' if abs(hierarchical_prob[0] - 0.5) > 0.3 else 'medium'
    }
