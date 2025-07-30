# 🏟️ AI Baseball Recruitment Pipeline: Complete D1 & Power 4 Prediction System

## Overview

This pipeline provides a complete baseball recruiting prediction system that processes raw player metrics through two main stages: **D1 Classification** followed by **Power 4 Prediction** for D1-eligible players. The system uses ensemble machine learning models and hierarchical approaches to deliver accurate recruiting intelligence.

## 📊 Complete Pipeline Architecture

```
Raw Player Data → Feature Engineering → D1 Ensemble Model → P4 Hierarchical Model → Final Predictions
                         ↓                     ↓                    ↓                      ↓
                   41 Engineered         D1 Probability      Elite Detection +       D1: 79.16% Acc
                    Features           (79.16% Accuracy)     P4 Classification       P4: 78.5% Acc
                                                                                   
                                                           Hierarchical Weighting
                                                          (Elite × 0.6 + P4 × 0.4)
```

## 🔄 Complete Workflow

### **Stage 1: D1 Eligibility Screening**
**Location**: `/backend/ml/models/models_d1_or_not_inf/`

**Input**: Raw player metrics
- Height, weight, 60-yard dash time
- Exit velocity max, infield velocity  
- Position, handedness, geographic region
- Power-speed composite, athletic indices

**Process**: 
- **Weighted Ensemble Model** with optimized weights:
  - XGBoost (35%), CatBoost (35%), LightGBM (25%), SVM (5%)
- **79.16% accuracy** for D1 vs Non-D1 classification
- **Soft voting** combines probability predictions from all models
- Includes feature scaling pipeline for SVM component

**Models Included**:
- `ensemble_model.pkl` - Main weighted ensemble classifier
- `ensemble_scaler.pkl` - StandardScaler for SVM preprocessing  
- `ensemble_metadata.pkl` - Performance metrics and configuration
- Individual model files: `xgboost_model.pkl`, `catboost_model.pkl`, etc.

**Output**: 
- `d1_probability`: Primary D1 probability (0-1)
- `d1_prediction`: Binary D1/Non-D1 classification
- `confidence_level`: High/Medium/Low confidence based on probability thresholds

---

### **Stage 2: Feature Engineering & Data Preparation**
**Location**: `/backend/ml/train/inf/inf_d1_p4_or_not/`

**Input**: D1-predicted players + D1 metadata

**Process**: Enhanced feature engineering
- **D1 Metadata Integration**: Incorporates D1 probabilities as meta-features
- **Percentile Calculations**: Converts raw metrics to percentile rankings
- **Advanced Ratios**: Power-per-pound, exit-to-sixty ratio, speed-size efficiency
- **Athletic Composites**: Multi-dimensional athletic indices
- **Elite Binary Features**: Threshold-based elite classifications
- **Position-Specific Features**: Tailored metrics for different positions
- **Regional Adjustments**: Geographic recruiting preferences

**Output**: 41 engineered features for hierarchical modeling

---

### **Stage 3: Hierarchical Power 4 Prediction** 
**Location**: `/backend/ml/models/models_p4_or_not_inf/`

#### **Sub-Stage 3A: Elite D1 Detection**

**Purpose**: Identify elite D1 players who are candidates for P4 recruitment

**Process**:
1. **Composite Elite Score Calculation**:
   ```python
   Elite Score = (Exit_Velo * 0.35) + (Speed * 0.25) + (Arm_Strength * 0.20) + 
                 (Size * 0.10) + (Power_Speed * 0.10)
   ```

2. **Elite Threshold**: Top 45% of D1 players (creates 55/45 balance)

3. **Elite Model Training**: XGBoost optimized for elite detection
   - **Input Features**: Raw athletic metrics, percentiles, ratios, composites
   - **Performance**: ~97.6% accuracy for elite vs regular D1 classification

**Output**: Elite D1 probability for each player

#### **Sub-Stage 3B: P4 Classification (Elite Subset Only)**

**Purpose**: Among elite D1 players, predict P4 offers

**Process**:
1. **Training Subset**: Only players classified as "Elite D1" 
   - Better class balance: ~33% P4, ~67% Non-P4 (vs original 33%/67% overall)
   - Cleaner signal: No noise from obvious non-P4 players

2. **P4 Model Training**: XGBoost optimized for P4 prediction
   - **Input Features**: All 41 engineered features
   - **Optimization Metric**: `((accuracy * 2) + recall_0 + (recall_1 * 3)) / 6`
   - **Performance**: ~60% accuracy, ~53% balanced accuracy on elite subset

**Output**: P4 probability for each player (trained on elite examples)

#### **Sub-Stage 3C: Soft Hierarchical Combination**

**Purpose**: Combine elite and P4 probabilities for final prediction

**Process**:
1. **Weighted Average Formula** (improved from multiplication):
   ```python
   Final_P4_Prob = (Elite_Prob * 0.6) + (P4_Prob * 0.4)
   ```

2. **Threshold Optimization**: Find optimal decision boundary for classification

3. **Final Classification**: Convert probabilities to binary P4/Non-P4 predictions

---

## 📈 Performance Metrics

### **D1 Ensemble Model Results**:
- **Test Accuracy**: 79.16%
- **Test F1-Score**: 0.6584
- **Test F-beta Score**: 0.7177 (β=0.7, emphasizes recall)
- **Model Weights**: XGBoost (35%), CatBoost (35%), LightGBM (25%), SVM (5%)

### **P4 Hierarchical Ensemble Results**:
- **Overall Accuracy**: 78.5%
- **P4 Precision**: 88.9% (when model predicts P4, it's right 89% of the time)
- **P4 Recall**: 42.9% (captures 43% of actual P4 players)
- **Balanced Accuracy**: 69.2%
- **Model Weights**: XGBoost (30%), CatBoost (35%), LightGBM (17.5%), SVM (17.5%)

### **End-to-End Pipeline Performance**:
| Stage | Model Type | Accuracy | Key Metric |
|-------|------------|----------|------------|
| D1 Classification | Weighted Ensemble | 79.16% | F-beta: 0.7177 |
| P4 Prediction | Hierarchical Ensemble | 78.5% | P4 Precision: 88.9% |
| **Combined System** | **Two-Stage Pipeline** | **~79% → 78.5%** | **High Precision Focus** |

---

## 🔧 Key Technical Innovations

### **1. Hierarchical Architecture**
- **Stage 1 (Elite Detection)**: Solves the easier problem of identifying elite vs regular D1 players
- **Stage 2 (P4 Classification)**: Focuses on the harder problem within the elite subset only
- **Stage 3 (Combination)**: Merges predictions using weighted average for balanced results

### **2. Improved Class Balance**
- **Original Problem**: 33% P4, 67% Non-P4 (challenging for ML)
- **Elite Subset**: 33% P4, 67% Non-P4 among elite players (cleaner signal)
- **Elite Detection**: 45% Elite, 55% Regular (more balanced binary classification)

### **3. Feature Engineering Pipeline**
- **D1 Metadata**: Leverages existing D1 model predictions as meta-features
- **Multi-Scale Features**: Raw metrics, percentiles, ratios, composites, binary thresholds
- **Domain Knowledge**: Baseball-specific features (position profiles, regional preferences)

### **4. Soft Hierarchical Weighting**
- **Non-Punitive**: Weighted average instead of multiplication preserves predictions for "diamond in the rough" players
- **Hierarchical Logic**: Elite probability still heavily influences final prediction (60% weight)
- **Recruiting-Realistic**: Allows for non-elite players with high P4 potential

---

## 📁 File Structure

```
/backend/ml/models/models_inf/
├── README.md                                    # This comprehensive documentation
├── models_d1_or_not_inf/                      # D1 Classification Models
│   ├── README.md                              # D1 model documentation  
│   ├── ensemble_model.pkl                     # Main weighted ensemble classifier
│   ├── ensemble_scaler.pkl                    # StandardScaler for SVM preprocessing
│   ├── ensemble_metadata.pkl                  # Performance metrics & configuration
│   ├── xgboost_model.pkl                      # Individual XGBoost model
│   ├── catboost_model.pkl                     # Individual CatBoost model
│   ├── lightgbm_model.pkl                     # Individual LightGBM model
│   ├── svm_model.pkl                          # Individual SVM model
│   └── [model_name]_params.pkl               # Hyperparameters for each model
│
└── models_p4_or_not_inf/                      # P4 Classification Models
    ├── README.md                              # P4 model documentation
    ├── elite_detection_xgboost_model.pkl      # Elite D1 detection model
    ├── p4_ensemble_xgboost_model.pkl          # XGBoost P4 classifier (30% weight)
    ├── p4_ensemble_catboost_model.pkl         # CatBoost P4 classifier (35% weight)
    ├── p4_ensemble_lightgbm_model.pkl         # LightGBM P4 classifier (17.5% weight)
    ├── p4_ensemble_svm_model.pkl              # SVM P4 classifier (17.5% weight)
    ├── feature_scaler_for_svm.pkl             # StandardScaler for SVM preprocessing
    ├── model_config_and_metadata.pkl          # Thresholds, features, metrics
    └── production_prediction_pipeline.py      # Reference implementation
```

---

## 🚀 Usage

### **For Production Inference**:
```python
import joblib
import pandas as pd
import numpy as np

# Load D1 models
d1_ensemble = joblib.load('models_d1_or_not_inf/ensemble_model.pkl')
d1_scaler = joblib.load('models_d1_or_not_inf/ensemble_scaler.pkl')
d1_metadata = joblib.load('models_d1_or_not_inf/ensemble_metadata.pkl')

# Load P4 models  
elite_model = joblib.load('models_p4_or_not_inf/elite_detection_xgboost_model.pkl')
p4_xgb = joblib.load('models_p4_or_not_inf/p4_ensemble_xgboost_model.pkl')
p4_cat = joblib.load('models_p4_or_not_inf/p4_ensemble_catboost_model.pkl')
p4_lgb = joblib.load('models_p4_or_not_inf/p4_ensemble_lightgbm_model.pkl')
p4_svm = joblib.load('models_p4_or_not_inf/p4_ensemble_svm_model.pkl')
p4_scaler = joblib.load('models_p4_or_not_inf/feature_scaler_for_svm.pkl')
p4_config = joblib.load('models_p4_or_not_inf/model_config_and_metadata.pkl')

# Complete prediction pipeline
def predict_player_d1_and_p4(player_data):
    # Step 1: Engineer features (41 features required)
    features = engineer_features(player_data)
    
    # Step 2: D1 prediction
    d1_probability = d1_ensemble.predict_proba([features])[0][1]
    d1_prediction = d1_probability >= 0.5
    
    # Step 3: If D1, proceed to P4 prediction
    if d1_prediction:
        # Elite detection
        elite_prob = elite_model.predict_proba([features])[0][1]
        
        # P4 ensemble predictions
        p4_xgb_prob = p4_xgb.predict_proba([features])[0][1]
        p4_cat_prob = p4_cat.predict_proba([features])[0][1] 
        p4_lgb_prob = p4_lgb.predict_proba([features])[0][1]
        p4_svm_prob = p4_svm.predict_proba([p4_scaler.transform([features])])[0][1]
        
        # Weighted ensemble
        p4_ensemble_prob = (p4_xgb_prob * 0.3 + p4_cat_prob * 0.35 + 
                           p4_lgb_prob * 0.175 + p4_svm_prob * 0.175)
        
        # Hierarchical combination
        hierarchical_p4_prob = (elite_prob * 0.6) + (p4_ensemble_prob * 0.4)
        p4_prediction = hierarchical_p4_prob >= p4_config['optimal_threshold']
        
        return {
            'd1_probability': d1_probability,
            'd1_prediction': d1_prediction,
            'p4_probability': hierarchical_p4_prob,
            'p4_prediction': p4_prediction,
            'elite_probability': elite_prob
        }
    else:
        return {
            'd1_probability': d1_probability,
            'd1_prediction': d1_prediction,
            'p4_probability': 0.0,  # Non-D1 players get 0% P4 chance
            'p4_prediction': False,
            'elite_probability': 0.0
        }
```

### **Input Requirements**:
- **Raw player metrics**: height, weight, 60-yard dash, exit velocity, infield velocity
- **Positional data**: primary position, handedness
- **Geographic data**: player region
- **Feature engineering**: Must create all 41 engineered features matching training pipeline

### **Output**:
- **D1 Classification**: probability and binary prediction
- **P4 Classification**: hierarchical probability and binary prediction (if D1-eligible)
- **Elite Detection**: probability of being elite D1 player
- **Confidence levels**: based on probability thresholds

---

## 🎯 Business Impact

### **Recruiting Advantages**:
1. **Higher Accuracy**: 75.7% vs 57.8% baseline - more reliable recruiting decisions
2. **Better Talent Capture**: 48.4% P4 recall vs 28.3% - identifies nearly half of actual P4 prospects  
3. **Reduced False Positives**: 72.3% P4 precision - fewer wasted recruiting resources
4. **Hierarchical Insights**: Separate elite detection and P4 classification provide additional recruiting intelligence

### **ROI Considerations**:
- **Confusion Matrix**: 94 True Positives, 36 False Positives, 129 False Negatives
- **Trade-off**: Recruit 36 extra players to capture 94 actual P4 prospects
- **Cost-Benefit**: Much better than missing 129 P4 players (False Negatives in baseline)

---

## 🔮 Future Enhancements

1. **Ensemble P4 Model**: Combine multiple algorithms for P4 prediction stage
2. **Position-Specific Models**: Separate P4 models for different positions
3. **Temporal Features**: Incorporate player development trends and age factors
4. **Competition Strength**: Weight predictions by high school competition level
5. **Transfer Portal Integration**: Extend model to predict transfer portal success

---

## 📚 Technical Notes

- **Random State**: All models use `random_state=42` for reproducibility
- **Cross-Validation**: 5-fold stratified CV for model validation
- **Hyperparameter Optimization**: Optuna framework with 25-30 trials per model
- **Data Preprocessing**: Outlier clipping, inf/nan handling, feature scaling where appropriate
- **Evaluation Metrics**: Comprehensive analysis including confusion matrices, classification reports, and business-relevant metrics

---

*This pipeline represents a significant advancement in baseball recruiting analytics, combining machine learning sophistication with domain expertise to deliver actionable recruiting intelligence.*