import os
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CatcherPredictionPipeline:
    def __init__(self, models_dir='models/'):
        """
        Initialize the catcher prediction pipeline.
        
        Args:
            models_dir: Directory containing the trained models and preprocessing artifacts
        """
        self.models_dir = models_dir
        self.d1_model = None
        self.p4_model = None
        self.d1_scaler = None
        self.p4_scaler = None
        self.d1_label_encoders = {}
        self.p4_label_encoders = {}
        self.feature_names = None
        
        # Load models and preprocessing artifacts
        self._load_models()
        
    def _load_models(self):
        """Load the trained models and preprocessing artifacts"""
        try:
            # Load D1 vs Non-D1 model and artifacts
            self.d1_model = joblib.load(os.path.join(self.models_dir, 'xgb_catcher_d1_or_not.pkl'))
            self.d1_scaler = joblib.load(os.path.join(self.models_dir, 'scalers_catcher.pkl'))
            self.d1_label_encoders = joblib.load(os.path.join(self.models_dir, 'label_encoders_catcher.pkl'))
            
            # Load Power 4 vs Non-Power 4 model and artifacts
            self.p4_model = joblib.load(os.path.join(self.models_dir, 'xgb_catcher_d1_p4_or_not.pkl'))
            self.p4_scaler = joblib.load(os.path.join(self.models_dir, 'scalers_catcher_d1_p4.pkl'))
            self.p4_label_encoders = joblib.load(os.path.join(self.models_dir, 'label_encoders_catcher_d1_p4.pkl'))
            
            print("Successfully loaded all models and preprocessing artifacts")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def _get_feature_columns(self) -> List[str]:
        """
        Get the list of feature columns used in training.
        Based on the exclusion logic from the training scripts.
        """
        # These are the columns that are excluded during training
        exclude_columns = [
            'hard_hit_p', 'position_velo',  # Keep c_velo
            'Unnamed: 0', 'name', 'link', 'commitment', 'college_location', 
            'conf_short', 'committment_group', 'high_school', 'class', 'positions',
            'player_section_of_region', 'confidence',
            'division', 'conference',  # Data leakage columns
            'three_section_commit_group'  # Target column
        ]
        
        # Add missing indicator columns (but keep number_of_missing and primary_position)
        missing_indicator_cols = [
            'missing_hand_speed_max', 'missing_bat_speed_max', 'missing_rot_acc_max',
            'missing_sixty_time', 'missing_thirty_time', 'missing_ten_yard_time',
            'missing_run_speed_max', 'missing_exit_velo_max', 'missing_exit_velo_avg',
            'missing_distance_max', 'missing_sweet_spot_p'
        ]
        exclude_columns.extend(missing_indicator_cols)
        
        # Define the expected feature columns in the exact order used during training
        # Based on the actual training output, these are the 21 features in the correct order
        feature_columns = [
            'primary_position', 'height', 'weight', 'throwing_hand', 'hitting_handedness',
            'hand_speed_max', 'bat_speed_max', 'rot_acc_max', 'sixty_time', 'thirty_time',
            'ten_yard_time', 'run_speed_max', 'exit_velo_max', 'exit_velo_avg', 'distance_max',
            'sweet_spot_p', 'c_velo', 'pop_time', 'player_region', 'number_of_missing'
        ]
        
        return feature_columns
    
    def _preprocess_input(self, input_data: Dict, is_d1_model: bool = True) -> np.ndarray:
        """
        Preprocess input data to match the format expected by the models.
        
        Args:
            input_data: Dictionary containing player statistics
            is_d1_model: Whether this is for the D1 model (True) or P4 model (False)
        
        Returns:
            Preprocessed feature array
        """
        # Get the feature columns for this model
        feature_columns = self._get_feature_columns()
        
        # Create a DataFrame with the input data
        df = pd.DataFrame([input_data])
        
        # Ensure all required features are present with reasonable defaults
        for col in feature_columns:
            if col not in df.columns:
                if col in ['height', 'weight', 'hand_speed_max', 'bat_speed_max', 
                          'rot_acc_max', 'sixty_time', 'thirty_time', 'ten_yard_time', 
                          'run_speed_max', 'exit_velo_max', 'exit_velo_avg', 'distance_max', 
                          'sweet_spot_p', 'c_velo', 'pop_time', 'number_of_missing']:
                    # Numerical features
                    default_values = {
                        'height': 70.0, 'weight': 170.0,
                        'hand_speed_max': 20.0, 'bat_speed_max': 70.0, 'rot_acc_max': 15.0,
                        'sixty_time': 7.0, 'thirty_time': 3.5, 'ten_yard_time': 1.8,
                        'run_speed_max': 20.0, 'exit_velo_max': 85.0, 'exit_velo_avg': 75.0,
                        'distance_max': 300.0, 'sweet_spot_p': 0.7, 'c_velo': 75.0,
                        'pop_time': 2.0, 'number_of_missing': 0
                    }
                    df[col] = default_values.get(col, 0.0)
                else:
                    # Categorical features
                    default_values = {
                        'player_state': 'Unknown', 'throwing_hand': 'R', 
                        'hitting_handedness': 'R', 'player_region': 'Unknown',
                        'primary_position': 'C'
                    }
                    df[col] = default_values.get(col, 'Unknown')
        
        # Select only the features used in training in the correct order
        df = pd.DataFrame(df[feature_columns])
        
        # Handle missing values
        df = self._handle_missing_values(df, is_d1_model)
        
        # Encode categorical variables
        df = self._encode_categorical_features(df, is_d1_model)
        
        # Scale numerical features
        df_scaled = self._scale_features(df, is_d1_model)
        
        return df_scaled
    
    def _handle_missing_values(self, df: pd.DataFrame, is_d1_model: bool) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df_processed = df.copy()
        
        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                # Use a reasonable default value for each metric
                default_values = {
                    'height': 70.0, 'weight': 170.0,
                    'hand_speed_max': 20.0,
                    'bat_speed_max': 70.0,
                    'rot_acc_max': 15.0,
                    'sixty_time': 7.0,
                    'thirty_time': 3.5,
                    'ten_yard_time': 1.8,
                    'run_speed_max': 20.0,
                    'exit_velo_max': 85.0,
                    'exit_velo_avg': 75.0,
                    'distance_max': 300.0,
                    'sweet_spot_p': 0.7,
                    'c_velo': 75.0,
                    'pop_time': 2.0,
                    'number_of_missing': 0  # Default to 0 missing values
                }
                df_processed[col].fillna(default_values.get(col, df[col].median()), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                # Use reasonable defaults for categorical variables
                default_values = {
                    'player_state': 'Unknown',
                    'throwing_hand': 'R',
                    'hitting_handedness': 'R',
                    'player_region': 'Unknown',
                    'primary_position': 'C'  # Default to catcher
                }
                mode_values = df[col].mode()
                fallback_value = default_values.get(col, mode_values.iloc[0] if len(mode_values) > 0 else 'Unknown')
                df_processed[col].fillna(fallback_value, inplace=True)
        
        return df_processed
    
    def _encode_categorical_features(self, df: pd.DataFrame, is_d1_model: bool) -> pd.DataFrame:
        """Encode categorical features using the saved label encoders"""
        df_encoded = df.copy()
        label_encoders = self.d1_label_encoders if is_d1_model else self.p4_label_encoders
        
        # Define categorical columns based on the training data
        categorical_cols = ['throwing_hand', 'hitting_handedness', 'player_region', 'primary_position']
        
        for col in categorical_cols:
            if col in df_encoded.columns and col in label_encoders:
                le = label_encoders[col]
                # Handle unseen categories by using a default value
                df_encoded[col] = df_encoded[col].astype(str)
                # Map unseen categories to the most common category
                df_encoded[col] = df_encoded[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
                df_encoded[col] = le.transform(df_encoded[col])
        
        return df_encoded
    
    def _scale_features(self, df: pd.DataFrame, is_d1_model: bool) -> np.ndarray:
        """Scale numerical features using the saved scaler"""
        scaler = self.d1_scaler if is_d1_model else self.p4_scaler
        if scaler is None:
            raise ValueError("Scaler not loaded. Please train the models first.")
        return scaler.transform(df)
    
    def predict(self, input_data: Dict) -> Dict:
        """
        Make predictions using the two-stage pipeline.
        
        Args:
            input_data: Dictionary containing player statistics
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Stage 1: Predict D1 vs Non-D1
            if self.d1_model is None:
                raise ValueError("D1 model not loaded. Please train the models first.")
            d1_features = self._preprocess_input(input_data, is_d1_model=True)
            d1_proba = self.d1_model.predict_proba(d1_features)[0]
            d1_prediction = self.d1_model.predict(d1_features)[0]
            
            # Get probability of being D1 (class 1)
            d1_probability = d1_proba[1]
            
            # Stage 2: If predicted as D1, predict Power 4 vs Non-Power 4
            if d1_prediction == 1:
                if self.p4_model is None:
                    raise ValueError("P4 model not loaded. Please train the models first.")
                p4_features = self._preprocess_input(input_data, is_d1_model=False)
                p4_proba = self.p4_model.predict_proba(p4_features)[0]
                p4_prediction = self.p4_model.predict(p4_features)[0]
                
                # Get probability of being Power 4 (class 1)
                p4_probability = p4_proba[1]
                
                # Calculate final probabilities
                # Non D1 probability = 1 - D1 probability
                # D1 Non P4 probability = D1 probability * (1 - P4 probability)
                # Power 4 probability = D1 probability * P4 probability
                
                final_probabilities = {
                    'Non D1': 1 - d1_probability,
                    'Non P4 D1': d1_probability * (1 - p4_probability),
                    'Power 4 D1': d1_probability * p4_probability
                }
                
                # Determine final prediction
                if p4_prediction == 1:
                    final_prediction = 'Power 4 D1'
                    confidence = p4_probability
                    stage = 'D1 + Power 4'
                else:
                    final_prediction = 'Non P4 D1'
                    confidence = 1 - p4_probability
                    stage = 'D1 + Non P4'
                
            else:
                # Predicted as Non D1
                final_probabilities = {
                    'Non D1': 1 - d1_probability,
                    'Non P4 D1': 0.0,
                    'Power 4 D1': 0.0
                }
                final_prediction = 'Non D1'
                confidence = 1 - d1_probability
                stage = 'Non D1'
            
            # Normalize probabilities to sum to 1.0
            prob_sum = sum(final_probabilities.values())
            if prob_sum > 0:
                for k in final_probabilities:
                    final_probabilities[k] = float(final_probabilities[k]) / prob_sum

            return {
                'prediction': final_prediction,
                'probabilities': final_probabilities,
                'confidence': confidence,
                'stage': stage
            }
            
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'prediction': None,
                'probabilities': None,
                'confidence': 0.0,
                'stage': 'Error'
            }
    
    def get_required_features(self) -> List[str]:
        """Get the list of required features for prediction"""
        return self._get_feature_columns()
    
    def get_feature_info(self) -> Dict:
        """Get information about the features used in the model"""
        return {
            'numerical_features': [
                'height', 'weight', 'hand_speed_max', 'bat_speed_max', 'rot_acc_max',
                'sixty_time', 'thirty_time', 'ten_yard_time', 'run_speed_max',
                'exit_velo_max', 'exit_velo_avg', 'distance_max', 'sweet_spot_p',
                'c_velo', 'pop_time', 'number_of_missing'
            ],
            'categorical_features': [
                'throwing_hand', 'hitting_handedness', 'player_region', 'primary_position'
            ],
            'feature_descriptions': {
                'height': 'Player height in inches',
                'weight': 'Player weight in pounds',
                'hand_speed_max': 'Maximum hand speed (mph)',
                'bat_speed_max': 'Maximum bat speed (mph)',
                'rot_acc_max': 'Maximum rotational acceleration',
                'sixty_time': '60-yard dash time (seconds)',
                'thirty_time': '30-yard dash time (seconds)',
                'ten_yard_time': '10-yard dash time (seconds)',
                'run_speed_max': 'Maximum running speed (mph)',
                'exit_velo_max': 'Maximum exit velocity (mph)',
                'exit_velo_avg': 'Average exit velocity (mph)',
                'distance_max': 'Maximum hit distance (feet)',
                'sweet_spot_p': 'Sweet spot percentage (0-1)',
                'c_velo': 'Catcher velocity (mph)',
                'pop_time': 'Pop time (seconds)',
                'number_of_missing': 'Number of missing values in player data',
                'throwing_hand': 'Throwing hand (L/R)',
                'hitting_handedness': 'Hitting handedness (L/R/S)',
                'player_region': 'Player region',
                'primary_position': 'Primary position (C)'
            }
        } 