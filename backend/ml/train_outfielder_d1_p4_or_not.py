import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

class OutfielderD1P4ModelTrainer:
    def __init__(self, data_path='../data/hitters/vae_outfielders.csv'):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        
    def load_and_preprocess_data(self, exclude_columns=None):
        """Load and preprocess the outfielder data, filtering to only D1 players"""
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        
        # Remove problematic columns with too many missing values
        if exclude_columns is None:
            exclude_columns = ['hard_hit_p', 'position_velo']  # Keep of_velo
        
        # Also remove non-predictive columns and data leakage columns
        non_predictive = ['Unnamed: 0', 'name', 'link', 'commitment', 'college_location', 
                         'conf_short', 'committment_group', 'high_school', 'class', 'positions',
                         'player_section_of_region', 'confidence', 'age', 'player_state']
        
        # Remove data leakage columns that directly indicate the target
        leakage_columns = ['division', 'conference']
        
        exclude_columns.extend(non_predictive)
        exclude_columns.extend(leakage_columns)
        
        # Remove columns with too many missing values
        missing_threshold = 0.5  # 50% missing
        missing_counts = df.isnull().sum() / len(df)
        high_missing_cols = missing_counts[missing_counts > missing_threshold].index.tolist()
        exclude_columns.extend(high_missing_cols)
        
        # Remove missing indicator columns (we'll handle missing values differently)
        missing_indicator_cols = [col for col in df.columns if col.startswith('missing_')]
        exclude_columns.extend(missing_indicator_cols)
        
        # Keep only relevant columns
        keep_columns = [col for col in df.columns if col not in exclude_columns]
        df = df[keep_columns]
        
        print(f"Original shape: {df.shape}")
        print(f"Excluded columns: {exclude_columns}")
        
        # Filter to only keep rows where of_velo is not missing
        if 'of_velo' in df.columns:
            df = df.dropna(subset=['of_velo'])
            print(f"Shape after filtering for valid of_velo: {df.shape}")
        
        # Filter to only D1 players (Non P4 D1 + Power 4 D1)
        target_col = 'three_section_commit_group'
        d1_mask = df[target_col].isin(['Non P4 D1', 'Power 4 D1'])
        df = df[d1_mask]
        print(f"Shape after filtering for D1 players only: {df.shape}")
        
        # Handle missing values for remaining columns
        df = self._handle_missing_values(df)
        
        # Create binary target: Power 4 D1 vs Non-Power 4 D1
        X = df.drop(columns=[target_col])
        y_original = df[target_col]
        
        # Create binary target: 0 = Non P4 D1, 1 = Power 4 D1
        y = (y_original == 'Power 4 D1').astype(int)
        
        # Store original target encoder for reference
        self.label_encoders['original_target'] = LabelEncoder().fit(y_original)
        self.label_encoders['binary_target'] = LabelEncoder().fit(y)
        
        print(f"Binary target distribution:")
        print(f"  Non P4 D1 (0): {sum(y == 0)} samples")
        print(f"  Power 4 D1 (1): {sum(y == 1)} samples")
        print(f"  Power 4 D1 percentage: {sum(y == 1) / len(y) * 100:.1f}%")
        
        # Encode categorical variables
        X, feature_encoders = self._encode_categorical_features(X)
        self.label_encoders.update(feature_encoders)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        X_train_scaled, X_test_scaled, self.scalers = self._scale_features(X_train, X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def _encode_categorical_features(self, X):
        """Encode categorical features"""
        label_encoders = {}
        X_encoded = X.copy()
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            label_encoders[col] = le
        
        return X_encoded, label_encoders
    
    def _scale_features(self, X_train, X_test):
        """Scale numerical features"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, scaler
    
    def _handle_class_imbalance(self, X_train, y_train, method='none'):
        """Handle class imbalance using various techniques"""
        if method == 'smote':
            # Use SMOTE with k_neighbors=3 for smaller minority class
            smote = SMOTE(random_state=42, k_neighbors=3)
            result = smote.fit_resample(X_train, y_train)
            X_resampled, y_resampled = result[0], result[1]
        else:
            X_resampled, y_resampled = X_train, y_train
        
        print(f"Original class distribution: {np.bincount(y_train)}")
        print(f"Resampled class distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def train_xgboost(self, X_train, y_train, X_test, y_test, feature_names):
        """Train XGBoost model"""
        print("\n=== Training XGBoost ===")
        
        # Calculate class weights based on training data proportions
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = total_samples / (len(class_counts) * class_counts)
        weight_dict = dict(zip(range(len(class_counts)), class_weights))
        
        print(f"Class distribution: {class_counts}")
        print(f"Class weights: {weight_dict}")
        
        # Handle class imbalance - try without SMOTE first
        X_resampled, y_resampled = self._handle_class_imbalance(X_train, y_train, 'none')
        
        # XGBoost parameters for binary classification with balanced approach
        xgb_params = {
            'objective': 'binary:logistic',
            'max_depth': 3,  # Very shallow to prevent overfitting
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'random_state': 42,
            'scale_pos_weight': 2.0,  # Moderate weight for minority class
            'min_child_weight': 5,  # Higher to prevent overfitting
            'reg_alpha': 0.5,  # L1 regularization
            'reg_lambda': 2.0  # L2 regularization
        }
        
        # Train model
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_resampled, y_resampled)
        
        # Predictions
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)
        
        # Evaluate
        self._evaluate_model(xgb_model, X_test, y_test, y_pred, y_pred_proba, 'XGBoost')
        
        # Feature importance
        self.feature_importance['XGBoost'] = dict(zip(feature_names, xgb_model.feature_importances_))
        
        self.models['XGBoost'] = xgb_model
        return xgb_model
    
    def _evaluate_model(self, model, X_test, y_test, y_pred, y_pred_proba, model_name):
        """Evaluate model performance"""
        print(f"\n{model_name} Results:")
        print("=" * 50)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # ROC AUC (one-vs-rest)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])  # Use positive class probability
            print(f"\nROC AUC Score: {roc_auc:.4f}")
        except:
            print("Could not calculate ROC AUC")
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def get_feature_importance_summary(self, top_n=10):
        """Get feature importance summary across all models"""
        print("\n=== Feature Importance Summary ===")
        
        for model_name, importance_dict in self.feature_importance.items():
            print(f"\n{model_name} - Top {top_n} Features:")
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:top_n]:
                print(f"  {feature}: {importance:.4f}")
    
    def save_models(self, output_dir='../ml/models/'):
        """Save trained models and preprocessing objects"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            joblib.dump(model, f"{output_dir}/xgb_outfield_d1_p4_or_not.pkl")
        
        # Save preprocessing objects
        joblib.dump(self.scalers, f"{output_dir}/scalers_outfield_d1_p4.pkl")
        joblib.dump(self.label_encoders, f"{output_dir}/label_encoders_outfield_d1_p4.pkl")
        
        print(f"Models saved to {output_dir}")

def main():
    """Main training function"""
    print("Starting Outfielder D1 Power 4 vs Non-Power 4 Model Training...")
    
    # Initialize trainer
    trainer = OutfielderD1P4ModelTrainer()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names = trainer.load_and_preprocess_data()
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Train XGBoost model
    trainer.train_xgboost(X_train, y_train, X_test, y_test, feature_names)
    
    # Get feature importance summary
    trainer.get_feature_importance_summary()
    
    # Save models
    trainer.save_models()
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 