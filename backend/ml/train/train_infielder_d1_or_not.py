import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import joblib
import warnings
warnings.filterwarnings('ignore')

class InfielderModelTrainer:
    def __init__(self, data_path='../../data/hitters/vae_infielders.csv'):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        
    def load_and_preprocess_data(self, exclude_columns=None):
        """Load and preprocess the infielder data"""
        print("Loading data...")
        df: pd.DataFrame = pd.read_csv(self.data_path)
        
        # Remove problematic columns with too many missing values
        if exclude_columns is None:
            exclude_columns = ['hard_hit_p', 'position_velo']  # Keep inf_velo
        
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
        df = pd.DataFrame(df[keep_columns])
        
        print(f"Original shape: {df.shape}")
        print(f"Excluded columns: {exclude_columns}")
        
        # Filter to only keep rows where inf_velo is not missing
        if 'inf_velo' in df.columns:
            df = pd.DataFrame(df.dropna(subset=['inf_velo']))
            print(f"Shape after filtering for valid inf_velo: {df.shape}")
        
        # Handle missing values for remaining columns
        df = self._handle_missing_values(df)
        
        # Create binary target: D1 vs Non-D1
        target_col = 'three_section_commit_group'
        X = df.drop(columns=[target_col])
        y_original = df[target_col]
        
        # Create binary target: 0 = Non D1, 1 = D1 (Non P4 D1 + Power 4 D1)
        y = (y_original != 'Non D1').astype(int)
        
        # Store original target encoder for reference
        self.label_encoders['original_target'] = LabelEncoder().fit(y_original)
        self.label_encoders['binary_target'] = LabelEncoder().fit(y)
        
        print(f"Binary target distribution:")
        print(f"  Non D1 (0): {sum(y == 0)} samples")
        print(f"  D1 (1): {sum(y == 1)} samples")
        print(f"  D1 percentage: {sum(y == 1) / len(y) * 100:.1f}%")
        
        # Encode categorical variables
        X, feature_encoders = self._encode_categorical_features(X)
        self.label_encoders.update(feature_encoders)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
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
    
    def _handle_class_imbalance(self, X_train, y_train, method='smote'):
        """Handle class imbalance using various techniques"""
        if method == 'smote':
            smote = SMOTE(random_state=42)
            result = smote.fit_resample(X_train, y_train)
            X_resampled, y_resampled = result[0], result[1]
        elif method == 'smoteenn':
            smoteenn = SMOTEENN(random_state=42)
            result = smoteenn.fit_resample(X_train, y_train)
            X_resampled, y_resampled = result[0], result[1]
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=42)
            result = undersampler.fit_resample(X_train, y_train)
            X_resampled, y_resampled = result[0], result[1]
        else:
            X_resampled, y_resampled = X_train, y_train
        
        print(f"Original class distribution: {np.bincount(y_train)}")
        print(f"Resampled class distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test, feature_names):
        """Train LightGBM model"""
        print("\n=== Training LightGBM ===")
        
        # Handle class imbalance
        X_resampled, y_resampled = self._handle_class_imbalance(X_train, y_train, 'smote')
        
        # LightGBM parameters
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,  # Suppress all LightGBM warnings
            'random_state': 42
        }
        
        # Train model with warnings suppressed
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lgb_model = lgb.LGBMClassifier(**lgb_params)
            lgb_model.fit(X_resampled, y_resampled)
        
        # Predictions
        y_pred = lgb_model.predict(X_test)
        y_pred_proba = lgb_model.predict_proba(X_test)
        
        # Evaluate
        self._evaluate_model(lgb_model, X_test, y_test, y_pred, y_pred_proba, 'LightGBM')
        
        # Store model and feature importance
        self.models['LightGBM'] = lgb_model
        self.feature_importance['LightGBM'] = dict(zip(feature_names, lgb_model.feature_importances_))
        
        return lgb_model

    def _evaluate_model(self, model, X_test, y_test, y_pred, y_pred_proba, model_name):
        """Evaluate model performance"""
        print(f"\n{model_name} Results:")
        print("=" * 50)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non D1', 'D1']))
        
        # Confusion matrix
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # ROC AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Cross-validation scores for both ROC AUC and accuracy
        cv_roc_auc = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc')
        cv_accuracy = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
        print(f"Cross-validation ROC AUC: {cv_roc_auc.mean():.4f} (+/- {cv_roc_auc.std() * 2:.4f})")
        print(f"Cross-validation Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
        
        return {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'cv_roc_auc': cv_roc_auc,
            'cv_accuracy': cv_accuracy
        }
    
    def get_feature_importance_summary(self, top_n=10):
        """Get feature importance summary for all models"""
        print("\nFeature Importance Summary:")
        print("=" * 50)
        
        for model_name, importance_dict in self.feature_importance.items():
            print(f"\n{model_name}:")
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_features[:top_n]):
                print(f"  {i+1}. {feature}: {importance:.4f}")
    
    def save_models(self, output_dir='../models/'):
        """Save trained models and preprocessing artifacts"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the LightGBM model
        best_model = self.models['LightGBM']
        joblib.dump(best_model, os.path.join(output_dir, 'lgb_infield_d1_or_not.pkl'))
        
        # Save scaler
        joblib.dump(self.scalers, os.path.join(output_dir, 'scalers_infield.pkl'))
        
        # Save label encoders
        joblib.dump(self.label_encoders, os.path.join(output_dir, 'label_encoders_infield.pkl'))
        
        print(f"Models saved to {output_dir}")
        print("Files saved:")
        print("  - lgb_infield_d1_or_not.pkl")
        print("  - scalers_infield.pkl")
        print("  - label_encoders_infield.pkl")

def main():
    """Main training function"""
    print("Starting Infielder D1 vs Non-D1 Model Training (LightGBM)")
    print("=" * 60)
    
    # Initialize trainer
    trainer = InfielderModelTrainer()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names = trainer.load_and_preprocess_data()
    
    print(f"\nFeature names: {feature_names}")
    print(f"Number of features: {len(feature_names)}")
    
    # Train LightGBM model
    trainer.train_lightgbm(X_train, y_train, X_test, y_test, feature_names)
    
    # Get feature importance summary
    trainer.get_feature_importance_summary()
    
    # Save models
    trainer.save_models()
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 