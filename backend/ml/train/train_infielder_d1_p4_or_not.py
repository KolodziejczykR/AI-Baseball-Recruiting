import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import catboost as cb
import joblib
import warnings
warnings.filterwarnings('ignore')

class InfielderD1P4CatBoostTrainer:
    def __init__(self, data_path='../../data/hitters/vae_infielders.csv'):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_importance = None

    def load_and_preprocess_data(self, exclude_columns=None):
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        if exclude_columns is None:
            exclude_columns = ['hard_hit_p', 'position_velo']
        non_predictive = ['Unnamed: 0', 'name', 'link', 'commitment', 'college_location',
                         'conf_short', 'committment_group', 'high_school', 'class', 'positions',
                         'player_section_of_region', 'confidence', 'age', 'player_state']
        leakage_columns = ['division', 'conference']
        exclude_columns.extend(non_predictive)
        exclude_columns.extend(leakage_columns)
        missing_threshold = 0.5
        missing_counts = df.isnull().sum() / len(df)
        high_missing_cols = missing_counts[missing_counts > missing_threshold].index.tolist()
        exclude_columns.extend(high_missing_cols)
        missing_indicator_cols = [col for col in df.columns if col.startswith('missing_')]
        exclude_columns.extend(missing_indicator_cols)
        keep_columns = [col for col in df.columns if col not in exclude_columns]
        df = df[keep_columns]
        print(f"Original shape: {df.shape}")
        print(f"Excluded columns: {exclude_columns}")
        if 'inf_velo' in df.columns:
            df = df.dropna(subset=['inf_velo'])
            print(f"Shape after filtering for valid inf_velo: {df.shape}")
        target_col = 'three_section_commit_group'
        d1_mask = df[target_col].isin(['Non P4 D1', 'Power 4 D1'])
        df = df[d1_mask]
        print(f"Shape after filtering for D1 players only: {df.shape}")
        df = self._handle_missing_values(df)
        X = df.drop(columns=[target_col])
        y_original = df[target_col]
        y = (y_original == 'Power 4 D1').astype(int)
        self.label_encoders['original_target'] = LabelEncoder().fit(y_original)
        self.label_encoders['binary_target'] = LabelEncoder().fit(y)
        print(f"Binary target distribution:")
        print(f"  Non P4 D1 (0): {sum(y == 0)} samples")
        print(f"  Power 4 D1 (1): {sum(y == 1)} samples")
        print(f"  Power 4 D1 percentage: {sum(y == 1) / len(y) * 100:.1f}%")
        X, feature_encoders = self._encode_categorical_features(X)
        self.label_encoders.update(feature_encoders)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        X_train_scaled, X_test_scaled, self.scaler = self._scale_features(X_train, X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

    def _handle_missing_values(self, df):
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        return df

    def _encode_categorical_features(self, X):
        label_encoders = {}
        X_encoded = X.copy()
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            label_encoders[col] = le
        return X_encoded, label_encoders

    def _scale_features(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler

    def _handle_class_imbalance(self, X_train, y_train):
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"Original class distribution: {np.bincount(y_train)}")
        print(f"Resampled class distribution: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled

    def train_catboost(self, X_train, y_train, X_test, y_test, feature_names):
        print("\n=== Training CatBoost ===")
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = total_samples / (len(class_counts) * class_counts)
        weight_dict = dict(zip(range(len(class_counts)), class_weights))
        print(f"Class distribution: {class_counts}")
        print(f"Class weights: {weight_dict}")
        X_resampled, y_resampled = self._handle_class_imbalance(X_train, y_train)
        cb_params = {
            'iterations': 200,
            'depth': 6,
            'learning_rate': 0.1,
            'loss_function': 'Logloss',
            'random_state': 42,
            'verbose': 0,
            'class_weights': [weight_dict[0], weight_dict[1]]
        }
        cb_model = cb.CatBoostClassifier(**cb_params)
        cb_model.fit(X_resampled, y_resampled)
        y_pred = cb_model.predict(X_test)
        y_pred_proba = cb_model.predict_proba(X_test)
        print("\nCatBoost Results:")
        print("=" * 50)
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non P4 D1', 'Power 4 D1']))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        print(f"ROC AUC: {roc_auc:.4f}")
        cv_roc_auc = cross_val_score(cb_model, X_test, y_test, cv=5, scoring='roc_auc')
        cv_accuracy = cross_val_score(cb_model, X_test, y_test, cv=5, scoring='accuracy')
        print(f"Cross-validation ROC AUC: {cv_roc_auc.mean():.4f} (+/- {cv_roc_auc.std() * 2:.4f})")
        print(f"Cross-validation Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
        self.model = cb_model
        self.feature_importance = dict(zip(feature_names, cb_model.feature_importances_))
        return cb_model

    def save_model(self, output_dir='../models/'):
        import os
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(output_dir, 'cb_infield_d1_p4_or_not.pkl'))
        joblib.dump(self.scaler, os.path.join(output_dir, 'scalers_infield_d1_p4.pkl'))
        joblib.dump(self.label_encoders, os.path.join(output_dir, 'label_encoders_infield_d1_p4.pkl'))
        print(f"Models saved to {output_dir}")
        print("Files saved:")
        print("  - cb_infield_d1_p4_or_not.pkl")
        print("  - scalers_infield_d1_p4.pkl")
        print("  - label_encoders_infield_d1_p4.pkl")

def main():
    print("Starting Infielder D1 Power 4 vs Non-Power 4 Model Training (CatBoost)")
    trainer = InfielderD1P4CatBoostTrainer()
    X_train, X_test, y_train, y_test, feature_names = trainer.load_and_preprocess_data()
    print(f"\nFeature names: {feature_names}")
    print(f"Number of features: {len(feature_names)}")
    trainer.train_catboost(X_train, y_train, X_test, y_test, feature_names)
    trainer.save_model()
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 