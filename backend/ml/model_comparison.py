import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelComparison:
    def __init__(self, models_dir='models/'):
        self.models_dir = models_dir
        self.models = {}
        self.results = {}
        
    def load_models(self):
        """Load trained models from disk"""
        import os
        
        model_files = {
            'XGBoost': 'xgboost_model.pkl',
            'LightGBM': 'lightgbm_model.pkl', 
            'CatBoost': 'catboost_model.pkl',
            'Random Forest': 'random_forest_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            filepath = os.path.join(self.models_dir, filename)
            if os.path.exists(filepath):
                self.models[model_name] = joblib.load(filepath)
                print(f"Loaded {model_name} model")
            else:
                print(f"Model file not found: {filepath}")
    
    def compare_models(self, X_test, y_test, feature_names):
        """Compare performance of all loaded models"""
        print("=== Model Comparison ===")
        
        comparison_results = []
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Metrics
            accuracy = (y_pred == y_test).mean()
            
            # ROC AUC
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except:
                roc_auc = None
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store results
            result = {
                'Model': model_name,
                'Accuracy': accuracy,
                'ROC_AUC': roc_auc,
                'CV_Mean': cv_mean,
                'CV_Std': cv_std
            }
            comparison_results.append(result)
            
            # Detailed classification report
            print(f"\n{model_name} Classification Report:")
            print(classification_report(y_test, y_pred))
            
            # Store for later analysis
            self.results[model_name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'accuracy': accuracy,
                'roc_auc': roc_auc
            }
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        print("\n=== Model Comparison Summary ===")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_confusion_matrices(self, y_test):
        """Plot confusion matrices for all models"""
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (model_name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(y_test, result['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Non D1', 'Non P4 D1', 'Power 4 D1'],
                       yticklabels=['Non D1', 'Non P4 D1', 'Power 4 D1'],
                       ax=axes[i])
            axes[i].set_title(f'{model_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{self.models_dir}confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_test):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, result in self.results.items():
            if result['roc_auc'] is not None:
                # One-vs-rest ROC curves
                from sklearn.preprocessing import label_binarize
                y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
                
                for i in range(3):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], result['y_pred_proba'][:, i])
                    plt.plot(fpr, tpr, label=f'{model_name} - Class {i} (AUC = {result["roc_auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - One-vs-Rest')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.models_dir}roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_importance(self, feature_names, top_n=15):
        """Analyze and plot feature importance across models"""
        importance_data = []
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, importance in enumerate(importances):
                    importance_data.append({
                        'Model': model_name,
                        'Feature': feature_names[i],
                        'Importance': importance
                    })
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            
            # Plot top features for each model
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, model_name in enumerate(self.models.keys()):
                model_importance = importance_df[importance_df['Model'] == model_name]
                top_features = model_importance.nlargest(top_n, 'Importance')
                
                axes[i].barh(range(len(top_features)), top_features['Importance'])
                axes[i].set_yticks(range(len(top_features)))
                axes[i].set_yticklabels(top_features['Feature'])
                axes[i].set_title(f'{model_name} - Top {top_n} Features')
                axes[i].set_xlabel('Importance')
            
            plt.tight_layout()
            plt.savefig(f'{self.models_dir}feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print top features for each model
            print("\n=== Feature Importance Summary ===")
            for model_name in self.models.keys():
                model_importance = importance_df[importance_df['Model'] == model_name]
                top_features = model_importance.nlargest(top_n, 'Importance')
                print(f"\n{model_name} - Top {top_n} Features:")
                for _, row in top_features.iterrows():
                    print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    def generate_report(self, output_file='model_comparison_report.txt'):
        """Generate a comprehensive comparison report"""
        with open(f'{self.models_dir}{output_file}', 'w') as f:
            f.write("=== INFIELDER MODEL COMPARISON REPORT ===\n\n")
            
            # Model performance summary
            f.write("MODEL PERFORMANCE SUMMARY:\n")
            f.write("-" * 50 + "\n")
            
            for model_name, result in self.results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  ROC AUC: {result['roc_auc']:.4f if result['roc_auc'] else 'N/A'}\n")
            
            # Recommendations
            f.write("\n\nRECOMMENDATIONS:\n")
            f.write("-" * 50 + "\n")
            
            # Find best model by accuracy
            best_accuracy_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
            f.write(f"Best model by accuracy: {best_accuracy_model[0]} ({best_accuracy_model[1]['accuracy']:.4f})\n")
            
            # Find best model by ROC AUC
            valid_roc_models = [(name, result) for name, result in self.results.items() if result['roc_auc'] is not None]
            if valid_roc_models:
                best_roc_model = max(valid_roc_models, key=lambda x: x[1]['roc_auc'])
                f.write(f"Best model by ROC AUC: {best_roc_model[0]} ({best_roc_model[1]['roc_auc']:.4f})\n")
            
            f.write("\nKey Insights:\n")
            f.write("1. Class imbalance is a significant challenge (Power 4 D1 is only 8.8% of data)\n")
            f.write("2. Consider ensemble methods for better performance\n")
            f.write("3. Feature engineering could improve model performance\n")
            f.write("4. Regular retraining with new data is recommended\n")
        
        print(f"Report saved to {self.models_dir}{output_file}")

def main():
    """Main comparison function"""
    print("Starting Model Comparison...")
    
    # Initialize comparison
    comparison = ModelComparison()
    
    # Load models
    comparison.load_models()
    
    if not comparison.models:
        print("No models found. Please train models first.")
        return
    
    # Load test data (you'll need to run the training script first)
    # This is a placeholder - you'll need to load your test data
    print("Note: This script requires test data. Please run the training script first.")
    
    # Example usage (uncomment when you have test data):
    # comparison.compare_models(X_test, y_test, feature_names)
    # comparison.plot_confusion_matrices(y_test)
    # comparison.plot_roc_curves(y_test)
    # comparison.analyze_feature_importance(feature_names)
    # comparison.generate_report()

if __name__ == "__main__":
    main() 