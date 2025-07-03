import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json

# === EDIT THESE VARIABLES ===
# List the feature column names to use for training
FEATURES = ["exit_velocity", "sixty_yard_dash", "height", "weight"]  # <-- Replace with actual feature names
# Name of the target column
TARGET = "group"  # <-- Replace with actual target column name
# ===========================

# Load data
DATA_PATH = "../data/hitter_training_data.csv"
df = pd.read_csv(DATA_PATH)

# Split data
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = CatBoostClassifier(verbose=0)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")


# TODO: Optimize model
"""
# Save model
joblib.dump(model, "../data/hitter_model.pkl")

# Save features list
with open("../data/hitter_features.json", "w") as f:
    json.dump(FEATURES, f) 
""" 