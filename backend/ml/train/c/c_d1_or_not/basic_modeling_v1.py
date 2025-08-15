import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, balanced_accuracy_score, precision_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import optuna

csv_path = '/Users/ryankolodziejczyk/Documents/AI Baseball Recruitment/code/backend/data/hitters/of_feat_eng_d1_or_not.csv'
df = pd.read_csv(csv_path)

