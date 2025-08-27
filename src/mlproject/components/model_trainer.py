from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass
from pathlib import Path
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
from datetime import datetime
today = datetime.now().strftime("%Y_%m_%d")
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.mlproject.utils import evaluate_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

@dataclass
class ModelTrainerConfig:
    model:Path = Path("artifacts/model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()



    def get_data_for_training(self, train_final, test_final):
        try:

            X_train = train_final.drop(columns=["case_status"])
            y_train = train_final["case_status"]

            X_test = test_final.drop(columns=["case_status"])
            y_test = test_final["case_status"]

            smt = SMOTEENN(random_state=42, sampling_strategy='minority')
            X_train_sampled, y_train_sampled = smt.fit_resample(X_train, y_train)
            X_test_sampled, y_test_sampled = X_test, y_test

            logging.info(f"Before sampling X_train: {X_train.shape}, y_train: {y_train.shape}")
            logging.info(f"#Before sampling X_test: {X_test.shape}, y_test: {y_test.shape}")
            logging.info(f"After sampling X_train: {X_train_sampled.shape}, y_train_sampled: {y_train_sampled.shape}")
            logging.info(f"After sampling X_test_sampled: {X_test_sampled.shape}, y_test_sampled: {y_test_sampled.shape}")

            #models
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "Support Vector Classifier": SVC(),
                "AdaBoost Classifier": AdaBoostClassifier()

            }

            #parameters
            params = {
                "XGBClassifier" : {
                    'max_depth': range(3, 10, 2),
                    'min_child_weight': range(1, 6, 2)
                    },

                "Random Forest" : {
                    "max_depth": [10, 12, None, 15, 20],
                    "max_features": ['sqrt', 'log2', None],
                    "n_estimators": [10, 50, 100, 200]
                    },

                "K-Neighbors Classifier" : {
                    "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    "weights": ['uniform', 'distance'],
                    "n_neighbors": [3, 4, 5, 7, 9],
                    },
                }

            report = evaluate_model(
                X_train_sampled, y_train_sampled, X_test_sampled, y_test_sampled, models, params
            )
            report.to_csv(f"artifacts/model_evaluation_report_{today}.csv", index=False)


        except Exception as e:
            raise CustomException(e)

