from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass
from pathlib import Path
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pandas as pd
from datetime import datetime
import ast
from src.mlproject.utils import evaluate_model, save_object

today = datetime.now().strftime("%Y_%m_%d")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

@dataclass
class ModelTrainerConfig:
    model: Path = Path("artifacts/model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def get_data_for_training(self, train_final, test_final):
        try:
            # Split features and target
            X_train = train_final.drop(columns=["case_status"])
            y_train = train_final["case_status"]
            X_test = test_final.drop(columns=["case_status"])
            y_test = test_final["case_status"]

            # Resample training data
            smt = SMOTEENN(random_state=42, sampling_strategy='minority')
            X_train_sampled, y_train_sampled = smt.fit_resample(X_train, y_train)
            X_test_sampled, y_test_sampled = X_test, y_test

            logging.info(f"Before sampling X_train: {X_train.shape}, y_train: {y_train.shape}")
            logging.info(f"Before sampling X_test: {X_test.shape}, y_test: {y_test.shape}")
            logging.info(f"After sampling X_train: {X_train_sampled.shape}, y_train_sampled: {y_train_sampled.shape}")
            logging.info(f"After sampling X_test: {X_test_sampled.shape}, y_test_sampled: {y_test_sampled.shape}")

            # Define models
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

            # Hyperparameter grids
            params = {
                "XGBClassifier" : {'max_depth': range(3, 10, 2),
                                   'min_child_weight': range(1, 6, 2)},
                "Random Forest" : {"max_depth": [10, 12, None, 15, 20],
                                   "max_features": ['sqrt', 'log2', None],
                                   "n_estimators": [10, 50, 100, 200]},
                "K-Neighbors Classifier" : {"algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                                            "weights": ['uniform', 'distance'],
                                            "n_neighbors": [3, 4, 5, 7, 9]}
            }

            # Evaluate models
            report = evaluate_model(
                X_train_sampled, y_train_sampled,
                X_test_sampled, y_test_sampled,
                models, params
            )
            report_path = f"artifacts/model_evaluation_report_{today}.csv"
            report.to_csv(report_path, index=False)
            logging.info(f"Model evaluation report saved at {report_path}")

            # Pick best model by Test_Accuracy
            best_model_row = report.sort_values(by="Test_Accuracy", ascending=False).iloc[0]
            best_model_name = best_model_row['Model']
            best_params_str = best_model_row.get('Best_Params', None)

            # Handle empty or malformed strings safely
            if not best_params_str or best_params_str in ["{}", "", "nan", "NaN"] or pd.isna(best_params_str):
                best_params = {}
            else:
                try:
                    best_params = ast.literal_eval(best_params_str)
                except Exception:
                    best_params = {}

            # Get original model instance and create a new one with best params
            model_instance = models[best_model_name]
            best_model_instance = type(model_instance)(**best_params)

            # Fit on full training data
            best_model_instance.fit(X_train_sampled, y_train_sampled)

            logging.info(f"Best Model Selected: {best_model_name}")
            logging.info(f"Best Hyperparameters: {best_params}")

            # Save for deployment
            save_object(self.config.model, best_model_instance)
            logging.info(f"Best model saved at {self.config.model}")

            # Return trained model for further use
            return best_model_instance

        except Exception as e:
            raise CustomException(e)
