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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
today = datetime.now().strftime("%Y_%m_%d")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
import mlflow
import dagshub
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

dagshub.init(repo_owner='srinu-nayak', repo_name='visa-application', mlflow=True)



@dataclass
class ModelTrainerConfig:
    model: Path = Path("artifacts/model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def eval_metrics(self, y_true, y_pred):
        """
        Evaluate classification performance.

        Parameters:
        y_true : array-like of shape (n_samples,)
            True labels.
        y_pred : array-like of shape (n_samples,)
            Predicted labels.

        Returns:
        accuracy, precision, recall, f1 : float
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')  # use 'macro' for multiclass
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')

        return accuracy, precision, recall, f1

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
            model = best_model_instance.fit(X_train_sampled, y_train_sampled)
            y_pred = model.predict(X_test_sampled)

            # After predicting best model
            y_pred = model.predict(X_test_sampled)

            # Probabilities for ROC curve (if supported)
            try:
                y_prob = model.predict_proba(X_test_sampled)[:, 1]
            except:
                y_prob = None

            # Metrics
            accuracy, precision, recall, f1 = self.eval_metrics(y_test_sampled, y_pred)

            with mlflow.start_run():
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)
                mlflow.log_param("model_name", best_model_name)

                # Confusion matrix
                cm = confusion_matrix(y_test_sampled, y_pred)
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title("Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.savefig("confusion_matrix.png")
                plt.close()
                mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")

                # ROC curve
                if y_prob is not None:
                    fpr, tpr, _ = roc_curve(y_test_sampled, y_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(6, 4))
                    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
                    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
                    plt.title("ROC Curve")
                    plt.xlabel("FPR")
                    plt.ylabel("TPR")
                    plt.legend()
                    plt.savefig("roc_curve.png")
                    plt.close()
                    mlflow.log_artifact("roc_curve.png", artifact_path="plots")

                # Feature importance
                if hasattr(model, "feature_importances_"):
                    feature_importances = model.feature_importances_
                    feature_names = X_train_sampled.columns
                    plt.figure(figsize=(6, 4))
                    sns.barplot(x=feature_importances, y=feature_names)
                    plt.title("Feature Importances")
                    plt.xlabel("Importance")
                    plt.ylabel("Features")
                    plt.savefig("feature_importance.png")
                    plt.close()
                    mlflow.log_artifact("feature_importance.png", artifact_path="plots")

            logging.info(f"Best Model Selected: {best_model_name}")
            logging.info(f"Best Hyperparameters: {best_params}")

            # Save for deployment
            save_object(self.config.model, best_model_instance)
            logging.info(f"Best model saved at {self.config.model}")

            # Return trained model for further use
            return best_model_instance

        except Exception as e:
            raise CustomException(e)
