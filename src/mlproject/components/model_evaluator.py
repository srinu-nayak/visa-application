import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
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
import ast



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

report = pd.read_csv("model_evaluation_report_2025_08_27.csv")
best_model_row = report.sort_values(by="Test_Accuracy", ascending=False).iloc[0]
best_model_name = best_model_row['Model']
best_params = ast.literal_eval(best_model_row['Best_Params'])  # convert string to dict

# Get the model instance from your models dictionary
model_instance = models[best_model_name]

# Create a new instance with best params
best_model_instance = type(model_instance)(**best_params)

print(best_model_instance)