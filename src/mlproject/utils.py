from dotenv import load_dotenv
import os
from pymongo import MongoClient
import pandas as pd
from urllib.parse import quote_plus
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from pathlib import Path
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import label_binarize

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Load .env file
load_dotenv()

# Get credentials
user = os.getenv("MONGO_USER")
password = os.getenv("MONGO_PASSWORD")
host = os.getenv("MONGO_HOST")
db_name = os.getenv("MONGO_DB")
collection_name = os.getenv("MONGO_COLLECTION")

# Escape username & password for URL safety
user = quote_plus(user)
password = quote_plus(password)

def connectingToMongo():
    try:
        logging.info("Connecting to MongoDB")

        # MongoDB connection string

        mongo_uri = f"mongodb+srv://{user}:{password}@{host}/{db_name}?retryWrites=true&w=majority"
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        # # Fetch data into DataFrame
        df = pd.DataFrame(list(collection.find()))

        if '_id' in df.columns:
            df = df.drop('_id', axis=1)

            logging.info("Sending dataframe to data_ingestion")
        return df


    except Exception as e:
        raise CustomException(e)


def save_object(filename, obj):
    try:
        logging.info("Saving preprocessor object")
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    except Exception as ex:
        raise CustomException(ex)


def evaluate_clf(y_true, y_pred, y_proba=None):
    """
    Evaluates a classifier's predictions.
    Returns: accuracy, f1, precision, recall, roc_auc
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")

    # ROC AUC
    try:
        if y_proba is not None:  # use predicted probabilities if available
            auc = roc_auc_score(
                label_binarize(y_true, classes=np.unique(y_true)),
                y_proba,
                multi_class="ovr"
            )
        else:
            auc = None
    except:
        auc = None

    return acc, f1, prec, rec, auc


def evaluate_model(X_train_sampled, y_train_sampled, X_test_sampled, y_test_sampled, models: dict, params: dict):
    """
    Evaluates multiple classifiers with optional hyperparameter tuning.
    Returns a DataFrame containing metrics and best parameters.
    """
    try:
        logging.info("Started evaluating models")
        report = []

        for modelname, model in models.items():
            params_grid = params.get(modelname, {})

            # Hyperparameter tuning
            if params_grid:
                random = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=params_grid,
                    n_iter=100,
                    cv=3,
                    verbose=1,
                    n_jobs=-1,
                    scoring='accuracy'
                )
                random.fit(X_train_sampled, y_train_sampled)
                best_model = random.best_estimator_
                best_params = random.best_params_
            else:
                best_model = model
                best_model.fit(X_train_sampled, y_train_sampled)
                best_params = {}

            # Predictions
            y_train_pred = best_model.predict(X_train_sampled)
            y_test_pred = best_model.predict(X_test_sampled)

            # Probabilities for ROC AUC
            y_train_proba = best_model.predict_proba(X_train_sampled) if hasattr(best_model, "predict_proba") else None
            y_test_proba = best_model.predict_proba(X_test_sampled) if hasattr(best_model, "predict_proba") else None

            # Metrics
            tr_acc, tr_f1, tr_prec, tr_rec, tr_auc = evaluate_clf(y_train_sampled, y_train_pred, y_train_proba)
            te_acc, te_f1, te_prec, te_rec, te_auc = evaluate_clf(y_test_sampled, y_test_pred, y_test_proba)

            report.append({
                "Model": modelname,
                "Best_Params": best_params,
                "Train_Accuracy": tr_acc,
                "Test_Accuracy": te_acc,
                "Train_F1": tr_f1,
                "Test_F1": te_f1,
                "Train_Precision": tr_prec,
                "Test_Precision": te_prec,
                "Train_Recall": tr_rec,
                "Test_Recall": te_rec,
                "Train_AUC": tr_auc,
                "Test_AUC": te_auc
            })

        return pd.DataFrame(report).sort_values(by="Test_Accuracy", ascending=False)

    except Exception as ex:
        raise CustomException(ex)
