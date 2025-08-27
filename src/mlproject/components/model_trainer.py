from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass
from pathlib import Path
from imblearn.combine import SMOTETomek, SMOTEENN




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

            logging.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            logging.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

            print("X_train shape: ", X_train.shape)
            print("y_train shape: ", y_train.shape)
            print("X_test shape: ", X_test.shape)
            print("y_test shape: ", y_test.shape)

        except Exception as e:
            raise CustomException(e)

