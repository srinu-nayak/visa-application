from src.mlproject.utils import connectingToMongo
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from pathlib import Path


@dataclass
class DataIngestionConfig:
    artifacts_dir:Path = Path("artifacts").mkdir(parents=True, exist_ok=True)
    raw_data_path: Path = Path("artifacts/raw.csv")
    train_data_path: Path = Path("artifacts/train.csv")
    test_data_path: Path = Path("artifacts/test.csv")

class DataIngestion:

    def __init__(self):
        self.config = DataIngestionConfig()

    def data_ingestion(self):
        try:
            df = connectingToMongo()

            logging.info(f"saved the raw data")
            df.to_csv(self.config.raw_data_path, index=False, header=True)

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            logging.info(f"saved the train data")
            train_df.to_csv(self.config.train_data_path, index=False, header=True)

            logging.info(f"saved the test data")
            test_df.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info(f"sendind train_df and test_df from data_ingestion to data_transforamtion {train_df.shape}, {test_df.shape}")
            return train_df, test_df

        except Exception as e:
            raise CustomException(e)

