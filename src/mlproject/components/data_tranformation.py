import logging
from datetime import date
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from src.mlproject.exception import CustomException
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataTransformationConfig:
    preprocessor:Path = Path("artifacts/preprocessor.pkl")

class DataTransformation():
    def __init__(self):
        self.config = DataTransformationConfig()

    @staticmethod
    def preprocess_df(df):
        df = df.drop(columns=['case_id'])
        current_year = date.today().year
        df['company_age'] = current_year - df['yr_of_estab']
        df = df.drop(columns=['yr_of_estab'])
        return df


    def data_transform(self, train_df, test_df):
        try:
            # Wrap in a FunctionTransformer
            preprocessor = FunctionTransformer(self.preprocess_df)
            # Apply to train and test
            train_transformed = preprocessor.fit_transform(train_df)
            test_transformed = preprocessor.transform(test_df)

            print(type(train_transformed))
            print(type(test_transformed))




        except Exception as e:
            raise CustomException(e)

