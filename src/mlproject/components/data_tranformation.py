import logging
from datetime import date
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, FunctionTransformer, PowerTransformer, \
    OrdinalEncoder
from datetime import date
import pandas as pd
from src.mlproject.utils import save_object


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
    def custom_preprocessor(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop case_id, compute company_age from yr_of_estab,
        drop yr_of_estab, return modified dataframe
        """
        df = df.drop(columns=['case_id'])
        current_year = date.today().year
        df['company_age'] = current_year - df['yr_of_estab']
        df = df.drop(columns=['yr_of_estab'])
        return df

    def build_pipeline(self, df: pd.DataFrame) -> Pipeline:
        df_processed = self.custom_preprocessor(df.copy())

        numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df_processed.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
        oh_columns = ['continent', 'unit_of_wage', 'region_of_employment']
        or_columns = [col for col in categorical_cols if col not in oh_columns]

        numeric_pipeline = Pipeline([
            ('power', PowerTransformer(method='yeo-johnson')),
            ('scaler', StandardScaler())
        ])

        ordinal_pipeline = Pipeline([
            ('ordinal_encoder', OrdinalEncoder())
        ])

        one_hot_pipeline = Pipeline([
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
        ])

        column_transformer = ColumnTransformer([
            ('num', numeric_pipeline, numeric_cols),
            ('ord', ordinal_pipeline, or_columns),
            ('ohe', one_hot_pipeline, oh_columns)
        ])

        pipeline = Pipeline([
            ('custom_preprocess', FunctionTransformer(self.custom_preprocessor)),
            ('transform', column_transformer)
        ])

        return pipeline

    def data_transform(self, train_df, test_df):
        try:

            input_feature_train_df = train_df.drop(columns=['case_status'])
            input_feature_test_df = test_df.drop(columns=['case_status'])

            target_feature_train_df = train_df['case_status']
            target_feature_test_df = test_df['case_status']

            label_encoder = LabelEncoder()
            target_feature_train_df_transformed = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df_transformed = label_encoder.transform(target_feature_test_df)

            preprocessor = self.build_pipeline(input_feature_train_df)
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Get transformed feature names
            feature_names = preprocessor.named_steps['transform'].get_feature_names_out()

            train_df_transformed = pd.DataFrame(input_feature_train_arr, columns=feature_names)
            test_df_transformed = pd.DataFrame(input_feature_test_arr, columns=feature_names)

            target_series = pd.Series(
                target_feature_train_df_transformed,
                name="case_status",
                index=train_df_transformed.index  # keep same index
            )

            train_final = pd.concat([train_df_transformed, target_series], axis=1)

            target_series_test = pd.Series(
                target_feature_test_df_transformed,
                name="case_status",
                index=test_df_transformed.index
            )

            test_final = pd.concat([test_df_transformed, target_series_test], axis=1)


            save_object(
                self.config.preprocessor,
                preprocessor

            )

            return train_final, test_final


        except Exception as e:
            raise CustomException(e)

