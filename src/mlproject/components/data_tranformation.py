from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer, PowerTransformer, LabelEncoder
from datetime import date
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from src.mlproject.utils import save_object
from src.mlproject.exception import CustomException


@dataclass
class DataTransformationConfig:
    preprocessor: Path = Path("artifacts/preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    @staticmethod
    def custom_preprocessor(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop 'case_id' safely, compute 'company_age', drop 'yr_of_estab'.
        """
        df = df.drop(columns=['case_id'], errors='ignore')  # avoids KeyError
        current_year = date.today().year
        if 'yr_of_estab' in df.columns:
            df['company_age'] = current_year - df['yr_of_estab']
            df = df.drop(columns=['yr_of_estab'])
        return df

    def build_pipeline(self, df: pd.DataFrame) -> Pipeline:
        # Apply custom preprocessing first
        df_processed = self.custom_preprocessor(df.copy())

        # Identify numeric and categorical columns
        numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df_processed.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

        # Explicitly define which categorical columns get one-hot vs ordinal
        one_hot_cols = ['continent', 'unit_of_wage', 'region_of_employment']
        ordinal_cols = [col for col in categorical_cols if col not in one_hot_cols]

        numeric_pipeline = Pipeline([
            ('power', PowerTransformer(method='yeo-johnson')),
            ('scaler', StandardScaler())
        ])

        ordinal_pipeline = Pipeline([
            ('ordinal', OrdinalEncoder())
        ])

        one_hot_pipeline = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
        ])

        column_transformer = ColumnTransformer([
            ('num', numeric_pipeline, numeric_cols),
            ('ord', ordinal_pipeline, ordinal_cols),
            ('ohe', one_hot_pipeline, one_hot_cols)
        ])

        pipeline = Pipeline([
            ('custom_preprocess', FunctionTransformer(self.custom_preprocessor, validate=False)),
            ('transform', column_transformer)
        ])

        return pipeline

    def data_transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        try:
            # Separate input and target
            X_train = train_df.drop(columns=['case_status'])
            X_test = test_df.drop(columns=['case_status'])
            y_train = train_df['case_status']
            y_test = test_df['case_status']

            # Encode target
            label_encoder = LabelEncoder()
            y_train_enc = label_encoder.fit_transform(y_train)
            y_test_enc = label_encoder.transform(y_test)

            # Build and fit pipeline
            preprocessor = self.build_pipeline(X_train)
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # Get feature names safely
            feature_names = preprocessor.named_steps['transform'].get_feature_names_out()
            train_transformed = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train.index)
            test_transformed = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test.index)

            # Combine with target
            train_final = pd.concat([train_transformed, pd.Series(y_train_enc, name='case_status', index=X_train.index)], axis=1)
            test_final = pd.concat([test_transformed, pd.Series(y_test_enc, name='case_status', index=X_test.index)], axis=1)

            # Save preprocessor
            save_object(self.config.preprocessor, preprocessor)

            return train_final, test_final

        except Exception as e:
            raise CustomException(e)
