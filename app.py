from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_tranformation import DataTransformation


if __name__ == "__main__":
    try:
        train_df, test_df = DataIngestion().data_ingestion()
        DataTransformation().data_transform(train_df, test_df)
    except Exception as e:
        raise CustomException(e)