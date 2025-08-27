from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_tranformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    try:
        train_df, test_df = DataIngestion().data_ingestion()
        train_final, test_final = DataTransformation().data_transform(train_df, test_df)
        ModelTrainer().get_data_for_training(train_final, test_final)
    except Exception as e:
        raise CustomException(e)