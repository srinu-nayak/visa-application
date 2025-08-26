from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    try:
        DataIngestion().data_ingestion()
    except Exception as e:
        raise CustomException(e)