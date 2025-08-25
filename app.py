from src.mlproject.exception import CustomException


if __name__ == "__main__":
    try:
        1/0
    except Exception as e:
        raise CustomException(e)