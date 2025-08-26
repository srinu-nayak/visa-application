from dotenv import load_dotenv
import os
from pymongo import MongoClient
import pandas as pd
from urllib.parse import quote_plus
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException

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
