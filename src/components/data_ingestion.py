import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts/data_ingestion", "train.csv")
    test_data_path = os.path.join("artifacts/data_ingestion", "test.csv")
    raw_data_path = os.path.join("artifacts/data_ingestion", "raw.csv")


# notebook/data/income_cleandata.csv

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initialte_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            logging.info("Data Reading using Pandas library from local system")
            data = pd.read_csv(os.path.join("notebook/data", "income_cleandata.csv"))
            logging.info("Data Reading completed")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)  # Just makes the directory not the actual file
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            
            logging.info("Data Splitted into train and test")
            train_set, test_set = train_test_split(data, test_size=0.3, random_state=42)
            

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.info("Error Occured in Data Ingestion Stage")
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initialte_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    modelTrainer = ModelTrainer()
    print(modelTrainer.initiate_model_trainer(train_array=train_array, test_array=test_arr))


# src/components/data_ingestion.py



