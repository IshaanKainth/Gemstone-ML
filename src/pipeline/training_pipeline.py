# src/pipeline/training_pipeline.py

import os
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer 
from src.components.model_evaluation import ModelEvaluation 

from src.logger import logging
from src.exceptions import CustomException


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        self.model_evaluation = ModelEvaluation()

    def run_data_ingestion(self):
        logging.info("Starting data ingestion...")
        train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed.")
        return train_data_path, test_data_path

    def run_data_transformation(self, train_path, test_path):
        logging.info("Starting data transformation...")
        train_arr, test_arr = self.data_transformation.initiate_data_transformation(train_path, test_path)
        logging.info("Data transformation completed.")
        return train_arr, test_arr

    def run_model_training(self, train_arr, test_arr):
        logging.info("Starting model training...")
        self.model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info("Model training completed.")

    def run_model_evaluation(self, train_arr, test_arr):
        logging.info("Starting model evaluation...")
        self.model_evaluation.initiate_model_evaluation(train_arr, test_arr)
        logging.info("Model evaluation completed.")

    def run_pipeline(self):
        try:
            train_data_path, test_data_path = self.run_data_ingestion()
            train_arr, test_arr = self.run_data_transformation(train_data_path, test_data_path)
            self.run_model_training(train_arr, test_arr)
            self.run_model_evaluation(train_arr, test_arr)
        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}")
            raise CustomException(e,sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
