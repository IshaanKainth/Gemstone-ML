import os
import sys
import pandas as pd
import numpy as np
from src.exceptions import CustomException
from src.logger import logging 
from src.utils.main_utils import save_object 
from dataclasses import dataclass
from pathlib import Path
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj=DataTransformationConfig()
    
    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated') 

            numerical_columns=['carat', 'depth','table', 'x', 'y', 'z']
            categorical_columns=['cut', 'color','clarity']

            cut_map=['Fair','Good','Very Good','Premium','Ideal']
            color_map=['D','E','F','G','H','I','J']
            clarity_map=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('pipeline initiated')

            num_pipeline=Pipeline(
                steps=[
                ('impute',SimpleImputer()),
                ('scaler',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('ordinal',OrdinalEncoder(categories=[cut_map,color_map,clarity_map])),
                ('scaler',StandardScaler())
                ]
            )
            preprocessor=ColumnTransformer([
            ("numerical_pipeline",num_pipeline,numerical_columns),
            ("categorical_pipeline",cat_pipeline,categorical_columns),
            ])

            return preprocessor
        
        except Exception as e:
            logging.info("Error in data transformation")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            preprocessor_obj=self.get_data_transformation()

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.preprocessor_obj.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            return(
                train_arr,
                test_arr
            )
            
        except Exception as e:
            logging.info("Error in data transformation")
            raise CustomException(e,sys) 
    

