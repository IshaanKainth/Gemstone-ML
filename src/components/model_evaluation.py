import os
import sys
import pandas as pd
import numpy as np
from src.exceptions import CustomException
from src.logger import logging 
from src.utils.main_utils import load_object
from dataclasses import dataclass
from pathlib import Path
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


@dataclass
class ModelEvaluationConfig:
    pass

class ModelEvaluation:
    def __init__(self):
        pass

    def evaluate_metrices(self,actual,predicted):
        try:
            rmse=np.sqrt(mean_squared_error(actual,predicted))
            mae=mean_absolute_error(actual,predicted)
            r2=r2_score(actual,predicted)
            return rmse,mae,r2
        
        except Exception as e:
            logging.info('Exception occured during model evaluation')
            raise CustomException(e,sys)
        
    def initiate_model_evaluation(self,train_arr,test_arr):
        try:
            X_test,y_test=(test_arr[:,:-1],test_arr[:,-1])
            model_path=os.path.join('artifacts','model.pkl')
            model=load_object(file_path=model_path)
            y_pred=model.predict(X_test)
            (rmse,mae,r2)=self.evaluate_metrices(y_test,y_pred)

        except Exception as e:
            logging.info()
            raise CustomException(e,sys) 
    