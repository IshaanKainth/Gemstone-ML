import os
import sys
import pandas as pd
import numpy as np
from src.exceptions import CustomException
from src.logger import logging 
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split

@dataclass
class DataValidationConfig:
    pass

class DataValidation:
    def __init__(self):
        pass
    def initiate_data_validation(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise CustomException(e,sys) 
    