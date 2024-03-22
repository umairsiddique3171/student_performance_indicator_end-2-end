import pickle
import sys
import os

import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException


def save_object(file_path,obj):

    try :
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logging.info("utils.save_object() used for saving object")

    except Exception as e: 
        raise CustomException(e,sys)