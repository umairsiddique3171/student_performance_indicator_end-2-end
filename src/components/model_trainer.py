import os 
import sys
import json
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_models,evaluate_models_with_tuning

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts","model.p")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,tuning=False):
        try:
            logging.info("Split training and test input data to features and target")
            X_train,y_train,X_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])

            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor(),
            }
            
            if tuning:
                with open(r'src\components\tuning_parameters.json', 'r') as file:
                    params = json.load(file)
                logging.info("Parameters Loaded")
                model_report:dict=evaluate_models_with_tuning(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            else:
                model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found with score greater the 60%")

            save_object(file_path=self.model_trainer_config.train_model_file_path,obj=best_model)

            predicted=best_model.predict(X_test)
            r2 = r2_score(y_test,predicted)

            return r2

        except Exception as e:
            raise CustomException(e,sys)