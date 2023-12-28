import pandas as pd
import numpy as np
from dataclasses import dataclass
import sys
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

from src.customer_segment.exception import customexception
from src.customer_segment.logger import logging
from src.customer_segment.utils.utils import (saveobj, evaluatemodel)



@dataclass
class model_trainer_config:
    model_path = os.path.join("artifect", 'model.pkl')
    model_evaluation_report = os.path.join("artifect", "evaluation_report.csv")

class model_trainer:
    def __init__(self):
        self.model_trainer_config = model_trainer_config

    def model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Now Splitting the data")
            x_variable_train = train_arr[:,:-1]
            y_variable_train = train_arr[:,-1]
            x_variable_test = test_arr[:,:-1]
            y_variable_test = test_arr[:,-1]

            over_sampler = SMOTE()
            xTrain,yTrain = over_sampler.fit_resample(x_variable_train, y_variable_train)
            logging.info("SMOTE DONE")
            model1 = {
                "DecisionTree" : DecisionTreeClassifier(),
                "RandomForest" : RandomForestClassifier(),
                "GradientBoostingClassifier" : GradientBoostingClassifier(),
                "AdaBoostClassifier" : AdaBoostClassifier(),
                "BaggingClassifier" : BaggingClassifier(),
                "LogisticRegression" : LogisticRegression()
                
            }

            model_report, my_model = evaluatemodel(xTrain= xTrain, yTrain= yTrain, xTest=x_variable_test, yTest=y_variable_test,models=model1)

            logging.info("Extracting model_report and final_model")
            saveobj(file_path = self.model_trainer_config.model_path,
                    obj = my_model)

            model_report.to_csv(self.model_trainer_config.model_evaluation_report)
            logging.info("Final_model_report is here and model.pkl file is also here")
            
            return my_model



            
        except Exception as e:
            raise customexception(e,sys)










































