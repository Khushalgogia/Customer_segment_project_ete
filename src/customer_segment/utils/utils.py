import sys
import os
import dill
import pandas as pd
import numpy as np

from src.customer_segment.logger import logging
from src.customer_segment.exception import customexception
from sklearn.metrics import (accuracy_score, recall_score, f1_score)




def saveobj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=  True)

        with open(file_path, 'wb') as object:
            dill.dump(obj, object)




    except Exception as e:
        raise customexception(e,sys)
    


def evaluatemodel(xTrain,yTrain,xTest,yTest,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(xTrain,yTrain)

            y_test_pred = model.predict(xTest)
            acc_test = accuracy_score(y_test_pred,yTest)
            acc_train = accuracy_score(model.predict(xTrain),yTrain)
            recall_test = recall_score(y_test_pred,yTest)
            recall_train = recall_score(model.predict(xTrain),yTrain)
            f1_test = f1_score(y_test_pred,yTest)
            f1_train = f1_score(model.predict(xTrain),yTrain)
            report[list(models.keys())[i]] = [acc_train,acc_test,recall_train,recall_test,f1_train,f1_test]
            report_df = pd.DataFrame(report).T
            report_df = report_df.rename(columns = {0 : 'Train_Accuracy_score', 1 : 'Test_Accuracy_score',
                                                    2 : "Train_Recall_score", 3 : "Test_Recall_score",
                                                    4 : "Train_f1_score", 5 : "Test_f1_score"})
            
        return report_df
            







    except Exception as e:
        raise customexception(e,sys)
