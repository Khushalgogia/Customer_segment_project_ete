from src.customer_segment.utils import utils
import pandas as pd
import numpy as np
from src.customer_segment.logger import logging
from src.customer_segment.exception import customexception
from src.customer_segment.components.data_transformation import data_transformation
from src.customer_segment.components.model_trainer import model_trainer

import sys
import os

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class dataingestionconfig:
    raw_data_path = os.path.join("artifect", "raw.csv")
    Train_path = os.path.join("artifect", "train.csv") 
    Test_path = os.path.join("artifect", "test.csv") 
     

class dataingestion:
    def __init__(self):
        self.dataingestionconfig = dataingestionconfig()

    def initiate_data_ingestion(self):
        logging.info("Starting the data ingestion")
        try:
            df = pd.read_csv(r"C:\Users\khush\Python, 12-7\Practice\Projects\Customer_segment_project_ete\notebooks\data\marketing_campaign.csv", sep = '\t')

            os.makedirs(os.path.dirname(self.dataingestionconfig.raw_data_path),exist_ok = True)

            df.to_csv(self.dataingestionconfig.raw_data_path, index = False, header = True)

            train_dataset,test_dataset = train_test_split(df,test_size = 0.25, random_state = 101)

            train_dataset.to_csv(self.dataingestionconfig.Train_path, index = False, header = True)
            test_dataset.to_csv(self.dataingestionconfig.Test_path, index = False, header = True)

            logging.info("data ingestion completed, hurray")

            return (
                self.dataingestionconfig.Train_path,
                self.dataingestionconfig.Test_path,
                
            )




        except Exception as e:
            raise customexception(e,sys)



if __name__ == "__main__":

    obj = dataingestion()
    train_dataset,test_dataset = obj.initiate_data_ingestion()

    obj1 = data_transformation()
    train_arr,test_arr = obj1.initiate_pipeline(train_dataset,test_dataset)

    obj2 = model_trainer()
    report = obj2.model_trainer(train_arr,test_arr)
    print(report)

















