import pandas as pd
import numpy as np
import os
import sys
from  src.customer_segment.logger import logging
from src.customer_segment.exception import customexception
from src.customer_segment.components_segmentation.data_transformation import data_transformation
from src.customer_segment.components_segmentation.model_trainer import model_trainer

from src.customer_segment.utils.utils import saveobj
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class dataingestionconfig:
    raw_data_path = os.path.join("artifects_segmentation", 'raw data.csv')
    

class data_ingestion:
    def __init__(self):
        self.dataingestionconfig = dataingestionconfig
    
    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data_ingestion")

            df = pd.read_csv(r"C:\Users\khush\Python, 12-7\Practice\Projects\Customer_segment_project_ete\notebooks\data\marketing_campaign.csv", sep = '\t')
            
            os.makedirs(os.path.dirname(self.dataingestionconfig.raw_data_path),exist_ok= True)
            df.to_csv(self.dataingestionconfig.raw_data_path,index = False, header=True)

            #return self.dataingestionconfig.raw_data_path
            return df




        except Exception as e:
            raise customexception(e,sys)
        



if __name__ == "__main__":
    obj = data_ingestion()
    df = obj.initiate_data_ingestion()

    obj2 = data_transformation()
    preprocessed_array = obj2.data_transformation_fn(df)

    obj3 = model_trainer()
    final_df = obj3.trainer(preprocessed_array)