import pandas as pd
import numpy as np

import os
import sys

from src.customer_segment.logger import logging
from src.customer_segment.exception import customexception
from src.customer_segment.utils.utils import saveobj

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class data_transformation_config:
    preprocessor_path = os.path.join("artifects_segmentation","preprocessor.pkl")
    preprocessed_data_path = os.path.join("artifects_segmentation","preprocessed_data.csv")

class data_transformation:
    def __init__(self):
        self.data_transformation_config = data_transformation_config

    def building_pipeline(self):
        try:
            logging.info("initiating pipeline")
            def Preprocessor_transformer(dataset):
        
                dataset = dataset.drop(columns = ['ID', 'Z_CostContact', 'Z_Revenue','Complain'])
                dataset['Income'] = dataset['Income'].fillna(dataset['Income'].median())
                
                dataset['Dt_Customer'] = pd.to_datetime(dataset['Dt_Customer'],format='%d-%M-%Y')
                max_date = max(dataset['Dt_Customer'])
                l1 = (max_date - dataset['Dt_Customer']).dt.days
                dataset['Dt_Customer'] = l1
                
                #Marriage
                status_update = {'Married' : "Couple", "Together" : "Couple",
                                "Alone" : "Single", "Absurd" : "Single", "YOLO": "Single", "Widow": "Single","Divorced" : "Single","Single":"Single"}

                dataset['Marital_Status'] = dataset['Marital_Status'].map(status_update)
                encoder_map = {"Couple":1, "Single" : 0}
                dataset['Marital_Status'] = dataset['Marital_Status'].map(encoder_map)
                
                # Education

                education_map = {"2n Cycle" : "High School", 'Basic' : "High School",
                                'Graduation' : "Bachelors",
                                'PhD': "Higher Education", "Master": "Higher Education"}
                
                dataset['Education'] = dataset['Education'].map(education_map)
                encoder_map = {"High School": 0, "Bachelors" : 1, "Higher Education" :2}
                dataset['Education'] = dataset['Education'].map(encoder_map)
                
                

                # kid_teenhome_combine
                
                dataset['childornot'] = dataset['Kidhome'] + dataset['Teenhome']
                dataset = dataset.drop(columns = ['Kidhome', 'Teenhome'])

                # Amount Spent
                dataset['Amount spend'] = dataset['MntWines'] + dataset['MntFruits'] + dataset['MntMeatProducts'] +dataset['MntFishProducts'] + dataset['MntSweetProducts'] + dataset['MntGoldProds']
                dataset = dataset.drop(columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'])
                
                # Total Purchases
                dataset['total_num_purchases'] = dataset['NumDealsPurchases'] + dataset['NumWebPurchases'] + dataset['NumCatalogPurchases'] + dataset['NumStorePurchases']
                dataset = dataset.drop(columns = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases'])
                
                # Find age from Year_Birth 

                current_year = 2023
                dataset['Year_Birth'] = current_year - dataset['Year_Birth'] 
                
                
                
                
                return dataset


            def targetvalues_transform(dataset):
                dataset['Final_Response'] = dataset['AcceptedCmp1'] + dataset['AcceptedCmp2'] + dataset['AcceptedCmp3'] + dataset['AcceptedCmp4'] + dataset['AcceptedCmp5'] + dataset['Response']
                dataset['Final_Response'] = dataset['Final_Response'].apply(lambda x : 1 if x >= 1 else 0)
                dataset = dataset.drop(columns = ['AcceptedCmp1', 'AcceptedCmp2','AcceptedCmp3','AcceptedCmp4', 'AcceptedCmp5', 'Response'])
                return dataset
                




            


            transformed_cols = ['Dt_Customer','Marital_Status','Education', 'Kidhome', 'Teenhome', 'MntWines', 'MntFruits', 
                                'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                            'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases','Year_Birth',
                            
                                'ID', 'Z_CostContact', 'Z_Revenue','Complain','Recency','NumWebVisitsMonth','Income']
            target_cols = ['AcceptedCmp1', 'AcceptedCmp2','AcceptedCmp3','AcceptedCmp4', 'AcceptedCmp5', 'Response']

            transformer = ColumnTransformer([
                
                ("Preprocessor_transformer", FunctionTransformer(Preprocessor_transformer), transformed_cols) ,
                ("targetvalues_transform", FunctionTransformer(targetvalues_transform), target_cols),
                
                
            ],remainder= 'passthrough')

            saveobj(file_path = self.data_transformation_config.preprocessor_path,obj = transformer)

            return transformer
        
        except Exception as e:
            raise customexception(e,sys)

    def data_transformation_fn(self,data_path):
        try:
            df = pd.read_csv(data_path)
            logging.info("Will transform the dataset")

            preprocessor = self.building_pipeline()

            preprocessed_data_array = preprocessor.fit_transform(df)
            preprocessed_columns = ['Dt_Customer','Marital_Status', 'Education', 'Current_Age',
                                        'Recency', 'NumWebVisitsMonth', 'Income', 'childornot',
                                        'total_amt_spent', 'totalnum_purchases','Target']

            my_df = pd.DataFrame(preprocessed_data_array,columns = preprocessed_columns)

            my_df.to_csv(self.data_transformation_config.preprocessed_data_path,header= True, index = False)



            return preprocessed_data_array
        except Exception as e:
            raise customexception(e,sys)

