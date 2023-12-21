from dataclasses import dataclass
import sys
import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE

from src.customer_segment.logger import logging
from src.customer_segment.exception import customexception


@dataclass
class datatransformationconfig:
    preprocesser_path = os.join.path("artifect", "prediction_preprocessor.pkl")

class data_transformation:
    def __init__(self):
        self.datatransformationconfig = datatransformationconfig()

        def get_data_transformer_obj(self):
            try:
                logging.info("starting the pipeline")
                def Preprocessor_transformer(dataset):
    
                    dataset = dataset.drop(columns = ['ID', 'Z_CostContact', 'Z_Revenue','Complain'])
                    dataset['Income'] = dataset['Income'].fillna(df['Income'].median())
                    
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
                    
                    #Target Variable

                    

                    dataset = normalize(dataset)
                    
                    
                    
                    return dataset


                def targetvalues_transform(dataset):
                    dataset['Final_Response'] = dataset['AcceptedCmp1'] + dataset['AcceptedCmp2'] + dataset['AcceptedCmp3'] + dataset['AcceptedCmp4'] + dataset['AcceptedCmp5'] + dataset['Response']
                    dataset['Final_Response'] = dataset['Final_Response'].apply(lambda x : 1 if x >= 1 else 0)
                    dataset = dataset.drop(columns = ['AcceptedCmp1', 'AcceptedCmp2','AcceptedCmp3','AcceptedCmp4', 'AcceptedCmp5', 'Response'])
                    return dataset
                    




                columns_for_SMOTE = ['Age', 'Income', 'Education', 'Total_Purchases']


                transformed_cols = ['Dt_Customer','Marital_Status','Education', 'Kidhome', 'Teenhome', 'MntWines', 'MntFruits', 
                                    'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                                'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases','Year_Birth',
                                
                                    'ID', 'Z_CostContact', 'Z_Revenue','Complain','Recency','NumWebVisitsMonth','Income']
                target_cols = ['AcceptedCmp1', 'AcceptedCmp2','AcceptedCmp3','AcceptedCmp4', 'AcceptedCmp5', 'Response']

                transformer = ColumnTransformer([
                    
                    ("Preprocessor_transformer", FunctionTransformer(Preprocessor_transformer), transformed_cols) ,
                    ("targetvalues_transform", FunctionTransformer(targetvalues_transform), target_cols),
                    
                    
                ],remainder= 'passthrough')



                cols = ['Income','Dt_Customer', 'Marital_Status', 'Education', 'Year_birth__Age',
                        'Childornot' ,'Amount spend' ,'total_num_purchases' ,'Final_Response' ,'Recency'  ,'NumWebVisitsMonth' ]
                logging.info("Pipeline build")

                return transformer

            
            except Exception as e:
                raise customexception(e,sys)

        def initiate_pipeline(self,train_path,test_path):
            try:
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
                logging.info("Completed reading the dataset")

                preprocessor_obj = self.get_data_transformer_obj()

                train_arr = preprocessor_obj.fit_transform(train_df)
                test_arr = preprocessor_obj.fit_transform(test_df)

                logging.info("Preprocessing done on dataset through our pipeline")

                















            except Exception as e:
                raise customexception(e,sys)




















