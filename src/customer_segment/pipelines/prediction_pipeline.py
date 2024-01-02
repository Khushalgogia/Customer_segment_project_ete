import pandas as pd
import numpy as np
from src.customer_segment.exception import customexception
from src.customer_segment.logger import logging
import sys
import os

from src.customer_segment.utils.utils import saveobj, load_model
from src.customer_segment.components.data_transformation import data_transformation

class PredictPipeline:
    def __init__(self):
        self.data_transformation = data_transformation

    def predict(self, data):
        try:
            logging.info("Starting my app.py file")
            model_path = os.path.join("artifect", "model.pkl")
            preprocessor_path = os.path.join("artifect", "prediction_preprocessor.pkl")
            transformed_data = os.path.join("pipeline", "transformed_data.csv")
            logging.info("preprocessor and transformed data path created")
            datatransformer = data_transformation()
            preprocessor = datatransformer.get_data_transformer_obj()
            model_predict = load_model(model_path)
            #preprocessor = load_model(preprocessor_path)
            logging.info("Now starting the transformation of preprocessor")
            transformed_data = preprocessor.fit_transform(data)
            logging.info("Data Transformed")
            transformed_data = transformed_data[:,:-1]
            logging.info("target column removed")
            
            predicted_data = model_predict.predict(transformed_data)
            logging.info(f"Answer predicted : {predicted_data}")
            return predicted_data
        except Exception as e:
            raise customexception(e,sys)
        


class df_converter:
    def __init__(self,ID, Year_Birth, Education, Marital_Status, Income, Kidhome, Teenhome, Dt_Customer,
                  Recency, MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds,
                    NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth,
                      AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, AcceptedCmp1, AcceptedCmp2, Complain, Z_CostContact,
                        Z_Revenue, Response):
        self.ID = ID
        self.Year_Birth = Year_Birth
        self.Education = Education
        self.Marital_Status = Marital_Status
        self.Income = Income
        self.Kidhome = Kidhome
        self.Teenhome = Teenhome
        self.Dt_Customer = Dt_Customer
        self.Recency = Recency
        self.MntWines = MntWines
        self.MntFruits = MntFruits
        self.MntMeatProducts = MntMeatProducts
        self.MntFishProducts = MntFishProducts
        self.MntSweetProducts = MntSweetProducts
        self.MntGoldProds = MntGoldProds
        self.NumDealsPurchases = NumDealsPurchases
        self.NumWebPurchases = NumWebPurchases
        self.NumCatalogPurchases = NumCatalogPurchases
        self.NumStorePurchases = NumStorePurchases
        self.NumWebVisitsMonth = NumWebVisitsMonth
        self.AcceptedCmp3 = AcceptedCmp3
        self.AcceptedCmp4 = AcceptedCmp4
        self.AcceptedCmp5 = AcceptedCmp5
        self.AcceptedCmp1 = AcceptedCmp1
        self.AcceptedCmp2 = AcceptedCmp2
        self.Complain = Complain
        self.Z_CostContact = Z_CostContact
        self.Z_Revenue = Z_Revenue
        self.Response = Response
    
    def convert_to_df(self):
        try:
            logging.info("creating the df")
            custom_dic = {
                'ID' : [self.ID], 'Year_Birth' : [self.Year_Birth], 'Education' : [self.Education], 'Marital_Status' : [self.Marital_Status], 
                'Income' : [self.Income], 'Kidhome' : [self.Kidhome], 
                'Teenhome' : [self.Teenhome], 'Dt_Customer' : [pd.to_datetime(self.Dt_Customer, format='%d-%m-%Y')],
                  'Recency': [self.Recency], 
                'MntWines': [self.MntWines], 'MntFruits': [self.MntFruits], 'MntMeatProducts': [self.MntMeatProducts],
                  'MntFishProducts': [self.MntFishProducts], 
                'MntSweetProducts': [self.MntSweetProducts], 'MntGoldProds': [self.MntGoldProds], 
                'NumDealsPurchases': [self.NumDealsPurchases], 'NumWebPurchases': [self.NumWebPurchases], 
                'NumCatalogPurchases': [self.NumCatalogPurchases], 
                'NumStorePurchases': [self.NumStorePurchases], 'NumWebVisitsMonth': [self.NumWebVisitsMonth],
                  'AcceptedCmp3': [self.AcceptedCmp3], 'AcceptedCmp4': [self.AcceptedCmp4], 'AcceptedCmp5': [self.AcceptedCmp5], 
                  'AcceptedCmp1': [self.AcceptedCmp1], 
                'AcceptedCmp2': [self.AcceptedCmp2], 'Complain': [self.Complain],
                  'Z_CostContact': [self.Z_CostContact], 'Z_Revenue': [self.Z_Revenue], 'Response': [self.Response]
            }
            logging.info("Values ingested")

            return pd.DataFrame(custom_dic)
        except Exception as e:
            raise customexception(e,sys)
        

if __name__ == "__main__":
    obj1 = df_converter(2174, 1954, 'Graduation', 'Single', 46344, 1, 1, '08-03-2014',
       38, 11, 1, 6, 2, 1, 6, 2, 1, 1, 2, 5, 0, 0, 0, 0, 0, 0, 3, 11, 0)
    df = obj1.convert_to_df()

    obj2 = PredictPipeline()
    ans = obj2.predict(df)
    print(ans[0])














