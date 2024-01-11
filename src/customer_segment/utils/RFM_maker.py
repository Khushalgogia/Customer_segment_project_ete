import pandas as pd
import numpy as np
import os
import sys
from src.customer_segment.exception import customexception

try:
#Reading the dataset
    df = pd.read_csv(r"C:\Users\khush\Python, 12-7\Practice\Projects\Customer_segment_project_ete\notebooks\data\E-com_Data.csv")

    # Taking out only the relevant columns

    df = df[['CustomerID', 'InvoieNo', 'Date of purchase','Price']]

    #Dropping the null values of CustomerID column

    df = df.dropna(subset = ['CustomerID'])

    #Dropping Duplicate values

    df = df[~(df.duplicated())]

    #Changing the dtype to datetime for further processing

    df['Date of purchase'] = pd.to_datetime(df['Date of purchase'])

    #Renaming the column

    df = df.rename(columns = {"Date of purchase" : "Date"})

    # Processing only first 5000 rows as processing the whole data will take alot of time
    df = df.iloc[:5000,:]

    # RFM Approach

    df = df.groupby("CustomerID").agg({"InvoieNo" : lambda x : x.count(), 
                                "Date" : lambda x : max(df['Date']) - x.max(),
                                "Price" : lambda x : x.sum()})
    df = df.rename(columns = {"Date" : "Recency","InvoieNo" : "Frequency", "Price" : "Monetary"})

    #Saving the file
   
    rfm_path = os.path.join("artifect", "RFM.csv")
    os.makedirs(os.path.dirname(rfm_path), exist_ok= True)
    df.to_csv(rfm_path, header = True, index = True)

except Exception as e:
    raise customexception(e,sys)











