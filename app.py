from flask import Flask, request, render_template

import numpy as np
import pandas as pd
import os
import sys

from src.customer_segment.pipelines.prediction_pipeline import PredictPipeline, df_converter
from src.customer_segment.logger import logging
from src.customer_segment.exception import customexception


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predict",methods = ['GET', 'POST'])
def predictor():
    if request.method == "GET":
        return render_template("home.html")
    elif request.method == "POST":
        try:
          logging.info("Ingesting values in DF")

          data = df_converter( ID = int(request.form.get("ID")),Year_Birth = int(request.form.get("Year_Birth")),
                              Education = request.form.get("Education"), Marital_Status = request.form.get("Marital_Status"),
                                Income = int(request.form.get("Income")), 
                              Kidhome = int(request.form.get("Kidhome")), Teenhome = int(request.form.get("Teenhome")),
                                Dt_Customer = request.form.get("Dt_Customer"),
                    Recency = int(request.form.get("Recency")), MntWines = int(request.form.get("MntWines")),
                      MntFruits = int(request.form.get("MntFruits")), MntMeatProducts= int(request.form.get("MntMeatProducts")), 
                      MntFishProducts= int(request.form.get("MntFishProducts")), MntSweetProducts= int(request.form.get("MntSweetProducts")),
                        MntGoldProds= int(request.form.get("MntGoldProds")),
                      NumDealsPurchases= int(request.form.get("NumDealsPurchases")), NumWebPurchases= int(request.form.get("NumWebPurchases")),
                        NumCatalogPurchases= int(request.form.get("NumCatalogPurchases")), NumStorePurchases= int(request.form.get("NumStorePurchases")),
                          NumWebVisitsMonth= int(request.form.get("NumWebVisitsMonth")),
                        AcceptedCmp3= int(request.form.get("AcceptedCmp3")), AcceptedCmp4= int(request.form.get("AcceptedCmp4")),
                          AcceptedCmp5= int(request.form.get("AcceptedCmp5")), AcceptedCmp1= int(request.form.get("AcceptedCmp1")), 
                          AcceptedCmp2= int(request.form.get("AcceptedCmp2")), Complain= int(request.form.get("Complain")), 
                          Z_CostContact= int(request.form.get("Z_CostContact")),
                          Z_Revenue= int(request.form.get("Z_Revenue")), Response= int(request.form.get("Response"))

          )
          logging.info("Now values ingested, converting to df")

          df = data.convert_to_df()
          logging.info("Converted to df")
          pipeline = PredictPipeline()
          logging.info("Now predicting the final answer")
          final_result = pipeline.predict(df)
          logging.info("Hurray, answer predicted")
          if final_result[0] == 0:
              response = "This customer is not gonna buy from us"
          elif final_result[0] == 1:
              response = "This is our customer, PAY ATTENTION!!"
          return render_template("home.html",results = response)
        except Exception as e:
            raise customexception(e,sys)

if __name__ == "__main__":
    app.run(debug = True)






































