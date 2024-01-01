from flask import Flask, request, render_template

import numpy as np
import pandas as pd

from src.customer_segment.pipelines.prediction_pipeline import PredictPipeline, df_converter

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predict",methods = ['GET', 'POST'])
def predictor():
    if request.method == "GET":
        return render_template("home.html")
    elif request.method == "POST":
        data = df_converter( ID = request.form.get("ID"),Year_Birth = request.form.get("Year_Birth"),
                            Education = request.form.get("Education"), Marital_Status = request.form.get("Marital_Status"),
                              Income = request.form.get("Income"), 
                            Kidhome = request.form.get("Kidhome"), Teenhome = request.form.get("Teenhome"),
                              Dt_Customer = request.form.get("Dt_Customer"),
                  Recency = request.form.get("Recency"), MntWines = request.form.get("MntWines"),
                    MntFruits = request.form.get("MntFruits"), MntMeatProducts= request.form.get("MntMeatProducts"), 
                    MntFishProducts= request.form.get("MntFishProducts"), MntSweetProducts= request.form.get("MntSweetProducts"),
                      MntGoldProds= request.form.get("MntGoldProds"),
                    NumDealsPurchases= request.form.get("NumDealsPurchases"), NumWebPurchases= request.form.get("NumWebPurchases"),
                      NumCatalogPurchases= request.form.get("NumCatalogPurchases"), NumStorePurchases= request.form.get("NumStorePurchases"),
                        NumWebVisitsMonth= request.form.get("NumWebVisitsMonth"),
                      AcceptedCmp3= request.form.get("AcceptedCmp3"), AcceptedCmp4= request.form.get("AcceptedCmp4"),
                        AcceptedCmp5= request.form.get("AcceptedCmp5"), AcceptedCmp1= request.form.get("AcceptedCmp1"), 
                        AcceptedCmp2= request.form.get("AcceptedCmp2"), Complain= request.form.get("Complain"), 
                        Z_CostContact= request.form.get("Z_CostContact"),
                        Z_Revenue= request.form.get("Z_Revenue"), Response= request.form.get("Response")

        )

        df = data.convert_to_df()

        pipeline = PredictPipeline()
        final_result = pipeline.predict(df)
        return render_template("home.html",results = final_result[0])

if __name__ == "__main__":
    app.run(debug = True)






































