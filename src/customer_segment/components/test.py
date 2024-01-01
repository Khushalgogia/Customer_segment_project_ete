import pandas as pd
import numpy as np
import os
import sys
from src.customer_segment.utils.utils import load_model
from src.customer_segment.pipelines.prediction_pipeline import df_converter

preprocessor_path = os.path.join("artifect", "prediction_preprocessor.pkl")
transformed_path = os.path.join("transformed.csv")

preprocessor = load_model(preprocessor_path)

obj1 = df_converter(2174, 1954, 'Graduation', 'Single', 46344, 1, 1, '08-03-2014',
       38, 11, 1, 6, 2, 1, 6, 2, 1, 1, 2, 5, 0, 0, 0, 0, 0, 0, 3, 11, 0)
df = obj1.convert_to_df()


#df = pd.DataFrame(2174, 1954, 'Graduation', 'Single', 46344, 1, 1, '08-03-2014',
       #38, 11, 1, 6, 2, 1, 6, 2, 1, 1, 2, 5, 0, 0, 0, 0, 0, 0, 3, 11, 0)
my_df = preprocessor.fit_transform(df)
my_df = pd.DataFrame(my_df)
my_df.to_csv(transformed_path,index = False)




















