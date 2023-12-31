import pandas as pd
import numpy as np
from dataclasses import dataclass
import sys
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

from src.customer_segment.exception import customexception
from src.customer_segment.logger import logging
from src.customer_segment.utils.utils import (saveobj, evaluatemodel)

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

@dataclass
class model_trainer_config:
    model_trainer_path = os.path.join("artifects_segmentation", "cluster_predict.pkl")
    segmented_data_path = os.path.join("artifects_segmentation", "segmented_data.csv")

class model_trainer:
    def __init__(self):
        self.model_trainer_config = model_trainer_config

    def trainer(self, data):
        try:
            # We are making two clusters
            
            agg = AgglomerativeClustering(n_clusters = 2)
            
            segment = agg.fit_predict(data)
            data['segments'] = segment

            # Segmenting both of these customers
            data['segments'] = data['segments'].apply(lambda x : "Bad Customers" if x == 1 else "Good Customers")

            data.to_csv(self.model_trainer_config.segmented_data_path, index = False, header = True)

            saveobj(file_path= self.model_trainer_config.model_trainer_path, obj = agg)

            return data
        
        except Exception as e:
            raise customexception(e,sys)



































