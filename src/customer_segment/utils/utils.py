import sys
import os
import dill
import pandas as pd
import numpy as np

from src.customer_segment.logger import logging
from src.customer_segment.exception import customexception


def saveobj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=  True)

        with open(file_path, 'wb') as object:
            dill.dump(obj, object)




    except Exception as e:
        raise customexception(e,sys)