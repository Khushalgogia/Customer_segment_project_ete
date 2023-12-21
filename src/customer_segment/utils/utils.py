import sys
import os
import dill
import pandas as pd
import numpy as np

from src.customer_segment.logger import logging
from src.customer_segment.exception import customexception


def saveobj(file_path, obj):
    try:
        pass
    except Exception as e:
        customexception(e,sys)