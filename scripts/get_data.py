# import data and load into dataframe
# functions for:
# - load raw data from data folder
# - split data into IQ and SJ
# - split data into test, cross-eval split
import random
import pandas as pd


from typing import List, Tuple, Dict


def split_data(data, city):
    # Split the data into two cities - IQ and SJ
    data = data[data["city"] == city]
    return data
