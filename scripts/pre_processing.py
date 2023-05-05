
import pandas as pd
import numpy as np
import datetime as dt 
from sklearn.model_selection import TimeSeriesSplit

from typing import List, Tuple, Dict


def merge_data(train_features: pd.DataFrame, train_target: pd.DataFrame, test_features: pd.DataFrame, inc_test=False) -> pd.DataFrame:
    """_summary_

    Args:
        train_features (pd.DataFrame): _description_
        train_target (pd.DataFrame): _description_
        test (pd.DataFrame): _description_
        inc_test (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """    
    # Combine the train features with the target into one data frame 
    merged_data = pd.concat([train_features, train_target['total_cases']], axis=1)
    
    # if setting met, then combine also the test for feature engineering etc
    if inc_test == True:
        merged_data = pd.concat([merged_data, test_features], axis=0) 
    
    return merged_data


def pre_process(data: pd.DataFrame, city: str, remove_anomalies=False, inc_test=False) -> pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        city (str): _description_

    Returns:
        pd.DataFrame: _description_
    """    
    # First split data by city 
    data = data.loc[data["city"] == city, :]

    # Remove any unnecessary columns
    data = data.drop(['city'], axis=1)

    # Change date to date_time 
    data['week_start_date'] = pd.to_datetime(data['week_start_date'])

    # Removing missing values
    if inc_test == True:
        data['total_cases'] = data['total_cases'].fillna(-1)
        data = data.fillna(method='ffill')
    else:
        data = data.fillna(method='ffill')

    # Remove anomalies 
    if remove_anomalies:
        if city == 'sj':
            data = data.loc[data['total_cases'] < 350, :] 
        elif city == 'iq':
            data = data.loc[data['total_cases'] < 80, :] 
            data = data.iloc[50:, :]
    
    return data



def train_cv_split(data: pd.DataFrame, city: str) -> Tuple:
    """_summary_

    Args:
        data (pd.DataFrame) 
        city (str): 'iq' or 'sj'

    Returns:
        Tuple of pd.DataFrames 

    Split the provided training data into a training and crossvalidation set
    Not performed for iq due to insufficient data 

    """
    x = data.loc[:, data.columns!=['total_cases']]
    y = data.loc[:, 'total_cases']
    if city == 'sj':
        X_train = x.iloc[:650, :]
        X_test = x.iloc[650:, :]
        y_train = y.iloc[:650, :]
        y_test = y.iloc[650:, :]
    
    elif city == 'iq':
        X_train = x
        X_test = x
        y_train = y
        y_test = y

    return X_train, y_train, X_test, y_test 
    
    





  

