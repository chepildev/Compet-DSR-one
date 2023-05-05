
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def regression_evaluation(y_train, y_test, pred_train, pred_test):
    '''
    Function for evaluation of regression models results 
    '''
    rmse_train = mean_squared_error(y_train, pred_train)**0.5
    rmse_test = mean_squared_error(y_test, pred_test)**0.5
    mae_train = mean_absolute_error(y_train, pred_train)
    mae_test = mean_absolute_error(y_test, pred_test)
        
    print(f'''
    Evaluation metrics:
        RMSE train: {rmse_train}
        RMSE test: {rmse_test}
        MAE train: {mae_train}
        MAE test: {mae_test} 
    ''')
        
    return rmse_train, rmse_test, mae_train, mae_test
        