import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
#from xgboost import XGBRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scripts.model_evaluation import regression_evaluation


def rf_model(X_train, y_train, X_test, y_test, params={}):

    # Fit model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Predict with model 
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    # Evaluate 
    rmse_train = mean_squared_error(y_train, pred_train)**0.5
    rmse_test = mean_squared_error(y_test, pred_test)**0.5
    mae_train = mean_absolute_error(y_train, pred_train)
    mae_test = mean_absolute_error(y_test, pred_test)
        
    print(f'''
    RandomForestRegressor with params: {params}
    Evaluation metrics:
        RMSE train: {rmse_train}
        RMSE test: {rmse_test}
        MAE train: {mae_train}
        MAE test: {mae_test} 
    ''')
    
    return rmse_train, rmse_test, mae_train, mae_test
    
    
    

