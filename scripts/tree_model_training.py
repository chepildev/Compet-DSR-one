import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
#from scripts.model_evaluation import regression_evaluation
from xgboost import XGBRegressor 




def xg_model(X: pd.DataFrame, y:pd.DataFrame, tss_splits=2, params={}) -> pd.DataFrame:
    """_summary_

    Args:
        X (pd.DataFrame): 
        y (pd.DataFrame): 
        tss_splits (int, optional): Number of time-series train-test splits. Defaults to 2.
        params (dict, optional): custom hyperparameters. Defaults to {}.

    Returns:
        pd.DataFrame: Error metrics and hyperparamters for each iteration 
    """
    tss = TimeSeriesSplit(n_splits=2) 
    i=1
    score = []

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    for train_index, test_index in tss.split(X_scaled):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        #params
        learning_rate=0.1
        n_estimators=[100,200]
        max_depth=5
        subsample=1.0
        colsample_bytree=1.0
        reg_lambda=2

        for n in n_estimators:

            xgm = XGBRegressor(
                    learning_rate=learning_rate,
                    n_estimators=n,
                    max_depth=max_depth,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_lambda=reg_lambda)
            
            xgm.fit(X_train, y_train)
            
            pred_train = xgm.predict(X_train)
            pred_test = xgm.predict(X_test)
            rmse_train = mean_squared_error(y_train, pred_train)**0.5
            rmse_test = mean_squared_error(y_test, pred_test)**0.5
            mae_train = mean_absolute_error(y_train, pred_train)
            mae_test = mean_absolute_error(y_test, pred_test)

            score.append({'TSS iteration': i, 
                        'rmse_test':rmse_test, 'rmse_train':rmse_train, 
                        'mae_train':mae_train, 'mae_test':mae_test,
                         'learning_rate':learning_rate, 'n_estimators':n,
                         'max_depth':max_depth, 'subsample':subsample,
                         'colsample_bytree':colsample_bytree,
                         'reg_lambda':reg_lambda
                        })

        i += 1

    return pd.DataFrame(score)



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
    
    
    

