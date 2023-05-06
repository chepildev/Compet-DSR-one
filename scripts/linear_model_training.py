import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def poisson_reg_model(X: pd.DataFrame, y:pd.DataFrame, tss_splits=2, params={}) -> pd.DataFrame:
    """_summary_
    Poisson regression model
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
        alpha=[0.6, 1]
        max_iter=[100, 200]

        for a in alpha:
            for mi in max_iter:

                model = PoissonRegressor(alpha=a, max_iter = mi, 
                                         fit_intercept=True, solver='lbfgs',
                                         )
                
                model.fit(X_train, y_train)
                
                pred_train = model.predict(X_train)
                pred_test = model.predict(X_test)
                rmse_train = mean_squared_error(y_train, pred_train)**0.5
                rmse_test = mean_squared_error(y_test, pred_test)**0.5
                mae_train = mean_absolute_error(y_train, pred_train)
                mae_test = mean_absolute_error(y_test, pred_test)

                score.append({'TSS iteration': i, 
                            'rmse_test':rmse_test, 'rmse_train':rmse_train, 
                            'mae_test':mae_test, 'mae_train':mae_train,
                            'alpha':a, 'mat_iter':mi})
        i += 1

    return pd.DataFrame(score)


def poisson_get_coefs(X: pd.DataFrame, y:pd.DataFrame, params={}) -> pd.DataFrame:
    """_summary_
    Poisson regression model - to acquire parameters for analysis 
    Args:
        X (pd.DataFrame): 
        y (pd.DataFrame): 
        params (dict, optional): custom hyperparameters. Defaults to {}.
    Returns:
        pd.DataFrame
    """

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    model = PoissonRegressor(alpha=0.8, max_iter = 100, fit_intercept=True, solver='lbfgs')
                
    model.fit(X_scaled, y)
    coefs = model.coef_

    return   pd.Series(coefs, index=X.columns)

