
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from xgboost import XGBRegressor 
import datetime as dt


def final_predict(X_test, X_train, y_train :pd.DataFrame,
                  city: str, model:str, include_PCA=False, params={}) -> pd.DataFrame:
    """_summary_

    Args:
        final_test (_type_): _description_
        X_train (_type_): _description_
        y_train (pd.DataFrame): _description_
        city (str): _description_
        model (str): _description_
        params (dict, optional): _description_. Defaults to {}.

    Returns:
        pd.DataFrame: _description_
    """    

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if include_PCA:
        pca = PCA(n_components=3)
        pca.fit(X_train_scaled)
        X_train_pca = pca.transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        X_train_scaled = np.concatenate((X_train_scaled, X_train_pca), axis=1)
        X_test_scaled = np.concatenate((X_test_scaled, X_test_pca), axis=1)

    m = eval(model + "(**params)")
    m.fit(X_train_scaled, y_train)
    final_preds = m.predict(X_test_scaled)

    return final_preds


def preds_for_plot(X_test, X_train, y_train :pd.DataFrame, model:str, params={}) -> pd.DataFrame:
    """_summary_
    Args:
        final_test (_type_): _description_
        X_train (_type_): _description_
        y_train (pd.DataFrame): _description_
        city (str): _description_
        model (str): _description_
        params (dict, optional): _description_. Defaults to {}.
    Returns:
        pd.DataFrame: _description_
    """    
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    m = eval(model + "(**params)")
    m.fit(X_train_scaled, y_train)
    preds_train = m.predict(X_train_scaled)

    return preds_train, y_train



def write_submission(final_iq: pd.DataFrame, final_sj: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        final_iq (pd.DataFrame): _description_
        final_sj (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """    
    
    final = pd.concat([final_sj, final_iq], axis=0)
    final['total_cases'] = final['total_cases'].astype(int)
    final.to_csv('for_submission.csv', index=False)
    print('Writing submission file to folder: ')
    
    return final