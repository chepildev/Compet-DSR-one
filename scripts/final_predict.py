
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor 
import datetime as dt


def final_predict(X_test, X_train, y_train :pd.DataFrame,
                  city: str, model:str, params={}) -> pd.DataFrame:
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
    m = eval(model + "(**params)")
    m.fit(X_train, y_train)
    final_preds = m.predict(X_test)

    return final_preds


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