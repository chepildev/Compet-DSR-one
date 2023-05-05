# Functions for adding different features
import pandas as pd
import numpy as np
import datetime as dt



def drop_date(data):

    # Currently just removing the date column so models can run
    try:
        data = data.drop(['week_start_date'], axis=1)
    except:
        pass
    try:
        data = data.drop(['date'], axis=1)
    except:
        pass

    return data


def cyclical_encode_date(df:pd.DataFrame) -> pd.DataFrame:
    """Add cyclical encoding to date column
    month and week of year are encoded into two variables each.
    new columns:
    - sin_month
    - cos_month
    - sin_week
    - cos_week
    Args:
        df (pd.DataFrame): the dataframe of features

    Returns:
        pd.DataFrame: the inputted dataframe with cyclically encoded date columns
    """
    df["date"] = pd.to_datetime(df.loc[:,"week_start_date"], format="%Y-%m-%d")
    month = df.loc[:,"date"].dt.month
    week_of_year = df.loc[:,"date"].dt.isocalendar().week

    # Encode both sin and cosine
    df["sin_month"] = np.sin(2 * np.pi * month / max(month)) 
    df['cos_month'] = np.cos(2 * np.pi * month / max(month))
    df['sin_week'] = np.sin(2 * np.pi * week_of_year / max(week_of_year))
    df['cos_week'] = np.cos(2 * np.pi * week_of_year / max(week_of_year))

    return df

if __name__ == "__main__":
    train_features = pd.read_csv('./data/dengue_features_train.csv')
    cyclical_encode_data(train_features)