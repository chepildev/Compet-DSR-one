# Functions for adding different features
import pandas as pd
import numpy as np
import datetime as dt


def drop_date(data):
    # Currently just removing the date column so models can run
    try: data = data.drop(["week_start_date"], axis=1)
    except: pass
    try: data = data.drop(["date"], axis=1)
    except: pass
    try: data = data.drop(["year"], axis=1)
    except: pass
    try: data = data.drop(["weekofyear"], axis=1)
    except: pass

    return data


def cyclical_encode_date(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical encoding to date column
    Encode the month and week of year are encoded into sin and cos variables.
    Set index to date
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
    #df["date"] = pd.to_datetime(df.loc[:, "week_start_date"], format="%Y-%m-%d")
    month = df.loc[:, "week_start_date"].dt.month
    week_of_year = df.loc[:, "week_start_date"].dt.isocalendar().week

    # Encode both sin and cosine
    df["sin_month"] = np.sin(2 * np.pi * month / max(month))
    df["cos_month"] = np.cos(2 * np.pi * month / max(month))
    df["sin_week"] = np.sin(2 * np.pi * week_of_year / max(week_of_year))
    df["cos_week"] = np.cos(2 * np.pi * week_of_year / max(week_of_year))

    # Set index to date
    df.set_index("week_start_date", inplace=True, drop=True)
    
    return df


def shift_features(df: pd.DataFrame, periods: int, merge: bool = True) -> pd.DataFrame:
    """Create a new set of features by shifting the data by a specified amount

    Args:
        df (pd.DataFrame): input dataframe
        shift (int): number of periods (weeks) to shift
        drop_original (bool, default=True): if True, the columns are appended to the original dataframe

    Returns:
        pd.DataFrame: either the shifted dataframe or the joined and shifted dataframe together
    """
    df = drop_date(df)

    def rename_col(name, periods):
        return f"{name}_{periods}up"

    df_shifted = df.shift(periods=periods, axis=0)
    df_shifted.rename(columns=lambda name: rename_col(name, periods), inplace=True)

    if merge:
        combined = df.join(df_shifted, how="left")
        # drop first n rows
        combined = combined.iloc[periods:, :]

    return combined


def rolling_avg(data, column, window_size):
    """Convert column total_cases to rolling average

    Args: Dataframe, column name, window size

    Returns: Dataframe with rolling average overwritting column

    """
    data[column] = data[column].rolling(window_size).mean()

    return data

if __name__ == "__main__":
    train_features = pd.read_csv("./data/dengue_features_train.csv")
    train = cyclical_encode_date(train_features)
    print(train)
