# Functions for adding different features
import pandas as pd
import numpy as np
import datetime as dt


def drop_date(data):
    # Currently just removing the date column so models can run
    try:
        data = data.drop(columns=["week_start_date"], axis=1)
    except:
        pass
    try:
        data = data.drop(columns=["date"], axis=1)
    except:
        pass
    try:
        data = data.drop(columns=["year"], axis=1)
    except:
        pass
    try:
        data = data.drop(columns=["weekofyear"], axis=1)
    except:
        pass

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
    # df["date"] = pd.to_datetime(df.loc[:, "week_start_date"], format="%Y-%m-%d")
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


def shift_features(
    df: pd.DataFrame, periods: int, merge: bool = True, fillna: bool = True
) -> pd.DataFrame:
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
    df_shifted.drop(f"total_cases_{periods}up", axis=1, inplace=True)

    if merge:
        combined = df.join(df_shifted, how="left")
        if fillna:
            combined = combined.fillna(method="bfill")
        else:
            combined = combined.iloc[periods:, :]  # drop first n rows

    return combined


def rolling_avg(data, column, window_size):
    """Convert column total_cases to rolling average

    Args: Dataframe, column name, window size

    Returns: Dataframe with rolling average overwritting column

    """
    data[column] = data[column].rolling(window_size).mean()

    return data


def exp_weight_moving(data, column, span):
    """Convert column total_cases to rolling average

    Args: Dataframe, column name, window size

    Returns: Dataframe with rolling average overwritting column

    """
    data[column] = data[column].ewm(span=span).mean()

    return data


def add_rolling(data: pd.DataFrame, city: str, fillna: bool = True) -> pd.DataFrame:
    """_summary_
    Args:
        data (pd.DataFrame): data to add new features to
        city (str): specify city 'iq' or 'sj'
        fillna (bool, optional): ffill and bfill null values after creating rolling averages. Defaults to True.

    Returns:
        pd.DataFrame: with new rolling average features
    """

    data["ndvi_ne_rolling"] = data.loc[:, "ndvi_ne"].rolling(20, center=False).mean()
    data["ndvi_nw_rolling"] = data.loc[:, "ndvi_nw"].rolling(20, center=False).mean()
    data["precipitation_amt_mm_rolling"] = (data.loc[:, "precipitation_amt_mm"].rolling(12, center=True).mean()    )    
    data["reanalysis_air_temp_k_rolling"] = (
        data.loc[:, "reanalysis_air_temp_k"].rolling(12, center=False).mean()   )    data["reanalysis_avg_temp_k_rolling"] = (
        data.loc[:, "reanalysis_avg_temp_k"].rolling(16, center=False).mean()   )   data["reanalysis_dew_point_temp_k_rolling"] = (
        data.loc[:, "reanalysis_dew_point_temp_k"].rolling(8, center=True).mean()   )    data["reanalysis_max_air_temp_k_rolling"] = (
        data.loc[:, "reanalysis_max_air_temp_k"].rolling(18, center=False).mean()   )   data["reanalysis_precip_amt_kg_per_m2_rolling"] = (
        data.loc[:, "reanalysis_precip_amt_kg_per_m2"].rolling(10, center=True).mean()   )   data["reanalysis_relative_humidity_percent_rolling"] = (
        data.loc[:, "reanalysis_relative_humidity_percent"]       .rolling(20, center=True)     .mean()  )  data["reanalysis_sat_precip_amt_mm_rolling"] = (
        data.loc[:, "reanalysis_sat_precip_amt_mm"].rolling(30, center=True).mean()
    )
    data["reanalysis_specific_humidity_g_per_kg_rolling"] = (    data.loc[:, "reanalysis_specific_humidity_g_per_kg"]     .rolling(8, center=False)        .mean()    )
    data["reanalysis_tdtr_k_rolling"] = (        data.loc[:, "reanalysis_tdtr_k"].rolling(24, center=False).mean()    )
    data["station_avg_temp_c_rolling"] = (        data.loc[:, "station_avg_temp_c"].rolling(12, center=False).mean()    )
    data["station_diur_temp_rng_c_rolling"] = (        data.loc[:, "station_diur_temp_rng_c"].rolling(16, center=False).mean()    )
    data["station_diur_temp_rng_c"] = (        data.loc[:, "station_max_temp_c"].rolling(12, center=False).mean()    )
    data["station_precip_mm_rolling"] = (        data.loc[:, "station_precip_mm"].rolling(16, center=True).mean()    )

    if city == "iq":
        data["ndvi_se_rolling"] = (            data.loc[:, "ndvi_se"].rolling(18, center=False).mean()        )
        data["ndvi_sw_rolling"] = (            data.loc[:, "ndvi_sw"].rolling(16, center=False).mean()        )
        data["reanalysis_min_air_temp_k_rolling"] = (            data.loc[:, "reanalysis_min_air_temp_k"].rolling(8, center=True).mean()        )
        data["station_min_temp_c_rolling"] = (            data.loc[:, "station_min_temp_c"].rolling(12, center=True).mean()        )

    elif city == "sj":
        data["ndvi_se_rolling"] = (            data.loc[:, "ndvi_se"].rolling(16, center=False).mean()        )
        data["ndvi_sw_rolling"] = (            data.loc[:, "ndvi_sw"].rolling(12, center=False).mean()        )
        data["reanalysis_min_air_temp_k_rolling"] = (            data.loc[:, "reanalysis_min_air_temp_k"].rolling(12, center=False).mean()        )
        data["station_min_temp_c_rolling"] = (            data.loc[:, "station_min_temp_c"].rolling(12, center=False).mean()        )

    if fillna:
        data = data.fillna(method="ffill")
        data = data.fillna(method="bfill")

    return data


def add_rolling2(data: pd.DataFrame, city: str, fillna: bool = True) -> pd.DataFrame:
    """_summary_
    Args:
        data (pd.DataFrame): data to add new features to
        city (str): specify city 'iq' or 'sj'
        fillna (bool, optional): ffill and bfill null values after creating rolling averages. Defaults to True.

    Returns:
        pd.DataFrame: with new rolling average features
    """
    recipe_all = {"ndvi_e": 20, 
                  "ndvi_nw": 20, 
                  "precipitation_amt_mm":12,
                  "reanalysis_air_temp_k":12,
                  "reanalysis_avg_temp_k":16,
                  "reanalysis_dew_point_temp_k":8,
                  "reanalysis_max_air_temp_k":18,
                  "reanalysis_precip_amt_kg_per_m2":10,
                  "reanalysis_relative_humidity_percent":20,
                  "reanalysis_sat_precip_amt_mm":30,
                  "reanalysis_specific_humidity_g_per_kg":8,
                  "reanalysis_tdtr_k":24,
                  "station_avg_temp_c":12,
                  "station_diur_temp_rng_c":16,
                  "station_max_temp_c":12,
                  "station_precip_mm":16
                  }
    
    recipe_iq = {"ndvi_se": 18, 
                 "ndvi_sw": 16,
                 "reanalysis_min_air_temp_k":8,
                 "station_min_temp_c":12}
    
    recipe_sj = {"ndvi_se": 16, 
                 "ndvi_sw": 12,
                 "reanalysis_min_air_temp_k":12,
                 "station_min_temp_c":12}

    for feature_name, periods in recipe_all.items:
        data[f"{feature_name}_{periods}week"] = (
            data.loc[:, feature_name].rolling(periods, center=False).mean()
        )

    if city == "iq":
        for feature_name, periods in recipe_iq.items:
            data[f"{feature_name}_{periods}week"] = (
                data.loc[:, feature_name].rolling(periods, center=False).mean()
            )
    if city == "sj":
        for feature_name, periods in recipe_sj.items:
            data[f"{feature_name}_{periods}week"] = (
                data.loc[:, feature_name].rolling(periods, center=False).mean()
            )

    if fillna:
        data = data.fillna(method="ffill")
        data = data.fillna(method="bfill")

    return data


def remove_original_cols(data: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        pd.DataFrame: input for columns to be removed from

    Returns:
        pd.DataFrame: DataFrame with columns removed

    For now keeping the original cols that were most correlated in benchmark:
    - reanalysis_specific_humidity_g_per_kg
    - reanalysis_dew_point_temp_k
    """
    cols_to_remove = ['reanalysis_specific_humidity_g_per_kg',
                      'reanalysis_dew_point_temp_k','ndvi_ne', 'ndvi_nw',
                        'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
                        'reanalysis_avg_temp_k',
                        'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
                        'reanalysis_precip_amt_kg_per_m2',
                        'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
                        'reanalysis_tdtr_k',
                        'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
                        'station_min_temp_c', 'station_precip_mm']
    ]

    data = data.drop(cols_to_remove, axis=1)

    return data


def create_binary_station_min_temp_c(data: pd.DataFrame, temp: float) -> pd.DataFrame:
    """create binary output column for min_temp_c less than temp: 19.5 (default)

    Args:
        data (pd.DataFrame): column to be transformed
        temp (float): temperature threshold

    Returns:
        pd.DataFrame: column output with binary output
    """
    data[f'station_min_temp_c_binary_{temp}_c'] = data['station_min_temp_c'].apply(lambda x: 1 if x < temp else 0)
    return data

if __name__ == "__main__":
    train_features = pd.read_csv("./data/dengue_features_train.csv")
    train = cyclical_encode_date(train_features)
    print(train)
