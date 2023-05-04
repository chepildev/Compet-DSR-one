# Functions for adding different features

def feature_eng(df):
    # drop week start date column
    df.drop('week_start_date', axis=1, inplace=True)
    # fill na features with ffills
    df.fillna(method='ffill', inplace=True)
    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'station_avg_temp_c', 
                 'station_min_temp_c']
    df = df[features]
    return df
