import pandas as pd 
import numpy as np



## Dealing with missing values
# Filling missing values with forward fill 
def fill_missing(data):
    data_no_missing = data.fillna(method='ffill', inplace=True)
    return data_no_missing

## Dealing with anomalies 

