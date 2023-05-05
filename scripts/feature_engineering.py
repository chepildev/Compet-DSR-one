# Functions for adding different features

import pandas as pd
import numpy as np
import datetime as dt



def feature_engineer_1(data):

    # Currently just removing the date column so models can run
    data = data.drop(['week_start_date'], axis=1)

    return data
