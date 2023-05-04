# import data and load into dataframe
# functions for:
# - load raw data from data folder
# - split data into IQ and SJ
# - split data into test, cross-eval split
import random
import pandas as pd


def load_data():
    """
    Load raw data from data folder.

    Returns:
    A pandas DataFrame containing the raw data.
    """
    # Load the data file into a pandas DataFrame
    data = pd.read_csv('data/data.csv')

    return data


def split_data(data, iq_col='IQ', sj_col='SJ'):
    """
    Split data into IQ and SJ.

    Args:
    data: pandas DataFrame, the data to split.
    iq_col: str, the column name for IQ data (default: 'IQ').
    sj_col: str, the column name for SJ data (default: 'SJ').

    Returns:
    Two pandas DataFrames, one for IQ data and one for SJ data.
    """
    iq_data = data[[iq_col]]
    sj_data = data[[sj_col]]

    return iq_data, sj_data


def split_train_test(data, test_ratio=0.2, random_state=None):
    """
    Split data into train and test sets.

    Args:
    data: pandas DataFrame, the data to split.
    test_ratio: float, the ratio of the test set size to the total dataset size (default: 0.2).
    random_state: int or None, the random seed for shuffling the data (default: None).

    Returns:
    Two pandas DataFrames, one for the train set and one for the test set.
    """
    # Shuffle the data
    if random_state is not None:
        random.seed(random_state)
    data = data.sample(frac=1)

    # Calculate the test set size
    test_size = int(len(data) * test_ratio)

    # Split the data into train and test sets
    test_data = data[:test_size]
    train_data = data[test_size:]

    return train_data, test_data
