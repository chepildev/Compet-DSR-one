# xgboost model
import pandas as pd
import numpy as np

# Paste this Code to model_training.py file

import xgboost as xgb
from sklearn.metrics import accuracy_score


def train_xgb_model(
    X_train, Y_train, Xtest, Y_test, max_depth=5, learning_rate=0.1, n_estimators=100
):
    """
    Trains an XGBoost model on the provided training data and evaluates its performance on the provided testing data.

    Parameters:
    Xtrain (pandas.DataFrame): Training features.
    Ytrain (pandas.Series): Training labels.
    Xtest (pandas.DataFrame): Testing features.
    Ytest (pandas.Series): Testing labels.
    max_depth (int): Maximum depth of a tree. Default is 5.
    learning_rate (float): Learning rate. Default is 0.1.
    n_estimators (int): Number of trees. Default is 100.

    Returns:
    xgb.Booster: Trained XGBoost model.
    float: Accuracy score of the trained model on the testing data.
    """
    # Define XGBoost model
    model = xgb.XGBClassifier(
        max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators
    )

    # Train model
    fitted_model = model.fit(X_train, Y_train)

    # Make predictions
    Y_pred = model.predict(Xtest)

    # Evaluate model
    accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy:", accuracy)

    return fitted_model, Y_pred, accuracy
