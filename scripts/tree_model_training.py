import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


def xg_model(X: pd.DataFrame, y: pd.DataFrame, tss_splits=2, params={}) -> pd.DataFrame:
    """_summary_
    XGBoostRegressor model function
    Args:
        X (pd.DataFrame):
        y (pd.DataFrame):
        tss_splits (int, optional): Number of time-series train-test splits. Defaults to 2.
        params (dict, optional): custom hyperparameters. Defaults to {}.

    Returns:
        pd.DataFrame: Error metrics and hyperparamters for each iteration
    """
    tss = TimeSeriesSplit(n_splits=2)
    i = 1
    score = []
    plots = []

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    for train_index, test_index in tss.split(X_scaled):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # params
        learning_rate = 0.05
        n_estimators = [100]
        max_depth = [4, 5, 6]
        subsample = 0.6
        colsample_bytree = 0.8
        reg_lambda = [5, 10, 20]
        gamma = 10

        for n in n_estimators:
            for md in max_depth:
                for rl in reg_lambda:
                    xgm = XGBRegressor(
                        learning_rate=learning_rate,
                        n_estimators=n,
                        max_depth=md,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        reg_lambda=rl,
                        gamma=gamma,
                    )

                    xgm.fit(X_train, y_train)

                    pred_train = xgm.predict(X_train)
                    pred_test = xgm.predict(X_test)
                    rmse_train = mean_squared_error(y_train, pred_train) ** 0.5
                    rmse_test = mean_squared_error(y_test, pred_test) ** 0.5
                    mae_train = mean_absolute_error(y_train, pred_train)
                    mae_test = mean_absolute_error(y_test, pred_test)

                    score.append(
                        {
                            "TSS iteration": i,
                            "rmse_test": rmse_test,
                            "rmse_train": rmse_train,
                            "mae_test": mae_test,
                            "mae_train": mae_train,
                            "learning_rate": learning_rate,
                            "n_estimators": n,
                            "max_depth": md,
                            "subsample": subsample,
                            "colsample_bytree": colsample_bytree,
                            "reg_lambda": rl,
                            "gamma": gamma,
                        }
                    )

                    plots.append(
                        {
                            "i": i,
                            "pred_train": pred_train,
                            "pred_test": pred_test,
                            "y_train": y_train,
                            "y_test": y_test,
                        }
                    )
        i += 1

    return pd.DataFrame(score), plots


def rforest_model(
    X: pd.DataFrame, y: pd.DataFrame, tss_splits=2, params={}
) -> pd.DataFrame:
    """_summary_
    Random Forest (sklearn) model function
    Args:
        X (pd.DataFrame):
        y (pd.DataFrame):
        tss_splits (int, optional): Number of time-series train-test splits. Defaults to 2.
        params (dict, optional): custom hyperparameters. Defaults to {}.

    Returns:
        pd.DataFrame: Error metrics and hyperparamters for each iteration
    """
    tss = TimeSeriesSplit(n_splits=2)
    i = 1
    score = []

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    for train_index, test_index in tss.split(X_scaled):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # params
        max_features = "sqrt"
        n_estimators = [100, 200]
        max_depth = [4, 8]
        min_samples_split = [2, 5]
        min_samples_leaf = [2]
        bootstrap = True

        for n in n_estimators:
            for md in max_depth:
                for mss in min_samples_split:
                    for msl in min_samples_leaf:
                        rfm = RandomForestRegressor(
                            n_estimators=n,
                            max_depth=md,
                            min_samples_split=mss,
                            min_samples_leaf=msl,
                            max_features=max_features,
                            bootstrap=bootstrap,
                        )

                        rfm.fit(X_train, y_train)

                        pred_train = rfm.predict(X_train)
                        pred_test = rfm.predict(X_test)
                        rmse_train = mean_squared_error(y_train, pred_train) ** 0.5
                        rmse_test = mean_squared_error(y_test, pred_test) ** 0.5
                        mae_train = mean_absolute_error(y_train, pred_train)
                        mae_test = mean_absolute_error(y_test, pred_test)

                        score.append(
                            {
                                "TSS iteration": i,
                                "rmse_test": rmse_test,
                                "rmse_train": rmse_train,
                                "mae_test": mae_test,
                                "mae_train": mae_train,
                                "n_estimators": n,
                                "max_depth": md,
                                "min_samples_split": mss,
                                "min_samples_leaf": msl,
                            }
                        )
        i += 1

    return pd.DataFrame(score)


def rf_feature_importance(X: pd.DataFrame, y: pd.DataFrame, params={}) -> pd.DataFrame:
    """_summary_
    Function to computer the Random Forest feature importances for analysis.
    Using the method: .feature_importances_
    Args:
        X (pd.DataFrame): Training features for model
        y (pd.DataFrame): Training output for model
        params (dict, optional): custom hyperparameters for model. Defaults to {}.

    Returns:
        pd.Series: RandomForest feature importances
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    rfm = RandomForestRegressor(n_estimators=100, max_depth=5)
    rfm.fit(X, y)

    importances = rfm.feature_importances_
    rf_feature_importances = pd.Series(importances, index=X.columns)

    return rf_feature_importances


def rf_model(X_train, y_train, X_test, y_test, params={}):
    # Fit model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Predict with model
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    # Evaluate
    rmse_train = mean_squared_error(y_train, pred_train) ** 0.5
    rmse_test = mean_squared_error(y_test, pred_test) ** 0.5
    mae_train = mean_absolute_error(y_train, pred_train)
    mae_test = mean_absolute_error(y_test, pred_test)

    print(
        f"""
    RandomForestRegressor with params: {params}
    Evaluation metrics:
        RMSE train: {rmse_train}
        RMSE test: {rmse_test}
        MAE train: {mae_train}
        MAE test: {mae_test} 
    """
    )

    return rmse_train, rmse_test, mae_train, mae_test
