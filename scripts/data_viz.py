import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def rolling_chart(data: pd.DataFrame, y: pd.DataFrame, feature: str, rolling=8, set_center=False):
    data1 = y
    data2 = data.loc[:, feature].rolling(rolling, center=set_center).mean()

    fig, ax1 = plt.subplots(figsize=(20,4))

    t = np.arange(data.shape[0])

    color = 'tab:red'
    ax1.set_xlabel('Weeks')
    ax1.set_ylabel('Daily cases', color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel(feature, color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    return plt.show()


def feature_output_chart(data: pd.DataFrame, y: pd.DataFrame, feature: str, city:str = 'Not defined'):
    data1 = y
    data2 = data.loc[:, feature]

    fig, ax1 = plt.subplots(figsize=(20,4))

    t = np.arange(data.shape[0])

    color = 'tab:red'
    ax1.set_xlabel('Weeks')
    ax1.set_ylabel('Daily cases', color=color)
    ax1.plot(t, data1, color=color, label='total_cases')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel(feature, color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color, label=feature)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.legend()
    ax1.set_title(f'Total cases and {feature} against time')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    return plt.show()


def plot_train_test(y_pred, y, test_or_train='undefined'):

    data1 = y_pred
    data2 = y
    fig, ax1 = plt.subplots(figsize=(16,4))
    t = np.arange(y_pred.shape[0])

    color = 'tab:red'
    ax1.set_xlabel('Weeks')
    ax1.set_ylabel('Predictions', color=color)
    ax1.plot(t, data1, color=color, label='Total cases predictions')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel(test_or_train, color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color, label=test_or_train)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.legend()
    ax1.set_title(f'Model predicts compared to {test_or_train} data against time')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    return plt.show()


