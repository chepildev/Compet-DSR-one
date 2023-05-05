import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import n_colors, hex_to_rgb
import plotly.io as pio
pio.renderers.default = "browser"

def plot_prediction(sj, iq):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles = ["San Juan", "Iquitos"],
     vertical_spacing=0.02)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#00cc99']

    for i, c in enumerate(sj.columns):
        fig.add_trace(go.Scatter(x=sj.index, 
                                y=sj.loc[:,c],
                                mode='lines', # 'lines' or 'markers'
                                name=c,
                                line_color=colors[i],
                                legendgroup=f'group{i}',
                                showlegend=False,),
                    
                    row=1, col=1)

    for i, c in enumerate(iq.columns):
        fig.add_trace(go.Scatter(x=iq.index, 
                                y=iq.loc[:,c],
                                mode='lines', # 'lines' or 'markers'
                                line_color=colors[i],
                                legendgroup=f'group{i}',
                                name=c),
                    row=2, col=1)
    fig.show()

if __name__ == "__main__":
    features_train = pd.read_csv('../data/dengue_features_train.csv')
    labels_train = pd.read_csv('../data/dengue_labels_train.csv')
    features_test = pd.read_csv('../data/dengue_features_test.csv')
    features_train.loc[:, "total_cases"] = labels_train.loc[:, "total_cases"]
    sj = features_train.loc[features_train.loc[:,"city"]=="sj"].drop(["city", "year", "weekofyear"], 
    axis=1)
    iq = features_train.loc[features_train.loc[:,"city"]=="iq"].drop(["city", "year", "weekofyear"], 
    axis=1)