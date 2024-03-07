import pandas as pd
import plotly
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots
import sys


# Set parms
tidy_folder = '2_tidy/'
fit_folder = '4_fit/'
forecast_folder = '5_forecast/'
ORG= 'onfido'
nrows=5

colors=[
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

# read tidy and forecast data from s3 and prep for plotting

dfBest = pd.read_csv('/UFE/output_files/onfido/dfBest.csv',
                     parse_dates=['ds'], index_col=0, date_parser=lambda d: pd.to_datetime(d, format='%d/%m/%Y', errors="coerce"))

tidy = pd.read_csv(f'/Users/tmb/Desktop/gzip_files/usage_27_Feb_24.csv',
                   parse_dates=['tm'],  index_col=0,  date_parser=lambda d: pd.to_datetime(d, format='%Y-%m-%dT%H:%M:%SZ', errors="coerce"))
tidy.reset_index(inplace=True)
tidy = tidy.rename(columns={'tm': 'ds'})

# concatenate actuals with forecasts in order to plot on the same plot
plot_df = pd.concat([tidy.set_index(['ts_id', 'ds']), dfBest.set_index(['ts_id', 'ds'])])
plot_df.reset_index(inplace=True)


nrows=20
unique_ids = plot_df['ts_id'].unique()[1000:1019]
fig = make_subplots(rows=nrows, cols=1, vertical_spacing=0.3/nrows)
                    #,subplot_titles=[i for i in unique_ids])

for n, id in zip(range(1,nrows), unique_ids):
    fig.append_trace(go.Scatter(name=id, x=plot_df[plot_df['ts_id']==id]['ds'],y=plot_df[plot_df['ts_id']==id]['y']), row=n, col=1)
    fig.append_trace(go.Scatter(name=id+'best_model', x=plot_df[plot_df['ts_id']==id]['ds'],y=plot_df[plot_df['ts_id']==id]['best_model'], showlegend=False), row=n, col=1)
    fig.append_trace(go.Scatter(name='Upper Bound', x=plot_df[plot_df['ts_id']==id]['ds'],y=plot_df[plot_df['ts_id']==id]['best_model-lo-95'], marker=dict(color="#444"), line=dict(width=0), showlegend=False), row=n, col=1)
    fig.append_trace(go.Scatter(name='Lower Bound', x=plot_df[plot_df['ts_id']==id]['ds'],y=plot_df[plot_df['ts_id']==id]['best_model-hi-95'], marker=dict(color="#444"), line=dict(width=0), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',showlegend=False), row=n, col=1)

fig['layout'].update(height=1500)
fig.show()
#
# if __name__ == "__main__":
#     freq = input("Hourly (1h) or Daily (1D) frequency: ")
#     key, metadatakey = get_keys()
#     dataloadcache, metadata_str = rs3.get_data(tidy_folder + freq + '/', key, metadatakey)