"""
Key Objects and Dataframes:
tags: Dictionary where each key is a level and its value contains tags associated to that level.
S_df:
df: cleaned input data
Y_train_df: training data
Y_test_df: test data
Y_h: observed data including aggregations
Y_hat_df: Base Forecasts NB predictions will not be coherent at this point; forcast future dates
Y_fitted_df: Fitted data from tmin - tmax (training time range)
Y_rec_df: Coherent reconciled predictions

"""

import sys
import importlib
import pandas as pd
import matplotlib.pyplot as plt

from dash import Dash, dcc, html, Input, Output
import plotly.express as px

app = Dash(__name__)


def simple_example():

    # gapminder population data 1704 rows
    # country
    # continent
    # year
    # lifeExp
    # pop
    # gdpPercap
    # iso_alpha
    # iso_num

    df = px.data.gapminder().query("continent=='Oceania'")
    fig = px.line(df, x="year", y="lifeExp", color='country')
    fig.show()
    return

def dash_example():
    app.layout = html.Div([
        html.H4('Life expentancy progression of countries per continents'),
        dcc.Graph(id="graph"),
        dcc.Checklist(
            id="checklist",
            options=["Asia", "Europe", "Africa", "Americas", "Oceania"],
            value=["Americas", "Oceania"],
            inline=True
        ),
    ])

    @app.callback(
        Output("graph", "figure"),
        Input("checklist", "value"))
    def update_line_chart(continents):
        df = px.data.gapminder()  # replace with your own data source
        mask = df.continent.isin(continents)
        fig = px.line(df[mask],
                      x="year", y="lifeExp", color='country')
        return fig

    app.run_server(debug=True)

def clean_raw_data(df):
    print(df.head())

    # Set parms for heat map
    #meter = input('Which meter? ')
    #startdate = input('Start date (YYYY-mm-dd HH:MM:SS format)? ')
    #enddate = input('End date (YYYY-mm-dd HH:MM:SS format)? ')
    meter = 'ingest'
    startdate = '2023-01-01 00:00:00'
    enddate = '2023-01-31 00:00:00'


    # greater than the start date and smaller than the end date
    datetime_mask = (df['tm'] > startdate) & (df['tm'] <= enddate)
    df = df.loc[datetime_mask]

    # Hard code sample of accounts
    accounts = ['Burst SMS - Staging','ClickHouse Production','Integ Tests Primary','Prompt Production',"Soner's Organization",'Tricentis Prod','m3terBilllingOrg Production']
    df = df[(df['account'].isin(accounts)) & (df['meter'] == meter)]
    df=df[['account', 'tm','y']]

    df['tm'] = pd.to_datetime(df['tm'])
    df.sort_values(by=['tm'], inplace=True)

    df = df.pivot(index='account', columns='tm', values='y')

    print(df.shape)
    return df

def heatmap(data):
    #plt.figure(figsize=(20, 20), facecolor='w', edgecolor='k')
    plt.figure(facecolor='w', edgecolor='k')
    #sns.set(font_scale=1.5)

    sns.heatmap(data,
                cmap="coolwarm",
                annot=True,
                fmt='.5g',
                cbar=False)
                #linewidths=2)
                #linecolor='black')

    plt.xlabel('Datetime', fontsize=9)
    plt.ylabel('Account', fontsize=9)
    plt.show()
    return

def transpose_data(df, freq, col):
    if 'ds' in df.columns:
        if freq == '1D':
            df['ds'] = pd.to_datetime(df['ds'], format="%Y-%m-%d")
        elif freq == '1h':
            df['ds'] = pd.to_datetime(df['ds'])
        # df.reset_index(drop=True)
        df.set_index('ds', inplace=True)

    print(df.head(1))
    models = df.columns[1:]

    # list of unique_ids
    unique_ids = df['unique_id'].unique()

    dfs = []

    for id in unique_ids:
        dftmp = df.loc[df['unique_id']==id]
        dfs.append(dftmp[col])

    #for i in dfs:
    #    print(i.head())

    dfAll = pd.concat(dfs, axis=1)
    # need to create columns for all models/accounts
    # model_ids = []
    # for model in models:
    #     model_ids_temp = [x + '_' + model for x in unique_ids]
    #     model_ids.append(model_ids_temp)
    # print(model_ids)

    dfAll.columns = unique_ids
    #print(dfAll.head())
    return dfAll

def plot_HW_forecast_vs_actuals(forecast, models: list):
    #print(forecast.head())
    print(str(forecast.isnull().any(axis=1).count()) + ' nulls in forecast' )
    forecast['y'] = forecast['y'].replace(0, 1)
    forecast.dropna(inplace=True)
    forecast.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/engine_p.csv')

    # set number of subplots = number of timeseries
    min_subplots = 2
    numb_ts = min_subplots if min_subplots > forecast['unique_id'].nunique() else forecast['unique_id'].nunique() # ensure the number of subplots is > 1

    # Plot model by model
    for model_ in models:
        mape_ = mean_absolute_percentage_error(forecast['y'].values, forecast[model_].values)
        print(f'{model_} MAPE: {mape_:.2%}')
        fig, ax = plt.subplots(numb_ts, 1, figsize=(1280 / (288/numb_ts), (90*numb_ts) / (288/numb_ts)))
        for ax_, device in enumerate(forecast['unique_id'].unique()):
            forecast.loc[forecast['unique_id'] == device].plot(x='ds', y='y', ax=ax[ax_], label='y', title=device, linewidth=2)
            forecast.loc[forecast['unique_id'] == device].plot(x='ds', y=model_, ax=ax[ax_], label=model_)
            ax[ax_].set_xlabel('Date')
            ax[ax_].set_ylabel('Sessions')
            ax[ax_].fill_between(forecast.loc[forecast['unique_id'] == device, 'ds'].values,
                                 forecast.loc[forecast['unique_id'] == device, f'{model_}-lo-95'],
                                 forecast.loc[forecast['unique_id'] == device, f'{model_}-hi-95'],
                                 alpha=0.2,
                                 color='orange')
            ax[ax_].set_title(f'{device} - Orange-ish band: 95% prediction interval')
            ax[ax_].legend()
        fig.tight_layout()
        plt.show()
        plt.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output_figs/eng_{}.jpg'.format(forecast['unique_id'][1]))
    return


def main():
    #rawdata = pd.read_csv('/UFE/data/dfUsage_28.csv')
    #data = clean_raw_data(rawdata)
    #heatmap(data)

    Y_train = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_train.csv', index_col='Unnamed: 0')
    Y_test = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_test.csv', index_col='Unnamed: 0')
    Y_df = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_df.csv') # observed data including aggregations
    Y_hat_df = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_hat_df.csv')  # Base Forecasts NB predictions will not be coherent at this point
    Y_fitted_df = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_fitted_df.csv') # Fitted data from tmin - tmax
    Y_rec_df = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_rec_df.csv') # hierarchial reconciled data

    Y_train_T = transpose_data(Y_train, '1D', 'y')
    Y_train_T.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_train_T.csv')

    Y_df_T = transpose_data(Y_df, '1D', 'y')
    Y_df_T.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_df_T.csv')

    Y_hat_df_T = transpose_data(Y_hat_df, '1D', 'AutoETS')
    Y_hat_df_T.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_hat_df_AE_T.csv')
    Y_hat_df_T = transpose_data(Y_hat_df, '1D', 'Naive')
    Y_hat_df_T.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_hat_df_N_T.csv')
    return


if __name__ == "__main__":
    main()