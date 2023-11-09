
import sys, os, importlib
from time import time
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
import readWriteS3 as rs3

from statsmodels.tsa.seasonal import seasonal_decompose
from statsforecast import StatsForecast
from sklearn.metrics import mean_absolute_percentage_error

from statsforecast.models import (
    SeasonalNaive, # model using the previous season's data as the forecast
    Naive, # Simple naive model using the last observed value as the forecast
    HistoricAverage, # Average of all historical data
    AutoETS, # Automatically selects best ETS model based on AIC
    AutoARIMA, # ARIMA model that automatically select the parameters for given time series with AIC and cross validation
    HoltWinters #HoltWinters ETS model
    )


# cache for dataload
dataloadcache = None

def datapath():
    tidyfilepath = '2_tidy/1h/'  # allow selection of data
    metadatakey = 'usage_meta.gz'
    key = 'usage.gz'
    forecastfilepath = '5_forecast/1h/'
    return tidyfilepath, key, metadatakey, forecastfilepath

def select_date_range():
    startdate_input = input('Start date (YYYY-mm-dd HH:MM:SS format)? ')
    if startdate_input == '':
        startdate = datetime.strptime('2023-10-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    else:
        startdate = startdate_input


    enddate_input = input('Start date (YYYY-mm-dd HH:MM:SS format)? ')
    if enddate_input == '':
        enddate = datetime.strptime('2023-11-07 00:00:00', '%Y-%m-%d %H:%M:%S')
    else:
        enddate = enddate_input

    return startdate, enddate

def clean_data(df: pd.DataFrame, datetime_col: str, y: str, startdate, enddate) -> pd.DataFrame:
    # filter dates
    datetime_mask = (df['tm'] > startdate) & (df['tm'] <= enddate)
    df = df.loc[datetime_mask]

    # Remove zeros
    #df['y'] = df['y'].replace(0,1) # replace zeros with ones in order for model to work - maybe should be min value of series?

    # Remove whitespace from account name
    df['account'] = df['account'].str.strip()

    # save unique combinations of account_id, meter, and measurement for formatting forecast file before saving
    df_ids = df[['account', 'account_id', 'meter', 'measurement']].drop_duplicates()
    df_ids.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/df_ids.csv')

    # format for fitting and forecasting - select subset of columns and add unique_id column
    df['unique_id'] = df.apply(
        lambda row: row.account + '_' + row.meter + '_' + row.measurement.split(' ')[0], axis=1)

    df = df[[datetime_col, y, 'unique_id']]
    df.columns = ['ds', 'y', 'unique_id']
    #df.sort_values(by=['unique_id', 'ds'], inplace=True)

    print(df.groupby(['unique_id']).size().reset_index(name='counts'))
    # Tell user number of time series
    print(str(df['unique_id'].nunique()) + ' Unique Time Series')

    # plot for inspection
    x = StatsForecast.plot(df)
    x.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output_figs/{}'.format('ts_eng_input_data'))

    return df, df_ids

def select_model_parms(datasize, freq, wpct=1)-> list:
    '''
    This function determines the model parameters. The input parameters are:
    Datasize = length of the data.  Or alternatively, the number of timeperiods in the data set
    Frequency = Usage frequency.  Options are h = hourly; d = daily; w = weekly; or enter in frequency as an integer.
    Window Size Percentage = Percentage of the data point used to train the model
        Default is 1
    '''

    models = ['autoETS'] # TODO organise for multiple models
    # configurable parameters
    # forecast window = time periods into the past used to train
    window = round((datasize * wpct))
    # set seasonality based on data frequency - this can also be determined by looking at the plot
    if freq == 'h':
        season = 24
    elif freq == 'd':
        season = 7
    elif freq == 'w':
        season = 52
    else: # user is defining the seasonality or has been calculated in another function....
        season = int(freq)
    return models, season, window

def split_data(df):
    #train = df.iloc[:int(0.75 * df['ds'].nunique())]  # TODO change 0.5 to horizon
    #valid = df.iloc[int(0.25 * df['ds'].nunique()) + 1:]  # TODO change 0.5*len(df) to horizon
    train = 1
    valid = 2
   # horizon (h)  = time periods into future for which a forecast will be made
    h = round((len(df) * 0.15))
    return train, valid, h

def only_forecast(df: pd.DataFrame, h, season):
    model_aliases=['AE'] #TODO this list needs to be created dynamically

    ts_models = [
        AutoETS(model=['Z','Z','Z'], season_length=season, alias='AE')
        #AutoARIMA(season_length=season, alias='AA'),
        #SeasonalNaive(season_length=season, alias='SN')
    ]

    # create the model object, for each model and let user know time required for each fit
    model = StatsForecast(
                        models=ts_models,
                          freq = 'H',
                          n_jobs=-1,
                          fallback_model = SeasonalNaive(season_length=24))

    # record time for benchnmarking

    forecast = model.forecast(df=df, h=h, level=[95])

    forecast_plot = StatsForecast.plot(df, forecast, engine='matplotlib')
    forecast_plot.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output_figs/{}'.format('forecast_plot'))
    #plot_HW_forecast_vs_actuals(forecast, model_aliases)
    return forecast, model_aliases

def fit(df: pd.DataFrame, h, season):
    model_aliases=['AE'] #TODO this list needs to be created dynamically

    ts_models = [
        AutoETS(model=['Z','Z','Z'], season_length=season, alias='AE')
        #AutoARIMA(season_length=season, alias='AA'),
        #SeasonalNaive(season_length=season, alias='SN')
    ]

    # create the model object, for each model and let user know time required for each fit
    model = StatsForecast(df=df,
                        models=ts_models,
                        freq = 'H',
                        n_jobs=-1,
                        verbose=True)

    model.fit(df)
    return model, model_aliases

def predict(model, df, h):
    # predict future h periods, with 95% confidence level
    prediction = model.predict(h=h, level=[95])
    print(prediction.head())

    # prep for plotting with confidence intervals
    #prediction_merge = prediction.reset_index().merge(df, on=['ds','unique_id'], how='left')
    #print(prediction_merge.head())
    return prediction

def plot(data: pd.DataFrame, account: str, meter: str):
    fig = px.line(data, x='tm', y='y', title='Account: {} & Meter: {}'.format(account, meter))
    fig.show()
    return

def SF_plot(df, forecast_df):
    #forecast_df.reset_index(inplace=True)
    df['ds'] = np.arange(1, len(df) + 1)
    forecast_df['ds'] = np.arange(len(df) + 1, len(df)+len(forecast_df) + 1)
    #df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/df.csv')
    #forecast_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/forecst_df.csv')
    x = StatsForecast.plot(df, forecast_df, level=[95])
    # Plot to unique_ids and some selected models
    x.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output_figs/{}.png'.format('ts_eng_forecast'))

def plot_HW_forecast_vs_actuals(forecast, models: list):
    print(forecast.head())
    # look for NaNs and remove for plotting
    print(str(forecast.isnull().any(axis=1).count()) + ' nulls in forecast' )
    forecast['y'] = forecast['y'].replace(0, 1)
    #print(p[p.isnull().any(axis=1)])
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

def prep_forecast_for_s3(df: pd.DataFrame, df_ids, model_aliases):
    df.reset_index(inplace=True)

    dashboard_cols = [
        'tm'  # timestamp
        , 'meter'
        , 'measurement'
        , 'account_id'  # account m3ter uid
        , 'account'
        , 'ts_id'  # ts unique id
        , 'z'  # prediction
        , 'z0'  # lower bound of 95% confidence interval
        , 'z1'  # lower bound of 95% confidence interval
        , '.model'  # model (e.g. model_)
    ]

    # add back meter, measurement, account and account_id
    df['account'] = df['unique_id'].str.split('_', expand=True)[0]
    df['meter'] = df['unique_id'].str.split('_', expand=True)[1]
    df['measurement'] = df['unique_id'].str.split('_', expand=True)[2]
    df = df.merge(df_ids[['account_id', 'account']], on='account_id', how='left')

    dfs = []

    for alias in model_aliases:
        iterator_list = ['df' + alias, alias, alias + '-lo-95', alias + '-hi-95']
        iterator_list[0] = df[['ds', 'meter', 'measurement', 'account_id', 'account', 'unique_id', iterator_list[1], iterator_list[2], iterator_list[3]]]
        iterator_list[0]['.model'] = alias
        iterator_list[0].columns = dashboard_cols
        dfs.append(iterator_list[0])

    dfAll = pd.concat(dfs, ignore_index=True)

    dfAll.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/all_forecasts.csv')
    return dfAll

def prep_meta_data_for_s3():
    # TODO change meta as dictionary passed parameter to function
    meta = {'nm': 'typ', 'meter': 'dim', 'measurement': 'dim', 'account': 'dim', 'account_id': 'dim', '.model': 'dim', 'z': 'measure', 'tm': 'time', '_intrvl': '1h', 'z0': 'measure', 'z1': 'measure'}
    meta_list = list(meta.items())
    with open ('/Users/tmb/PycharmProjects/data-science/UFE/output_files/tmbmeta', 'w') as file: # TODO change to local temp folder
        for i in meta_list:
            file.write(','.join(map(str, i))+'\n')

    print(type(file))
    print(file)
    return file

def main(dfUsage):
    ##################################
    # plot time series  ##############
    ##################################
    plot(dfUsage, dfUsage['account'].iloc[0], dfUsage['meter'].iloc[0])

    ##################################
    # Data Wrangle #########
    ##################################
    startdate, enddate = select_date_range()
    dfUsage_clean, df_ids = clean_data(dfUsage, 'tm', 'y', startdate, enddate)
    dfUsage_clean.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/dfUsage.csv')
    models, season, window = select_model_parms(round(dfUsage_clean['ds'].nunique()), 'h', 1 ) # function parameters datasize, freq, hpct=0.5,
    h = round(dfUsage_clean['ds'].nunique() * 0.15) # forecast horizon

    ##################################
    # split data #########
    ##################################
    #train,valid, h = split_data(dfUsage_clean)
    #train.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/train.csv')

    ##################################
    # fit, predict, forecast #########
    ##################################
    # fit then forcast
    init_fit = time()
    model, model_aliases = fit(dfUsage_clean, h, season)
    end_fit=time()
    print(f'Fit Minutes: {(end_fit - init_fit) / 60}')
    #model = rs3.read_model_from_s3()
    init_predict = time()
    forecast = predict(model, dfUsage_clean, h)
    end_predict=time()
    print(f'Predict Minutes: {(end_predict - init_predict) / 60}')

    # forcast only
    init_foreonly = time()
    forecast_only, model_aliases = only_forecast(dfUsage_clean, h, season)
    # dfUsage_clean[0:round(len(dfUsage_clean)*0.75)
    end_foreonly = time()
    print(f'Forecast Only Minutes: {(end_foreonly - init_foreonly) / 60}')

    # plot predictions
    # StatsForecast plot routine
    SF_plot(dfUsage_clean, forecast_only)
    #plot_HW_forecast_vs_actuals(forecast, model_aliases)

    ##################################
    # save to s3 if directed #########
    ##################################
    if savetos3 in ['Y','yes','Yes', 'YES', 'y']:
        forecast_to_save = prep_forecast_for_s3(forecast, df_ids, model_aliases)
        rs3.write_csv_to_s3(forecast_to_save, forecastfilepath, 'tmb_mixed_model_'+key)
        metadatafile = prep_meta_data_for_s3()
        rs3.write_meta_to_s3(metadatafile, forecastfilepath,'tmb_mixed_model_usage_meta.gz')
    else:
        forecast.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/only_forecst.csv')
    return

if __name__ == "__main__":
    data_loc = input("Data location (local or s3)? ")
    savetos3 = input("Save to s3? ")
    dataloadcache= pd.DataFrame()

    while True:
        if dataloadcache.empty:
            if data_loc == 's3':
                tidyfilepath, key, metadatakey, forecastfilepath = datapath()
                dataloadcache = rs3.get_data(tidyfilepath, key, metadatakey)
            elif data_loc == 'local':
                dataloadcache = rs3.get_data_local()
        data = rs3.select_ts(dataloadcache)
        main(data)
        print("Press enter to re-run the script, CTRL-C to exit")
        sys.stdin.readline()
        importlib.reload(rs3)