
import sys, os, importlib
from pandas import DataFrame, Series
from time import time
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
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
    startdate = input('Start date (YYYY-mm-dd HH:MM:SS format)? ')
    enddate = input('Start date (YYYY-mm-dd HH:MM:SS format)? ')
    return startdate, enddate

def clean_data(df: DataFrame, datetime_col: str, y: str, startdate, enddate) -> DataFrame:
    if startdate == 0 and enddate == 0:
        startdate = df['tm'].min()
        enddate = df['tm'].max()

    # filter dates
    datetime_mask = (df['tm'] > startdate) & (df['tm'] <= enddate)
    df['y'] = df['y'].replace(0,1) # replace zeros with ones in order for model to work - maybe should be min value of series?
    df = df.loc[datetime_mask]

    # save unique combinations of account_id, meter, measurement for later
    df_ids = df[['account', 'account_id', 'meter', 'measurement']].drop_duplicates()
    df_ids.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/df_ids.csv')

    # format for fitting and forecasting - select subset of columns and add unique_id column
    init = time()
    #df['unique_id'] = df.apply(lambda row: row.account.split(' ')[0] + '_' + row.meter + '_' +row.measurement.split(' ')[0], axis=1)
    df['unique_id'] = df.apply(
        lambda row: row.account_id + '_' + row.meter + '_' + row.measurement.split(' ')[0], axis=1)
    end = time()
    # Let user know the time to fit all models
    print(f'Data Wrangle Minutes: {(end - init) / 60}')

    df = df[[datetime_col, y, 'unique_id']]
    df.columns = ['ds', 'y', 'unique_id']

    # Tell user number of time series
    print(str(df['unique_id'].nunique()) + ' Unique Time Series')

    return df, df_ids

def select_model_parms(datasize, freq, hpct=0.50, wpct=1)-> list:
    '''
    This function determines the model parameters. The input parameters are:
    Datasize = length of the data.  Or alternatively, the number of timeperiods in the data set
    Frequency = Usage frequency.  Options are h = hourly; d = daily; w = weekly; or enter in frequency as an integer.
    Horizon Percentage =  Percentage of data points into the future the horizon should be.
        For example, if there are 100 usage measurements, and you would like to predict 10 points into the future enter 0.10.
        Default is 0.5
    Window Size Percentage = Percentage of the data point used to train the model
        Default is 1
    '''

    models = ['autoETS'] # TODO organise for multiple models
    # configurable parameters
    # horizon  = time periods into future for which a forecast will be made
    horizon = round((datasize * hpct))
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
    return models, season, horizon, window

def fit(df: DataFrame, season):

    train = df.iloc[:int(0.75*df['ds'].nunique())] # TODO change 0.5 to horizon
    valid = df.iloc[int(0.25*df['ds'].nunique())+1:] # TODO change 0.5*len(df) to horizon

    # plot the training data
    #plot_HW_train(train, df['unique_id'].iloc[0], 'train') #Need to do something to show plotly plots?
    StatsForecast.plot(train, engine='matplotlib')
    plt.show()

    model_aliases=['AE', 'SN'] #TODO this list needs to be created dynamically

    ts_models = [
        AutoETS(model=['Z','Z','Z'], season_length=season, alias='AE'),
        SeasonalNaive(season_length=season, alias='SN')
    ]
        #AutoARIMA(season_length=season, alias='AA')]



    # create the model object, for each model and let user know time required for each fit
    model = StatsForecast(
                        models=ts_models,
                          freq = 'H',
                          n_jobs=-1)
                          #fallback_model = SeasonalNaive(season_length=24))

    # Save model to S3 with some identifying name

    # record time for benchnmarking
    init = time()
    model.fit(train)
    #forecast = model.forecast(h=1500, level=[95])
    end=time()

    # Let user know the time to fit all models
    print(f'Forecast Minutes: {(end - init) / 60}')

    h = round((valid['ds'].nunique()) * 0.5)
    forecast = model.predict(h=h, level=[95])
    forecast = forecast.reset_index().merge(valid, on=['ds','unique_id'], how='left')

    model.plot(df, forecast, engine='plotly')
    print(model.fitted_)

    plot_HW_forecast_vs_actuals(forecast, model_aliases)
    return forecast, model_aliases

def predict(model, valid):
    h = round((valid['ds'].nunique()) * 0.5) # prediction 50% of the valid set's time interval into the future
    # Set up models to fit: e.g. additive (HW_A) & multiplicative (HW_M)
    model_aliases = ['AE']
    # predict future h periods, with 95% confidence level
    p = model.predict(h=h, level=[95])
    p = p.reset_index().merge(valid, on=['ds','unique_id'], how='left')

    # plot predictions
    plot_HW_forecast_vs_actuals(p, model_aliases)
    return

def plot(data: DataFrame, account: str, meter: str):
    fig = px.line(data, x='tm', y='y', title='Account: {} & Meter: {}'.format(account, meter))
    fig.show()
    return

def plot_HW_train(data: DataFrame, TimeSeries: str, plot_type: str):
    fig = px.line(data, x='ds', y='y', title='Time Series: {} {}'.format(TimeSeries, plot_type))
    fig.show()
    return

def plot_HW_forecast_vs_actuals(forecast, models: list):
    # look for NaNs and remove for plotting
    print(str(forecast.isnull().any(axis=1).count()) + ' nulls in forecast' )
    forecast['y'] = forecast['y'].replace(0, 1)
    #print(p[p.isnull().any(axis=1)])
    forecast.dropna(inplace=True)
    forecast.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/engine_p.csv')

    # set number of subplots = number of timeseries
    min_subplots = 2
    numb_ts = min_subplots if min_subplots > forecast['unique_id'].nunique() else forecast['unique_id'].nunique() # ensure the number of subplots is > 1
    print(numb_ts)
    print(type(numb_ts))

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
    df['account_id'] = df['unique_id'].str.split('_', expand=True)[0]
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

def main(dfUsage):
    plot(dfUsage, dfUsage['account'].iloc[0], dfUsage['meter'].iloc[0])
    startdate, enddate = select_date_range()
    dfUsage_clean, df_ids = clean_data(dfUsage, 'tm', 'y', startdate, enddate)

    models, season, horizon, window = select_model_parms(round(dfUsage_clean['ds'].nunique()), 'h', .5, 1 ) # function parameters datasize, freq, hpct=0.5, wpct=1
    forecast, model_aliases = fit(dfUsage_clean, season)

    #forecast = predict(model, valid)

    forecast_to_save = prep_forecast_for_s3(forecast, df_ids, model_aliases)
    rs3.write_to_s3(forecast_to_save, forecastfilepath, 'tmb_mixed_model_'+key)
    return

if __name__ == "__main__":
    data_loc = input("Data location (local or s3)? ")
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