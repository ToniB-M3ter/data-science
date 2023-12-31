
import os, sys, importlib
from os import path
from time import time
import pandas as pd
from tabulate import tabulate

import readWriteS3 as rs3
import error_analysis as err
import pre_process as pp

from statsforecast import StatsForecast
from statsforecast.models import (
    SeasonalNaive, # model using the previous season's data as the forecast
    Naive, # Simple naive model using the last observed value as the forecast
    HistoricAverage, # Average of all historical data
    AutoETS, # Automatically selects best ETS model based on AIC
    AutoARIMA, # ARIMA model that automatically select the parameters for given time series with AIC and cross validation
    HoltWinters #HoltWinters ETS model
    )

import logging
import warnings
from logging.config import fileConfig

logging.captureWarnings(True)
log_file_path = path.join(path.dirname(path.abspath(__file__)), 'log.config')
logging.config.fileConfig(log_file_path)
logger = logging.getLogger(__name__)

# parameters
dataloadcache = None

tidy_folder = '2_tidy/'
fit_folder = '4_fit/'
forecast_folder = '5_forecast/'

# comment out os.environ lines when running on aws as env variables will be set in lambda config
#os.environ['FIT_FORECAST'] = 'BOTH'
#os.environ['MODEL'] = 'AutoETS'

# FIT = fit and save fit; then forecast
# FORECAST = only forecast
# BOTH = fit and forecast in one step
global FIT_FORECAST
FIT_FORECAST=os.getenv('FIT_FORECAST')
MODEL=os.getenv('MODEL').split(',')
FREQ=os.getenv('FREQ')
USER=os.getenv('USER')

def get_keys():
    metadatakey = 'usage_meta.gz'
    key = 'usage.gz'
    return key, metadatakey

def get_season(data_freq):
    if 'D' in data_freq:
        data_freq = 'D'
        season = 7
    elif 'h' in data_freq:
        data_freq = 'H'
        season = 24

    return season

def select_models(data_freq: str, model_aliases: list) -> list:
    season=get_season(data_freq)
    all_models = [
        AutoETS(model=['Z', 'Z', 'Z'], season_length=season, alias='AutoETS'),
        AutoARIMA(season_length=season, alias='AutoARIMA'),
        SeasonalNaive(season_length=season, alias='SeasonalNaive'),
        Naive(alias='Naive')
    ]

    ts_models = []
    for model in model_aliases:
        for i in all_models:
            if model == str(i):
                ts_models.append(i)
    return ts_models

def only_forecast(df: pd.DataFrame, h, season, freq, ts_models):
    # create the model object, for each model and let user know time required for each fit
    model = StatsForecast(
        df = df,
        models=ts_models,
        freq =freq,
        n_jobs=-1,
        fallback_model=SeasonalNaive(season_length=season))

    forecast = model.forecast(df=df, h=h, level=[95])
    return forecast, model

def fit(df: pd.DataFrame, h, season, freq, ts_models):
    # create the model object, for each model and let user know time required for each fit
    model = StatsForecast(df=df,
                        models=ts_models,
                        freq = freq,
                        n_jobs=-1,
                        verbose=True)

    model.fit(df)   #(model = #<class 'statsforecast.core.StatsForecast'>
    return model

def predict(model, df, h):
    # predict future h periods, with level of confidence
    prediction = model.predict(h=h, level=[95])
    return prediction

def plot_forecasts(model, df, forecast, model_aliases):
    # Plot specific unique_ids and models
    forecast_plot_small = model.plot(
        df,
        forecast,
        models=model_aliases,
        level=[95],
        engine='plotly')

    forecast_plot = StatsForecast.plot(df, forecast, engine='matplotlib')
    forecast_plot.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output_figs/{}'.format('forecast_plot'))
    return

def plot(data: pd.DataFrame, account: str, meter: str):
    #fig = px.line(data, x='tm', y='y', title='Account: {} & Meter: {}'.format(account, meter))
    #fig.show()
    pass

def prep_forecast_for_s3(df: pd.DataFrame, df_ids, model_aliases):
    df.reset_index(inplace=True)

    dashboard_cols = [
        'tm'  # timestamp
        , 'meter'
        , 'measurement'
        , 'account_id'  # account m3ter uid
        , 'account'
        #, 'ts_id'  # ts unique id
        , 'z'  # prediction
        , 'z0'  # lower bound of 95% confidence interval
        , 'z1'  # lower bound of 95% confidence interval
        , '.model'  # model (e.g. model_)
    ]

    pat = "|".join(df_ids.account)
    df.insert(0, 'account', df['unique_id'].str.extract("(" + pat + ')', expand=False))
    df = df.merge(df_ids[['meter','measurement', 'account_id', 'account']], on='account', how='left')
    for alias in model_aliases:
        model_cols = [ alias, alias + '-lo-95', alias + '-hi-95'] # TODO make prediction interval bands dynamic
    df_cols = ['ds','meter','measurement', 'account_id', 'account']
    for col in model_cols:
        df_cols.append(col)
    df = df[df.columns.intersection(df_cols)]

    dfs = []

    for alias in model_aliases:
        iterator_list = ['df' + alias, alias, alias + '-lo-95', alias + '-hi-95']
        iterator_list[0] = df[['ds', 'meter', 'measurement', 'account_id', 'account', iterator_list[1], iterator_list[2], iterator_list[3]]]
        iterator_list[0]['.model'] = alias # TODO fix so not getting SettingWithCopyWarning
        iterator_list[0].columns = dashboard_cols
        dfs.append(iterator_list[0])

    dfAll = pd.concat(dfs, ignore_index=True)
    dfAll.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/all_forecasts_test.csv')
    return dfAll

def prep_meta_data_for_s3():
    # TODO change meta as dictionary passed parameter to function
    meta = {'nm': 'typ', 'meter': 'dim', 'measurement': 'dim', 'account': 'dim', 'account_id': 'dim', '.model': 'dim', 'z': 'measure', 'tm': 'time', '_intrvl': '1h', 'z0': 'measure', 'z1': 'measure'}
    meta_list = list(meta.items())
    with open ('/tmp/tmbmeta.txt', 'w') as file:
        for i in meta_list:
            file.write(','.join(map(str, i))+'\n')  # file type => _io.TextIOWrapper
    return file

def main(dfUsage, freq, metadata_str, account):
    ##################################
    # plot raw time series  ##########
    ##################################
    #plot(dfUsage, dfUsage['account'].iloc[0], dfUsage['meter'].iloc[0])

    ##################################
    # Data Wrangle #########
    ##################################
    startdate, enddate = pp.select_date_range(freq)
    dfUsage_clean, df_ids = pp.clean_data(dfUsage, 'tm', 'y', startdate, enddate)

    if USER is None:
        rs3.write_csv_log_to_S3(dfUsage_clean, 'dfUsage_clean')
    else:
        dfUsage_clean.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/dfUsage.csv')

    # get parameters and models
    season = get_season(freq)
    h = round(dfUsage_clean['ds'].nunique() * 0.10) # forecast horizon
    df_to_forecast, df_naive = pp.filter_data(dfUsage_clean)

    # Stationarity and Seasonality tests
    stationarity_df=pp.test_stationarity_dickey_fuller(df_to_forecast)
    seasonality_df=pp.seasonality_test(df_to_forecast, season)
    tests = pd.merge(stationarity_df, seasonality_df, on='unique_id')
    logger.info('Stationarity/Seasonality Results')
    logger.info(tabulate(tests.head(), headers="keys", tablefmt="psql"))
    #pp.decompose(dfUsage_clean)

    ##################################
    # fit, predict, forecast #########
    ##################################
    # forecast naive time series
    naive_model_aliases = ['Naive']
    ts_naive_model = select_models(freq, naive_model_aliases)
    if len(df_naive) > 0:
        forecast_only_naive, naive_model = only_forecast(df_naive, h, season, freq, ts_naive_model)
        # plot_forecasts(naive_model, dfUsage_clean, forecast_only_naive, naive_model_aliases)

    # set up model(s) for rest of data
    model_aliases = ['AutoETS']
    ts_models = select_models(freq, model_aliases)

    if FIT_FORECAST == 'FIT':
        init_fit = time()
        model = fit(dfUsage_clean, h, season, freq, ts_models)
        end_fit=time()
        logger.info(f'Fit Minutes: {(end_fit - init_fit) / 60}')
        rs3.write_model_to_s3(model, fit_folder+freq, account + freq + model_aliases[0] + '.pkl')
        forecast = predict(model, dfUsage_clean, h)
    elif FIT_FORECAST == 'FORECAST':
        # Read model from s3
        model = rs3.read_model_from_s3(fit_folder+freq, MODEL+'.pkl')
        init_predict = time()
        forecast = predict(model, dfUsage_clean, h)
        end_predict = time()
        logger.info(f'Forecast Minutes: {(end_predict - init_predict) / 60}')
    elif FIT_FORECAST == 'BOTH':
        if len(df_to_forecast)>0:
            init_foreonly = time()
            forecast_only, model = only_forecast(df_to_forecast, h, season, freq, ts_models)
            end_foreonly = time()
            logger.info(f'Forecast Only Minutes:  + {(end_foreonly - init_foreonly) / 60}')

        # plot and analyse
        #plot_forecasts(model, dfUsage_clean, forecast_only, model_aliases)
        cv_rmse_df = err.cross_validate(dfUsage_clean, model, h)
        if USER is None:
            rs3.write_csv_log_to_S3(cv_rmse_df, 'cv_rmse_df')
        else:
            dfUsage_clean.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/dfUsage.csv')

    ##################################
    # save to s3 if directed #########
    ##################################
    if USER is None:
        # metadatafile = prep_meta_data_for_s3() TODO remove or prep fields for saving

        # Naive forecasts
        if len(df_naive)>0:
            naive_forecast_to_save = prep_forecast_for_s3(forecast_only_naive, df_ids, naive_model_aliases)
            rs3.write_gz_csv_to_s3(naive_forecast_to_save, forecast_folder + freq + '/', account + '_' + freq + '_' + naive_model_aliases[0] + '_' + 'usage.gz') # save file for naive forecasts
            rs3.write_meta_to_s3(metadata_str, freq, forecast_folder + freq + '/',account + '_' + freq + '_' + naive_model_aliases[0] + '_' + 'usage_meta.gz')

        # other forecasts
        if len(df_to_forecast)>0:
            forecast_to_save = prep_forecast_for_s3(forecast_only, df_ids, model_aliases)
            rs3.write_gz_csv_to_s3(forecast_to_save, forecast_folder + freq + '/', account + '_' + freq + '_' + model_aliases[0] + '_' + 'usage.gz')
            rs3.write_meta_to_s3(metadata_str, freq, forecast_folder + freq+'/',account + '_' + freq + '_' + model_aliases[0] + '_' + 'usage_meta.gz')
    else:
        if len(df_naive)>0:
            forecast_only_naive.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/forecast{}.csv'.format(naive_model_aliases[0]))
        if len(df_to_forecast)>0:
            forecast_only.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/forecast{}.csv'.format(model_aliases[0]))
    return

def lambda_handler(event, context):
    freq = os.getenv(FREQ)
    key, metadatakey = get_keys()
    dataloadcache, metadata_str = rs3.get_data(tidy_folder + freq + '/', key, metadatakey)
    data, account = pp.select_ts(dataloadcache)
    main(data, freq, metadata_str, account)
    return

if __name__ == "__main__":
    freq = input("Hourly (1h) or Daily (1D) frequency: ")
    dataloadcache= pd.DataFrame()

    while True:
        if dataloadcache.empty:
            key, metadatakey = get_keys()
            dataloadcache, metadata_str = rs3.get_data(tidy_folder + freq + '/', key, metadatakey)
        data, account = pp.select_ts(dataloadcache)
        main(data, freq, metadata_str, account)
        print("Press enter to re-run the script, CTRL-C to exit")
        sys.stdin.readline()
        importlib.reload(rs3)