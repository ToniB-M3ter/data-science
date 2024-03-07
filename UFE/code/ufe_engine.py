
import os, sys, importlib
from os import path
from time import time
import pandas as pd
from tabulate import tabulate
from datetime import datetime as dt

import readWriteS3 as rs3
import pre_process as preproc
import post_process as postproc

from statsforecast import StatsForecast
from statsforecast.models import (
    SeasonalNaive, # model using the previous season's data as the forecast
    Naive, # Simple naive model using the last observed value as the forecast
    HistoricAverage, # Average of all historical data
    AutoETS, # Automatically selects best ETS model based on AIC
    AutoARIMA, # ARIMA model that automatically select the parameters for given time series with AIC and cross validation
    HoltWinters, #HoltWinters ETS model
    AutoCES, # Auto Complex Exponential Smoothing
    MSTL,
    OptimizedTheta,
    AutoTheta
    )
from utilsforecast.losses import scaled_crps, mse, rmse

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
forecast_folder = '4_forecast/'

# comment out os.environ lines when running on aws as env variables will be set in lambda config
os.environ['ORG'] = 'onfido'
os.environ['FIT_FORECAST'] = 'FIT'
os.environ['MODEL_ALIASES'] = 'AutoETS,AutoARIMA,HoltWinters,,SeasonalNaive,Naive,HistoricAverage,AutoTheta'

# FIT = fit and save fit; then forecast
# FORECAST = only forecast
# BOTH = fit and forecast in one step
global FIT_FORECAST
FIT_FORECAST=os.getenv('FIT_FORECAST')
MODEL_ALIASES=os.getenv('MODEL_ALIASES').split(',')
freq=os.getenv('FREQ')
USER=os.getenv('USER')
ORG=os.getenv('ORG')

def get_keys():
    metadatakey = 'hier_2024_03_04_usage_meta.gz'
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
    ts_models = [
        Naive(alias='Naive'),
        HistoricAverage(),
        SeasonalNaive(season_length=season, alias='SeasonalNaive'),
        MSTL(season_length=[season, season*4]),
        HoltWinters(season_length=season, error_type="A", alias="HWAdd"),
        #HoltWinters(season_length=season, error_type="M", alias="HWMult"),
        AutoETS(model=['Z', 'Z', 'Z'], season_length=season, alias='AutoETS'),
        AutoCES(season_length=season, alias='AutoCES'),
        AutoARIMA(season_length=season, alias='AutoARIMA'),
        AutoTheta(season_length=season,
                   decomposition_type="additive",
                   model="STM"),
        OptimizedTheta(season_length=season,
                       decomposition_type="additive", alias="ThetaAdd"),
        OptimizedTheta(season_length=season,
                       decomposition_type="multiplicative", alias="ThetaMult")
    ]

    #ts_models = []

    # for model in model_aliases:
    #     for i in all_models:
    #         if model == str(i):
    #             ts_models.append(i)
    return ts_models

def only_forecast(df: pd.DataFrame, h, season, freq, ts_models):
    # create the model object, for each model and let user know time required for each fit
    model = StatsForecast(
        df = df,
        models=ts_models,
        freq =freq,
        n_jobs=-1,
        fallback_model=Naive()
    )

    forecast = model.forecast(df=df, h=h, level=[95])
    return forecast, model

def fit(df: pd.DataFrame, h, season, freq, ts_models):
    # create the model object, for each model and let user know time required for each fit
    model = StatsForecast(df=df,
                        models=ts_models,
                        freq = freq,
                        n_jobs=1,
                        verbose=True)

    model.fit(df)
    return model

def make_prediction(model, df, h):
    # predict future h periods, with level of confidence
    prediction = model.predict(h=h, level=[95])
    return prediction

def plot_forecasts(model, df, forecast, model_aliases):
    # Plot specific unique_ids and models
    forecast_plot_small = model.plot(
        df,
        forecast,
        models=MODEL_ALIASES,
        level=[95],
        engine='plotly')

    forecast_plot = StatsForecast.plot(df, forecast, engine='matplotlib')
    plotname = 'forecast_plot'
    forecast_plot.savefig(f'/Users/tmb/PycharmProjects/data-science/UFE/output_figs/{ORG}/{plotname}')
    return


def prep_forecast_for_s3(df: pd.DataFrame, df_ids, evals):
    df.reset_index(inplace=True)
    evals.reset_index(inplace=True)

    dashboard_cols = [
        'tm'  # timestamp
        , 'meter'
        , 'measure'
        , 'account_cd'  # account m3ter uid
        , 'account_nm'
        , 'ts_id' # time series unique id
        , 'z'  # prediction
        , 'z0'  # lower bound of 95% confidence interval
        , 'z1'  # lower bound of 95% confidence interval
        , '.model'  # model (e.g. model_)
    ]

    #print(tabulate(df.head(5), headers='keys', tablefmt='psql'))
    #print(tabulate(df_ids.head(5), headers='keys', tablefmt='psql'))

    #df[['account_cd','meter','measure']] = df['unique_id'].str.split('%', expand=True)
    # df.columns ['unique_id', 'ds', 'best_model', 'best_model-hi-95','best_model-lo-95']
    # df_ids.columns ['account_cd', 'account_nm', 'meter', 'measure', 'ts_id']

    df = pd.merge(df, evals[['unique_id', 'best_model']], how='left', on='unique_id')
    df_ids.columns = ['account_cd', 'account_nm', 'meter', 'measure', 'unique_id']
    df = pd.merge(df, df_ids, how='left', on='unique_id')
    df.rename(columns={'unique_id': 'ts_id', 'best_model_y': '.model'}, inplace=True)

    # for alias in model_aliases:
    #     model_cols = [ alias, alias + '-lo-95', alias + '-hi-95'] # TODO make prediction interval bands dynamic
    # df_cols = ['ds','meter','measure', 'account_cd', 'account_nm']
    # for col in model_cols:
    #     df_cols.append(col)
    # df = df[df.columns.intersection(df_cols)]

    df.sort_values(['account_cd', 'meter', 'ds'], inplace=True)

    print(df.tail(10))

    # dfs = []
    # for alias in model_aliases:
    #     iterator_list = ['df' + alias, alias, alias + '-lo-95', alias + '-hi-95']
    #     iterator_list[0] = df[['ds', 'meter', 'measurement', 'account_cd', 'account_nm', iterator_list[1], iterator_list[2], iterator_list[3]]]
    #     iterator_list[0]['.model'] = alias # TODO fix so not getting SettingWithCopyWarning
    #     iterator_list[0].columns = dashboard_cols
    #     dfs.append(iterator_list[0])

    #dfAll = pd.concat(dfs, ignore_index=True)

    #dfAll['tm'] = pd.to_datetime(df["tm"].dt.strftime('%Y-%m-%dT%H:%M:%SZ'))  #format='%Y-%m-%dT%H:%M:%SZ'
    #dfAll['tm'] = df["ds"].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    if USER is None:
        pass
        rs3.write_csv_log_to_S3(df, 'dfBest')
    else:
        df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/dfBest.csv')
    return df

def prep_meta_data_for_s3():
    # TODO change meta as dictionary passed parameter to function
    meta = {'nm': 'typ', 'meter': 'dim', 'measurement': 'dim', 'account_nm': 'dim', 'account_cd': 'dim', '.model': 'dim', 'z': 'measure', 'tm': 'time', '_intrvl': '1h', 'z0': 'measure', 'z1': 'measure'}
    meta_list = list(meta.items())
    with open ('/tmp/tmbmeta.txt', 'w') as file:
        for i in meta_list:
            file.write(','.join(map(str, i))+'\n')  # file type => _io.TextIOWrapper
    return file

def main(data, freq, dimkey_list, account):
    ##################################
    # plot raw time series  ##########
    ##################################
    #plot(dfUsage, dfUsage['account'].iloc[0], dfUsage['meter'].iloc[0])

    ##################################
    # Data Wrangle #########
    ##################################
    startdate, enddate = preproc.select_date_range(freq)
    dfUsage_clean, df_ids = preproc.clean_data(data, 'tm', 'y', startdate, enddate)

    if USER is None:
        rs3.write_csv_log_to_S3(dfUsage_clean, 'dfUsage_clean')
    else:
        dfUsage_clean.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/dfUsage.csv', index=False)
        df_ids.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/df_ids.csv', index=False)

    # get parameters and models
    season = get_season(freq)
    h = round(dfUsage_clean['ds'].nunique() * 0.15) # forecast horizon
    #df_to_forecast, df_naive = ppreprocp.filter_data(dfUsage_clean)  # TMB testing
    df_naive = pd.DataFrame()                                  # TMB testing
    df_to_forecast = dfUsage_clean                             # TMB testing

    # Stationarity and Seasonality tests - cannot run on lambda as it will time out
    #stationarity_df=preproc.test_stationarity_dickey_fuller(df_to_forecast)
    #seasonality_df=preproc.seasonality_test(df_to_forecast, season)
    #tests = pd.merge(stationarity_df, seasonality_df, on='unique_id')
    #logger.info('Stationarity/Seasonality Results')
    #logger.info(tabulate(tests.head(), headers="keys", tablefmt="psql"))
    #preproc.decompose(dfUsage_clean)

    ##################################
    # fit, predict, forecast #########
    ##################################
    # forecast naive time series
    naive_model_alias = ['Naive']
    ts_naive_model = select_models(freq, naive_model_alias)

    naive_init_fit = time()
    if len(df_naive) > 0:
        forecast_naive, naive_model = only_forecast(df_naive, h, season, freq, ts_naive_model)
        # plot_forecasts(naive_model, dfUsage_clean, forecast_only_naive, naive_model_aliases)
    naive_end_fit = time()
    logger.info(f'Naive Fit Minutes: {(naive_end_fit - naive_init_fit) / 60}')

    # set up model(s) for rest of data
    model_aliases = MODEL_ALIASES
    ts_models = select_models(freq, model_aliases)

    if FIT_FORECAST == 'FIT':
        init_fit = time()
        model = fit(df_to_forecast, h, season, freq, ts_models)
        end_fit=time()
        logger.info(f'Fit Minutes: {(end_fit - init_fit) / 60}')
        rs3.write_model_to_s3(model, fit_folder+freq+'/', model_aliases[0] + '.pkl')
        #forecasts = make_prediction(model, df_to_forecast, h) # Should we only fit here?
    elif FIT_FORECAST == 'FORECAST':
        # Read model from s3
        model = rs3.read_model_from_s3(fit_folder+freq+"/", model_aliases[0]+'.pkl')
        init_predict = time()
        forecasts = make_prediction(model, df_to_forecast, h)
        end_predict = time()
        logger.info(f'Forecast Minutes: {(end_predict - init_predict) / 60}')
    elif FIT_FORECAST == 'BOTH':
        if len(df_to_forecast)>0:
            logger.info(f'Start Fit Minutes: {naive_init_fit}')
            init_foreonly = time()
            forecasts, model = only_forecast(df_to_forecast, h, season, freq, ts_models)
            end_foreonly = time()
            logger.info(f'Forecast Only Minutes:  + {(end_foreonly - init_foreonly) / 60}')


        forecasts.to_csv(
            f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/forecasts.csv')

        # plot and analyse
        #plot_forecasts(model, dfUsage_clean, forecast_only, model_aliases)

        ids = ['c82f6cc26578128fe40e32e657ef5dbdc5bffb300122292619d4f15c4bde508f'
            , '3a2b5a0934b8f866ab36c8818313c90722eb0e6ca38e5103ebaf3e6c70762f01'
            ,'07ba99e5b1dd640d4d9f8d3a914a7af5f0e64710c5668a2b37be6d8e4b20cda5']

        # cross validate
        if USER is None:
            pass #TODO cannot run cross validation on lambda as it will time out - must convert to EC2
            #cv_rmse_df = err.cross_validate(dfUsage_clean, model, h)
            #rs3.write_csv_log_to_S3(cv_rmse_df, 'cv_rmse_df')
        else:
            crossvalidation_df = postproc.cross_validate_simple(dfUsage_clean, model, h)
            crossvalidation_df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/cv_rmse_df.csv')
            evaluation_df = postproc.evaluate_cross_validation(crossvalidation_df, rmse)
            evaluation_df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/evaluation_df.csv')

            summary_df = evaluation_df.groupby('best_model').size().sort_values().to_frame()
            summary_df.reset_index().columns = ["Model", "Nr. of unique_ids"]
            summary_df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/summary_df.csv')

            best_forecast_per_ts = postproc.get_best_model_forecast(forecasts, evaluation_df)
            best_forecast_per_ts.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/best_forecast_per_ts.csv')


    ##################################
    # save to s3 if directed #########
    ##################################
    #if USER is None:
    if 2>1:
        # metadatafile = prep_meta_data_for_s3() TODO remove or prep fields for saving

        # Naive forecasts
        if len(df_naive)>0:
            naive_forecast_to_save = prep_forecast_for_s3(forecast_naive, df_ids, naive_model_alias)

        else:
            naive_forecast_to_save = pd.DataFrame()

        # other forecasts
        if len(df_to_forecast)>0:
            forecast_to_save = prep_forecast_for_s3(best_forecasts, df_ids, evaluation_df)
        else:
            forecast_to_save = pd.DataFrame()

        combined_forecasts = pd.concat([naive_forecast_to_save,forecast_to_save], ignore_index=True)
        rs3.write_gz_csv_to_s3(combined_forecasts, forecast_folder + freq + '/',
                                'best_' + dt.today().strftime("%Y_%d_%m") + '_' + 'usage.gz')
        rs3.write_meta_to_s3(metadata_str, freq, forecast_folder + freq + '/',
                              'best_' + dt.today().strftime("%Y_%d_%m") + '_' + 'hier_2024_03_04_usage_meta.gz')

    else:
        name = model_aliases[0]
        if len(df_naive)>0:
            naive_forecast_to_save = prep_forecast_for_s3(forecast_naive, df_ids, naive_model_alias) #TODO convert naive_model_alias to df
            naive_forecast_to_save.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/forecast{name}.csv')
        if len(df_to_forecast)>0:
            if FIT_FORECAST == 'FORECAST':
                forecast_to_save = prep_forecast_for_s3(best_forecasts, df_ids, evaluation_df )
                forecast_to_save.to_csv(
                    f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/forecast{name}.csv')
            elif FIT_FORECAST == 'BOTH':
                forecast_to_save = prep_forecast_for_s3(best_forecasts, df_ids, evaluation_df)
                forecast_to_save.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/forecast{name}.csv')
    return

def lambda_handler(event, context):
    key, metadatakey = get_keys()
    dataloadcache, metadata_str = rs3.get_data(tidy_folder + freq + '/', key, metadatakey)
    meta_dict = preproc.meta_str_to_dict(metadata_str)
    dimkey_list = preproc.meta_to_dim_list(meta_dict)
    rs3.write_meta_tmp(metadata_str)
    data, account = preproc.select_ts(dataloadcache)
    main(data, freq, metadata_str, account) #TODO feed in dimkey_list to main?
    return

if __name__ == "__main__":
    freq = input("Hourly (1h) or Daily (1D) frequency: ")
    dataloadcache= pd.DataFrame()

    while True:
        if dataloadcache.empty:
            key, metadatakey = get_keys()
            dataloadcache, metadata_str = rs3.get_data(tidy_folder + freq + '/', key, metadatakey)
        meta_dict = preproc.meta_str_to_dict(metadata_str)
        dimkey_list = preproc.meta_to_dim_list(meta_dict)
        rs3.write_meta_tmp(metadata_str)
        data, account = preproc.select_ts(dataloadcache)
        main(data, freq, dimkey_list, account)  #TODO feed in dimkey_list to main?

        print("Press enter to re-run the script, CTRL-C to exit")
        sys.stdin.readline()
        importlib.reload(rs3)