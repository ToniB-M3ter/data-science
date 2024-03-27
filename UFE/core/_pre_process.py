import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

import _readWrite as rw

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

module_logger = logging.getLogger('UFE.pre-process')
logger = logging.getLogger('UFE.pre-process')

USER = os.getenv('USER')

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

def select_models(season: int, indices: list) -> list:
    all_models = [
        Naive(alias='Naive'),
        HistoricAverage(),
        SeasonalNaive(season_length=season, alias='SeasonalNaive'),
        MSTL(season_length=[season, season*2]),
        HoltWinters(season_length=season, error_type="A", alias="HWAdd"),
        HoltWinters(season_length=season, error_type="M", alias="HWMult"),
        AutoETS(model=['Z', 'Z', 'Z'], season_length=season, alias='AutoETS'),
        AutoCES(season_length=season, alias='AutoCES'),
        AutoARIMA(season_length=season, alias='AutoARIMA'),
        AutoTheta(season_length=season,decomposition_type="additive",model="STM"),
        OptimizedTheta(season_length=season, decomposition_type="additive", alias="ThetaAdd"),
        OptimizedTheta(season_length=season, decomposition_type="multiplicative", alias="ThetaMult")
    ]

    ts_models=[]

    for i in indices:
        ts_models.append(all_models[i])

    return ts_models

def select_date_range(data_freq: str)-> datetime:
    if USER is None: # if running on aws automatically set freq
        if 'D' in data_freq:
            startdate = datetime.today() - relativedelta(months=6)
        elif 'h' in data_freq:
            startdate = datetime.today() - relativedelta(months=1)

        enddate = datetime.today() - relativedelta(days=1)
    else:
        # if we are running locally ask user for input
        startdate_input = input('Start date (YYYY-mm-dd HH:MM:SS format)? ')
        enddate_input = input('End date (YYYY-mm-dd HH:MM:SS format)? ')

        if startdate_input == '':
            # Select date range depending on frequency of data
            if 'D' in data_freq:
                startdate = datetime.today() - relativedelta(months=6)
            elif 'h' in data_freq:
                startdate = datetime.today() - relativedelta(months=1)
        else:
            startdate=startdate_input

        if enddate_input == '':
            enddate = datetime.today() - relativedelta(days=1)
        else:
            enddate=enddate_input

    return startdate, enddate

def clean_data(raw_df: pd.DataFrame, datetime_col: str, y: str, startdate, enddate) -> pd.DataFrame:
    # filter dates
    datetime_mask = (raw_df['tm'] > startdate) & (raw_df['tm'] <= enddate)
    df = raw_df.loc[datetime_mask]

    logger.info('clean_data from '  + str(startdate) +' to '+ str(enddate))

    # save unique combinations of account_id, meter, and measurement for formatting forecast file before saving
    # TODO allow for all combination of dimensions
    df_ids = df[['account_cd', 'account_nm', 'meter', 'measure', 'ts_id']].drop_duplicates()
    df_ts_ids = df['ts_id'].unique()

    print('compare df_ids, ' + str(len(df_ids)) + ' to df_ts_ids, ' + str(len(df_ts_ids)))

    if USER is None:
        rw.write_csv_log_to_S3(df_ids, 'df_ids')
    else:
        df_ids.to_csv('/Users/tmb/PycharmProjects/data-science/UFEPOC/output_files/df_ids.csv')

    df['unique_id'] = df['ts_id']
    df = df[[datetime_col, y, 'unique_id']]
    df.columns = ['ds', 'y', 'unique_id']

    # Tell user number of time series
    logger.info(str(df['unique_id'].nunique()) + ' Unique Time Series')

    return df, df_ids

def filter_data(df: pd.DataFrame, zero_threshold: float):

    z = df['ds'].nunique()
    # Score time series based on percentage of zeros
    dfZeros = df.groupby('unique_id').agg(lambda x:x.eq(0).sum()).reset_index()
    dfZeros['pct_zeros'] = dfZeros['y']/z

    # If less than threshold of values are non-zero, separate with view to forecast with naive model
    keep = dfZeros[dfZeros['pct_zeros'] < zero_threshold]
    df_to_forecast = pd.merge(df, keep['unique_id'], left_on='unique_id',
                             right_on='unique_id', how='right')

    df_naive = df[~df['unique_id'].isin(df_to_forecast['unique_id'])]

    logger.info(str( df['unique_id'].nunique() - df_to_forecast['unique_id'].nunique() ) + ' time series will be forecast with naive model out of ' + str(df['unique_id'].nunique()))
    print(str
          (df['unique_id'].nunique() - df_to_forecast['unique_id'].nunique()) + ' time series will be forecast with naive model out of ' + str(df['unique_id'].nunique())
          )
    return df_to_forecast, df_naive

def add_noise(Y_df):
    # MinT along with other methods require a positive definite covariance matrix
    # for the residuals, when dealing with 0s as residuals the methods break
    # data is augmented with minimal normal noise to avoid this error.

    # Additionally when forecasting base forecasts with many zeros,
    # will avoid failure of models that cannot tolerate zeros in time series
    Y_df['y'] = Y_df['y'] + np.random.normal(loc=0.0, scale=0.01, size=len(Y_df))
    return Y_df