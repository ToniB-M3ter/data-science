from datetime import datetime
from dateutil.relativedelta import relativedelta

import os
import pandas as pd
import numpy as np

from scipy.stats import kruskal
from statsforecast import StatsForecast
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

import readWriteS3 as rs3
USER = os.getenv('USER')

import logging
module_logger = logging.getLogger('ts_engine.pre-process')
logger = logging.getLogger('ts_engine.pre-process')

def split_data(df):
    #train = df.iloc[:int(0.75 * df['ds'].nunique())]  # TODO change 0.5 to horizon
    #valid = df.iloc[int(0.25 * df['ds'].nunique()) + 1:]  # TODO change 0.5*len(df) to horizon
    train = 1
    valid = 2
   # horizon (h)  = time periods into future for which a forecast will be made
    h = round((len(df) * 0.10))
    return train, valid, h

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

    logger.info('Fit from '  + str(startdate) +' to '+ str(enddate))

    # Remove whitespace from account name
    tmp_df = df['account'].copy()
    #tmp_df.replace(' ', '', regex=True, inplace=True)
    df.loc[:, 'account'] = tmp_df

    # save unique combinations of account_id, meter, and measurement for formatting forecast file before saving
    df_ids = df[['account', 'account_id', 'meter', 'measurement']].drop_duplicates()

    if USER is None:
        rs3.write_csv_log_to_S3(df_ids, 'df_ids')
    else:
        df_ids.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/df_ids.csv')

    # format for fitting and forecasting - select subset of columns and add unique_id column
    df.loc[:,'unique_id'] = df.apply(
        lambda row: row.account + '_' + row.meter + '_' + row.measurement.split(' ')[0], axis=1)

    df = df[[datetime_col, y, 'unique_id']]
    df.columns = ['ds', 'y', 'unique_id']

    # Tell user number of time series
    logger.info(str(df['unique_id'].nunique()) + ' Unique Time Series')

    # plot for inspection
    #x = StatsForecast.plot(df)
    #x.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output_figs/{}'.format('ts_eng_input_data'))

    return df, df_ids

def filter_data(clean_df: pd.DataFrame):
    z = clean_df['ds'].nunique()
    # Score time series based on percentage of zeros
    dfZeros = clean_df.groupby('unique_id').agg(lambda x:x.eq(0).sum()).reset_index()
    dfZeros['pct_zeros'] = dfZeros['y']/z

    # If less that 5% of values are non-zero, forecast with naive model
    df_naive_list = dfZeros[dfZeros['pct_zeros']> 0.94]['unique_id']
    df_naive = clean_df[clean_df['unique_id'].isin(df_naive_list)]
    logger.info(str(len(df_naive_list)) + ' time series will be forecast with naive model out of ' + str(clean_df['unique_id'].nunique()))

    # of the remaining time series
    df_forecast_list = dfZeros[dfZeros['pct_zeros']<= 0.94]['unique_id']
    df_to_forecast = clean_df[clean_df['unique_id'].isin(df_forecast_list)]

    return df_to_forecast, df_naive

def decompose(df: pd.DataFrame) -> pd.Series:
    dfdecompose = df.set_index('ds')
    unique_ids = df['unique_id'].unique()
    for id in unique_ids:
        dfdecompose = dfdecompose[dfdecompose['unique_id']==df['unique_id'].unique()[id]]
        result = seasonal_decompose(dfdecompose['y'],model='additive')
        #print(result.trend)
        #print(result.seasonal)
        #print(result.resid)
        #print(result.observed)

        #result.plot()
        #plt.show()
    return

def test_stationarity_dickey_fuller(df: pd.DataFrame) -> pd.DataFrame:
    unique_ids = df['unique_id'].unique()
    stationarity_list = []
    stationarity_sccore = []
    unique_ids_list = []
    for id in unique_ids:
        stationarity = False
        result = adfuller(df[df['unique_id']==id]['y'],
                          autolag='AIC')
        #print(id)
        #print('ADF Statistic:', result[0])
        #print('p-value:', result[1])
        #print('Critical Values:', result[4])
        unique_ids_list.append(id)
        if result[1]<0.05:
            stationarity = True
        stationarity_list.append(stationarity)
        stationarity_sccore.append(result[1])
    stationarity_df = pd.DataFrame({'unique_id': unique_ids_list, 'stationarity': stationarity_list, 'stationarity_score': stationarity_sccore})
    return stationarity_df

def seasonality_test(df, season):
    unique_ids = df['unique_id'].unique()
    idx = np.arange(df['ds'].nunique()) % season
    seasonal_list = []
    seasonal_score = []
    unique_ids_list = []
    for id in unique_ids:
        seasonal = False
        H_statistic, p_value = kruskal(df[df['unique_id']==id]['y'], idx)
        if p_value <= 0.05:
            seasonal = True
        unique_ids_list.append(id)
        seasonal_list.append(seasonal)
        seasonal_score.append(p_value)
    seasonality_df = pd.DataFrame({'unique_id': unique_ids_list, 'seasonal': seasonal, 'seasonal_score': seasonal_score})
    return seasonality_df
