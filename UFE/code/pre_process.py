from datetime import datetime
from dateutil.relativedelta import relativedelta

import os, shutil, gzip
import pandas as pd
import numpy as np
import sklearn

from scipy.stats import kruskal
from statsforecast import StatsForecast
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

import readWriteS3 as rs3
USER = os.getenv('USER')
METER = os.getenv('METER')

import logging
module_logger = logging.getLogger('ts_engine.pre-process')
logger = logging.getLogger('ts_engine.pre-process')

def meta_str_to_dict(metadata_str):
    meta_dict={}
    meta_tmp = metadata_str.split('\n')
    for i in meta_tmp:
        if len(i.split(","))==2:
            meta_dict[i.split(",")[0]]=i.split(",")[1]
    return meta_dict

def meta_to_dim_list(meta_dict):
    dimkey_list = []
    for k,v in meta_dict.items():
        if v == 'dim':
            dimkey_list.append(k)
    return dimkey_list

def meta_dict_to_tmp_txt_file(meta_dict):
    if 'z' not in meta_dict.keys(): # if the forecasts keys are not in dictionary add them
        meta_dict['z']= 'measure'
        meta_dict['z0'] = 'measure'
        meta_dict['z1'] = 'measure'

    #f_meta = open('/Users/tmb/PycharmProjects/data-science/UFE/data/meta_to_txt_file.txt', "w")
    f_meta = open('tmp/meta_txt_file.txt', "w")
    f_meta.write("\n")
    for k in meta_dict.keys():
        f_meta.write("{}, {}\n".format(k, meta_dict[k]))
    f_meta.close()

    convert_text_to_zip() # convert tmp file to .gz for loading to s3
    return

def convert_text_to_zip():
    with open("/tmp/{}".format('meta_txt_file.txt'), 'rb') as f_in: # probably can stay hardcoded as its a temp file
        with gzip.open("/tmp/{}".format('meta_txt_file.gz'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            #f.write(content.encode("utf-8"))
    return

def split_data(Y_df: pd.DataFrame, train_splt: float):
    h = round(Y_df['ds'].nunique() * train_splt)  # forecast horizon
    Y_test_df = Y_df.groupby('unique_id').tail(h)
    Y_train_df = Y_df.drop(Y_test_df.index)
    return Y_train_df, Y_test_df, h

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

def select_ts(df):
    # If interactive, select Account(s) otherwise select all accounts
    logger.info(str(df['account_cd'].nunique()) + ' Unique accounts')

    all = ['all','All','ALL']
    if USER is None:
        account = 'all'
    else:
        account = input('Enter an account, all or small: ' )

    if account in all:
        accounts = df['account_cd'].unique()
        df = df.loc[(df['account_cd'].isin(accounts))]
    elif account == 'small':
        accounts = ['AssembledHQ Prod',
                    'BurstSMS - Production',
                    'Burst SMS - Local Test',
                    'ClickHouse QA',
                    'Sift Forecasting',
                    'Sift Production',
                    'Onfido Dev',
                    'Onfido Prod',
                    'Patagona - Sandbox',
                    'Patagona - Production',
                    'Prompt QA',
                    'Regal.io Prod',
                    'm3terBilllingOrg Production',
                    'TherapyIQ Production',
                    'Tricentis Prod',
                    'Unbabel Staging'] # subset of accounts that are known to work
        df = df.loc[(df['account_cd'].isin(accounts))]
    else:
        try:
            df = df.loc[(df['account_cd'] == account)]
        except:
            logger.error("Account %s doesn't exist" % (account))

    # Select meter
    if USER is None:
        meter = METER
    else:
        print(df['meter'].unique())
        meter = input('Enter a meter? ')

    if meter in all:
        meters = df['meter'].unique()
        df = df.loc[(df['meter'].isin(meters))]
    else:
        try:
            df = df.loc[df['meter'] == meter]
        except:
            logger.error("Meter %s doesn't exist" % (meter))

    logger.info(str(len(df)) + ' records from ' + str(df['tm'].min()) + ' to ' + str(df['tm'].max()) )
    return df, account.replace( ' ', '')

def clean_data(raw_df: pd.DataFrame, datetime_col: str, y: str, startdate, enddate) -> pd.DataFrame:
    # filter dates
    datetime_mask = (raw_df['tm'] > startdate) & (raw_df['tm'] <= enddate)
    df = raw_df.loc[datetime_mask]

    logger.info('clean_data from '  + str(startdate) +' to '+ str(enddate))

    # save unique combinations of account_id, meter, and measurement for formatting forecast file before saving
    df_ids = df[['account_cd', 'account_nm', 'meter', 'measure', 'ts_id']].drop_duplicates()
    df_ts_ids = df['ts_id'].unique()

    print('compare df_ids, ' + str(len(df_ids)) + ' to df_ts_ids, ' + str(len(df_ts_ids)))

    if USER is None:
        rs3.write_csv_log_to_S3(df_ids, 'df_ids')
    else:
        df_ids.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/df_ids.csv')

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
    Y_df['y'] = Y_df['y'] + np.random.normal(loc=0.0, scale=0.01, size=len(Y_df))
    return Y_df

class feature_eng:
    def __init__(self, value):
        pass

    def add_day_of_the_week(df):
        # df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')
        df['weekno'] = df['ds'].apply(lambda x: x.weekday())
        return df

    def hash_function(row):
        return (sklearn.utils.murmurhash3_32(row.education))

    def hash_meter(self, df):
        meter_feature = df.groupby(by=["meter"]).count().reset_index()["meter"].to_frame()
        meter_feature["meter"] = meter_feature.apply(self.hash_function, axis=1)
        return meter_feature

    def mod_function(self, row):
        return(abs(row.meter_has) % self.n_features)

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
