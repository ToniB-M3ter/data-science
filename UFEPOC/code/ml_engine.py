import random
import asyncio
import time
import pandas as pd
import re

import lightgbm as lgb
import xgboost as xgb

#from tabulate import tabulate
import matplotlib.pyplot as plt

#import readWriteS3 as rs3
#import error_analysis as err
#import pre_process as pp

from mlforecast import MLForecast
from mlforecast.target_transforms import Differences, LocalStandardScaler
from mlforecast.lag_transforms import RollingMean, ExpandingStd
from window_ops.expanding import expanding_mean
from window_ops.shift import shift_array
from sklearn.linear_model import LinearRegression

from utilsforecast.plotting import plot_series
from datasetsforecast.m4 import M4, M4Evaluation, M4Info

# Select models
models = [
    lgb.LGBMRegressor(verbosity=-1),
    xgb.XGBRegressor()
    #RandomForestRegressor(random_state=0)
]

def make_plot(name, df, data=None, max_insample_length=None):
    # visualise forecast

    fig = plot_series(df=df, forecasts_df=data, max_insample_length=max_insample_length)
    fig.savefig('/Users/tmb/PycharmProjects/data-science/UFEPOC/output_figs/{}.png'.format(name+'_plot'))

def get_data():
    #await M4.async_download('data', group='Hourly')
    # df, *_ = M4.load('data', 'Hourly')
    # uids = df['unique_id'].unique()
    # random.seed(0)
    # sample_uids = random.choices(uids, k=4)
    # df = df[df['unique_id'].isin(sample_uids)].reset_index(drop=True)
    # df['ds'] = df['ds'].astype('int64')
    df = pd.read_csv('/UFEPOC/output_files/hierarchical/onfido/Y_df.csv', index_col=0)
    df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')
    fig = plot_series(df, max_insample_length=20 * 7, engine='matplotlib')
    fig.savefig('/Users/tmb/PycharmProjects/data-science/UFEPOC/output_figs/{}.png'.format('raw'))
    return df

def process(df):
    fcst = MLForecast(
        models=[],  # we're not interested in modeling yet
        freq='D',  # our series have integer timestamps, so we'll just add 1 in every timestep
        lags=[1, 7], # changed from 24 to 7
        target_transforms=[Differences([7])], # changed from 24 to 7
    )
    prep = fcst.preprocess(df)
    print(prep.head(10))
    print(prep.drop(columns=['unique_id', 'ds']).corr()['y'])
    return prep

def hour_index(times):
    return times % 7 # changed from 24 to 7

def fit_model(df):
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    #df['y'] = df['y'] + abs(min(df['y']))
    lgb_params = {
        'verbosity': -1,
        'num_leaves': 512,
    }
    # fit model
    fcst = MLForecast(
        models={
            'avg': lgb.LGBMRegressor(**lgb_params),
            'q75': lgb.LGBMRegressor(**lgb_params, objective='quantile', alpha=0.75),
            'q25': lgb.LGBMRegressor(**lgb_params, objective='quantile', alpha=0.25),
            #'fish': lgb.LGBMRegressor(**lgb_params, objective='poisson', num_trees=2, verbose=3)
            'huber': lgb.LGBMRegressor(**lgb_params, objective='huber', alpha=1),
        },
        freq='D',
        target_transforms=[Differences([7])], # changed from 24 to 7
        lags=[1, 7], # changed from 24 to 7
        #lag_transforms={1:[ExpandingStd()]}, # removed lag transforms
        date_features=['dayofweek'], # changed from hour_index
    )
    fcst.fit(df)
    preds = fcst.predict(24)
    fig = plot_series(df, preds, max_insample_length=24 * 7, engine='matplotlib')
    fig.savefig('/Users/tmb/PycharmProjects/data-science/UFEPOC/output_figs/{}.png'.format('preds'))
    return preds

def main():

    df = get_data()
    prep = process(df)
    preds = fit_model(df)

if __name__ == "__main__":
    main()