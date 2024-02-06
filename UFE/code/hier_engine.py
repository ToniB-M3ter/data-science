"""
Key Objects and Dataframes:
tags: Dictionary where each key is a level and its value contains tags associated to that level.
S_df:
df: cleaned input data
Y_train_df: training data
Y_test_df: test data
Y_h: observed data including aggregations
Y_hat_df: Base Forecasts NB predictions will not be coherent at this point
Y_fitted_df: Fitted data from tmin - tmax
Y_rec_df: Coherent reconciled predictions

"""
import numpy as np
import pandas as pd
import json
import os
from time import time
import matplotlib.pyplot as plt

import readWriteS3 as rs3
import error_analysis as err
import pre_process as pp

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# compute base forecast not coherent
from statsforecast.core import StatsForecast
from statsforecast.models import (
    SeasonalNaive, # model using the previous season's data as the forecast
    Naive, # Simple naive model using the last observed value as the forecast
    HistoricAverage, # Average of all historical data
    AutoETS, # Automatically selects best ETS model based on AIC
    AutoARIMA, # ARIMA model that automatically select the parameters for given time series with AIC and cross validation
    HoltWinters, #HoltWinters ETS model
    GARCH,     # capture heteroscedasticity (changes in variance / volatility)
    MSTL,      # multiple seasonalities
    OptimizedTheta,   # decomposed ts then keeps long-term trend and seasonality and uses noise to adjust short term forecasts
    Theta,
    AutoTheta
    )

#obtain hierarchical reconciliation methods and evaluation
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.evaluation import HierarchicalEvaluation
from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut, MinTrace
from hierarchicalforecast.utils import aggregate, HierarchicalPlot

# this makes it so that the outputs of the predict methods have the id as a column
# instead of as the index
os.environ['NIXTLA_ID_AS_COL'] = '1'
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

def add_noise(Y_df):
    # MinT along other methods require a positive definite covariance matrix
    # for the residuals, when dealing with 0s as residuals the methods break
    # data is augmented with minimal normal noise to avoid this error.
    Y_df['y'] = Y_df['y'] + np.random.normal(loc=0.0, scale=0.01, size=len(Y_df))
    return Y_df

def get_spec():
    Spc = [
        ['org'],
        ['org', 'account'],
        ['org','meter'],
        ['org','account','meter']
    ]
    return Spc

def get_aggregates(Y_df, Spc):
    Y_df, S_df, tags = aggregate(Y_df, Spc)
    Y_df = Y_df.reset_index()

    #print('hier_levels')
    #print(hier_levels)
    #print('hier_idxs')
    #print(hier_idxs[0:10])
    hier_linked_idxs = S_df.values.T
    #print(hier_linked_idxs)
    return Y_df, S_df, tags

def hier_prep(df: pd.DataFrame, startdate, enddate) -> pd.DataFrame:
    # filter dates
    datetime_mask = (df['tm'] > startdate) & (df['tm'] <= enddate)
    df = df.loc[datetime_mask]
    print('Fit from '  + str(startdate) +' to '+ str(enddate))

    # Remove whitespace from account name
    #tmp_df = df['account_cd'].copy()
    #tmp_df.replace(' ', '', regex=True, inplace=True)  #account_cd=account code (no spaces); account_nm=account name w/spaces
    #df[:, 'account_cd'] = tmp_df

    # drop unneeded columns & add column for org and rename tm --> ds
    df = df[['tm','account_cd','meter', 'measurement', 'y']]
    # if <aggregated> accounts, meter or measurements exist remove them as these aggregations will be created with hier aggregations
    df = df[(df.account_cd != '<aggregated>') & (df.meter != '<aggregated>') & (df.measurement != '<aggregated>')]
    df.insert(0,'org', 'onfido')  # TODO parmeterise org name
    df= df.rename(columns={'tm':'ds', 'account_cd':'account'})
    df=df[['org','account','meter','ds','y']]
    print('length of df: ' + str(len(df)))

    #df_ids = df[['account', 'meter']].drop_duplicates()
    #df_ids.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/df_ids.csv')
    return df

def hier_filter_df(df):
    """Filter out time series that are vastly zeros e.g. 75% or > zeros"""
    z = df['ds'].nunique()
    dfZeros = df.groupby(['org', 'account', 'meter']).agg(lambda x: x.eq(0).sum()).reset_index()
    dfZeros['pct_zeros'] = dfZeros['y'] / z
    keep = dfZeros[dfZeros['pct_zeros'] < 0.25]
    df_to_Forecast = pd.merge(df, keep[['account', 'meter', 'pct_zeros']], left_on=['account', 'meter'],
                        right_on=['account', 'meter'], how='right')

    df_to_Forecast.to_csv(
        '/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/df_to_Forecast.csv')
    return df_to_Forecast

def split_data(Y_df):
    h = round(Y_df['ds'].nunique() * 0.15)  # forecast horizon
    Y_test_df = Y_df.groupby('unique_id').tail(h)
    Y_train_df = Y_df.drop(Y_test_df.index)
    return Y_train_df, Y_test_df, h

def base_forecasts(Y_train_df, data_freq, h):
    season = get_season(data_freq)

    fcst = StatsForecast(
        df=Y_train_df,
        models=[AutoETS(season_length=season),
                #SeasonalNaive(season_length=season)
                #AutoARIMA(season_length=season),
                #AutoTheta(season_length=season,
                #          decomposition_type="additive",
                #          model="STM")
                ],
        freq=data_freq,
        n_jobs=-1
    )
    Y_hat_df = fcst.forecast(h=h, level=[95], fitted=True) #forecast after tmax of training set
    Y_fitted_df = fcst.forecast_fitted_values() #fitted data from tmin - tmax
    return Y_hat_df, Y_fitted_df

def reconcile_forecasts(Y_hat_df, Y_fitted_df, Y_train_df, S_df, tags):
    reconcilers = [
        BottomUp(),
        #TopDown(method='average_proportions'),  #options forecast_proportions, average_proportions, proportion_averages
        MinTrace(method='mint_shrink', nonnegative=True),
        MinTrace(method='ols', nonnegative=True)
    ]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    try: Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df,
                              Y_df=Y_train_df, #Y_fitted_df
                              S=S_df,
                              tags=tags)
    except Exception as error:
        print("An exception has occured when trying to reconcile the hierarchy with MinT Shrink method, "
              "reconciliation will occur by Bottoms up and MinTrace(OLS) methods", error)
        reconcilers = [BottomUp(), MinTrace(method='ols', nonnegative=True)]
        hrec = HierarchicalReconciliation(reconcilers=reconcilers)
        Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df,
                                  Y_df=Y_train_df, #Y_fitted_df
                                  S=S_df,
                                  tags=tags)
    return Y_rec_df

def evaluate_forecasts(Y_df, Y_rec_df, Y_test_df, Y_train_df, tags):
    eval_tags = {}
    eval_tags['org'] = tags['org']
    eval_tags['account'] = tags['org/account']
    eval_tags['meter'] = tags['org/meter']
    eval_tags['account_meter'] = tags['org/account/meter']
    eval_tags['All'] = np.concatenate(list(tags.values()))

    evaluator = HierarchicalEvaluation(evaluators=[err.rmse_calc])
    evaluation = evaluator.evaluate(
        Y_hat_df=Y_rec_df,  # feeding in reconciled values, e.g. Y_rec_df, not just forecast values
        Y_test_df=Y_test_df.set_index('unique_id'),
        tags=tags,
        Y_df= Y_df    #Y_train_df.set_index('unique_id'),
    )
    #evaluation = evaluation.drop('Overall')
    #evaluation.columns = ['Base', 'BottomUp', 'MinTrace(mint_shrink)', 'MinTrace(ols)']

    #evaluation = evaluation.applymap('{:.2f}'.format)
    evaluation = evaluation.map('{:.2f}'.format)
    print(evaluation)
    #print(evaluation.query('metric == "rmse_calc"'))
    return evaluation

def select_best_model(Y_rec_df, evaluation): #TODO programatically select best model
    Y_rec_df.reset_index(inplace=True)
    best_model = Y_rec_df[['unique_id', 'ds','AutoETS-lo-95', 'AutoETS-hi-95', 'AutoETS/MinTrace_method-ols_nonnegative-True']]
    return best_model

def prep_forecast_for_s3(df_best_model: pd.DataFrame, tags, model_name):
    """"df_best_model should contain forecast values only for model determined to be best fit
    e.g. df_best_model = Y_rec_df[['unique_id','ds','AutoETS/MinTrace_method-ols', 'AutoETS-lo-95','AutoETS-hi-95']]"""

    splt_tag = []
    for tag in tags:
        splt_tag.append(tag.split("/"))

    orgs=[]
    accounts=[]
    meters=[]
    measurements=[]

    # construct lists of hierarchical levels for final file
    for tag in splt_tag:
        if len(tag) == 4:
            measurements.append('check_count')
            meters.append(tag[2])
            accounts.append(tag[1])
            orgs.append(tag[0])
        elif len(tag) == 3:
            measurements.append('check_count')
            meters.append(tag[2])
            accounts.append(tag[1])
            orgs.append(tag[0])
        elif len(tag) == 2:
            measurements.append('check_count')
            meters.append('ALL')
            accounts.append(tag[1])
            orgs.append(tag[0])
        elif len(tag) == 1:
            measurements.append('check_count')
            meters.append('ALL')
            accounts.append('ALL')
            orgs.append(tag[0])
        else:
            print(len(tag))
            print(tag)
            print('unknown tag')

    hier_forecast = pd.DataFrame({"unique_id": tags, "org": orgs, "account_cd": accounts, "meter": meters, 'measurement': measurements})

    #df_best_model.reset_index(inplace=True)
    #df_best_model['z'] = df_best_model.iloc[:,-1:] #get last column of df
    print(df_best_model.columns)
    df_best_model.columns = ['unique_id','ds', 'z0','z1','z']
    df_best_model['tm'] = df_best_model["ds"].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    df_best_model['.model'] = model_name[0]

    hier_forecast = pd.merge(hier_forecast, df_best_model, on='unique_id')
    hier_forcast = hier_forecast[['tm'  # timestamp
        , 'meter'
        , 'measurement'
        , 'account_cd'  # account m3ter uidw
        , 'account_nm'
        , 'z'  # prediction
        , 'z0'  # lower bound of 95% confidence interval
        , 'z1'  # lower bound of 95% confidence interval
        , '.model'  # model (e.g. model_)
            ]]
    print(hier_forecast.head())

    if USER is None:
        rs3.write_csv_log_to_S3(hier_forecast, 'hier_forecast')
    else:
        hier_forecast.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hier_forecast.csv')
    return hier_forecast

def plot_correlations(df):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    plot_acf(df['y'],
             lags=28,
             alpha=0.1,
             ax=axs[0],
             use_vlines=True,
             color='lime')
    axs[0].set_title("Autocorrelation")
    plot_pacf(df['y'],
             lags=28,
             alpha=0.1,
             ax=axs[1],
             use_vlines=True,
             color='fuchsia')
    axs[1].set_title("Partial Autocorrelation")
    plt.show()
    return

def other_plots(S_df, tags, Y_df, Y_hat_df):

    # hplot.plot_hierarchical_predictions_gap( # plots of aggregated predictions at different levels of the hierarchical structure. aggregation() method used to aggregate
    #     Y_df=Y_hat_df,
    #     models='AutoETS',
    #     xlabel='Day',
    #     ylabel='Predictions'
    # )
    # plt.show()

    time_series = Y_df['unique_id'].unique().tolist()
    hplot = HierarchicalPlot(S=S_df, tags=tags)  # plotting class containing plotting methods
    hplot.plot_summing_matrix()  # plot hierarchical aggregation contraints matrix, S
    hplot.plot_hierarchically_linked_series( # plot collection of hierarchically linked series plots associated with the bottom_series and filetered modls and prediction interval level
        bottom_series=time_series[271],
        Y_df=Y_df.set_index('unique_id')
        )
    plt.show()

    return

def forecast_plots(S_df, tags, Y_df, Y_hat_df, Y_rec_df, series):
    hplot = HierarchicalPlot(S=S_df, tags=tags) # plotting class containing plotting methods

    plot_df = pd.concat([Y_df.set_index(['unique_id', 'ds']),
                         Y_rec_df.set_index('ds', append=True)], axis=1)
    plot_df = plot_df.reset_index('ds')

    # plot single series with filtered models and prediction interval
    plot_df = pd.concat([Y_df.set_index(['unique_id', 'ds']),
                         Y_rec_df.set_index('ds', append=True)], axis=1)
    plot_df = plot_df.reset_index('ds')
    hplot.plot_series(
        series=series,
        Y_df=plot_df,
        models=['y', 'AutoETS', 'SeasonalNaive','AutoETS/BottomUp',	'SeasonalNaive/BottomUp', 'AutoETS/MinTrace_method-ols_nonnegative-True', 'SeasonalNaive/MinTrace_method-ols_nonnegative-True'],
        level=[90]
    )
    plt.xticks(rotation=90)
    plt.show()
    return

def main(data, freq, metadata_str, account):
    # Clean and Prepare Data
    startdate, enddate = pp.select_date_range(freq)
    df = hier_prep(data, startdate, enddate)
    df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/df.csv')

    # Filter time series with > 90% zeros out of analysis
    df_to_forecast = hier_filter_df(df)

    Spc = get_spec()
    Y_df, S_df, tags = get_aggregates(df_to_forecast, Spc) # unique_id column created ; if filtering use df_to_forecast
    Y_df = add_noise(Y_df)
    Y_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/Y_df.csv')
    S_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/S_df.csv')

    #plot_correlations(Y_df) # Partial and Auto Correlation plots
    Y_train_df, Y_test_df, h = split_data(Y_df)

    # Fit Base Forecasts
    init_fit = time()
    Y_hat_df, Y_fitted_df= base_forecasts(Y_train_df, freq, h) # change to Y_df for prod
    end_fit=time()
    print(f'Forecast Minutes: {(end_fit - init_fit) / 60}')

    Y_hat_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/Y_hat_df.csv')
    Y_fitted_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/Y_fitted_df.csv')

    # Reconcile
    Y_rec_df = reconcile_forecasts(Y_hat_df, Y_fitted_df, Y_train_df, S_df, tags)
    Y_rec_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/Y_rec_df.csv')
    rs3.write_csv_log_to_S3(Y_rec_df, 'reconciled_forecasts')

    # Evaluate
    #evaluation = evaluate_forecasts(Y_df, Y_rec_df, Y_test_df, Y_train_df, tags)
    #evaluation.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/evaluation.csv')

    # TODO select best model/reconciliation method
    evaluation = pd.DataFrame()
    best_model=select_best_model(Y_rec_df, evaluation)

    # Prep for save
    prep_forecast_for_s3(best_model, S_df.index,['AutoETS'])

    # Visualise data
    other_plots(S_df, tags, Y_df, Y_hat_df)
    cnt = 0
    while cnt < 20:
        series = input("which series? ")
        if not series:
            pass
        else:
            forecast_plots(S_df, tags, Y_df, Y_hat_df, Y_rec_df, series)
        cnt = cnt+1

if __name__ == "__main__":
    #freq = input("Hourly (1h) or Daily (1D) frequency: ")
    freq = '1D'
    dataloadcache= pd.DataFrame()

    key, metadatakey = get_keys()
    dataloadcache, metadata_str = rs3.get_data('2_tidy/' + freq + '/', key, metadatakey)
    data, account = pp.select_ts(dataloadcache)
    main(data, freq, metadata_str, account)
