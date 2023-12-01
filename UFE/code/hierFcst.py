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
from time import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

import readWriteS3 as rs3
import error_analysis as err
import pre_process as pp

# compute base forecast not coherent
from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA, Naive, AutoETS

#obtain hierarchical reconciliation methods and evaluation
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.evaluation import HierarchicalEvaluation
from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut, MinTrace
from hierarchicalforecast.utils import aggregate, HierarchicalPlot

def get_keys():
    metadatakey = 'usage_meta.gz'
    key = 'usage.gz'
    return key, metadatakey

def clean_data(df: pd.DataFrame, startdate, enddate) -> pd.DataFrame:
    # filter dates
    datetime_mask = (df['tm'] > startdate) & (df['tm'] <= enddate)
    df = df.loc[datetime_mask]
    print('Fit from '  + str(startdate) +' to '+ str(enddate))

    # Remove whitespace from account name
    tmp_df = df['account'].copy()
    tmp_df.replace(' ', '', regex=True, inplace=True)
    df['account'] = tmp_df

    # drop unneeded columns & add column for org and rename tm --> ds
    #df.drop(columns=['n_loads', 'n_events', 'account_id', 'meter', 'measurement'], inplace=True)
    df = df[['tm','account','y']]
    df.insert(0,'org', 'm3ter')
    df= df.rename(columns={'tm':'ds'})
    df=df[['org','account','ds','y']]

    print('length of df: ' + str(len(df)))
    return df

def get_spec():
    Spc = [
        ['org'],
        ['org', 'account']
        #['org','meter']
        #['org','account','meter']
    ]
    return Spc

def get_aggregates(Y_df, Spc):
    Y_df, S_df, tags = aggregate(Y_df, Spc)
    Y_df = Y_df.reset_index()
    return Y_df, S_df, tags

def split_data(Y_df):
    h = round(Y_df['ds'].nunique() * 0.15)  # forecast horizon
    Y_test_df = Y_df.groupby('unique_id').tail(h)
    Y_train_df = Y_df.drop(Y_test_df.index)
    return Y_train_df, Y_test_df, h

def base_forecasts(Y_train_df, data_freq, h):
    if 'D' in data_freq:
        data_freq = 'D'
        season = 7
    elif 'h' in data_freq:
        data_freq = 'H'
        season = 24

    fcst = StatsForecast(
        df=Y_train_df,
        models=[AutoETS(season_length=season), Naive()],
        freq=data_freq,
        n_jobs=-1
    )
    Y_hat_df = fcst.forecast(h=h, fitted=True) #forecast after tmax of training set
    Y_fitted_df = fcst.forecast_fitted_values() #fitted data from tmin - tmax
    return Y_hat_df, Y_fitted_df

def reconcile_forecasts(Y_hat_df, Y_fitted_df, S_df, tags):
    reconcilers = [
        BottomUp(),
        #TopDown(method='average_proportions'),  #options forecast_proportions, average_proportions, proportion_averages
        MinTrace(method='mint_shrink'),
        MinTrace(method='ols')
    ]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df,
                              Y_df=Y_fitted_df,
                              S=S_df,
                              tags=tags)
    return Y_rec_df

def evaluate_forecasts(Y_rec_df, Y_test_df, Y_train_df, tags):
    eval_tags = {}
    eval_tags['org'] = tags['org']
    eval_tags['account'] = tags['org/account']
    #eval_tags['meter'] = tags['org/meter']
    #eval_tags['measure'] = tags['org/meter/measure']
    #eval_tags['Bottom'] = tags['Country/State/Region/Purpose']
    eval_tags['All'] = np.concatenate(list(tags.values()))

    evaluator = HierarchicalEvaluation(evaluators=[err.mse, err.rmse, err.mase])
    evaluation = evaluator.evaluate(
        Y_hat_df=Y_rec_df,
        Y_test_df=Y_test_df.set_index('unique_id'),
        tags=eval_tags,
        Y_df=Y_train_df.set_index('unique_id')
    )
    #evaluation = evaluation.drop('Overall')
    #evaluation.columns = ['Base', 'BottomUp', 'MinTrace(mint_shrink)', 'MinTrace(ols)']
    #evaluation.columns = ['Base', 'BottomUp']

    evaluation = evaluation.applymap('{:.2f}'.format)
    #€print(evaluation)
    print(evaluation.query('metric == "mase"'))
    print(evaluation.query('metric == "rmse"'))
    return evaluation

def generate_plots(S_df, tags, Y_df, Y_hat_df):
    #print(Y_df.columns)#Index(['unique_id', 'ds', 'y'], dtype='object')

    time_series = ['m3ter/AssembledHQProd',
                   'm3ter/BurstSMS-LocalTest',
                   'm3ter/BurstSMS-Production',
                   'm3ter/OnfidoDev',
                   'm3ter/OnfidoProd',
                   'm3ter/Patagona-Production',
                   'm3ter/Patagona-Sandbox',
                   'm3ter/Regal.ioProd',
                   'm3ter/SiftForecasting',
                   'm3ter/TricentisProd']

    hplot = HierarchicalPlot(S=S_df, tags=tags) # plotting class containing plotting methods

    #hplot.plot_summing_matrix() # plot hierarchical aggregation contraints matrix, S

    for i in time_series:
        # plot single series with filtered models and prediction interval
        hplot.plot_series(
            series='m3ter/BurstSMS-Production',
            Y_df=Y_hat_df
            #models=AutoETS
            #level=95
        )
        plt.show()

        hplot.plot_hierarchically_linked_series( # plot collection of hierarchically linked series plots associated with the bottom_series and filetered modls and prediction interval level
            bottom_series=i,
            Y_df=Y_df.set_index('unique_id')
            )
        plt.show()

    hplot.plot_hierarchical_predictions_gap( # plots of aggregated predictions at different levels of the hierarchical structure. aggregation() method used to aggregate
        Y_df=Y_hat_df,
        models='AutoETS',
        xlabel='Day',
        ylabel='Predictions',
    )
    plt.show()

    return

def main(data, freq, metadata_str, account):
    # Clean and Prepare Data
    startdate, enddate = pp.select_date_range(freq)
    df = clean_data(data, startdate, enddate)
    df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/df.csv')
    Spc = get_spec()
    Y_df, S_df, tags = get_aggregates(df, Spc)
    Y_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_df.csv')
    print(tags)
    S_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/S_df.csv')
    Y_train_df, Y_test_df, h = split_data(Y_df)
    Y_test_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_test.csv')
    Y_train_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_train.csv')

    # Fit Base Forecasts
    init_fit = time()
    Y_hat_df, Y_fitted_df= base_forecasts(Y_train_df, freq, h)
    end_fit=time()
    print(f'Forecast Minutes: {(end_fit - init_fit) / 60}')

    Y_hat_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_hat_df.csv')
    Y_fitted_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_fitted_df.csv')

    # Reconcile
    Y_rec_df = reconcile_forecasts(Y_hat_df, Y_fitted_df, S_df, tags)
    Y_rec_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_rec_df.csv')

    # Evaluate
    evaluation = evaluate_forecasts(Y_rec_df, Y_test_df, Y_train_df, tags)
    evaluation.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/evaluation.csv')


    # Visualise data
    generate_plots(S_df, tags, Y_df, Y_hat_df)

if __name__ == "__main__":
    data_loc = input("Data location (local or s3)? ")
    #savetos3 = input("Save to s3? ")
    freq = input("Hourly (1h) or Daily (1D) frequency: ")
    dataloadcache= pd.DataFrame()

    key, metadatakey = get_keys()
    dataloadcache, metadata_str = rs3.get_data('2_tidy/' + freq + '/', key, metadatakey)
    data, account = rs3.select_ts(dataloadcache)
    main(data, freq, metadata_str, account)

    # while True:
    #     if dataloadcache.empty:
    #         if data_loc == 's3':
    #             key, metadatakey = get_keys()
    #             dataloadcache, metadata_str = rs3.get_data('2_tidy/'+freq+'/', key, metadatakey)
    #         elif data_loc == 'local':
    #             dataloadcache = rs3.get_data_local()
    #     data, account = rs3.select_ts(dataloadcache)
    #     main(data, freq, metadata_str, account)
    #     print("Press enter to re-run the script, CTRL-C to exit")
    #     sys.stdin.readline()
    #     importlib.reload(rs3)