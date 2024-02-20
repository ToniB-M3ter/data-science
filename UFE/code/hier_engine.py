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
import os
import json
from time import time
from datetime import datetime as dt
import matplotlib.pyplot as plt
from tabulate import tabulate
from datetime import datetime as dt

import readWriteS3 as rs3
import post_process as postproc
import pre_process as preproc

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
from hierarchicalforecast.utils import aggregate, HierarchicalPlot, is_strictly_hierarchical
from hierarchicalforecast.evaluation import scaled_crps, msse, energy_score


# this makes it so that the outputs of the predict methods have the id as a column
# instead of as the index
os.environ['NIXTLA_ID_AS_COL'] = '1'
os.environ['ORG'] = 'onfido'
USER=os.getenv('USER')
ORG=os.getenv('ORG')

tidy_folder = '2_tidy/'
fit_folder = '3_fit/'
forecast_folder = '4_forecast/'

# base forecast file paths
yhat_file = f'NO/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/Y_hat_df.csv'
yfitted_file = f'NO//Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/Y_fitted_df.csv'

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

def get_spec():
    Spc = [
        ['org'],
        ['org', 'account_cd'],
        ['org','meter'],
        ['org','account_cd','meter']
    ]
    return Spc

def get_aggregates(Y_df, Spc):
    Y_df, S_df, tags = aggregate(Y_df, Spc)
    Y_df = Y_df.reset_index()
    return Y_df, S_df, tags

def save_tags(tags):
    dict_with_lists = {k:v.tolist() for k,v in tags.items()}
    with open(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/tags.json', 'w') as fp:
        json.dump(dict_with_lists, fp, indent=4)
    return

def hier_prep(df: pd.DataFrame, dimkeys, startdate, enddate) -> pd.DataFrame:
    # filter dates
    datetime_mask = (df['tm'] > startdate) & (df['tm'] <= enddate)
    df = df.loc[datetime_mask]
    print('Fit from '  + str(startdate) +' to '+ str(enddate))

    # dimkeys ['meter', 'measure', 'account_cd', 'account_nm']
    ts_id = ['ts_id']
    cols = dimkeys+ts_id
    df_ids = df[cols].drop_duplicates()

    # remove account_nm as it is redundant with account_cd
    cols.remove('account_nm') #['meter', 'measure', 'account_cd', 'ts_id']
    measurement_cols = ['tm','y']
    cols = cols+measurement_cols #['meter', 'measure', 'account_cd', 'ts_id', 'tm', 'y']

    # drop unneeded columns & add column for org and rename tm --> ds for forecasting
    df = df[cols]
    # if <aggregated> accounts, meter or measurements exist remove them as these aggregations will be created with hier aggregations
    df = df[(df.account_cd != '<aggregated>') & (df.meter != '<aggregated>') & (df.measure != '<aggregated>')]
    df.insert(0,'org',ORG)
    cols.append('org') #['org','meter', 'measure', 'account_cd', 'ts_id', 'tm', 'y']
    df=df[cols]
    df = df.rename(columns={'tm': 'ds'})
    print(df.tail(10))
    return df, df_ids

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
    Y_hat_df.set_index('unique_id', inplace=True)
    Y_fitted_df.set_index('unique_id', inplace=True)

    if is_strictly_hierarchical(S=S_df.values.astype(np.float32),
                                tags={key: S_df.index.get_indexer(val) for key, val in tags.items()}):
        reconcilers = [
            BottomUp(),
            TopDown(method='average_proportions'),
            TopDown(method='proportion_averages'),
            MinTrace(method='ols', nonnegative=True),
            MinTrace(method='wls_var', nonnegative=True),
            MinTrace(method='mint_shrink', nonnegative=True)
        ]
    else:
        reconcilers = [
            BottomUp(),
            MinTrace(method='ols', nonnegative=True),
            MinTrace(method='wls_var', nonnegative=True),
            MinTrace(method='mint_shrink', nonnegative=True)
        ]

    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    try: Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df,
                                   Y_df=Y_fitted_df, #Y_fitted_df Y_train_df
                                   S=S_df,
                                   tags=tags,
                                   level=[95],
                                   intervals_method = 'normality')
                                   #intervals_method='permbu') # use bootstrap method if levels are not strictly hierarchical
    except Exception as error:
        print("An exception has occured when trying to reconcile the hierarchy "
              "reconciliation will occur by Bottoms up, MinTrace(wls_var) and MinTrace(OLS) methods", error)
        reconcilers = [
            BottomUp(),
            MinTrace(method='ols', nonnegative=True),
            MinTrace(method='wls_var', nonnegative=True)
        ]
        hrec = HierarchicalReconciliation(reconcilers=reconcilers)

        Y_rec_df = hrec.bootstrap_reconcile(Y_hat_df=Y_hat_df,
                                            Y_df=Y_fitted_df,
                                            S_df=S_df, tags=tags,
                                            level=[95],
                                            intervals_method='normality',
                                            num_samples=10, num_seeds=10)

        #Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df,
    #                         Y_df=Y_fitted_df, #Y_fitted_df Y_train_df
    #                        S=S_df,
    #                       tags=tags)
                                  #level=[95])
                                  #intervals_method='permbu') # use bootstrap method if levels are not strictly hierarchical
    return Y_rec_df

def evaluate_forecasts(Y_df, Y_rec_df, Y_test_df, Y_train_df, tags):
    eval_tags = {}
    eval_tags['org'] = tags['org']
    eval_tags['account'] = tags['org/account_cd']
    eval_tags['meter'] = tags['org/meter']
    eval_tags['account_meter'] = tags['org/account_cd/meter']
    eval_tags['All'] = np.concatenate(list(tags.values()))

    evaluator = HierarchicalEvaluation(evaluators=[postproc.rmse_calc])
    evaluation = evaluator.evaluate(
        Y_hat_df=Y_rec_df,  # feeding in reconciled values, e.g. Y_rec_df, not just forecast values
        Y_test_df=Y_test_df.set_index('unique_id'),
        tags=tags,
        Y_df= Y_df #.set_index('uniques_id')    #Y_train_df.set_index('unique_id'),
    )

    evaluation = evaluation.map('{:.2f}'.format)
    print(evaluation.query('metric == "rmse_calc"'))
    return evaluation

def select_rec_method(Y_rec_df, evaluation):
    # Select best model based on evaluation matrix
    # Drop metric; hi / lo intervals columns
    #evaluation.drop(list(evaluation.filter(regex='metric')), axis=1, inplace=True)
    evaluation.reset_index(level=1, drop=True, inplace=True)
    evaluation.drop(list(evaluation.filter(regex='-lo')), axis=1, inplace=True)
    evaluation.drop(list(evaluation.filter(regex='-hi')), axis=1, inplace=True)
    evaluation.loc['mean']=evaluation.mean(numeric_only=True)
    eval_trans = evaluation.T
    print(tabulate(eval_trans, headers='keys', tablefmt='psql'))
    rec_meth = eval_trans.idxmin()['Overall']
    # Now filter values from the reconciled forecasts
    Y_rec_df.reset_index(inplace=True)
    Y_best_rec_df = Y_rec_df[['unique_id', 'ds',
                           rec_meth,
                           rec_meth+'-lo-95',
                           rec_meth+'-hi-95']]
    return rec_meth, Y_best_rec_df

def prep_forecast_for_s3(df_best_rec: pd.DataFrame, tags, df_ids, model_name):
    """"df_best_model should contain forecast values only for model determined to be best fit
    along with the prediction intervals
    e.g. df_best_rec = Y_rec_df[['unique_id','ds','AutoETS/MinTrace_method-ols', 'AutoETS/MinTrace_method-ols-lo-95','AutoETS/MinTrace_method-ols-hi-95']]"""

    splt_tag = []
    for tag in tags:
        splt_tag.append(tag.split("/"))

    orgs=[]
    accounts=[]
    meters=[]
    measures=[]

    # construct lists of hierarchical levels for final file # TODO fix for any amount of dimensions
    for tag in splt_tag:
        if len(tag) == 4:
            measures.append('check_count') # onfido only have one measure so hard coding this
            meters.append(tag[2])
            accounts.append(tag[1])
            orgs.append(tag[0])
        elif len(tag) == 3:
            measures.append('check_count')
            meters.append(tag[2])
            accounts.append(tag[1])
            orgs.append(tag[0])
        elif len(tag) == 2:
            measures.append('check_count')
            meters.append('<aggregated>')
            accounts.append(tag[1])
            orgs.append(tag[0])
        elif len(tag) == 1:
            measures.append('check_count')
            meters.append('<aggregated>')
            accounts.append('<aggregated>')
            orgs.append(tag[0])
        else:
            print(len(tag))
            print(tag)
            print('unknown tag')

    hier_forecast = pd.DataFrame({"unique_id": tags, "org": orgs, "account_cd": accounts, "meter": meters, 'measure': measures}) #TODO add ts_id and join on it
    hier_forecast = pd.merge(hier_forecast, df_ids, on=['account_cd', 'meter', 'measure'])

    df_best_rec.columns = ['unique_id', 'ds', 'z','z0', 'z1']
    df_best_rec['tm'] = df_best_rec["ds"].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    df_best_rec['.model'] = model_name

    hier_frcst = pd.merge(hier_forecast[['unique_id','account_cd', 'account_nm', 'org','meter','measure']],
                          df_best_rec[['unique_id', 'tm', 'z0', 'z1', 'z', '.model']], on='unique_id')
    #hier_frcst = pd.merge(df_best_rec[['unique_id', 'tm', 'z', 'z0', 'z1', '.model']], df_ids, left_on='unique_id', right_on='ts_id')
    hier_frcst.drop("unique_id", axis=1, inplace=True)
    fin_hier_forcast = hier_frcst[['ts_id', 'account_cd', 'account_nm', 'meter', 'measure', 'tm', 'z', 'z0', 'z1', '.model']]
    print(tabulate(fin_hier_forcast.tail(), headers='keys', tablefmt='psql'))

    return fin_hier_forcast

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
        bottom_series='onfido/001240000097LPXAA2/document_report_standard_hybrid',
        Y_df=Y_df.set_index('unique_id')
        )
    plt.show()

    return

def forecast_plots(S_df, tags, Y_df, Y_hat_df, Y_rec_df, ser):
    hplot = HierarchicalPlot(S=S_df, tags=tags) # plotting class containing plotting methods

    cols = Y_rec_df.columns
    recs = [item for item in cols if '-lo' not in item]
    recs = [item for item in cols if '-hi' not in item]
    recs = [item for item in cols if '-index' not in item]
    recs = [item for item in cols if '-sample' not in item]

    plot_df = pd.concat([Y_df.set_index(['unique_id', 'ds']),
                         Y_rec_df.set_index('ds', append=True)], axis=1)
    plot_df = plot_df.reset_index('ds')

    print(plot_df.tail())

    hplot.plot_series(
        series=ser,
        Y_df=plot_df,
        models=recs.append('y'),
        level=[95]
    )
    plt.xticks(rotation=90)
    plt.show()
    return

def main(data, freq, dimkeys_list, account):
    # Clean and Prepare Data
    startdate, enddate = preproc.select_date_range(freq)
    df, df_ids = hier_prep(data, dimkeys_list, startdate, enddate)
    df_ids.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/df_ids.csv')
    df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/df.csv')

    # Filter time series with > 90% zeros out of analysis
    df_to_forecast, df_naive = preproc.filter_data(df, 0.90, ['ts_id']) #TODO parameterise to all dimensions
    df_to_forecast.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/df_to_forecast.csv')

    Spc = get_spec()
    Y_df, S_df, tags = get_aggregates(df_to_forecast, Spc) # unique_id column created ; if filtering use df_to_forecast
    Y_df = preproc.add_noise(Y_df)
    save_tags(tags)
    Y_df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/Y_df.csv')
    S_df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/S_df.csv')
    #tags_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in tags.items()]))


    #plot_correlations(Y_df) # Partial and Auto Correlation plots
    Y_train_df, Y_test_df, h = preproc.split_data(Y_df, 0.18)
    Y_train_df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/Y_train_df.csv')
    Y_test_df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/Y_test_df.csv')

    if os.path.exists(yhat_file): # if we have a base forecasts use them to save time
        Y_hat_df = pd.read_csv(yhat_file, index_col=0)
        Y_fitted_df = pd.read_csv(yfitted_file, index_col=0)
    else:
        print('Fitting base models')
        # Fit Base Forecasts
        init_fit = time()
        Y_hat_df, Y_fitted_df= base_forecasts(Y_train_df, freq, h) # change to Y_df for prod
        end_fit=time()
        print(f'Forecast Minutes: {(end_fit - init_fit) / 60}')

        # Save
        Y_hat_df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/Y_hat_df.csv')
        Y_fitted_df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/Y_fitted_df.csv')

    # Reconcile
    Y_rec_df = reconcile_forecasts(Y_hat_df, Y_fitted_df, Y_train_df, S_df, tags)
    Y_rec_df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/Y_rec_df.csv')
    rs3.write_csv_log_to_S3(Y_rec_df, 'reconciled_forecasts')

    # Evaluate
    evaluation = evaluate_forecasts(Y_df, Y_rec_df, Y_test_df, Y_train_df, tags)
    evaluation.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/evaluation.csv')
    # And select best rconciliation method
    best_rec_meth, Y_best_rec_df=select_rec_method(Y_rec_df, evaluation)
    Y_best_rec_df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/Y_best_rec_df.csv')

    # Prep for save
    hier_forecast=prep_forecast_for_s3(Y_best_rec_df, S_df.index, df_ids, best_rec_meth)
    if USER is None:
        rs3.write_gz_csv_to_s3(hier_forecast, forecast_folder + freq + '/',
                               'hier' + '_' + dt.today().strftime("%Y_%d_%m") + '_' + 'usage.gz')
        rs3.write_meta_to_s3(metadata_str, freq, forecast_folder + freq + '/',
                             'hier' + '_' + dt.today().strftime("%Y_%d_%m") + '_' + 'usage_meta.gz')
    else:
        hier_forecast.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/hier_forecast.csv')

    # Visualise data
    #other_plots(S_df, tags, Y_df, Y_hat_df)
    cnt = 0
    while cnt < 20:
        ser = input("which series? ")
        if not ser:
            pass
        else:
            forecast_plots(S_df, tags, Y_df, Y_hat_df, Y_rec_df, ser)
            pass
        cnt = cnt+1

if __name__ == "__main__":
    #freq = input("Hourly (1h) or Daily (1D) frequency: ")
    freq = '1D'
    dataloadcache= pd.DataFrame()

    key, metadatakey = get_keys()
    dataloadcache, metadata_str = rs3.get_data('2_tidy/' + freq + '/', key, metadatakey)
    dimkey_list = preproc.meta_str_to_dict(metadata_str)
    data, account = preproc.select_ts(dataloadcache)
    main(data, freq, dimkey_list, account)
