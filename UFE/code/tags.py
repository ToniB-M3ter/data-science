import numpy as np
import pandas as pd
import os
import json


import readWriteS3 as rs3
import pre_process as pp

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
fit_folder = '4_fit/'
forecast_folder = '5_forecast/'

# base forecast file paths
yhat_file = f'NO/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/Y_hat_df.csv'
yfitted_file = f'NO//Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/Y_fitted_df.csv'

def get_keys():
    metadatakey = 'usage_meta.gz'
    key = 'usage.gz'
    return key, metadatakey

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
    return Y_df, S_df, tags

def hier_prep(df: pd.DataFrame, startdate, enddate) -> pd.DataFrame:
    # filter dates
    datetime_mask = (df['tm'] > startdate) & (df['tm'] <= enddate)
    df = df.loc[datetime_mask]
    print('Fit from '  + str(startdate) +' to '+ str(enddate))

    df_ids = df[['account_cd', 'account_nm', 'meter', 'measurement']].drop_duplicates() #TODO change to all generic 'dimensions'
    df_ids.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/df_ids.csv')

    # drop unneeded columns & add column for org and rename tm --> ds for forecasting
    df = df[['tm','account_cd','meter','measurement','y']]
    # if <aggregated> accounts, meter or measurements exist remove them as these aggregations will be created with hier aggregations
    df = df[(df.account_cd != '<aggregated>') & (df.meter != '<aggregated>') & (df.measurement != '<aggregated>')]
    df.insert(0,'org','onfido')  # TODO parmeterise org name
    df= df.rename(columns={'tm':'ds', 'account_cd':'account'})
    df=df[['org','account','meter','ds','y']]
    print('length of df: ' + str(len(df)))

    return df, df_ids

def dict_to_json(obj):
    dict_with_lists = {k:v.tolist() for k,v in obj.items()}
    with open('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/tags.json', 'w') as fp:
        json.dump(dict_with_lists, fp, indent=4)

    with open('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/tags.json', 'r') as f:
        tags_dict = json.load(f)
    tags = {k: np.array(v) for k, v in tags_dict.items()}
    return

def main(data, freq, metadata_str, account):
    # Clean and Prepare Data
    startdate, enddate = pp.select_date_range(freq)
    df, df_ids = hier_prep(data, startdate, enddate)
    df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/df.csv')

    # Filter time series with > 90% zeros out of analysis
    df_to_forecast, df_naive = pp.filter_data(df, 0.90, ['org','account','meter'])
    #df_to_forecast.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/df_to_forecast.csv')

    Spc = get_spec()
    Y_df, S_df, tags = get_aggregates(df_to_forecast, Spc) # unique_id column created ; if filtering use df_to_forecast
    #Y_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/Y_df.csv')
    #S_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/S_df.csv')

    dict_to_json(tags)
    tags_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in tags.items()]))
    tags_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/onfido/tags_testing_df.csv')

if __name__ == "__main__":
    # freq = input("Hourly (1h) or Daily (1D) frequency: ")
    freq = '1D'
    dataloadcache = pd.DataFrame()

    key, metadatakey = get_keys()
    dataloadcache, metadata_str = rs3.get_data('2_tidy/' + freq + '/', key, metadatakey)
    data, account = pp.select_ts(dataloadcache)
    main(data, freq, metadata_str, account)