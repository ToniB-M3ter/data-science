
import sys, importlib
from time import time
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

import readWriteS3 as rs3
import error_analysis as err
import pre_process as pp

from statsforecast import StatsForecast
#from sklearn.metrics import mean_absolute_percentage_error

from statsforecast.models import (
    SeasonalNaive, # model using the previous season's data as the forecast
    Naive, # Simple naive model using the last observed value as the forecast
    HistoricAverage, # Average of all historical data
    AutoETS, # Automatically selects best ETS model based on AIC
    AutoARIMA, # ARIMA model that automatically select the parameters for given time series with AIC and cross validation
    HoltWinters #HoltWinters ETS model
    )


# cache for dataload
dataloadcache = None

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

def select_models(data_freq):
    season=get_season(data_freq)

    model_aliases = ['AutoETS']

    ts_models = [
        AutoETS(model=['Z', 'Z', 'Z'], season_length=season, alias='AutoETS')
        # AutoARIMA(season_length=season, alias='AA'),
        # SeasonalNaive(season_length=season, alias='SN'),
        # Naive(alias='N')
    ]
    return ts_models, model_aliases

def only_forecast(df: pd.DataFrame, h, data_freq):
    season = get_season(data_freq)

    ts_models, model_aliases = select_models(data_freq)

    # create the model object, for each model and let user know time required for each fit
    model = StatsForecast(
                        df = df,
                        models=ts_models,
                          freq =data_freq,
                          n_jobs=-1,
                          fallback_model=SeasonalNaive(season_length=season))
                          #fallback_model = SeasonalNaive(season_length=24))

    forecast = model.forecast(df=df, h=h, level=[95])

    # Plot specific unique_ids and models
    forecast_plot_small = model.plot(
        df,
        forecast,
        models=model_aliases,
        #unique_ids=["BurstSMS - Local Test_billing_bill", "Patagona - Production_billing_bill", "Sift Forecasting_billing_bill"],
        level=[95],
        engine='plotly')
    #forecast_plot_small.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output_figs/{}'.format('forecast_plot_small'))
    forecast_plot = StatsForecast.plot(df, forecast, engine='matplotlib')
    forecast_plot.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output_figs/{}'.format('forecast_plot'))

    ts = df['unique_id'].values[0]
    res_rmse = err.cross_validate(df, model, h, ts)

    #plot_HW_forecast_vs_actuals(forecast, model_aliases)
    return forecast, model_aliases

def fit(df: pd.DataFrame, h, data_freq):
    season = get_season(data_freq)

    ts_models, model_aliases = select_models(data_freq)

    # create the model object, for each model and let user know time required for each fit
    model = StatsForecast(df=df,
                        models=ts_models,
                        freq = data_freq,
                        n_jobs=-1,
                        verbose=True)

    model.fit(df)   #(model = #<class 'statsforecast.core.StatsForecast'>
    return model, model_aliases

def predict(model, df, h):
    # predict future h periods, with level of confidence
    prediction = model.predict(h=h, level=[95])

    # prep for plotting with prediction intervals
    #prediction_merge = prediction.reset_index().merge(df, on=['ds','unique_id'], how='left')
    #print(prediction_merge.head())
    return prediction

def plot(data: pd.DataFrame, account: str, meter: str):
    fig = px.line(data, x='tm', y='y', title='Account: {} & Meter: {}'.format(account, meter))
    fig.show()
    return

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
    df = df.merge(df_ids[['meter','measurement', 'account_id', 'account']], on='account')
    for alias in model_aliases:
        model_cols = [ alias, alias + '-lo-95', alias + '-hi-95'] # TODO make prediciton interval bands dynamic
    df_cols = ['ds','meter','measurement', 'account_id', 'account']
    for col in model_cols:
        df_cols.append(col)
    df = df[df.columns.intersection(df_cols)]

    dfs = []

    for alias in model_aliases:
        iterator_list = ['df' + alias, alias, alias + '-lo-95', alias + '-hi-95']
        iterator_list[0] = df[['ds', 'meter', 'measurement', 'account_id', 'account', iterator_list[1], iterator_list[2], iterator_list[3]]]
        iterator_list[0]['.model'] = alias
        iterator_list[0].columns = dashboard_cols
        dfs.append(iterator_list[0])

    dfAll = pd.concat(dfs, ignore_index=True)
    dfAll.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/all_forecasts_test.csv')
    return dfAll

def prep_meta_data_for_s3():
    # TODO change meta as dictionary passed parameter to function
    meta = {'nm': 'typ', 'meter': 'dim', 'measurement': 'dim', 'account': 'dim', 'account_id': 'dim', '.model': 'dim', 'z': 'measure', 'tm': 'time', '_intrvl': '1h', 'z0': 'measure', 'z1': 'measure'}
    meta_list = list(meta.items())
    with open ('/Users/tmb/PycharmProjects/data-science/UFE/output_files/tmbmeta.txt', 'w') as file: # TODO change to local temp folder
        for i in meta_list:
            file.write(','.join(map(str, i))+'\n')  # file type => _io.TextIOWrapper
    return file

def main(dfUsage, freq, metadata_str, account):
    # define folders
    fit_folder = '4_fit/'
    forecast_folder = '5_forecast/'
    ##################################
    # plot time series  ##############
    ##################################
    plot(dfUsage, dfUsage['account'].iloc[0], dfUsage['meter'].iloc[0])
    ##################################
    # Data Wrangle #########
    ##################################
    startdate, enddate = pp.select_date_range(freq)
    dfUsage_clean, df_ids = pp.clean_data(dfUsage, 'tm', 'y', startdate, enddate)
    dfUsage_clean.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/dfUsage.csv')
    # models=['AE'] TODO select models interactively.....?
    h = round(dfUsage_clean['ds'].nunique() * 0.10) # forecast horizon
    ##################################
    # split data #########
    ##################################
    #train,valid, h = split_data(dfUsage_clean)
    #train.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/train.csv')
    ##################################
    # fit, predict, forecast #########
    ##################################
    # fit
    init_fit = time()
    #model, model_aliases = fit(dfUsage_clean, h, freq)
    end_fit=time()
    #print(f'Fit Minutes: {(end_fit - init_fit) / 60}')
    #rs3.write_model_to_s3(model, fit_folder+freq, account+freq+'ETS_model.pkl')
    # forecast
    # Read model from s3 if one exists for faster forecasting
    #model = rs3.read_model_from_s3(fit_folder+freq, 'Prompt_model.pkl')

    init_predict = time()
    #forecast = predict(model, dfUsage_clean, h)
    end_predict=time()
   # print(f'Predict Minutes: {(end_predict - init_predict) / 60}')

    # forcast only
    init_foreonly = time()
    forecast_only, model_aliases = only_forecast(dfUsage_clean, h, freq)
    end_foreonly = time()
    print(f'Forecast Only Minutes: {(end_foreonly - init_foreonly) / 60}')

    #plot_HW_forecast_vs_actuals(forecast, model_aliases)

    ##################################
    # save to s3 if directed #########
    ##################################

    if savetos3 in ['Y','yes','Yes', 'YES', 'y']:
        forecast_to_save = prep_forecast_for_s3(forecast_only, df_ids, model_aliases)
        rs3.write_csv_to_s3(forecast_to_save, forecast_folder+freq+'/', account+'_'+freq+'_'+'ETS'+'_'+'usage.gz')
        #metadatafile = prep_meta_data_for_s3() TODO remove or prep fields for saving
        rs3.write_meta_to_s3(metadata_str, forecast_folder+freq+'/',account+'_'+freq+'_'+'ETS'+'_'+'usage_meta.gz')
    else:
        forecast_only.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/only_forecst.csv')
    return

if __name__ == "__main__":
    data_loc = input("Data location (local or s3)? ")
    savetos3 = input("Save to s3? ")
    freq = input("Hourly (1h) or Daily (1D) frequency: ")
    dataloadcache= pd.DataFrame()

    while True:
        if dataloadcache.empty:
            if data_loc == 's3':
                key, metadatakey = get_keys()
                dataloadcache, metadata_str = rs3.get_data('2_tidy/'+freq+'/', key, metadatakey)
            elif data_loc == 'local':
                dataloadcache = rs3.get_data_local()
        data, account = rs3.select_ts(dataloadcache)
        main(data, freq, metadata_str, account)
        print("Press enter to re-run the script, CTRL-C to exit")
        sys.stdin.readline()
        importlib.reload(rs3)