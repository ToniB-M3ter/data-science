
import sys, importlib
from time import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

import readWriteS3 as rs3
import error_analysis as err

from statsmodels.tsa.seasonal import seasonal_decompose
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

def select_date_range(data_freq):
    startdate_input = input('Start date (YYYY-mm-dd HH:MM:SS format)? ')
    if startdate_input == '':
        # Select date range depending on frequency of data
        if 'D' in data_freq:
            startdate = datetime.today() - relativedelta(months=6)
        elif 'h' in data_freq:
            startdate = datetime.today() - relativedelta(months=1)
    else:
        startdate = startdate_input

    enddate_input = input('End date (YYYY-mm-dd HH:MM:SS format)? ')
    if enddate_input == '':
        enddate = datetime.today() - relativedelta(days=1)
    else:
        enddate = enddate_input

    return startdate, enddate

def clean_data(df: pd.DataFrame, datetime_col: str, y: str, startdate, enddate) -> pd.DataFrame:
    # filter dates
    datetime_mask = (df['tm'] > startdate) & (df['tm'] <= enddate)
    df = df.loc[datetime_mask]

    print('Fit from '  + str(startdate) +' to '+ str(enddate))

    # Remove whitespace from account name
    tmp_df = df['account'].copy()
    #tmp_df.replace(' ', '', regex=True, inplace=True)
    df['account'] = tmp_df

    # save unique combinations of account_id, meter, and measurement for formatting forecast file before saving
    df_ids = df[['account', 'account_id', 'meter', 'measurement']].drop_duplicates()
    df_ids.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/df_ids.csv')

    # format for fitting and forecasting - select subset of columns and add unique_id column
    df['unique_id'] = df.apply(
        lambda row: row.account + '_' + row.meter + '_' + row.measurement.split(' ')[0], axis=1)

    df = df[[datetime_col, y, 'unique_id']]
    df.columns = ['ds', 'y', 'unique_id']

    print(df.groupby(['unique_id']).size().reset_index(name='counts'))

    # Tell user number of time series
    print(str(df['unique_id'].nunique()) + ' Unique Time Series')

    # plot for inspection
    x = StatsForecast.plot(df)
    #x.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output_figs/{}'.format('ts_eng_input_data'))

    return df, df_ids

def split_data(df):
    #train = df.iloc[:int(0.75 * df['ds'].nunique())]  # TODO change 0.5 to horizon
    #valid = df.iloc[int(0.25 * df['ds'].nunique()) + 1:]  # TODO change 0.5*len(df) to horizon
    train = 1
    valid = 2
   # horizon (h)  = time periods into future for which a forecast will be made
    h = round((len(df) * 0.10))
    return train, valid, h

def only_forecast(df: pd.DataFrame, h, data_freq):
    if 'D' in data_freq:
        data_freq = 'D'
        season = 7
    elif 'h' in data_freq:
        data_freq = 'H'
        season = 24

    model_aliases=['AE'] #TODO this list needs to be created dynamically

    ts_models = [
        AutoETS(model=['Z','Z','Z'], season_length=season, alias='AE'),
        #AutoARIMA(season_length=season, alias='AA'),
        SeasonalNaive(season_length=season, alias='SN'),
        Naive(alias='N')
    ]

    # create the model object, for each model and let user know time required for each fit
    model = StatsForecast(
                        df = df,
                        models=ts_models,
                          freq =data_freq,
                          n_jobs=-1,
                          fallback_model=SeasonalNaive(season_length=24))
                          #fallback_model = SeasonalNaive(season_length=24))

    forecast = model.forecast(df=df, h=h, level=[95])

    # Plot specific unique_ids and models
    forecast_plot_small = model.plot(
        df,
        forecast,
        models=["AutoETS", "SeasonalNaive"], unique_ids=["BurstSMS - Local Test_billing_bill", "Patagona - Production_billing_bill", "Sift Forecasting_billing_bill"],
        level=[95],
        engine='plotly')
    forecast_plot_small.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output_figs/{}'.format('forecast_plot_small'))
    forecast_plot = StatsForecast.plot(df, forecast, engine='matplotlib')
    forecast_plot.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output_figs/{}'.format('forecast_plot'))

    ts = df['unique_id'].values[0]
    res_rmse = err.cross_validate(df, model, h, ts)

    #plot_HW_forecast_vs_actuals(forecast, model_aliases)
    return forecast, model_aliases

def fit(df: pd.DataFrame, h, data_freq):
    if 'D' in data_freq:
        data_freq = 'D'
        season = 7
    elif 'h' in data_freq:
        data_freq='H'
        season = 24

    model_aliases=['AE'] #TODO this list needs to be created dynamically

    ts_models = [
        AutoETS(model=['Z','Z','Z'], season_length=season, alias='AE')
        #AutoARIMA(season_length=season, alias='AA'),
        #SeasonalNaive(season_length=season, alias='SN')
    ]

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

def plot_HW_forecast_vs_actuals(forecast, models: list):
    #print(forecast.head())
    print(str(forecast.isnull().any(axis=1).count()) + ' nulls in forecast' )
    forecast['y'] = forecast['y'].replace(0, 1)
    forecast.dropna(inplace=True)
    forecast.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/engine_p.csv')

    # set number of subplots = number of timeseries
    min_subplots = 2
    numb_ts = min_subplots if min_subplots > forecast['unique_id'].nunique() else forecast['unique_id'].nunique() # ensure the number of subplots is > 1

    # Plot model by model
    for model_ in models:
        mape_ = mean_absolute_percentage_error(forecast['y'].values, forecast[model_].values)
        print(f'{model_} MAPE: {mape_:.2%}')
        fig, ax = plt.subplots(numb_ts, 1, figsize=(1280 / (288/numb_ts), (90*numb_ts) / (288/numb_ts)))
        for ax_, device in enumerate(forecast['unique_id'].unique()):
            forecast.loc[forecast['unique_id'] == device].plot(x='ds', y='y', ax=ax[ax_], label='y', title=device, linewidth=2)
            forecast.loc[forecast['unique_id'] == device].plot(x='ds', y=model_, ax=ax[ax_], label=model_)
            ax[ax_].set_xlabel('Date')
            ax[ax_].set_ylabel('Sessions')
            ax[ax_].fill_between(forecast.loc[forecast['unique_id'] == device, 'ds'].values,
                                 forecast.loc[forecast['unique_id'] == device, f'{model_}-lo-95'],
                                 forecast.loc[forecast['unique_id'] == device, f'{model_}-hi-95'],
                                 alpha=0.2,
                                 color='orange')
            ax[ax_].set_title(f'{device} - Orange-ish band: 95% prediction interval')
            ax[ax_].legend()
        fig.tight_layout()
        plt.show()
        plt.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output_figs/eng_{}.jpg'.format(forecast['unique_id'][1]))
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
    df = df[['ds','meter','measurement', 'account_id', 'account', 'AE', 'AE-lo-95', 'AE-hi-95']]

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
    startdate, enddate = select_date_range(freq)
    dfUsage_clean, df_ids = clean_data(dfUsage, 'tm', 'y', startdate, enddate)
    dfUsage_clean.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/dfUsage.csv')
    # models=['AE'] TODO select models interactively.....?
    h = round(dfUsage_clean['ds'].nunique() * 0.1) # forecast horizon
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
        forecast_to_save = prep_forecast_for_s3(forecast, df_ids, model_aliases)
        rs3.write_csv_to_s3(forecast_to_save, forecast_folder+freq+'/', account+'_'+freq+'_'+'ETS'+'_'+'usage.gz')
        #metadatafile = prep_meta_data_for_s3()
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