
import sys, os
import importlib
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

from statsforecast import StatsForecast
from statsforecast.models import HoltWinters
from sklearn.metrics import mean_absolute_percentage_error

from pandas import DataFrame, Series, to_datetime
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import readWriteS3 as rw2s3

# cache for dataload
dataloadcache = None


def datapath():
    filepath = '2_tidy/1h/'  # allow selection of data
    metakey = 'usage_meta.gz'
    key = 'usage.gz'
    return filepath, metakey, key

def select_date_range():
    startdate = input('Start date (YYYY-mm-dd HH:MM:SS format)? ')
    enddate = input('Start date (YYYY-mm-dd HH:MM:SS format)? ')
    return startdate, enddate

def prep_for_ts(account, df: DataFrame, datetime_col: str, y: str, startdate, enddate):
    datetime_mask = (df['tm'] > startdate) & (df['tm'] <= enddate)
    df['y'] = df['y'].replace(0,1) # replace zeros with ones in order for model to work - maybe should be min value of series?
    df = df.loc[datetime_mask]
    df = df[[datetime_col, y]]
    # TODO need to separate prep for HW from exp smoothing: if clause or separate function....
    df['unique_id'] = account.split(' ')[0] + '_HW'
    df.columns = ['ds', 'y', 'unique_id']
    #df.set_index(datetime_col, inplace=True) #date time must be passed as a colum for statsforecasting/Nixtla
    return df

def decompose(dfUsage: DataFrame)-> Series:
    result = seasonal_decompose(dfUsage['y'],model='additive')
    print(type(result))
    result.plot()
    plt.show()
    return

def expSmooth(dfUsage: DataFrame)-> Series:
    ses = SimpleExpSmoothing(dfUsage)
    alpha = 0.2
    model = ses.fit(smoothing_level=alpha, optimized=False)
    forecastperiods = 3
    forecast = model.forecast(forecastperiods)
    print('Exponential Smoothing')
    print(forecast)
    return forecast

def ETS(df: DataFrame):
    print(df.info())
    train_percent = 0.50
    num_pts = len(df)
    train_pts = round(num_pts*train_percent)
    train = df.iloc[:train_pts]
    valid = df.iloc[train_pts+1:]
    h = round((valid['ds'].nunique()) * 0.5) # prediction 50% of the time interval into the future

    print(len(train))
    print(len(valid))
    print(h)

    # plot the training data
    plot_HW_train(train, df['unique_id'].iloc[0], 'train')

    # Set up models to fit: e.g. additive (HW_A) & multiplicative (HW_M)
    model_aliases = ['HW_A','HW_M']
    model = StatsForecast(models=[HoltWinters(season_length=24, error_type='A', alias='HW_A'),
                              HoltWinters(season_length=24, error_type='M', alias='HW_M')],
                              freq='H', n_jobs=-1)
    # fit model
    model.fit(train)

    # predict future h periods, with 95% confidence level
    p = model.predict(h=h, level=[95])
    p = p.reset_index().merge(valid, on=['ds','unique_id'], how='left')

    # plot predictions
    plot_HW_forecast_vs_actuals(p, model_aliases)
    return

def plot(data: DataFrame, account: str, meter: str):
    fig = px.line(data, x='tm', y='y', title='Account: {} & Meter: {}'.format(account, meter))
    fig.show()
    return

def plot_HW_train(data: DataFrame, TimeSeries: str, plot_type: str):
    fig = px.line(data, x='ds', y='y', title='Time Series: {} {}'.format(TimeSeries, plot_type))
    fig.show()
    return

def plot_HW_forecast_vs_actuals(p, models: list):
    # look for NaNs and remove for plotting
    print(p.isnull().any(axis=1).count())
    p['y'] = p['y'].replace(0, 1)
    print(p[p.isnull().any(axis=1)])
    p.dropna(inplace=True)
    p.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/exp_sm_p.csv')

    # set number of subplots = number of timeseries
    numb_ts = 1

    # Plot model by model
    for model_ in models:
        mape_ = mean_absolute_percentage_error(p['y'].values, p[model_].values)
        print(f'{model_} MAPE: {mape_:.2%}')
        fig, ax = plt.subplots(3, 1, figsize=(1280 / 96, 720 / 96))
        fig, ax = plt.subplots(3, 1, figsize=(1280 / 96, 720 / 96))
        for ax_, device in enumerate(p['unique_id'].unique()):
            p.loc[p['unique_id'] == device].plot(x='ds', y='y', ax=ax[ax_], label='y', title=device, linewidth=2)
            p.loc[p['unique_id'] == device].plot(x='ds', y=model_, ax=ax[ax_], label=model_)
            ax[ax_].set_xlabel('Date')
            ax[ax_].set_ylabel('Sessions')
            ax[ax_].fill_between(p.loc[p['unique_id'] == device, 'ds'].values,
                                 p.loc[p['unique_id'] == device, f'{model_}-lo-95'],
                                 p.loc[p['unique_id'] == device, f'{model_}-hi-95'],
                                 alpha=0.2,
                                 color='orange')
            ax[ax_].set_title(f'{device} - Orange-ish band: 95% prediction interval')
            ax[ax_].legend()
        fig.tight_layout()
        plt.show()
        plt.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output_figs/exp_sm_{}.jpg'.format(p['unique_id'][1]))
    return

def prep_forecast_for_s3(dfForecast):
    # dfForecast columns
    # idx, unique_id, ds, HW_A, HW_A - lo - 95, HW_A - hi - 95, HW_M, HW_M - lo - 95, HW_M - hi - 95, y

    cols = [
        'meter'
        'measurement'
        'account'
        'account_id'  # account m3ter uid
        'ts_id'  # ts unique id
        '.model'  # model (e.g. model_)
        'z'  # prediction
        'tm'  # timestamp
        'z0'  # lower bound of 95% confidence interval
        'z1'  # lower bound of 95% confidence interval
    ]

    # rename columns to match Shiny dashboard
    dfForecast.rename(columns={'ds':'tm', 'HW_A': 'z', '':'', '':''}, inplace=True)



    return dfForecast

def main(dfUsage):
    plot(dfUsage, dfUsage['account'].iloc[0], dfUsage['meter'].iloc[0])
    startdate, enddate = select_date_range()
    dfUsage_clean = prep_for_ts(dfUsage['account'].iloc[0], dfUsage, 'tm', 'y', startdate, enddate)

    decomposition = decompose(dfUsage_clean)
    #expSmoothForecast = expSmooth(dfUsage_clean)
    forecast = ETS(dfUsage_clean)

    #prep_forecast_for_s3(forecast)
    #rw2s3.write_to_s3(forecast)

    return

if __name__ == "__main__":
    data_loc = input("Data location (local or s3)? ")
    dataloadcache= pd.DataFrame()

    while True:
        if dataloadcache.empty:
            if data_loc == 's3':
                filepath, key, metadatakey = datapath()
                dataloadcache = rw2s3.get_data(filepath, key, metadatakey)
            elif data_loc == 'local':
                print('')
                dataloadcache = rw2s3.get_data_local()
        data = rw2s3.select_ts(dataloadcache)
        main(data)
        print("Press enter to re-run the script, CTRL-C to exit")
        sys.stdin.readline()
        importlib.reload(rw2s3)