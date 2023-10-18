
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
import readFromS3 as rs3

# cache for dataload
dataloadcache = None

# df: index = Datetime; col = Hourly_Temp
#df = pd.read_csv("/Users/tmb/PycharmProjects/data-science/UFE/data/MLTempDataset1.csv",
 #                index_col = 'Datetime', usecols=['Datetime', 'Hourly_Temp'])

def get_data():
    data = rs3.main()
    return data

def select_date_range():
    startdate = input('Start date (YYYY-mm-dd HH:MM:SS format)? ')
    enddate = input('Start date (YYYY-mm-dd HH:MM:SS format)? ')
    return startdate, enddate

def prep_for_ts(account, df: DataFrame, datetime_col: str, y: str, startdate, enddate):
    datetime_mask = (df['tm'] > startdate) & (df['tm'] <= enddate)
    df['y'] = df['y'].replace(0,1)
    df = df.loc[datetime_mask]
    df = df[[datetime_col, y]]
    # TODO need to separate prep for HW from exp smoothing: if clause or separate function....
    df['unique_id'] = account[0:5] + '_HW'
    df.columns = ['ds', 'y', 'unique_id']
    #df.set_index(datetime_col, inplace=True) #date time must be passed as a colum for statsforecasting/Nixtla
    return df

def decompose(dfUsage: DataFrame)-> Series:
    result = seasonal_decompose(dfUsage['y'],model='additive')
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

def ETS(df):
    #print(df.columns)
    train_percent = 0.15
    num_pts = len(df)
    train_pts = round(num_pts*train_percent)
    train = df.iloc[:train_pts]
    valid = df.iloc[train_pts+1:]
    h = valid['ds'].nunique()
    # plot the training data
    plot_HW_train(train, df['unique_id'].iloc[0], 'train')

    # Set up models to fit: e.g. additive (HW_A) & multiplicative (HW_M)
    model_aliases = ['HW_A','HW_M']
    model = StatsForecast(models=[HoltWinters(season_length=24, error_type='A', alias='HW_A'),
                              HoltWinters(season_length=24, error_type='M', alias='HW_M')],
                              freq='H', n_jobs=-1)
    # fit model
    model.fit(train)

    # predict future h periods, with 90% confidence level
    p = model.predict(h=h, level=[90])
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
    print(p.head(10))

    # set number of subplots = number of timeseries
    numb_ts = 1

    # Plot model by model
    for model_ in models:
        mape_ = mean_absolute_percentage_error(p['y'].values, p[model_].values)
        print(f'{model_} MAPE: {mape_:.2%}')
        fig, ax = plt.subplots(3, 1, figsize=(1280 / 96, 720 / 96))
        for ax_, device in enumerate(p['unique_id'].unique()):
            p.loc[p['unique_id'] == device].plot(x='ds', y='y', ax=ax[ax_], label='y', title=device, linewidth=2)
            p.loc[p['unique_id'] == device].plot(x='ds', y=model_, ax=ax[ax_], label=model_)
            ax[ax_].set_xlabel('Date')
            ax[ax_].set_ylabel('Sessions')
            ax[ax_].fill_between(p.loc[p['unique_id'] == device, 'ds'].values,
                                 p.loc[p['unique_id'] == device, f'{model_}-lo-90'],
                                 p.loc[p['unique_id'] == device, f'{model_}-hi-90'],
                                 alpha=0.2,
                                 color='orange')
            ax[ax_].set_title(f'{device} - Orange-ish band: 90% prediction interval')
            ax[ax_].legend()
        fig.tight_layout()
        plt.show()
    return

def main(dfUsage):
    #dfUsage = get_data()
    plot(dfUsage, dfUsage['account'].iloc[0], dfUsage['meter'].iloc[0])
    startdate, enddate = select_date_range()
    dfUsage_clean = prep_for_ts(dfUsage['account'].iloc[0], dfUsage, 'tm', 'y', startdate, enddate)
    #decomposition = decompose(dfUsage_clean)
    #expSmoothForecast = expSmooth(dfUsage_clean)
    ETSForecast = ETS(dfUsage_clean)

    #write_to_s3(forecast)


    return

if __name__ == "__main__":

    filepath = '2_tidy/1h/'  # allow selection of data
    metadatakey = 'usage_meta.gz'
    key = 'usage.gz'

    dataloadcache= pd.DataFrame()

    while True:
        if dataloadcache.empty:
            dataloadcache = rs3.get_data(filepath, key, metadatakey)
        data = rs3.select_ts(dataloadcache)
        main(data)
        print("Press enter to re-run the script, CTRL-C to exit")
        sys.stdin.readline()
        importlib.reload(rs3)