# running code from CL
# poetry run python UFE/code/tmbtesting.py


# plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import sklearn
import sys

# data wrangling
import numpy as np
import pandas as pd
import pandas_datareader as pdr

# predicting
from statsmodels.tsa.api import acf, graphics, pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL


plt.style.use('Solarize_Light2')
# set up for matplotlib/seaborn plotting
sns.set_style("darkgrid")
pd.plotting.register_matplotlib_converters()
# Default figure size
sns.mpl.rc("figure", figsize=(16, 6))
sns.mpl.rc("font", size=14)

def plot_raw(data: pd.DataFrame, title: str):
    fig, ax = plt.subplots()
    ax = data.plot(ax=ax).set(title=title)
    plt.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output/{}.jpg'.format(title[0:7]))
    plt.show()
    #fig = px.line(data, x=data.index, y=data['Volume'], title='Yahoo Stock')
    #fig.show()
    return

def plot_forecast(res, title):
    fig, ax = plt.subplots()
    fig = res.plot_predict(720,840)
    fig.suptitle(title)
    plt.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output/{}.png'.format(title[0:7]))
    #res.predict().plot(ax=ax)
    plt.show()
    return

def heatmap():
    pass

def get_housing_data():
    data = pdr.get_data_fred("HOUSTNSA", "1959-01-01", "2019-06-01")
    title= 'Percent Change in Housing Prices\nFrom 1959-01-01 to 2019-06-01'
    housing = data.HOUSTNSA.pct_change().dropna()
    # Scale by 100 to get percentages
    housing = 100 * housing.asfreq("MS")
    plot_raw(housing, title)
    return data, title

def get_nixtla_data():
    Y_df = pd.read_parquet('https://datasets-nixtla.s3.amazonaws.com/m4-hourly.parquet')
    print(Y_df.info())
    Y_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/data/nixtla_sample_data.csv')
    return

def get_engine_data():
    df = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/engine_p.csv')
    df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d %H:%M:%S')
    return df

def convert_to_dashboard_format(df: pd.DataFrame, model_aliases):
    dashboard_cols = [
        'tm'  # timestamp
        #'meter'
        #'measurement'
        #'account'
        #'account_id'  # account m3ter uid
        ,'ts_id'  # ts unique id
        ,'z'  # prediction
        ,'z0'  # lower bound of 95% confidence interval
        ,'z1'  # lower bound of 95% confidence interval
        , '.model'  # model (e.g. model_)
    ]

    dfs = []

    for alias in model_aliases:
        iterator_list = ['df'+alias, alias, alias+'-lo-95', alias+'-hi-95']
        iterator_list[0] = df[['ds','unique_id', iterator_list[1], iterator_list[2], iterator_list[3] ]]
        iterator_list[0]['.model'] = alias
        iterator_list[0].columns = dashboard_cols
        dfs.append(iterator_list[0])

    dfAll = pd.concat(dfs, ignore_index=True)

    dfAll.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/forecasts.csv')
    return dfAll

def smoothing(data):
    fit1 = SimpleExpSmoothing(data[0:200]).fit(smoothing_level=0.2, optimized=False)
    fit2 = SimpleExpSmoothing(data[0:200]).fit(smoothing_level=0.8, optimized=False)
    plt.figure(figsize=(18, 8))
    plt.title('Smoothing Levels 0.2 vs 0.8')
    plt.plot(data[0:200], marker='o', color="black", label="original")
    plt.plot(fit1.fittedvalues, marker="o", color="b", label="0.2")
    plt.plot(fit2.fittedvalues, marker="o", color="r", label="0.8")
    plt.legend()
    plt.xticks(rotation="vertical")
    plt.show()
    return

def decompose(data):
    print(data.describe())
    print(data.head())
    stl = STL(data, period='M', seasonal=7)
    res = stl.fit()
    fig = res.plot()
    return

def fit_data(data):
    # select model
    mod = AutoReg(data, 3, old_names=False)
    # fit model
    res = mod.fit()
    print(res.summary())


    # with covariance estimators as OLS
    #res2 = mod.fit(cov_type='HC0')
    #print(res2.summary())
    # select model order
    sel = ar_select_order(data, 13, old_names=False)
    sel.ar_lags
    res = sel.model.fit()
    print(res.summary())
    return res

def main():
    #get_nixtla_data()
    df = get_engine_data()
    model_aliases=['AE', 'SN']
    convert_to_dashboard_format(df, model_aliases)
    #data, title = get_housing_data()
    #heatmap()
    #smoothing(data)
    #decompose(data)
    #res = fit_data(data)
    #plot_forecast(res, title)
    return


if __name__ == "__main__":
    main()


######################################################################################################################
# Deprecated functions
#
######################################################################################################################

# def make_forecasts(df: DataFrame, horizon, season, window):
#     print('horizon ' + str(horizon) + '  season ' + str(season) + ' window ' + str(window))
#     # prep data / choose train
#     train = df.iloc[:horizon]
#     valid = df.iloc[horizon + 1:]
#     h = round((valid['ds'].nunique()) * 0.5)  # prediction 50% of the valid set's time interval into the future
#
#     # plot the training data
#     plot_HW_train(train, df['unique_id'].iloc[0], 'train')
#
#     # Set up models to fit: e.g. additive (HW_A) & multiplicative (HW_M)
#     model_aliases = [
#         'SN',
#         'N',
#         # 'HA',
#         'AA',
#         'AE',
#         'HW_M',
#         'HW_A'
#     ]
#
#     ts_models = [
#         SeasonalNaive(season_length=season, alias='SN'),
#         Naive(alias='N'),
#         # HistoricAverage(alias='HA'),
#         AutoARIMA(season_length=season, alias='AA'),
#         AutoETS(model=['Z', 'Z', 'Z'], season_length=season, alias='AE'),
#         HoltWinters(season_length=season, error_type='M', alias='HW_M'),
#         HoltWinters(season_length=season, error_type='A', alias='HW_A')
#     ]
#
#     model = StatsForecast(models=ts_models,
#                           freq='H',
#                           n_jobs=-1)
#
#     # fit model and record time
#     init = time()
#     model.fit(train)
#     end = time()
#     print(f'Forecast Minutes: {(end - init) / 60}')
#
#     # predict future h periods, with 95% confidence level
#     p = model.predict(h=h, level=[95])
#     p = p.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')
#
#     # plot predictions
#     plot_HW_forecast_vs_actuals(p, model_aliases)
#     return



# def decompose(dfUsage: DataFrame)-> Series:
#     result = seasonal_decompose(dfUsage['y'],model='additive')
#     result.plot()
#     plt.show()
#     return