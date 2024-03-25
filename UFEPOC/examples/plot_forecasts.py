import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA, Naive, AutoETS
from statsforecast.models import (
    SeasonalNaive, # model using the previous season's data as the forecast
    Naive, # Simple naive model using the last observed value as the forecast
    HistoricAverage, # Average of all historical data
    AutoETS, # Automatically selects best ETS model based on AIC
    AutoARIMA, # ARIMA model that automatically select the parameters for given time series with AIC and cross validation
    HoltWinters, #HoltWinters ETS model
    AutoCES, # Auto Complex Exponential Smoothing
    MSTL,
    OptimizedTheta,
    AutoTheta
    )

ORG='onfido'
horizon = 20
data_freq = '1D'

if 'D' in data_freq:
    data_freq = 'D'
    season = 7
elif 'h' in data_freq:
    data_freq = 'H'
    season = 24


# base forecast file paths
dfUsage_file = f'/UFEPOC/output_files/{ORG}/dfUsage.csv'
forecasts_file = f'/UFEPOC/output_files/{ORG}/forecasts.csv'


fileLocs = [ dfUsage_file, forecasts_file]
fileNames = ['dfUsage', 'forecasts']

file_dict = {}

for filename, fileloc in zip(fileNames, fileLocs):
    file_dict[filename] = pd.read_csv(fileloc, index_col=0)
# Get dfs
dfUsage = file_dict['dfUsage']
forecasts = file_dict['forecasts']

dfUsage.reset_index(inplace=True)  # unique_id is index at this point, need to reset to make multi index in next line

# models
ts_models = [
    Naive(alias='Naive'),
    HistoricAverage(),
    SeasonalNaive(season_length=season, alias='SeasonalNaive'),
    MSTL(season_length=[season, season*4]),
    HoltWinters(season_length=season, error_type="A", alias="HWAdd"),
    HoltWinters(season_length=season, error_type="M", alias="HWMult"),
    AutoETS(model=['Z', 'Z', 'Z'], season_length=season, alias='AutoETS'),
    AutoCES(season_length=season, alias='AutoCES'),
    AutoARIMA(season_length=season, alias='AutoARIMA'),
    AutoTheta(season_length=season,
               decomposition_type="additive",
               model="STM"),
    OptimizedTheta(season_length=season,
                   decomposition_type="additive", alias="ThetaAdd"),
    OptimizedTheta(season_length=season,
                   decomposition_type="multiplicative", alias="ThetaMult")
]

def read_model_from_local():
    with open('/UFEPOC/output_files/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def plot_forecasts(model, df, forecasts, model_aliases):
    # Plot specific unique_ids and models
    forecast_plot_small = model.plot(
        df,
        forecasts,
        models=model_aliases,
        level=[95],
        engine='plotly')

    forecast_plot = StatsForecast.plot(df, forecasts, engine='matplotlib')
    plt.xticks(fontsize="8", rotation=45)
    plt.legend(fontsize="8", loc="lower left")
    plt.show()

    #plotname = 'forecast_plot'
    #forecast_plot.savefig(f'/Users/tmb/PycharmProjects/data-science/UFEPOC/output_figs/{ORG}/{plotname}')
    return

model=read_model_from_local()

cnt = 0
while cnt < 3:
    try:
        series = input("which series? ")
        if not series:
            pass
        else:
            # plot_forecasts(dfUsage_clean, forecast_only_naive, naive_model_aliases)
            plot_forecasts(dfUsage, forecasts, ts_models)
            cnt = cnt + 1
    except:
        print('hit an error')


