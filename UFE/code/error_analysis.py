import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsforecast import StatsForecast # required to instantiate StastForecast object and use cross-validation method
from datasetsforecast.losses import rmse as df_rmse
from utilsforecast.evaluation import evaluate


def mse_calc(y, y_hat):
    return np.mean((y-y_hat)**2)

def rmse_calc(y, y_hat):
    return np.mean(np.sqrt(np.mean((y-y_hat)**2, axis=1)))

def mase_calc(y, y_hat, y_insample, seasonality=24):
    errors = np.mean(np.abs(y - y_hat), axis=1)
    scale = np.mean(np.abs(y_insample[:, seasonality:] - y_insample[:, :-seasonality]), axis=1)
    return np.mean(errors / scale)

def r_sqed_calc():
    """
    R-squared represents the proportion of variance in the dependent variable
    explained by the linear regression model

    y_i =  y at position i
    y_hat = prediction for y
    y_bar = mean y
    Sigma = sum from i to N

    R^2 = 1 - (Sigma(y_i - y_hat)^2/ Sigma(y_i - y_bar)^2)
    """
    pass

def r_sqed_adjed_calc():
    """
    R-squared-adjusted is a modified version of R-squared, adjusted for the number
    of independent variables in the model.


    R-squared represents the proportion of variance in the dependent variable
    explained by the linear regression model
    n = number of observations in the data
    k = number of independent variables in data



        R_adj^2 = 1 - ((1-R^2)(n-1)/(n-k-1))
        """
    pass

def cross_validate(Y_df, model, h, ts=None):
    """
    Once the StatsForecastobject has been instantiated, we can use the cross_validation method, which takes the following arguments:
    df: training data frame with StatsForecast format
    h (int): represents the h steps into the future that will be forecasted
    step_size (int): step size between each window, meaning how often do you want to run the forecasting process.
    n_windows (int): number of windows used for cross-validation, meaning the number of forecasting processes in the past you want to evaluate.


    The crossvaldation_df object is a new data frame that includes the following columns:
    unique_id: index. If you dont like working with index just run crossvalidation_df.resetindex()
    ds: datestamp or temporal index
    cutoff: the last datestamp or temporal index for the n_windows.
    y: true value
    "model": columns with the model’s name and fitted value.


    The function to compute the RMSE takes two arguments:
    The actual values.
    The forecasts, in this case, AutoETS.
    """

    if ts:
        print(ts)
        df = Y_df[Y_df['unique_id'] == ts]  # select time series
    else:
        df=Y_df

    StatsForecast.plot(Y_df, engine='plotly')

    crossvalidation_df = model.cross_validation(
        df=df,
        h=h,
        step_size=h,
        n_windows=5
    )

    crossvalidation_df.rename(columns={'y': 'actual'}, inplace=True)  # rename actual values

    cutoff = crossvalidation_df['cutoff'].unique()

    crossvalidation_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/crossvalidation_df.csv')

    for k in range(len(cutoff)):
        cv = crossvalidation_df[crossvalidation_df['cutoff'] == cutoff[k]]
        x = StatsForecast.plot(df, cv.loc[:, cv.columns != 'cutoff'])
        x.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output_figs/{}'.format('xval.png'))

    AE_rmse_res = df_rmse(crossvalidation_df['actual'], crossvalidation_df['AE'])
    SN_rmse_res = df_rmse(crossvalidation_df['actual'], crossvalidation_df['SN'])
    #AA_rmse_res = df_rmse(crossvalidation_df['actual'], crossvalidation_df['AA'])
    N_rmse_res = df_rmse(crossvalidation_df['actual'], crossvalidation_df['N'])
    print("AE, SN, AA, N RMSE using cross-validation: ", AE_rmse_res, SN_rmse_res, N_rmse_res)#, N_rmse_res)