from time import time
import os
#os.environ['MPLCONFIGDIR']= tempfile.gettempdir()
import pandas as pd
import numpy as np

from statsforecast import StatsForecast # required to instantiate StastForecast object and use cross-validation method
from datasetsforecast.losses import rmse as df_rmse
from utilsforecast.evaluation import evaluate

import readWriteS3 as rs3

import logging
module_logger = logging.getLogger('ts_engine.error_analysis')
logger = logging.getLogger('ts_engine.error_analysis')

USER = os.getenv('USER')

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

def  cross_validate(Y_df, model, h):
    """
    Once the StatsForecastobject has been instantiated, we can use the cross_validation method, which takes the following arguments:
    df: training data frame with StatsForecast format
    h (int): represents the h steps into the future that will be forecasted
    step_size (int): step size between each window, meaning how often do you want to run the forecasting process.
    n_windows (int): number of windows used for cross-validation, meaning the number of forecasting processes in the past you want to evaluate.


    The crossvaldation_df object is a new data frame that includes the following columns:
    unique_id: index. If you don't like working with index just run crossvalidation_df.resetindex()
    ds: datestamp or temporal index
    cutoff: the last datestamp or temporal index for the n_windows.
    y: true value
    "model": columns with the model’s name and fitted value.


    The function to compute the RMSE takes two arguments:
    The actual values.
    The forecasts, in this case, AutoETS.
    """

    unique_ids = Y_df['unique_id'].unique()
    cvs = []
    cv_scores = []

    for i in unique_ids:
        df = Y_df[Y_df['unique_id'] == i]  # select time series
        # StatsForecast.plot(Y_df, engine='plotly')
        crossvalidation_df = model.cross_validation( # model defines what the fit and fitted values are
            df=df,
            h=h,
            step_size=h,
            n_windows=5 # TODO parameterise
        )

        crossvalidation_df.rename(columns={'y': 'actual'}, inplace=True)  # rename actual values
        cvs.append(crossvalidation_df.copy())

        cutoff = crossvalidation_df['cutoff'].unique()
        for k in range(len(cutoff)):
            cv = crossvalidation_df[crossvalidation_df['cutoff'] == cutoff[k]]
            #x = StatsForecast.plot(df, cv.loc[:, cv.columns != 'cutoff'])
            #x.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output_figs/xval/xval_{}_{}.png'.format(i,str(model)))

        rmse_score = df_rmse(crossvalidation_df['actual'], crossvalidation_df['AutoETS'])
        cv_scores.append(rmse_score)

    # save scores and create df from ids and scores
    cvs_scores_df = pd.DataFrame(
        {'unique_id': unique_ids, 'cvs rmse scores': cv_scores})

    # save all cvs'
    cvs_all = pd.concat(cvs, ignore_index=True)
    if USER is None:
        rs3.write_csv_log_to_S3(cvs_all, 'cvs_all')
    else:
        cvs_all.to_csv(
            '/Users/tmb/PycharmProjects/data-science/UFE/output_files/crossvalidation_{}.csv'.format(id))
    return cvs_scores_df

def evaluate_cross_validation(df, metric):
    models = df.drop(columns=['unique_id', 'ds', 'cutoff', 'y']).columns.tolist()
    evals = []
    # Calculate loss for every unique_id and cutoff.
    for cutoff in df['cutoff'].unique():
        eval_ = evaluate(df[df['cutoff'] == cutoff], metrics=[metric], models=models)
        evals.append(eval_)
    evals = pd.concat(evals)
    evals = evals.groupby('unique_id').mean(numeric_only=True) # Averages the error metrics for all cutoffs for every combination of model and unique_id
    evals['best_model'] = evals.idxmin(axis=1)
    return evals

def get_best_model_forecast(forecasts_df, evaluation_df):
    df = forecasts_df.set_index('ds', append=True).stack().to_frame().reset_index(level=2) # Wide to long
    df.columns = ['model', 'best_model_forecast']
    df = df.join(evaluation_df[['best_model']])
    df = df.query('model.str.replace("-lo-90|-hi-90", "", regex=True) == best_model').copy()
    df.loc[:, 'model'] = [model.replace(bm, 'best_model') for model, bm in zip(df['model'], df['best_model'])]
    df = df.drop(columns='best_model').set_index('model', append=True).unstack()
    df.columns = df.columns.droplevel()
    df = df.reset_index(level=1)
    return df

def main():
    crossvaldation_df = cross_validate()
    evaluation_df = evaluate_cross_validation(crossvaldation_df.reset_index(), 'mse')
    logger.info(evaluation_df.head())
