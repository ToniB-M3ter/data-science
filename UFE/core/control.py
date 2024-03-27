
import os, sys, importlib
from time import time
import pandas as pd
from datetime import datetime as dt

import _readWrite as rw
import _pre_process as preproc
from _stats_fit_forecast import FitForecast as ff
import _combine_forecasts as comb
import _evaluation as eval


from statsforecast import StatsForecast
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

from utilsforecast.losses import (
    mse,   # mean square error
    mape,  # mean absolute percentage error
    mae,   # mean absolute error
    mase,  # mean absolute scaled error
    rmse,  # root mean square error
    mqloss, # multi-quantile loss
    scaled_crps, # scaled continues ranked probability score
    )

import logging
from logging.config import fileConfig

logging.captureWarnings(True)
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log.config')
logging.config.fileConfig(log_file_path)
logger = logging.getLogger(__name__)

############################################# Set parameters #############################################
# TODO: Parameters to be set with config files
# comment out os.environ parameter lines when running on aws as env variables will be set in lambda config
os.environ['ORG'] = 'onfido'
os.environ['FIT_FORECAST'] = 'BOTH'

# FIT = fit and save fit to S3
# FORECAST = only forecast
# BOTH = fit and forecast in one step
global FIT_FORECAST
FIT_FORECAST=os.getenv('FIT_FORECAST')
freq=os.getenv('FREQ')
USER=os.getenv('USER')
ORG=os.getenv('ORG')

# folder
tidy_folder = '2_tidy/'
fit_folder = '4_fit/'
forecast_folder = '4_forecast/'

# fit/forecast parms
n_jobs = -1
hpct = 0.15
predict_int = 95
add_noise = 'Y'
models_indices = [0, 2, 3, 4, 5, 6, 7, 8, 9]
# evaluation parms
xval= 'Y'
n_win = 3
#################################################################################################################
def main(freq):
    """
    model indices
    0   Naive
    1   HistoricAverage
    2   SeasonalNaive
    3   MSTL
    4   HoltWinters
    5   HoltWinters
    6   AutoETS
    7   AutoCES
    8   AutoARIMA
    9   AutoTheta
    10  OptimizedTheta
    11  OptimizedTheta
    """

    # Housekeeping Tasks
    key, metadatakey = preproc.get_keys()
    metadata_str, cols = rw.metadata.get_metadata(tidy_folder + freq + '/', metadatakey)
    meta_dict = rw.metadata.meta_str_to_dict(metadata_str)
    dimkey_list = rw.metadata.meta_to_dim_list(meta_dict)

    # Wrangle Data
    tidydata = rw.tsdata.get_data(tidy_folder + freq + '/', key, cols)
    startdate, enddate = preproc.select_date_range(freq) # TODO programtically select best date range
    dfUsage_clean, df_ids = preproc.clean_data(tidydata, 'tm', 'y', startdate, enddate)

    logger.info(str(tidydata['account_cd'].nunique()) + ' Unique accounts')
    logger.info(str(len(tidydata)) + ' total records from ' + str(tidydata['tm'].min()) + ' to ' + str(tidydata['tm'].max()))

    if USER is None:
        rw.write_csv_log_to_S3(dfUsage_clean, 'dfUsage_clean')
        rw.write_csv_log_to_S3(dfUsage_clean, 'df_ids')
    else:
        dfUsage_clean.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFEPOC/output_files/{ORG}/dfUsage.csv', index=False)
        df_ids.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFEPOC/output_files/{ORG}/df_ids.csv', index=False)

    # Get Parms and Models
    season = preproc.get_season(freq)
    ts_models = preproc.select_models(season, models_indices)
    h = round(dfUsage_clean['ds'].nunique() * hpct)  # forecast horizon
    if add_noise == 'Y':
        df_to_forecast = preproc.add_noise(dfUsage_clean)
    else:
        df_to_forecast = dfUsage_clean


    # Fit and Forecast
    if FIT_FORECAST == 'FIT':
        init_fit = time()
        model = ff.fit(dfUsage_clean, h, season, freq, ts_models, n_jobs, predict_int)
        end_fit = time()
        logger.info(f'Fit Minutes: {(end_fit - init_fit) / 60}')
        rw.model.write_model_to_s3(model, fit_folder + freq + '/', ts_models + '.pkl') # Write model from s3
    elif FIT_FORECAST == 'FORECAST':
        model = rw.model.read_model_from_s3(fit_folder + freq + "/", ts_models + '.pkl') # Read model from s3
        init_predict = time()
        forecasts = ff.make_prediction(model, df_to_forecast, h, predict_int)
        end_predict = time()
        logger.info(f'Forecast Minutes: {(end_predict - init_predict) / 60}')
        print(f'Forecast Minutes: {(end_predict - init_predict) / 60}')
    elif FIT_FORECAST == 'BOTH':
        init_foreonly = time()
        forecasts, model = ff.both(df_to_forecast, h, season, freq, ts_models, n_jobs, predict_int)
        end_foreonly = time()
        logger.info(f'Forecast Only Minutes:  + {(end_foreonly - init_foreonly) / 60}')

    # Add UUID to model

    # Combine
    forecasts = comb.avg_models(forecasts)

    # Save forecasts
    if USER is None:
        rw.logs.write_csv_log_to_S3(forecasts) #TODO add S3 path, and pass file name e.g. all forecasts
    else:
        forecasts.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/forecasts.csv')

    # Validate
    if xval=='Y':
        init_xval = time()
        crossvalidation_df = eval.Evaluate.cross_validate_simple(df_to_forecast, model, h, n_win)
        end_xval = time()
        logger.info(f'Cross Validation Minutes:  + {(end_xval - init_xval) / 60}')
    else:
        init_xval = time()
        crossvalidation_df = eval.Evaluate.cross_validate_simple(df_to_forecast, model, h, 1)
        end_xval = time()
        logger.info(f'Cross Validation Minutes:  + {(end_xval - init_xval) / 60}')

    crossvalidation_df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFEPOC/output_files/{ORG}/cv_rmse_df.csv')
    evaluation_df = eval.evaluate_cross_validation(crossvalidation_df, [mse,  # mean square error
                                                                            mape,  # mean absolute percentage error
                                                                            mae,  # mean absolute error
                                                                            mase,  # mean absolute scaled error
                                                                            rmse,  # root mean square error
                                                                            mqloss,  # multi-quantile loss
                                                                            scaled_crps # scaled continues ranked probability score
                                                                            ]
                                                       )
    evaluation_df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFEPOC/output_files/{ORG}/evaluation_df.csv')

    # TODO reformat evaluation files and load to S3

    summary_df = evaluation_df.groupby('best_model').size().sort_values().to_frame()
    summary_df.reset_index().columns = ["Model", "Nr. of unique_ids"]
    summary_df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFEPOC/output_files/{ORG}/summary_df.csv')

    best_forecasts = eval.Evaluate.get_best_model_forecast(forecasts, evaluation_df)
    best_forecasts.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFEPOC/output_files/{ORG}/best_forecasts.csv')


    # Save

if __name__ == "__main__":
    freq = input("Hourly (1h) or Daily (1D) frequency: ")
    main(freq)