import os, sys, importlib
from time import time
import pandas as pd
from datetime import datetime as dt

import utils
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
logs_folder = 'logs/'

# fit/forecast parms
n_jobs = -1
hpct = 0.15
predict_int = 95
add_noise = 'Y'
models_indices = [0, 2, 3, 4, 5, 6, 7, 8, 9] #[0,2]

# evaluation parms
xval= 'Y'
n_win = 3
metrics =[mse,  # mean square error
                mape,  # mean absolute percentage error
                #mae,  # mean absolute error
                #mase,  # mean absolute scaled error
                rmse,  # root mean square error
                #mqloss,  # multi-quantile loss
                #scaled_crps # scaled continues ranked probability score
                ]
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

    #if USER is None:
    if 1==1:
        rw.logs.write_csv_log_to_S3(dfUsage_clean, 'dfUsage_clean', logs_folder)
        rw.logs.write_csv_log_to_S3(dfUsage_clean, 'df_ids', logs_folder)
    else:
        dfUsage_clean.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/dfUsage.csv', index=False)
        df_ids.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/df_ids.csv', index=False)

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
    #forecasts.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/base_forecasts.csv')

    # UUID for file
    file_UID = utils.generate_uid()

    # Combine
    if FIT_FORECAST in ['FORECAST', 'BOTH']:  # only proceed if we have generated forecasts
        all_forecasts = comb.avg_models(forecasts)  # Create new forecast that is a combination of other forecasts
        # Save forecasts
        if USER is None:
        #if 1==1:
            rw.logs.write_csv_log_to_S3(all_forecasts, 'forecasts', forecast_folder)
        else:
            all_forecasts.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/all_forecasts-{file_UID}.csv')

        # Validate
        scores_df = eval.Evaluate.evaluate_simple(dfUsage_clean, all_forecasts, metrics)

        # if we are cross validating there are extra steps
        if xval=='Y':
            init_xval = time()
            crossvalidation_df = eval.Evaluate.cross_validate(df_to_forecast, model, h, n_win)
            end_xval = time()
            logger.info(f'Cross Validation Minutes:  + {(end_xval - init_xval) / 60}')
            xval_eval = eval.Evaluate.score_cross_validation(crossvalidation_df, metrics)
            xval_eval.to_csv(
                f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/xval_eval_{file_UID}.csv')
            scores_df = pd.merge(xval_eval, scores_df[['unique_id', 'metric','Combined']], on=['unique_id', 'metric'], how='left')

        scores_df.to_csv(
            f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/scores_{file_UID}.csv')

        # select best model for display in dashboard
        best_forecasts, evaluation_w_best_model, summary_df = eval.Evaluate.best_model_forecast(all_forecasts, scores_df) # evaluation_df best model name
        evaluation_w_best_model.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/eval_w_bm.csv')
        best_forecasts.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/best_forecasts.csv') # to save for dashboard

        # reformat evaluation data and save]
        # Add UUID to model
        UID_list = [utils.generate_uid() for x in ts_models]
        model_codes = pd.DataFrame({'model': ts_models, 'UUID': UID_list})
        model_codes.loc[len(model_codes.index)] = ['Combined', utils.generate_uid()] #add entry for combined
        eval_reformat = eval.Evaluate.reformat(scores_df)

        if USER is None:
        #if 1==1:
        # reformat for saving to s3
            try:
                rw.logs.write_csv_log_to_S3(eval_reformat, 'eval_reformat', forecast_folder)
            except:
                logger.error("Cannot write %s to S3" % (eval_reformat))
        else:
            eval_reformat.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/eval_reformat_{file_UID}.csv')


        # Save
        if len(all_forecasts) > 0:
            forecast_to_save = rw.tsdata.prep_forecast_for_s3(best_forecasts, df_ids, evaluation_w_best_model) # save best forecasts for dashboard
            # TODO Then save forecasts model by model to forecasts folder
            #if USER is None:
            if 1==1:
                rw.tsdata.write_gz_csv_to_s3(forecast_to_save, forecast_folder + freq + '/', 'best_' + dt.today().strftime("%Y_%m_%d") + '_usage.gz')
                storedFileNameBase=rw.metadata.write_dict_to_textfile() # pass metaDict if created, e.g. metadata_str
                rw.metadata.write_meta_tmp(storedFileNameBase)
                rw.metadata.gz_upload(storedFileNameBase, forecast_folder + freq + '/')
                #rs3.write_meta_to_s3(metadata_str, freq, forecast_folder + freq + '/', 'best_' + dt.today().strftime("%Y_%d_%m") + '_' + 'hier_2024_03_04_usage_meta.gz')
            else:
                forecast_to_save.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/best_forecast_{file_UID}.csv') # TODO define UID

if __name__ == "__main__":
    freq = input("Hourly (1h) or Daily (1D) frequency: ")
    main(freq)