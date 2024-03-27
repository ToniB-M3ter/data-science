from time import time

from utilsforecast.losses import (
    mse,   # mean square error
    mape,  # mean absolute percentage error
    mae,   # mean absolute error
    mase,  # mean absolute scaled error
    rmse,  # root mean square error
    mqloss, # multi-quantile loss
    scaled_crps, # scaled continues ranked probability score
    )


# cross validate
if USER is None:
    pass  # TODO cannot run cross validation on lambda as it will time out - must convert to EC2
    # cv_rmse_df = err.cross_validate(dfUsage_clean, model, h)
    # rs3.write_csv_log_to_S3(cv_rmse_df, 'cv_rmse_df')
else:
    init_xval = time()
    crossvalidation_df = postproc.cross_validate(df_to_forecast, model, h, 1)
    end_xval = time()
    logger.info(f'Cross Validation Minutes:  + {(end_xval - init_xval) / 60}')

    crossvalidation_df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/cv_rmse_df.csv')
    evaluation_df = postproc.evaluate_cross_validation(crossvalidation_df, [rmse, msse, energy_score,
                                                                            scaled_crps])  # other err fxns: mse, mqloss, rel_mse, msse, scaled_crps, energy_score
    evaluation_df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/evaluation_df.csv')

    summary_df = evaluation_df.groupby('best_model').size().sort_values().to_frame()
    summary_df.reset_index().columns = ["Model", "Nr. of unique_ids"]
    summary_df.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/summary_df.csv')

    best_forecasts = postproc.get_best_model_forecast(forecasts, evaluation_df)
    best_forecasts.to_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/{ORG}/best_forecasts.csv')