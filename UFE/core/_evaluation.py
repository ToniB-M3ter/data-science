import os
import logging
import pandas as pd
import numpy as np
import utils

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import (
    mse,   # mean square error
    mape,  # mean absolute percentage error
    mae,   # mean absolute error
    mase,  # mean absolute scaled error
    rmse,  # root mean square error
    mqloss, # multi-quantile loss
    scaled_crps, # scaled continues ranked probability score
    )


class ErrorAnalysis():
    def mse_calc(y, y_hat):
        return np.mean((y - y_hat) ** 2)

    def rmse_calc(y, y_hat):
        return np.mean(np.sqrt(np.mean((y - y_hat) ** 2, axis=1)))

    def mase_calc(y, y_hat, y_insample, seasonality=24):
        errors = np.mean(np.abs(y - y_hat), axis=1)

        scale = np.mean(np.abs(y_insample[:, seasonality:] - y_insample[:, :-seasonality]), axis=1)
        return np.mean(errors / scale)

    def rel_mse(y, y_hat, y_train, mask=None):
        if mask is None:
            mask = np.ones_like(y)
        n_series, n_hier, horizon = y.shape

        eps = np.finfo(float).eps
        y_naive = np.repeat(y_train[:, :, [-1]], horizon, axis=2)
        norm = mse(y=y, y_hat=y_naive)
        loss = mse(y=y, y_hat=y_hat, weights=mask)
        loss = loss / (norm + eps)
        return loss

    def msse(y, y_hat, y_train, mask=None):
        if mask is None:
            mask = np.ones_like(y)
        n_series, n_hier, horizon = y.shape

        eps = np.finfo(float).eps
        y_in_sample_naive = y_train[:, :, :-1]
        y_in_sample_true = y_train[:, :, 1:]
        norm = mse(y=y_in_sample_true, y_hat=y_in_sample_naive)
        loss = mse(y=y, y_hat=y_hat, weights=mask)
        loss = loss / (norm + eps)
        return loss


class Evaluate():
    def __init__(self,
                 df: pd.DataFrame,
                 h: int,  # forecast horizon
                ):
        self.df = df
        self.h = h

    def evaluate_simple(actuals, forecasts, metrics):
        print(forecasts.columns)
        # check if combine lo/hi cols in forecasts, and if so drop them
        comb_cols = [x for x in forecasts.columns if 'combined' in x]
        print(comb_cols)
        if comb_cols:
            forecasts.drop(['combined-lo', 'combined-hi'], axis=1, inplace=True)
        print(forecasts.columns)

        valid = pd.merge(forecasts, actuals, on=['unique_id', 'ds'], how='outer') # combine actuals and forecasts for evaluation
        valid.fillna(0, inplace=True)
        valid.sort_values(['unique_id', 'ds'], inplace=True)
        evals = evaluate(valid, metrics=metrics)
        return evals

    def cross_validate(df, model, h, n_win):
        crossvalidation_df = model.cross_validation(
            df=df, # Y_df
            h=h,
            step_size=h,
            n_windows=n_win
        )
        return crossvalidation_df

    def score_cross_validation(df, metrics: list, train=None) -> pd.DataFrame:
        df.reset_index(inplace=True)
        models = df.drop(columns=['unique_id', 'ds', 'cutoff', 'y']).columns.tolist()
        evals = []
        # Calculate loss for every unique_id and cutoff.
        for cutoff in df['cutoff'].unique():
            eval_ = evaluate(df[df['cutoff'] == cutoff], metrics=metrics, models=models)
            #eval_ = evaluate(df[df['cutoff'] == cutoff], metrics=metrics, models=models, train_df=train)
            evals.append(eval_)
        evals = pd.concat(evals)
        #evals = evals.groupby('unique_id').mean(numeric_only=True)  # Averages the error metrics for all cutoffs for every combination of model and unique_id
        #evals['best_model'] = evals.idxmin(axis=1)
        return evals

    def get_best_model_forecast(forecasts_df, evaluation_df):
        df = forecasts_df.set_index('ds', append=True).stack().to_frame().reset_index(level=2)  # Wide to long
        df.columns = ['model', 'best_model_forecast']
        df = df.join(evaluation_df[['best_model']])
        df = df.query('model.str.replace("-lo-95|-hi-95", "", regex=True) == best_model').copy()
        df.loc[:, 'model'] = [model.replace(bm, 'best_model') for model, bm in zip(df['model'], df['best_model'])]
        df = df.drop(columns='best_model').set_index('model', append=True).unstack()
        df.columns = df.columns.droplevel()
        df = df.reset_index(level=1)
        return df

    def best_model_forecast(forecasts_df, evals):
        evals = evals.groupby('unique_id').mean(numeric_only=True)  # Averages the error metrics for all cutoffs for every combination of model and unique_id
        evals['best_model'] = evals.idxmin(axis=1)
        df = forecasts_df.set_index('ds', append=True).stack().to_frame().reset_index(level=2)  # Wide to long
        df.columns = ['model', 'best_model_forecast']
        df = df.join(evals[['best_model']])
        df = df.query('model.str.replace("-lo-95|-hi-95", "", regex=True) == best_model').copy()
        df.loc[:, 'model'] = [model.replace(bm, 'best_model') for model, bm in zip(df['model'], df['best_model'])]
        df = df.drop(columns='best_model').set_index('model', append=True).unstack()
        df.columns = df.columns.droplevel()
        df = df.reset_index(level=1)
        return df

    def reformat(evaluation_df):
        eval_reformatted = pd.melt(evaluation_df, id_vars=['unique_id', 'metric'], var_name='.model',
                                   value_name='value')
        df_models = eval_reformatted[['unique_id', '.model']].drop_duplicates()
        df_models['UUID'] = [utils.generate_uid() for x in range(len(df_models.index))]
        eval_reformatted = pd.merge(eval_reformatted, df_models, on=['unique_id', '.model'], how='left')
        #eval_reformatted.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/onfido/eval_reformatted.csv')
        return eval_reformatted

