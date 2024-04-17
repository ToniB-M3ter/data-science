"""Averages values from n base models into new 'combined model' values"""
from pandas import DataFrame

def avg_models(base_forecasts: DataFrame)-> DataFrame:
    other_cols = ['ds', 'meter', 'measure', 'account_cd', 'account_nm', '.model', 'unique_id']
    lo_cols = [x for x in base_forecasts.columns if 'lo' in x]
    hi_cols = [x for x in base_forecasts.columns if 'hi' in x]
    base_cols = [y for y in base_forecasts if y not in lo_cols and y not in hi_cols and y not in other_cols]

    coltypes = [base_cols, lo_cols, hi_cols]
    coltypesnames = ['base', 'lo-95', 'hi-95']
    for coltype, coltypename in zip(coltypes, coltypesnames):
        base_forecasts['Combined-' + coltypename ] = round(base_forecasts[list(coltype)].sum(axis=1)/len(list(coltype)), 3)
    base_forecasts.rename(columns={'Combined-base':'Combined'}, inplace=True)
    return base_forecasts
