"""Averages values from n base models into new 'combined model' values"""
from pandas import DataFrame

def avg_models(base_forecasts: DataFrame, models:list=None)-> DataFrame:
    other_cols = ['ds', 'meter', 'measure', 'account_cd', 'account_nm', '.model', 'unique_id']

    if models:
        for model in models:
            all_cols = [x for x in base_forecasts.columns if model in x]

        lo_cols = [x for x in all_cols if 'lo' in x]
        hi_cols = [x for x in all_cols if 'hi' in x]
        base_cols = [y for y in base_forecasts if y not in lo_cols and y not in hi_cols]
    else:
        lo_cols = [x for x in base_forecasts.columns if 'lo' in x]
        hi_cols = [x for x in base_forecasts.columns if 'hi' in x]
        base_cols = [y for y in base_forecasts if y not in lo_cols and y not in hi_cols and y not in other_cols]

    coltypes = [lo_cols, hi_cols, base_cols]
    coltypesnames = ['lo', 'hi', 'base']
    for coltype, coltypename in zip(coltypes, coltypesnames):
        base_forecasts['combined-' + coltypename] = round(base_forecasts[list(coltype)].sum(axis=1)/len(list(coltype)), 3)
    return base_forecasts
