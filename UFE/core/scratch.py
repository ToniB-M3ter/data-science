import pandas as pd
import numpy as np
import utils
from functools import partial
import _pre_process as preproc
import _combine_forecasts as comb
import _evaluation as eval
import _stats_fit_forecast as fitfrct

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

"""
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

dfUsage = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/onfido/dfUsage.csv',  parse_dates=['ds'])
ids = ['980576b982cf369a3c00735bb78feb7575304eac08e53d56e07e420296ae0b7d',
       '2c82fa602e85fa401b9d8aca2b590c1549c3dfbbeff2cf1b20b3b60adcf8515c',
        '86ec7c5729d2ec4a39a29a5efb0b28387d711984ee6f80db758aa492494b49e1',
        '0cf695a1c18d56bdb5829bbf492ea6f91a5a2c7fb5cccc2b139e0709ad3a75ae',
        '8de3de0593ff3ead1e4a8956c3faabce50eb222a971316201b2f37328aff30ba',
        '3a58ca5729354fd8547d20e8b22a4aa805da2ad422f5578ac8bffc5b042e4ea8',
        'ff8ede0773e8dcc8aa42624a1432d6fc240312ef824091fd24dbd01c6472da7c',
        'b0e5e7c6168446137e2b66185cce9471b2ddc4f8cf88b7c148b4d4ac9a0145ca',
        '3d74b677ca513f9cb5d01310ef086848547ed69052a88248f5b1d2f94efaaf5f',
        '5f6424814760c5679de6ab39a0af023c255fab33af6ccbf39bbd076291fe5c39',
        'e213b991de56a2e14345cd1a9252d87a4ab36394f97fa75887ecacf49241509d',
        'ea221906d61cfc42ffa3bbd8017a75b8cb7228a7ad3d371b02f82afc393820f8',
        'eddb02df728a239afb9ca767f0a3742a406bfe3c7a05bfd5fc33bacb490a7ada',
        'db9b215f48ace36dfae0641ec6138619beda20c8d3ed533991ef58cc9e6300a4',
       ]

#dfUsage = dfUsage[dfUsage['unique_id'].isin(ids)]


def simple_evaluation():
    # train_df (actuals)
    train = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/onfido/dfUsage.csv',
                          parse_dates=['ds'])
    # forecasts
    forecasts = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/onfido/forecasts-d2185c80-8ec1-4af6-a517-59ab59a73c86.csv',
                          parse_dates=['ds'])

    # combine actuals and forecasts for evaluation
    valid = pd.merge(forecasts, train, on=['unique_id', 'ds'], how='outer')
    valid.fillna(0, inplace=True)
    valid.sort_values(['unique_id', 'ds'], inplace=True)
    valid.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/onfido/valid.csv')

    # metrics
    metrics = [mse, rmse, mape]

    # models - could select specific models to evaluate if desired

    eval_ = evaluate(valid, metrics=metrics)
    eval_.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/onfido/eval.csv')

def reformat_evaluation():
    eval = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/onfido/evaluation_aa53d2d1-98e7-4fa8-8df1-f06e0434bfc7.csv', index_col=0)
    eval_reformatted = pd.melt(eval, id_vars=['unique_id', 'metric'], var_name='.model', value_name='value' )
    print(eval_reformatted.head())
    eval_reformatted.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/onfido/eval_reformatted.csv')

def main():
    freq = '1D'
    indices = [0, 2, 5]

    # create decoding dataframe
    #ts_models = preproc.select_models(freq, indices)
    #UID_list = [utils.generate_uid() for x in ts_models]
    #model_codes = pd.DataFrame({'model':ts_models, 'UUID':UID_list})
    #print(model_codes)

    #test_model = fitfrct.FitForecast.both(dfUsage, 18, 7, '1D', ts_models, -1, 95)

    #forecasts = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/onfido/forecasts.csv')
    #comb_forecasts = comb.avg_models(forecasts)
    #comb_forecasts.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/onfido/comb_forecasts.csv')

    #simple_evaluation()
    reformat_evaluation()
if __name__ == "__main__":
    main()