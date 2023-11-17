import numpy as np
import pandas as pd
from tabulate import tabulate

#obtain hierarchical dataset
from datasetsforecast.hierarchical import HierarchicalData

# compute base forecast no coherent
from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA, Naive

#obtain hierarchical reconciliation methods and evaluation
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.evaluation import HierarchicalEvaluation
from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut


# Load TourismSmall dataset
Y_df, S, tags = HierarchicalData.load('./data', 'TourismSmall')
Y_df['ds'] = pd.to_datetime(Y_df['ds'])
Y_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_df.csv')
print(tabulate(Y_df.head(), headers="keys", tablefmt="psql"))

def run_example():
    #split train/test sets
    Y_test_df  = Y_df.groupby('unique_id').tail(4)
    Y_train_df = Y_df.drop(Y_test_df.index)

    # Compute base auto-ARIMA predictions
    fcst = StatsForecast(df=Y_train_df,
                         models=[AutoARIMA(season_length=4), Naive()],
                         freq='Q', n_jobs=-1)
    Y_hat_df = fcst.forecast(h=4)

    # Reconcile the base predictions
    reconcilers = [
        BottomUp(),
        TopDown(method='forecast_proportions'),
        MiddleOut(middle_level='Country/Purpose/State',
                  top_down_method='forecast_proportions')
    ]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df, Y_df=Y_train_df,
                              S=S, tags=tags)
    print(tabulate(Y_rec_df.head(10), headers="keys", tablefmt="psql"))
    Y_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_rec_df.csv')
    return

def main():
    run_example()
    return

if __name__ == "__main__":
        main()