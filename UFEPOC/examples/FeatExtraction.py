import matplotlib.pylab as plt
import pandas as pd

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import mutual_info_regression

from tsfresh.examples import robot_execution_failures

# Set parms
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 25)

def tsfresh_example():
    robot_execution_failures.download_robot_execution_failures()
    df, y = robot_execution_failures.load_robot_execution_failures()
    print(df.head())
    df.to_csv('/Users/tmb/PycharmProjects/data-science/UFEPOC/data/robot_exec_fail.csv')
    #print(df.info())

    print(df[df.id == 3][['time', 'F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z']])
    print(df[df.id == 20][['time', 'F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z']])

    df[df.id == 3][['time', 'F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z']].plot(x='time', title='Success example (id 3)', figsize=(12, 6))
    #plt.show()
    df[df.id == 20][['time', 'F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z']].plot(x='time', title='Failure example (id 20)', figsize=(12, 6))
    #plt.show()

    extraction_settings = ComprehensiveFCParameters()

    X = extract_features(df, column_id='id', column_sort='time',
                         default_fc_parameters=extraction_settings,
                         # we impute = remove all NaN features automatically
                         impute_function=impute)

    X.to_csv('/Users/tmb/PycharmProjects/data-science/UFEPOC/data/tsfreshFeat.csv')
    return

def create_lags(df: pd.DataFrame, lags: int) -> pd.DataFrame:
    usage = df['y']
    cols = df.columns
    df_lag = pd.concat([usage.shift(10), usage.shift(9), usage.shift(8), usage.shift(7), usage.shift(6), usage.shift(5), usage.shift(4), usage.shift(3), usage.shift(2), usage.shift(1), usage], axis=1)
    df_lag.columns = ['t-10', 't-9', 't-8', 't-7', 't-6', 't-5', 't-4', 't-3', 't-2', 't-1', 'y']
    return df_lag

def tsfresh_create_lags(df: pd.DataFrame, lags: int) -> pd.DataFrame:
    df_shift, y = make_forecasting_frame(x, kind="price", max_timeshift=10, rolling_direction=1)
    make_forecasting_frame()
    return df_shift, y

def mutual_info(df: pd.DataFrame):
    # split data
    X_train, X_test, y_train, y_test = train_test_split(df.drop(labels=['SalePrice'], axis=1),
                                                        df['SalePrice'],
                                                        test_size=0.3,
                                                        random_state=0)
    # test for mutual info
    mutual_info = mutual_info_regression(X_train.fillna(0), y_train)
    mutual_info
    return


def main():
    usage = pd.read_csv(f'/UFEPOC/output_files/hierarchical/onfido/df.csv', index_col=0)
    lags = create_lags(usage, 10)
    print(lags.head(10))
    usage_lags = tsfresh_create_lags(usage, 10)
    print(usage_lags.head(10))




if __name__ == "__main__":
    main()