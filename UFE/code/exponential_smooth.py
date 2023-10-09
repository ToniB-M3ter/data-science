
import sys
import importlib
from statsmodels.tsa.api import SimpleExpSmoothing
from pandas import DataFrame, Series, to_datetime
import plotly.express as px
import readFromS3 as rs3

# cache for dataload
dataloadcache = None
#from statsforecast import StatsForecast
#from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt, seasonal_decompose
#from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# df: index = Datetime; col = Hourly_Temp
#df = pd.read_csv("/Users/tmb/PycharmProjects/data-science/UFE/data/MLTempDataset1.csv",
 #                index_col = 'Datetime', usecols=['Datetime', 'Hourly_Temp'])

def get_data():
    data = rs3.main()
    return data

def prep_for_ts(df: DataFrame, datetime_col: str, y: str):
    df = df[[datetime_col, y]]
    df.set_index(datetime_col, inplace=True)
    return df

def forecast(dfUsage: DataFrame)-> str:
    ses = SimpleExpSmoothing(dfUsage)
    alpha = 0.2
    model = ses.fit(smoothing_level=alpha, optimized=False)
    forecastperiods = 10
    forecast = model.forecast(forecastperiods)
    print(forecast)
    return forecast

def plot(data: DataFrame, account: str, meter: str):
    fig = px.line(data, x='tm', y='y', title='Account: {} & Meter: {}'.format(account, meter))
    fig.show()
    return


def main():
    dfUsage = get_data()
    plot(dfUsage, dfUsage['account'].iloc[0], dfUsage['meter'].iloc[0])
    dfUsage_clean = prep_for_ts(dfUsage, 'tm', 'y')
    forecast(dfUsage_clean)
    return


if __name__ == "__main__":

    filepath = '2_tidy/1h/'  # allow selection of data
    metadatakey = 'usage_meta.gz'
    key = 'usage.gz'

    while True:
        if not dataloadcache:
            dataloadcache = rs3.get_data(filepath, key, metadatakey)
        rs3.select_ts(dataloadcache)
        print("Press enter to re-run the script, CTRL-C to exit")
        sys.stdin.readline()
        importlib.reload(rs3)