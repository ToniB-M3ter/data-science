

import sys
import importlib
from pandas import DataFrame
import pandas as pd
import readWriteS3 as rs3
import plotly.express as px

def get_stats(dfUsage):

    return

def plot(data: DataFrame, account: str, meter: str):
    fig = px.line(data, x='tm', y='y', title='Account: {} & Meter: {}'.format(account, meter))
    fig.show()
    return

def prep_for_ts(df: DataFrame, datetime_col: str, y: str):
    df = df[[datetime_col, y]]
    df.set_index(datetime_col, inplace=True)
    return df

def main(dfUsage):
    #dfUsage = get_data()
    plot(dfUsage, dfUsage['account'].iloc[0], dfUsage['meter'].iloc[0])
    dfUsage_clean = prep_for_ts(dfUsage, 'tm', 'y')
    return


if __name__ == "__main__":

    filepath = '2_tidy/1h/'  # allow selection of data
    metadatakey = 'usage_meta.gz'
    key = 'usage.gz'

    dataloadcache= pd.DataFrame()

    while True:
        if dataloadcache.empty:
            dataloadcache = rs3.get_data(filepath, key, metadatakey)
        data = rs3.select_ts(dataloadcache)
        print(data.head())
        main(data)
        print("Press enter to re-run the script, CTRL-C to exit")
        sys.stdin.readline()
        importlib.reload(rs3)