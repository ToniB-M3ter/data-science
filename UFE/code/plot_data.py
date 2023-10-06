import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

def convert(df):

    # df['timeStamp'] = pd.to_datetime(df['timeStamp'])
    # df.set_index('timeStamp', inplace=True)
    # #print(df.index.dtype)
    #
    # df = df.sort_index(ascending=False)
    # df.index = df.index.to_period('D')
    # print(df.index.dtype)

    # Break up into individual dfs
    orgs = df['org'].unique()
    orgDict = {elem: pd.DataFrame() for elem in orgs}
    for key in orgDict.keys():
        orgDict[key] = df[:][df.org == key]

    for org in orgs:
        dforg = df[org]
        dforg['timeStamp'] = pd.to_datetime(dforg['timeStamp'])
        dforg.set_index('timeStamp', inplace=True)

        dforg = dforg.sort_index(ascending=False)
        dforg.index = dforg.index.to_period('D')
        #print(df.index.dtype)

        dfgroupbyindex = df.groupby([df.index])[['usage']].count()
        print(dfgroupbyindex)


    #resampled = dfbyorg.resample('W').usage.count()
    return



def single_TS():

    return

def main(df):

    convert(df)


    return


if __name__ == "__main__":
    df = pd.read_csv("/UFE/data/dfUsage.csv")
    main(df)