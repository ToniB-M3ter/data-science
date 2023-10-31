
import sys
import importlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import readWriteS3 as rs3

def clean_data(df):
    print(df.head())

    # Set parms for heat map
    #meter = input('Which meter? ')
    #startdate = input('Start date (YYYY-mm-dd HH:MM:SS format)? ')
    #enddate = input('End date (YYYY-mm-dd HH:MM:SS format)? ')
    meter = 'ingest'
    startdate = '2023-01-01 00:00:00'
    enddate = '2023-01-31 00:00:00'


    # greater than the start date and smaller than the end date
    datetime_mask = (df['tm'] > startdate) & (df['tm'] <= enddate)
    df = df.loc[datetime_mask]

    # Hard code sample of accounts
    accounts = ['Burst SMS - Staging','ClickHouse Production','Integ Tests Primary','Prompt Production',"Soner's Organization",'Tricentis Prod','m3terBilllingOrg Production']
    df = df[(df['account'].isin(accounts)) & (df['meter'] == meter)]
    df=df[['account', 'tm','y']]

    df['tm'] = pd.to_datetime(df['tm'])
    df.sort_values(by=['tm'], inplace=True)

    df = df.pivot(index='account', columns='tm', values='y')

    print(df.shape)
    return df


def heatmap(data):
    #plt.figure(figsize=(20, 20), facecolor='w', edgecolor='k')
    plt.figure(facecolor='w', edgecolor='k')
    #sns.set(font_scale=1.5)

    sns.heatmap(data,
                cmap="coolwarm",
                annot=True,
                fmt='.5g',
                cbar=False)
                #linewidths=2)
                #linecolor='black')

    plt.xlabel('Datetime', fontsize=9)
    plt.ylabel('Account', fontsize=9)
    plt.show()
    return

def main():
    data = pd.read_csv('/UFE/data/dfUsage.csv')
    data = clean_data(data)
    heatmap(data)
    return


if __name__ == "__main__":
    main()