from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
from statsforecast import StatsForecast

def select_date_range(data_freq: str)-> datetime:
    startdate_input = input('Start date (YYYY-mm-dd HH:MM:SS format)? ')
    if startdate_input == '':
        # Select date range depending on frequency of data
        if 'D' in data_freq:
            startdate = datetime.today() - relativedelta(months=6)
        elif 'h' in data_freq:
            startdate = datetime.today() - relativedelta(months=1)
    else:
        startdate = startdate_input

    enddate_input = input('End date (YYYY-mm-dd HH:MM:SS format)? ')
    if enddate_input == '':
        enddate = datetime.today() - relativedelta(days=1)
    else:
        enddate = enddate_input

    return startdate, enddate

def clean_data(raw_df: pd.DataFrame, datetime_col: str, y: str, startdate, enddate) -> pd.DataFrame:
    # filter dates
    datetime_mask = (raw_df['tm'] > startdate) & (raw_df['tm'] <= enddate)
    df = raw_df.loc[datetime_mask]

    print('Fit from '  + str(startdate) +' to '+ str(enddate))

    # Remove whitespace from account name
    tmp_df = df['account'].copy()
    #tmp_df.replace(' ', '', regex=True, inplace=True)
    df['account'] = tmp_df

    # save unique combinations of account_id, meter, and measurement for formatting forecast file before saving
    df_ids = df[['account', 'account_id', 'meter', 'measurement']].drop_duplicates()
    df_ids.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/df_ids.csv')

    # format for fitting and forecasting - select subset of columns and add unique_id column
    df['unique_id'] = df.apply(
        lambda row: row.account + '_' + row.meter + '_' + row.measurement.split(' ')[0], axis=1)

    df = df[[datetime_col, y, 'unique_id']]
    df.columns = ['ds', 'y', 'unique_id']

    print(df.groupby(['unique_id']).size().reset_index(name='counts'))

    # Tell user number of time series
    print(str(df['unique_id'].nunique()) + ' Unique Time Series')

    # plot for inspection
    x = StatsForecast.plot(df)
    #x.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output_figs/{}'.format('ts_eng_input_data'))

    return df, df_ids

def split_data(df):
    #train = df.iloc[:int(0.75 * df['ds'].nunique())]  # TODO change 0.5 to horizon
    #valid = df.iloc[int(0.25 * df['ds'].nunique()) + 1:]  # TODO change 0.5*len(df) to horizon
    train = 1
    valid = 2
   # horizon (h)  = time periods into future for which a forecast will be made
    h = round((len(df) * 0.10))
    return train, valid, h

