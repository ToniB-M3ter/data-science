import random
import pandas as pd
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output
from dash import Dash, dcc, html, Input, Output


import sys
sys.path.insert(1, '/Users/tmb/PycharmProjects/data-science/UFE/code')
import readWriteS3 as rs3

# Set parms
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 25)

ORG= 'onfido'
freq='1D'

tidy_folder = '2_tidy/'
fit_folder = '4_fit/'
forecast_folder = '4_forecast/'

def get_keys(level, nodays):
    if level=='tidy':
        metadatakey = 'usage_meta.gz'
        key = 'usage.gz'
    elif level=='hier':
        #plot_date = dt.today() - relativedelta(days=nodays)
        metadatakey = 'hier_2024_03_06_usage_meta.gz'
        key = 'hier_2024_03_06_usage.gz'
        #metadatakey = 'hier' + '_' + plot_date.strftime("%Y_%m_%d") + '_' + 'usage_meta.gz'
        #key = 'hier' + '_' + plot_date.strftime("%Y_%m_%d") + '_' + 'usage.gz'
    return key, metadatakey

def combine(tidy_data, forecasts):
    # read tidy and forecast data from s3 and prep for plotting
    #plot_df = pd.read_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/plot_df_28Feb.csv')
    tidy_data.to_csv('tidy_data.csv') # ts_id and tm --> columns; tm = datetime(ns)
    tidy_data.loc[:, '.model'] = 'actuals' # add .model column to tidy data
    forecasts.to_csv('forecasts.csv')

    plot_df = pd.concat([tidy_data.set_index(['ts_id', 'tm']), forecasts.set_index(['ts_id', 'tm'])])
    print(plot_df.loc['000111d79d89ee128ff0b685c298147df65d976ee68fdf2fceb5c86277791aa8'])
    plot_df.reset_index(inplace=True)
    plot_df.sort_values(['ts_id', '.model', 'tm'], inplace=True, ascending=True)
    print(plot_df[plot_df['ts_id']=='000111d79d89ee128ff0b685c298147df65d976ee68fdf2fceb5c86277791aa8'])
    plot_df.to_csv('plot_df.csv')

    df_ids = plot_df[['account_cd', 'account_nm', 'meter', 'measure', 'ts_id']].drop_duplicates()
    print('There are '+str(len(df_ids))+' unique time series in df_ids')
    unique_ids = plot_df['ts_id'].unique()
    print('There are '+str(len(unique_ids))+' unique time series')
    return plot_df, df_ids, unique_ids

def select_ts_to_plot(plot_df, df_ids, unique_ids):
    # which times series to plot
    dim_ids = pd.DataFrame()
    cnt=0
    ts = ' '
    ts = input('Which time series? ')
    try:
        if ts in ['meter','meters', 'm3ters', 'Meters']:
            nrows = 21
            account_cds = ['<aggregated>']*nrows
            account_nms = ['<aggregated>']*nrows
            meters = ['applicant_fraud_report_standard',   #meter level
                    'autofill',
                    'document_report_standard_auto',
                    'document_report_standard_hybrid',
                    'document_report_standard_manual',
                    'document_report_video',
                    'facial_similarity_report_motion',
                    'facial_similarity_report_photo_fully_auto',
                    'facial_similarity_report_standard',
                    'facial_similarity_report_video',
                    'identity_report_standard',
                    'india_pan_report_standard',
                    'known_faces_report_standard',
                    'phone_verification_report_standard',
                    'proof_of_address_report_standard',
                    'right_to_work_report_standard',
                    'street_level_report_standard',
                    'us_driving_licence_report_standard',
                    'watchlist_report_aml',
                    'watchlist_report_full',
                    'watchlist_report_kyc']
            measures = ['check_count']*nrows
            dim_ids = pd.DataFrame({'account_cd':account_cds, 'account_nm':account_nms, 'meter':meters, 'measure':measures})
            df_unq_ids = pd.merge(df_ids, dim_ids, on=['account_cd', 'meter', 'measure'])
            ids=df_unq_ids['ts_id'].tolist()
        elif ts in ['ord', 'ordered']:
            nrows = 15
            ids = plot_df['unique_id'].unique()[2500:2514]
        elif ts in [' ', 'random']:
            nrows = 15
            t = random.sample(range(len(unique_ids)), 15)  # 15 random numbers for indices to plot
        elif ts == 'def':
            ids = [
                   'onfido/0010800003F94PgAAJ/watchlist_report_full',
                   'onfido/0010800003F94PgAAJ/document_report_standard_hybrid',
                   'onfido/0010800003F94PgAAJ/facial_similarity_report_standard',
                   'onfido/0010800003F94PgAAJ/<aggregated>']
            nrows = len(ids)
    except Exception as error:
        cnt = cnt +1
        print("An exception has occured "
              "retry entering time series", error)
        while cnt <4:
            ts = input('Which time series? ')
    return nrows, ids, df_unq_ids

def plot(nrows, ids, plot_df):
    ids = ids[0:5]
    fig = make_subplots(rows=nrows, cols=1, vertical_spacing=0.3/nrows,
                        subplot_titles=[i for i in ids])

    #full_fig = fig.full_figure_for_development()
    #yaxis_range = full_fig.layout.yaxis.range[1] - full_fig.layout.yaxis.range[1]


    annotations_list=[]

    for n, id in zip(range(1,nrows), ids):
        fig.append_trace(go.Scatter(name=id, x=plot_df['tm'],y=plot_df[plot_df['ts_id']==id]['y'], marker=dict(color="#2ca02c")), row=n, col=1)
        #fig.append_trace(go.Scatter(name=id+'AutoETS', x=plot_df['tm'],y=plot_df[plot_df['ts_id']==id]['AutoETS'], marker=dict(color="#ff7f0e"), showlegend=False), row=n, col=1)
        #fig.append_trace(go.Scatter(name=id + 'AutoETS/BottomUp', x=plot_df['tm'], y=plot_df[plot_df['ts_id'] == id]['AutoETS/BottomUp'], marker=dict(color="#17becf"),  showlegend=False), row=n, col=1)
        fig.append_trace(go.Scatter(name=id +'AutoETS/MinTrace nonneg', x=plot_df['tm'], y=plot_df[plot_df['ts_id'] == id]['z'], marker=dict(color="#e377c2")), row=n, col=1)
        fig.append_trace(go.Scatter(name='Upper Bound', x=plot_df['tm'],y=plot_df[plot_df['ts_id']==id]['z0'], marker=dict(color="#444"), line=dict(width=0), showlegend=False), row=n, col=1)
        fig.append_trace(go.Scatter(name='Lower Bound', x=plot_df['tm'],y=plot_df[plot_df['ts_id']==id]['z1'], marker=dict(color="#444"), line=dict(width=0), fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',showlegend=False), row=n, col=1)

        # annot_dict=dict(x=datetime.today() - relativedelta(months=3),
        #                 y=yaxis_range/2, # 17000
        #                 xref='x'+str(n),
        #                 yref='y'+str(n),
        #                 text=id,
        #                 )
        # annotations_list.append(annot_dict.copy())


    fig['layout'].update(title='Hierarchical Reconciliation Method Comparison<br>Green = Actuals; Orange=BaseForecast; Blue=AutoETS Bottoms Up; Pink=MinTrace Reconciliation'
                         , height=(nrows*200)
                         #annotations=annotations_list
                         )
    fig.show()


def main():
    nodays=0
    key, metadatakey = get_keys('tidy', nodays)
    metadata_str, cols = rs3.get_metadata(tidy_folder + freq + '/', metadatakey)
    tidy_data = rs3.get_data(tidy_folder + freq + '/', key, cols)

    hierkey, hiermetadtakey = get_keys('hier', nodays)
    #metadata_str, cols = rs3.get_metadata(forecast_folder + freq + '/', hiermetadtakey)
    cols = [
            'ts_id',
            'account_cd',
            'account_nm',
            'meter',
            'measure',
            'tm',
            'z',
            'z0',
            'z1',
            '.model']
    forecasts = rs3.get_data(forecast_folder + freq + '/', hierkey, cols)  #get forecasts  get_data(filepath, key, metadatakey)

    plot_df, df_ids, unique_ids = combine(tidy_data, forecasts)
    nrows, ids, df_unq_ids = select_ts_to_plot(plot_df, df_ids, unique_ids)
    plot(nrows, ids, plot_df)


if __name__ == "__main__":
    main()