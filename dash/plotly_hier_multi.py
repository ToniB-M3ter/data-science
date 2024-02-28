import pandas as pd
import numpy as np
import random
import plotly
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dash.dependencies import Input, Output
from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots
import cufflinks as cf
cf.go_offline()

# Set parms
ORG= 'onfido'
nrows=15

colors=[
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

# read tidy and forecast data from s3 and prep for plotting
plot_df = pd.read_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/plot_df_28Feb.csv')
unique_ids = plot_df['unique_id'].unique()
t = random.sample(range(len(unique_ids)), 15) # 15 random numbers for indices to plot
ids=['onfido/applicant_fraud_report_standard',   #meter level
        'onfido/autofill',
        'onfido/document_report_standard_auto',
        'onfido/document_report_standard_hybrid',
        'onfido/document_report_standard_manual',
        'onfido/document_report_video',
        'onfido/facial_similarity_report_motion',
        'onfido/facial_similarity_report_photo_fully_auto',
        'onfido/facial_similarity_report_standard',
        'onfido/facial_similarity_report_video',
        'onfido/identity_report_standard',
        'onfido/india_pan_report_standard',
        'onfido/known_faces_report_standard',
        'onfido/phone_verification_report_standard',
        'onfido/proof_of_address_report_standard',
        'onfido/right_to_work_report_standard',
        'onfido/street_level_report_standard',
        'onfido/us_driving_licence_report_standard',
        'onfido/watchlist_report_aml',
        'onfido/watchlist_report_full',
        'onfido/watchlist_report_kyc']

fig = make_subplots(rows=nrows, cols=1, vertical_spacing=0.02,
                    subplot_titles=[i for i in ids])

#ids= plot_df['unique_id'].unique()
#ids = plot_df['unique_id'].unique()[t]

#full_fig = fig.full_figure_for_development()
#yaxis_range = full_fig.layout.yaxis.range[1] - full_fig.layout.yaxis.range[1]

annotations_list=[]

for n, id in zip(range(1,nrows), ids):
    fig.append_trace(go.Scatter(name=id, x=plot_df['ds'],y=plot_df[plot_df['unique_id']==id]['y'], marker=dict(color="#2ca02c")), row=n, col=1)
    fig.append_trace(go.Scatter(name=id+'AutoETS', x=plot_df['ds'],y=plot_df[plot_df['unique_id']==id]['AutoETS'], marker=dict(color="#ff7f0e"), showlegend=False), row=n, col=1)
    fig.append_trace(go.Scatter(name=id + 'AutoETS/BottomUp', x=plot_df['ds'], y=plot_df[plot_df['unique_id'] == id]['AutoETS/BottomUp'], marker=dict(color="#17becf"),  showlegend=False), row=n, col=1)
    fig.append_trace(go.Scatter(name=id +'AutoETS/MinTrace nonneg', x=plot_df['ds'], y=plot_df[plot_df['unique_id'] == id]['AutoETS/MinTrace_method-mint_shrink_nonnegative-True'], marker=dict(color="#e377c2")), row=n, col=1)
    fig.append_trace(go.Scatter(name='Upper Bound', x=plot_df['ds'],y=plot_df[plot_df['unique_id']==id]['AutoETS/MinTrace_method-mint_shrink_nonnegative-True-lo-95'], marker=dict(color="#444"), line=dict(width=0), showlegend=False), row=n, col=1)
    fig.append_trace(go.Scatter(name='Lower Bound', x=plot_df['ds'],y=plot_df[plot_df['unique_id']==id]['AutoETS/MinTrace_method-mint_shrink_nonnegative-True-hi-95'], marker=dict(color="#444"), line=dict(width=0), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',showlegend=False), row=n, col=1)

    # annot_dict=dict(x=datetime.today() - relativedelta(months=3),
    #                 y=yaxis_range/2, # 17000
    #                 xref='x'+str(n),
    #                 yref='y'+str(n),
    #                 text=id,
    #                 )
    # annotations_list.append(annot_dict.copy())


fig['layout'].update(title='Hierarchical Reconciliation Method Comparison<br>Green = Actuals; Orange=BaseForecast; Blue=AutoETS Bottoms Up; Pink=MinTrace Reconciliation'
                     , height=2000
                     #annotations=annotations_list
                     )
fig.show()


