import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objs as go
import datetime
from dash.dependencies import Input, Output
from dash import Dash, dcc, html, Input, Output
import cufflinks as cf
cf.go_offline()

# Set parms
ORG= 'onfido'
id = 'onfido/0010800002mn0xmAAA'
# needs work? ['onfido/0010800002mn0xmAAA'
#'onfido/0010800003HykxlAAB'
#'onfido/00108000037ScsuAAC/facial_similarity_report_motion'
#'onfido/phone_verification_report_standard'
# #'onfido/proof_of_address_report_standard'
# #'onfido/0012400001N11ncAAB'
# #'onfido/phone_verification_report_standard'

plot_df_direct = pd.read_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/plot_df.csv')
unique_ids = plot_df_direct['unique_id'].unique()
plot_df_direct['ds'] = pd.to_datetime(plot_df_direct['ds'])
plot_df_id = plot_df_direct[plot_df_direct['unique_id']== id]

p = go.Figure([
    go.Scatter(
        name='Actuals',
        x=plot_df_id['ds'],
        y=plot_df_id['y'],
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
    ),
    go.Scatter(
        name='Usage Forecast',
        x=plot_df_id['ds'],
        y=plot_df_id['AutoETS/MinTrace_method-mint_shrink_nonnegative-True_mint_shr_ridge-2e-05'],
        mode='lines',
        line=dict(color='#9467bd'),
    ),
    go.Scatter(
        name='Upper Bound',
        x=plot_df_id['ds'],
        y=plot_df_id['AutoETS/MinTrace_method-mint_shrink_nonnegative-True_mint_shr_ridge-2e-05-hi-95'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='Lower Bound',
        x=plot_df_id['ds'],
        y=plot_df_id['AutoETS/MinTrace_method-mint_shrink_nonnegative-True_mint_shr_ridge-2e-05-lo-95'],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    )
])

fig = go.Figure(p)
fig.update_layout(
    yaxis_title='Usage',
    title=f'Usage with prediction interval for {id}',
    hovermode="x"
    )

fig.show()