from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px

ORG = 'onfido'

app = Dash(__name__)

app.layout=html.Div([
    html.H4('Usage Hier data'),
    dcc.Graph(id="hierarchy_fig"),
    dcc.Checklist(
        id="Ids",
        options=["onfido","onfido/facial_similarity_report"],
        value=["onfido"],
        inline=True
    ),
])

@app.callback(
    Output("hierarchy_fig", "figure"),
    Input("Ids", "value"))

def processData(Ids):
    plot_df = pd.read_csv(f'/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/{ORG}/plot_df.csv')
    plot_df.fillna(0)
    #unique_ids = plot_df['unique_id'].unique()
    mask = plot_df.unique_id.isin(Ids)
    return plot_df[mask]

def update_chart(Ids):
    plot_df=processData(Ids)
    plot_df.to_json()
    hierfig = go.Figure([
        go.Scatter(
            name='Actuals',
            x=plot_df['ds'],
            y=plot_df['y'],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Usage Forecast',
            x=plot_df['ds'],
            y=plot_df['AutoETS'],
            mode='lines',
            line=dict(color='#9467bd'),
        ),
        go.Scatter(
            name='Upper Bound',
            x=plot_df['ds'],
            y=plot_df['AutoETS-lo-95'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=plot_df['ds'],
            y=plot_df['AutoETS-hi-95'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    hierfig.update_layout(
        yaxis_title='Usage',
        title='Usage with prediction interval',
        hovermode="x"
    )
    return hierfig


if __name__ == "__main__":
    app.run_server(debug=True)
