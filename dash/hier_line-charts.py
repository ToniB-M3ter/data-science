from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)


app.layout = html.Div([
    html.H4('Hierarchical Time Seris'),
    dcc.Graph(id="line-charts-x-graph"),
    dcc.Checklist(
    id="line-charts-x-checklist",
        options=["m3ter","m3ter/AssembledHQProd",	"m3ter/BurstSMS-LocalTest",	"m3ter/BurstSMS-Production", "m3ter/OnfidoDev",	"m3ter/OnfidoProd"	
                 "m3ter/Patagona-Production",	"m3ter/Patagona-Sandbox",	"m3ter/Regal.ioProd",	"m3ter/SiftForecasting",	"m3ter/TricentisProd"],
        value=["m3ter"],
        inline=True
    ),
])


@app.callback(
    Output("line-charts-x-graph", "figure"), 
    Input("line-charts-x-checklist", "value"))
def update_line_chart(unique_ids):
    df = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/hierarchical/Y_hat_df.csv')
    mask = df.unique_id.isin(unique_ids)
    fig = px.line(df[mask],
        x="ds", y="AutoETS", color='unique_id')
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
