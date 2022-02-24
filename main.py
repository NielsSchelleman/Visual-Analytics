import pandas as pd
from Model import buildModel
from dash import Dash, html, dcc, Input, Output, exceptions
import numpy as np
from itertools import combinations
import plotly.graph_objects as go

def getRanges():
    # acceptable search spaces, assume you cannot go from some value to any special value eg. 26 --> -9
    return {"ExternalRiskEstimate": range(30, 100),
            "MSinceOldestTradeOpen": range(0,1000,10),
            "MSinceMostRecentTradeOpen": range(0, 400,4),
            "AverageMInFile": range(0, 400,4),
            "NumSatisfactoryTrades": range(0, 100),
            "NumTrades60Ever/DerogPubRec": range(0, 20),
            "NumTrades90Ever/DerogPubRec": range(0, 20),
            "PercentTradesNeverDelq": range(0, 100),
            "MSinceMostRecentDelq": range(0, 100),
            "MaxDelq/PublicRecLast12M": range(0, 12),
            "MaxDelqEver": range(0, 10),
            "NumTotalTrades": range(0, 125, 2),
            "NumTradesOpeninLast12M": range(0, 25),
            "PercentInstallTrades": range(0, 100),
            "MSinceMostRecentInqexcl7days": range(0, 30),
            "NumInqLast6M": range(0, 75),
            "NumInqLast6Mexcl7days": range(0, 75),
            "NetFractionRevolvingBurden": range(0, 250, 3),
            "NetFractionInstallBurden": range(0, 500, 5),
            "NumRevolvingTradesWBalance": range(0, 40),
            "NumInstallTradesWBalance": range(1, 25),
            "NumBank/NatlTradesWHighUtilization": range(0, 25),
            "PercentTradesWBalance": range(0, 100)}

def rangeSearchChecklist():
    return ["ExternalRiskEstimate",
            "MSinceOldestTradeOpen",
            "MSinceMostRecentTradeOpen",
            "AverageMInFile",
            "NumSatisfactoryTrades",
            "NumTrades60Ever/DerogPubRec",
            "NumTrades90Ever/DerogPubRec",
            "PercentTradesNeverDelq",
            "MSinceMostRecentDelq",
            "MaxDelq/PublicRecLast12M",
            "MaxDelqEver",
            "NumTotalTrades",
            "NumTradesOpeninLast12M",
            "PercentInstallTrades",
            "MSinceMostRecentInqexcl7days",
            "NumInqLast6M",
            "NumInqLast6Mexcl7days",
            "NetFractionRevolvingBurden",
            "NetFractionInstallBurden",
            "NumRevolvingTradesWBalance",
            "NumInstallTradesWBalance",
            "NumBank/NatlTradesWHighUtilization",
            "PercentTradesWBalance"]


def create_grid(current, ranges, checklist):
    gridbase = []
    counts = []
    for nr, (val, name, valrange) in enumerate(zip(current[0], ranges.keys(), ranges.values())):
        if name in checklist:
            counts.append(nr)
            gridbase.append(valrange)
        else:
            gridbase.append(val)
    grid = np.array(np.meshgrid(*gridbase, indexing='ij'))
    return pd.DataFrame(grid.reshape(grid.shape[0], -1).T), counts


def plot_heatmaps(counts, data, ranges):
    axes = list(combinations(counts, 2))
    heatmaps = []
    for combo in axes:
        tempdata = data[[combo[0], combo[1], 'num_outcome']]
        heatmapdata = tempdata.groupby(list(combo)).mean().unstack()
        fig = go.Figure(layout=dict(xaxis_title=f'{list(ranges.keys())[combo[1]]}',
                                    yaxis_title=f'{list(ranges.keys())[combo[0]]}',
                                    title='fraction of Good values'),
                        data=go.Heatmap(
                            z=heatmapdata.values,
                            x=list(zip(*heatmapdata.columns))[1],
                            y=list(heatmapdata.index),
                            hoverongaps=False))
        heatmaps.append(dcc.Graph(figure=fig))
    return heatmaps

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    features = pd.read_csv('data/heloc_dataset_v1.csv')
    # the columns that stores the labels
    labelDimension = "RiskPerformance"
    # build a random forest classifier
    model = buildModel(features, labelDimension)
    app = Dash(__name__)

    app.layout = html.Div([
        html.Div([
            "ExternalRiskEstimate: ",
            dcc.Input(id='EXRE', value=-9, type='number'),
            html.Br(),
            "MSinceOldestTradeOpen: ",
            dcc.Input(id='MOTO',  value=-9, type='number'),
            html.Br(),
            "MSinceMostRecentTradeOpen: ",
            dcc.Input(id='MRTO',  value=-9, type='number'),
            html.Br(),
            "AverageMInFile: ",
            dcc.Input(id='AMIF', value=-9, type='number'),
            html.Br(),
            "NumSatisfactoryTrades: ",
            dcc.Input(id='NSAT',  value=-9, type='number'),
            html.Br(),
            "NumTrades60Ever/DerogPubRec",
            dcc.Input(id='T60D',  value=-9, type='number'),
            html.Br(),
            "NumTrades90Ever/DerogPubRec",
            dcc.Input(id='T90D', value=-9, type='number'),
            html.Br(),
            "PercentTradesNeverDelq",
            dcc.Input(id='PTND', value=-9, type='number'),
            html.Br(),
            "MSinceMostRecentDelq",
            dcc.Input(id='MMRD', value=-9, type='number'),
            html.Br(),
            "MaxDelq/PublicRecLast12M",
            dcc.Input(id='D12M', value=-9, type='number'),
            html.Br(),
            "MaxDelqEver",
            dcc.Input(id='MADE', value=-9, type='number'),
            html.Br(),
            "NumTotalTrades",
            dcc.Input(id='NUTT', value=-9, type='number'),
            html.Br(),
            "NumTradesOpeninLast12M",
            dcc.Input(id='TO12', value=-9, type='number'),
            html.Br(),
            "PercentInstallTrades",
            dcc.Input(id='PEIT', value=-9, type='number'),
            html.Br(),
            "MSinceMostRecentInqexcl7days",
            dcc.Input(id='MRI7', value=-9, type='number'),
            html.Br(),
            "NumInqLast6M",
            dcc.Input(id='NIL6', value=-9, type='number'),
            html.Br(),
            "NumInqLast6Mexcl7days",
            dcc.Input(id='I6E7', value=-9, type='number'),
            html.Br(),
            "NetFractionRevolvingBurden",
            dcc.Input(id='NFRB', value=-9, type='number'),
            html.Br(),
            "NetFractionInstallBurden",
            dcc.Input(id='NFIB', value=-9, type='number'),
            html.Br(),
            "NumRevolvingTradesWBalance",
            dcc.Input(id='RTWB', value=-9, type='number'),
            html.Br(),
            "NumInstallTradesWBalance",
            dcc.Input(id='ITWB', value=-9, type='number'),
            html.Br(),
            "NumBank/NatlTradesWHighUtilization",
            dcc.Input(id='TWHU', value=-9, type='number'),
            html.Br(),
            "PercentTradesWBalance",
            dcc.Input(id='PTWB', value=-9, type='number'),
            html.Br()
        ]),
        html.Br(),
        html.Div(id='current_eval'),


        html.H6("Which values should be modified? ( between 2 and 5)"),
        dcc.Checklist(options = rangeSearchChecklist(), id='rangeSearchChecklist', labelStyle = dict(display='block')),
        html.Button('Run Search', id='button_counterexample_run', n_clicks=0),
        html.Div(id='heatmaps')

    ])

    @app.callback(
        Output(component_id='current_eval', component_property='children'),
        Output('button_counterexample_run', 'n_clicks'),
        Output('heatmaps','children'),
        Input('EXRE', 'value'),
        Input('MOTO', 'value'),
        Input('MRTO', 'value'),
        Input('AMIF', 'value'),
        Input('NSAT', 'value'),
        Input('T60D', 'value'),
        Input('T90D', 'value'),
        Input('PTND', 'value'),
        Input('MMRD', 'value'),
        Input('D12M', 'value'),
        Input('MADE', 'value'),
        Input('NUTT', 'value'),
        Input('TO12', 'value'),
        Input('PEIT', 'value'),
        Input('MRI7', 'value'),
        Input('NIL6', 'value'),
        Input('I6E7', 'value'),
        Input('NFRB', 'value'),
        Input('NFIB', 'value'),
        Input('RTWB', 'value'),
        Input('ITWB', 'value'),
        Input('TWHU', 'value'),
        Input('PTWB', 'value'),
        Input(component_id="button_counterexample_run",component_property='n_clicks'),
        Input('rangeSearchChecklist', 'value')
    )
    def counterExampleSearch(EXRE, MOTO, MRTO, AMIF, NSAT, T60D, T90D, PTND, MMRD, D12M, MADE, NUTT, TO12, PEIT, MRI7,
                             NIL6, I6E7, NFRB, NFIB, RTWB, ITWB, TWHU, PTWB,
                             button,
                             checklist):
        if button == 0:
            raise exceptions.PreventUpdate

        else:
            current = np.array([[EXRE, MOTO, MRTO, AMIF, NSAT, T60D, T90D, PTND, MMRD, D12M, MADE, NUTT, TO12, PEIT,
                                 MRI7, NIL6, I6E7, NFRB, NFIB, RTWB, ITWB, TWHU, PTWB]], dtype='object')
            prediction = model.predict(current)
            if len(checklist)>5 or len(checklist)<2:
                return f'Output: {prediction}', 0, 0

            # create a grid with all the data for a gridsearch
            ranges = getRanges()
            newdata, counts = create_grid(current, ranges, checklist)

            # format the grid so that we can feed it to the model
            outcomes = model.predict(newdata)

            # reformat for plotting into heatmaps
            newdata['outcome'] = outcomes
            # sets bad =0 and good = 1 by turning strings into ints alphabetically
            newdata.outcome = pd.Categorical(newdata.outcome)
            newdata['num_outcome'] = newdata.outcome.cat.codes

            # find all axes for the plot
            heatmaps = plot_heatmaps(counts, newdata, ranges)
            return f'Output: {prediction}', 0, html.Div(heatmaps)

    app.run_server(debug=True)