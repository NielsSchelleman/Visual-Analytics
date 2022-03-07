import pandas as pd
from dash import Dash, html, dcc, Input, Output, exceptions
import numpy as np
from itertools import combinations
import plotly.graph_objects as go
import pickle
from scipy.spatial import distance



def getRanges(percentages, current_vals):
    #Indices of chosen variables
    non_zero_idx = np.nonzero(np.array(percentages))

    #Keep only variables which ranges we want to change
    variable_list = np.array(rangeSearchChecklist())[non_zero_idx]
    current_vals = np.array(current_vals).flatten()[non_zero_idx]
    percentages = np.array(percentages)[non_zero_idx]

    # acceptable search spaces, assume you cannot go from some value to any special value eg. 26 --> -9
    ranges = {"ExternalRiskEstimate": range(30, 100),
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
    #Update ranges for selected variables
    for variable, value, percentage in zip(variable_list, current_vals, percentages):
        #Only viable for valid observations
        if value > 0 and percentage > 0:
            #For now using initial ranges
            stepsize = np.round((ranges[variable][-1] - ranges[variable][0]) / len(ranges[variable]))
            if percentage <= 1:
                min = np.floor((1-(percentage))*value)
                max = np.ceil((1+(percentage))*value)
            else: #Absolute values range search
                min = value - int(percentage)
                max = value + int(percentage)

            ranges[variable] = range(int(min), int(max), int(stepsize))
    return ranges

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


def get_most_similar(current, checklist, prediction, features):
    opposing_group = features[features['RiskPerformance'] != prediction[0]].reset_index(drop=True)
    important_features = opposing_group[checklist]
    col_idx = [features.columns.get_loc(c) - 1 for c in checklist]
    important_values = current[0][col_idx]


    important_values = important_values.astype(float).reshape((1,-1))
    important_features = np.array(important_features).astype(float)

    dist = np.linalg.norm(important_features-important_values, axis=1)

    idx_min = np.argmin(dist)


    return opposing_group.loc[idx_min]





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
                            zmax=1,
                            zmin=0,
                            hoverongaps=False))
        heatmaps.append(dcc.Graph(figure=fig))
    return heatmaps

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    features = pd.read_csv('data/heloc_dataset_v1.csv')
    # the columns that stores the labels
    labelDimension = "RiskPerformance"
    # build a random forest classifier
    model = pickle.load(open('rf_mod.sav', 'rb'))
    margedict = {'margin-top': '3px'}
    # Non label columns
    column_names = list(features.columns)
    column_names.remove(labelDimension)
    # id lists
    variable_values_ids = ['EXRE', 'MOTO', 'MRTO', 'AMIF', 'NSAT', 'T60D', 'T90D', 'PTND', 'MMRD', 'D12M', 'MADE',
                           'NUTT', 'TO12',
                           'PEIT', 'MRI7', 'NIL6', 'I6E7', 'NFRB', 'NFIB', 'RTWB', 'ITWB', 'TWHU', 'PTWB']
    percentage_ids = list(column_names)

    app = Dash(__name__)

    app.layout = html.Div([
        html.Div(["Fill in between 2-5 items to check the range, all original values and % of total range to check"]),


        html.Div([
                dcc.Checklist(options=rangeSearchChecklist(),
                              id='rangeSearchChecklist',
                              labelStyle={'display': 'block', 'height': '22px', 'width': '300px'})],
                 style={'width': '300px', 'display': 'inline-block'}),

        html.Div([j for i in variable_values_ids for j in [html.Div([dcc.Input(id=i, value=-9, type='number',
                                                                     style={'height': '15.9px',
                                                                            'width': '140px'})],
                                                                    style={'height': '22px',
                                                                           'width': '150px'})]],
                 style={'width': '160px', 'display': 'inline-block'}),



        html.Div(id='percentages',style={'width': '160px', 'display': 'inline-block'}),
        html.Div(id='current_eval'),
        html.H6("Which values should be modified? ( between 2 and 5)"),


        html.Button('Run Search', id='button_counterexample_run', n_clicks=0),
        html.Div(id='heatmaps'),


    ])

    @app.callback(
        Output('percentages','children'),
        Input('rangeSearchChecklist', 'value'),
    )
    def GenInputs(checklist):
        if not checklist:
            checklist = []
        children = []
        for i in rangeSearchChecklist():
            if i in checklist:
                children.append(html.Div([
                    dcc.Input(id=i+'_%', value=10, type='number',
                              style={'display': 'block', 'height': '15.9px', 'width': '140px',
                                     'padding-left': '2px'})],
                    style={'height': '22px', 'width': '150px'}
                ))
            else:
                children.append(html.Div([dcc.Input(id=i+'_%', value=0, type='number', style={'display': 'None'}),
                                          html.Br()], style={'height': '22px',
                                                             'width': '150px'}))
        return children

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
        Input('rangeSearchChecklist', 'value'),
        Input('percentages', 'children')
    )
    def counterExampleSearch(EXRE, MOTO, MRTO, AMIF, NSAT, T60D, T90D, PTND, MMRD, D12M, MADE, NUTT, TO12, PEIT, MRI7,
                             NIL6, I6E7, NFRB, NFIB, RTWB, ITWB, TWHU, PTWB,
                             button,
                             checklist,
                             children
                             ):
        if button == 0:
            raise exceptions.PreventUpdate

        else:
            current = np.array([[EXRE, MOTO, MRTO, AMIF, NSAT, T60D, T90D, PTND, MMRD, D12M, MADE, NUTT, TO12, PEIT,
                                 MRI7, NIL6, I6E7, NFRB, NFIB, RTWB, ITWB, TWHU, PTWB]], dtype='object')
            prediction = model.predict(current)

            most_similar = get_most_similar(current, checklist, prediction, features)
            print('test')

            if len(checklist)>5 or len(checklist)<2:
                return f'Output: {prediction}', 0, 0

            #Get all percentage ranges
            percentages = []
            for i in range(len(children)):
                    percentages.append(children[i]['props']['children'][0]['props']['value'])

            # create a grid with all the data for a gridsearch
            if len(np.array(percentages)[np.array(percentages)<0]) > 0:
                #No negative ranges
                #return f'Ranges should be positive', 0, 0
                return html.H1([
                    html.Span("Ranges should only be positive", style={'color':'red'})
                ]), 0, 0


            ranges = getRanges(percentages, current)
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