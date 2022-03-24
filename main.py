import dash
import pandas as pd
from dash import Dash, html, dcc, Input, Output, exceptions
import numpy as np
from itertools import combinations
import plotly.graph_objects as go
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import plotly.express as px


def getRanges(percentages, current_vals):
    # Indices of chosen variables
    non_zero_idx = np.nonzero(np.array(percentages))

    # Keep only variables which ranges we want to change
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
    # Update ranges for selected variables
    for variable, value, percentage in zip(variable_list, current_vals, percentages):
        # Only viable for valid observations
        if value > 0 and percentage > 0:
            # For now using initial ranges
            stepsize = np.round((ranges[variable][-1] - ranges[variable][0]) / len(ranges[variable]))
            if percentage <= 1:
                min = np.floor((1-(percentage))*value)
                max = np.ceil((1+(percentage))*value)
            else: #Absolute values range search
                min = value - int(percentage)
                max = value + int(percentage)
            ranges[variable] = range(int(min), int(max), int(stepsize))
        if value == 0:
            stepsize = np.round((ranges[variable][-1] - ranges[variable][0]) / len(ranges[variable]))
            if percentage <= 1:
                min = 0
                max = 0 + np.round((ranges[variable][-1] - ranges[variable][0])*(1-percentage))
            else:  # Absolute values range search
                min = 0
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


def prep_lda(features):
    X = features.drop('RiskPerformance', axis=1)
    y = features['RiskPerformance']
    y_hat = model.predict(X)

    lda = LDA(n_components=3)
    y2 = [i if i == j else 'type1' if i == 'Bad' else 'type2' for i, j in zip(y, y_hat)]
    dims = ['LD1', 'LD2', 'LD3']
    components = lda.fit_transform(X, y2)
    comp_df = pd.DataFrame(components, columns=dims)
    comp_df['size'] = 1
    comp_df['labels'] = y2
    return comp_df, lda, dims


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    features = pd.read_csv('data/heloc_dataset_v1.csv')
    # the columns that stores the labels
    labelDimension = "RiskPerformance"
    # build a random forest classifier
    model = pickle.load(open('rf_mod.sav', 'rb'))
    margedict = {'margin-top': '3px'}

    ldfa_df, lda_model, lda_dims = prep_lda(features)

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
        html.Div(id='toptext'),

        html.Div([
                dcc.Checklist(options=rangeSearchChecklist(),
                              id='rangeSearchChecklist',
                              inputStyle={'display':'none'},
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
        html.Div(id='in-between-counterexample',style={'display':'none'}),

        html.Div(id='searchbar', children=[
            html.Button('Run Grid Search', id='button_counterexample_run', n_clicks=0, style={'margin-right': '3px'}),
            html.Button('Perform LIME', id='button_LIME', n_clicks=0, style={'margin-right': '3px'}),
            html.Button('Perform SHAP', id='button_SHAP', n_clicks=0, style={'margin-right': '3px'}),
            html.Button('Show Interactive LDA', id='button_LDA', n_clicks=0, style={'margin-right': '3px'})
        ], style={'width': '90%', 'display': 'block', 'background-color': '#e9e9ed', 'padding': '10px',
                  'border-radius': '5px'}),

        # store information of current client
        dcc.Store(id='store_person'),
        # store a counter
        html.Button(id='tally', n_clicks=0, style={'display':'none'}),



        # store plots and only show last one
        dcc.Store(id='heatmaps_plt', data=[None, -10, None]),
        dcc.Store(id='LIME_plt', data=[None, -10, None]),
        dcc.Store(id='SHAP_plt', data=[None, -10, None]),
        dcc.Store(id='LDA_plt', data=[None, -10, None]),
        #plot into here
        html.Div(id='misc_persist', children=[
            dcc.Checklist(id='lda_checklist',
                          options=['Bad','Good','Type1','Type2','Client'],
                          style={'display': 'none'})
        ]),
        html.Div(id='plot'),

    ])
    # callbacks_rangesearch.

    @app.callback(
        Output('rangeSearchChecklist', 'inputStyle'),
        Output('button_counterexample_run', 'style'),
        Output('button_counterexample_run', 'children'),
        Output('button_counterexample_run', 'n_clicks'),
        Output('rangeSearchChecklist','value'),
        Output('toptext', 'children'),
        Input('button_counterexample_run', 'n_clicks'),
    )
    def Intermediate(button):
        if button == 1 or button == 3:
            return {'display': 'inline-block'},\
                   {'background-color': 'yellow', 'margin-right': '3px'},\
                   'select columns', \
                   1, \
                   [], \
                   ["Fill in between 2-5 items to check the range, all original values and % of total range to check"]
        elif button == 2:
            return {'display': 'none'},\
                   {'background-color': '#e9e9ed', 'margin-right': '3px'},\
                   'Run Grid Search', \
                   2, \
                   dash.no_update, \
                   []

        else:
            raise exceptions.PreventUpdate

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
        Output('store_person','data'),
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
        Input('PTWB', 'value'))
    def create_profile(EXRE, MOTO, MRTO, AMIF, NSAT, T60D, T90D, PTND, MMRD, D12M, MADE, NUTT, TO12, PEIT, MRI7,
                       NIL6, I6E7, NFRB, NFIB, RTWB, ITWB, TWHU, PTWB):
        return np.array([[EXRE, MOTO, MRTO, AMIF, NSAT, T60D, T90D, PTND, MMRD, D12M, MADE, NUTT, TO12, PEIT,
                          MRI7, NIL6, I6E7, NFRB, NFIB, RTWB, ITWB, TWHU, PTWB]], dtype='object')


    @app.callback(
        Output(component_id='current_eval', component_property='children'),
        Output('heatmaps_plt','data'),
        Input('store_person','data'),
        Input(component_id="button_counterexample_run",component_property='n_clicks'),
        Input('rangeSearchChecklist', 'value'),
        Input('percentages', 'children'),
        Input('tally','n_clicks')
    )
    def counterExampleSearch(current, button, checklist, children, tally):
        if button != 2:
            raise exceptions.PreventUpdate

        else:
            prediction = model.predict(current)

            # most_similar = get_most_similar(current, checklist, prediction, features)

            if len(checklist) > 5 or len(checklist) < 2:
                return f'Output: {prediction}', [0, tally-1]

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
            return f'Output: {prediction}', [html.Div(heatmaps), tally+1, 'heatmap']

    @app.callback(
        Output('LDA_plt','data'),
        Output('button_LDA', 'n_clicks'),
        Input('store_person', 'data'),
        Input('button_LDA', 'n_clicks'),
        Input('tally', 'n_clicks'),
        Input('lda_checklist', 'value')
    )
    def LDA_plotting(person, button, tally, checklist):
        if button == 0:
            raise exceptions.PreventUpdate
        person_trans = lda_model.transform(pd.DataFrame(np.array(person[0]).reshape(1, 23), columns=features.keys()[1:]))[0]

        ldfa_df.loc[len(ldfa_df.index)] = {'LD1': person_trans[0],
                                           'LD2': person_trans[1],
                                           'LD3': person_trans[2],
                                           'size': 10,
                                           'labels': 'Client'}

        labels = {
            str(i): f"LD {i + 1} ({var:.1f}%)"
            for i, var in enumerate(lda_model.explained_variance_ratio_ * 100)
        }

        fig = px.scatter_matrix(
            ldfa_df,
            labels=labels,
            size='size',
            size_max=5,
            dimensions=lda_dims,
            color='labels'
        )
        fig.update_traces(diagonal_visible=False)
        fig.update_traces(marker=dict(line=dict(width=0)))

        if checklist:
            fig.for_each_trace(
                lambda trace: trace.update(visible='legendonly') if trace.name in checklist else ()
            )

        return [html.Div(dcc.Graph(figure=fig)), tally+1, 'lda'], 0


    @app.callback(
        Output('tally', 'n_clicks'),
        Output('plot', 'children'),
        Output('misc_persist', 'children'),
        Input('heatmaps_plt', 'data'),
        Input('LIME_plt', 'data'),
        Input('SHAP_plt', 'data'),
        Input('LDA_plt', 'data'),
    )
    def ShowPlot(heatmap, LIME, SHAP, LDA):

        curr_tally = -10
        curr_plot = None
        curr_name = None
        for plot, tally, name in [heatmap,LIME,SHAP,LDA]:
            if tally>curr_tally:
                curr_tally = tally
                curr_plot = plot

        if curr_name == 'lda':
            misc = html.Div(id='misc_persist', children=[
                dcc.Checklist(id='lda_checklist',
                              options=['Bad', 'Good', 'Type1', 'Type2', 'Client'],
                              style={'display': 'inline-block'})])
        else:
            misc = html.Div(id='misc_persist', children=[
                dcc.Checklist(id='lda_checklist', style={'display': 'none'})])

        return curr_tally, curr_plot, misc

    app.run_server(debug=True)