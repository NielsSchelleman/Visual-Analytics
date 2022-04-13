import dash
import pandas as pd
from dash import Dash, html, dcc, Input, Output, exceptions, State
import dash_bootstrap_components as dbc
import numpy as np
from itertools import combinations
import plotly.graph_objects as go
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import plotly.express as px
from statsmodels.stats.proportion import proportion_confint
import Lime_shap
import base64
from sklearn.preprocessing import StandardScaler

def get_model():
    """"This function returns the random forest model that was given in the assignment"""
    try:  # we try to load the model, rather than recreating it if it was already created before
        model = pickle.load(open('rf_mod.sav', 'rb'))
    except:  # we build the model, this is the code that was given to us with the assignment.
        print("the file for the model wasn't found, so it is being built now" )
        from Model import buildModel

        # load in the dataset
        features = pd.read_csv('heloc_dataset_v1.csv')

        # the columns that stores the labels
        labelDimension = "RiskPerformance"

        # build a random forest classifier
        model = buildModel(features, labelDimension)

        pickle.dump(model, open('rf_mod.sav', 'wb'))
    return model

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
        max_range = ranges[variable][-1]
        min_range = ranges[variable][0]
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
            if max > max_range:
                max = max_range
            elif min < min_range:
                min = min_range
            ranges[variable] = range(int(min), int(max), int(stepsize))
        if value == 0 and percentage > 0:
            stepsize = np.round((ranges[variable][-1] - ranges[variable][0]) / len(ranges[variable]))
            if percentage <= 1:
                min = 0
                max = 0 + np.round((ranges[variable][-1] - ranges[variable][0]) * (percentage))
                if (max - min) % stepsize == 0:
                    max += 1
            else:  # Absolute values range search
                min = 0
                max = value + int(percentage)
                if (max - min) % stepsize == 0:
                    max += 1
            if max > max_range:
                max = max_range
            elif min < min_range:
                min = min_range
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

def plot_Lime(inputvalues):
    """returns the object of an image of a lime explainer plot"""
    lime_model = Lime_shap.get_lime_model(features) # features is taken from outside the function
    Lime_shap.lime_explain(lime_model, model, inputvalues)
    encoded_image = base64.b64encode(open('lime_explain.jpg', 'rb').read()).decode('ascii')
    return html.Img(id='limeimage',src='data:image/png;base64,{}'.format(encoded_image))

def plot_Shap(inputvalues):
    """returns the object of an image of a shap waterfall plot"""
    shap_model = Lime_shap.get_shap_model(model)  # model is taken from outside the function
    shapvalue = Lime_shap.calculate_shap_value(shap_model, inputvalues)
    Lime_shap.shap_waterfall_plot(shap_model, shapvalue, inputvalues, column_names)  # column_names is taken from outside the function
    encoded_image = base64.b64encode(open('shap_waterfall.png', 'rb').read()).decode('ascii')
    return html.Img(id='shapimage',src='data:image/png;base64,{}'.format(encoded_image))

def plot_Shap_Summary():
    try:  # if the image already exists
        encoded_image = base64.b64encode(open('shap_summary.png', 'rb').read()).decode('ascii')
        #return html.Img(src='data:image/png;base64,{}'.format(encoded_image))
        return encoded_image
    except:  # otherwise we first create it
        features_ = features.drop('RiskPerformance', axis=1)
        shap_model = Lime_shap.get_shap_model(model)  # model is taken from outside the function
        shap_values = Lime_shap.get_shap_values(shap_model, features_)  # features is taken from outside the function
        Lime_shap.shap_summary_plot(shap_values, features_)# features is taken from outside the function
        encoded_image = base64.b64encode(open('shap_summary.png', 'rb').read()).decode('ascii')
        #return html.Img(id='shapsummary', src='data:image/png;base64,{}'.format(encoded_image))
        return encoded_image

def plot_most_similar(current,  same_group=False, normalize=True):
    columns = list(features.columns)
    current = np.array(current[0]).reshape(1,-1)
    #Select counter group
    if same_group:
        opposing_group = features[features['RiskPerformance'] == prediction[0]].reset_index(drop=True).drop('RiskPerformance', axis=1)
    else:
        opposing_group = features[features['RiskPerformance'] != 'Bad'].reset_index(drop=True).drop('RiskPerformance', axis=1)
    scaler = StandardScaler()
    opposing_group = pd.DataFrame(scaler.fit_transform(opposing_group))
    current = scaler.transform(current)
    dist = np.linalg.norm(opposing_group - current, axis=1)
    idx_min = np.argmin(dist)
    most_similar = np.array(opposing_group.loc[idx_min]).reshape(1, -1)

    if not normalize:
        #Show normalized values
        current = scaler.inverse_transform(current)
        most_similar = scaler.inverse_transform(most_similar)

    #else show original values
    fig = go.Figure(layout=dict(xaxis_title='Features', yaxis_title='Values', title='Comparison to most similar person in '
                                                                                    'group with loan accepted'),
    data=[go.Bar(name = 'Me', x=opposing_group.columns, y=current[0]),
                        go.Bar(name='Other', x=opposing_group.columns, y=most_similar[0])])
    fig.update_layout(barmode='group')
    mini = min(current[0])-1
    maxi = max(current[0])+1
    if mini<0 and maxi<0:
        fig.update_yaxes(range=[mini, 0])
    elif mini<0 and maxi>0:
        fig.update_yaxes(range=[mini, maxi])
    else:
        fig.update_yaxes(range=[0, maxi])
    return dcc.Graph(figure=fig, style={'width': '1000px', 'display': 'inline-block'})

def plot_heatmaps(counts, data, ranges):
    axes = list(combinations(counts, 2))
    heatmaps = []
    for combo in axes:
        tempdata = data[[combo[0], combo[1], 'num_outcome']]
        heatmapdata = tempdata.groupby(list(combo)).mean().unstack()
        axis1 = list(zip(*heatmapdata.columns))[1]
        axis2 = list(heatmapdata.index)

        fig = go.Figure(layout=dict(xaxis_title=f'{list(ranges.keys())[combo[1]]}',
                                    yaxis_title=f'{list(ranges.keys())[combo[0]]}',


                                    title='fraction of Good values'),
                        data=go.Heatmap(
                            z=heatmapdata.values,
                            x=axis1,
                            y=axis2,
                            zmax=1,
                            zmin=0,
                            hoverongaps=False))
        fig.update_layout(width=900,height=900)
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

def CI(values, prediction, model):
    map = {'Good': 1, 'Bad': 0}
    estimators = model.estimators_
    prediction = map[prediction[0]]

    same_classfied = []

    for estimator in estimators:
        new_pred = int(estimator.predict(values))

        if new_pred == prediction:
            same_classfied.append(1)
        else:
            same_classfied.append(0)

    count = sum(same_classfied)
    ci = proportion_confint(count, len(same_classfied))
    ci = (round(ci[0], 3), round(ci[1], 3))
    return ci

def plot_correlations():
    X = features.drop('RiskPerformance', axis=1).corr()
    dfs = []
    for i in np.linspace(0, 1, 21):
        d = X.copy()
        for col in X.keys():
            d[col] = d[col].where(d[col] >= i, other=0)
        dfs.append(d)
    # generate the frames. NB name
    frames = [
        go.Frame(data=go.Heatmap(z=df.values, x=df.columns, y=df.index, colorscale='Blues'), name=i / 20)
        for i, df in enumerate(dfs)
    ]

    corrmap = go.Figure(data=frames[0].data, frames=frames).update_layout(

        # iterate over frames to generate steps... NB frame name...
        sliders=[{"steps": [{"args": [[f.name], {"frame": {"duration": 0, "redraw": True},
                                                 "mode": "immediate", }, ],
                             "label": f.name, "method": "animate", }
                            for f in frames],
                  'currentvalue': {"prefix": "minimum correlation treshold: "}}],
        height=800,
        xaxis={"title": 'Correlation Matrix', "tickangle": 30, 'side': 'top'},
        font ={'size':14},
        title_x=0.5,

    )
    return dcc.Graph(figure=corrmap,style={'width': '1000px', 'display': 'inline-block'})

def plot_violins(column):
    X = features[column]
    X = X.append(X)
    y = features['RiskPerformance']
    y_hat = model.predict(features.drop('RiskPerformance',axis=1))
    y2 = [i if i == j else 'type1' if i == 'Bad' else 'type2' for i, j in zip(y, y_hat)]
    y2 = pd.Series(y2).append(pd.Series(['total']*len(y2)))
    return dcc.Graph(figure=px.violin(y=X, x=y2, color=y2, box=True, points="all", title=f'disribution of {column}'))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    features = pd.read_csv('data/heloc_dataset_v1.csv')
    # the columns that stores the labels
    labelDimension = "RiskPerformance"
    # build a random forest classifier
    model = get_model()
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

    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
    #app.css.append_css({'external_url': 'resetstyle.css'})
    app.layout = html.Div([
        html.Div(id='topbar', children=[
            html.P(id='info', children="Please first enter the client's data  |"),
            html.Div(id='toptext')
            ]),


        html.Div([
                dcc.Checklist(options=rangeSearchChecklist(),
                              id='rangeSearchChecklist',
                              inputStyle={'display': 'none'},
                              labelStyle={'display': 'block', 'height': '22px', 'width': '300px'})],
                 style={'width': '300px', 'display': 'inline-block'}),

        html.Div([j for i in variable_values_ids for j in [html.Div([dcc.Input(id=i, value=-9, type='number',
                                                                     style={'height': '21px',
                                                                            'width': '140px'})],
                                                                    style={'height': '22px',
                                                                           'width': '150px'})]],
                 style={'width': '160px', 'display': 'inline-block'}),



        html.Div(id='percentages',style={'width': '160px', 'display': 'inline-block'}),

        html.Img(id='shapsum', src='data:image/png;base64,{}'.format(plot_Shap_Summary()),
                 style={'width':'625px','margin-left':'100px', 'margin-top':'-490px'}),

        html.Div(id='current_eval'),
        html.Div(id='in-between-counterexample', style={'display': 'none'}),

        html.Div(id='searchbar', children=[
            html.Div(id='global', children=[
                html.P(id='global_text', children='global explainers',style={'display':'inline-block','margin-right':'10px'}),
                html.Div(id='dd_vals_div',children=[
                dcc.Dropdown(rangeSearchChecklist(), placeholder="See feature distribution",
                             id='dd_vals', style={'margin-right': '3px', 'display': 'inline-block', 'width': '350px',
                                                  'height': '30px', 'margin-bottom': '-10px', 'color': '#000000'}),
                html.Button('?', id='Q_feature', n_clicks=0,
                            style={'margin-right': '3px', "font-weight": "bold"}),
                ]),
                html.Div(id='corr_div',children=[
                html.Button('Correlation Matrix', id='button_corr', n_clicks=0, style={'margin-right': '3px'}),
                html.Button('?', id='Q_corr', n_clicks=0,
                            style={'margin-right': '3px', 'border-radius': '50%', "font-weight": "bold"}),
                ]),
                html.Div(id='lda_div', children=[
                html.Button('Show Interactive LDA', id='button_LDA', n_clicks=0, style={'margin-right': '3px'}),
                html.Button('?', id='Q_lda', n_clicks=0,
                            style={'margin-right': '3px', 'border-radius': '50%', "font-weight": "bold"}),
                ]),

                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Linear Discriminant Analysis")),
                        dbc.ModalBody("""This button plots three linear discriminants into a scatterplot matrix using
                        linear discriminant analysis 
                        (https://www.analyticsvidhya.com/blog/2021/08/a-brief-introduction-to-linear-discriminant-analysis/).
                        The plots show values which are true positives (good), true negatives (bad), 
                        type 1 errors (type1), type 2 errors (type2) as well as a bigger dot showing the currently entered
                        client.
                        """),
                    ],
                    id="A_LDA",
                    size="lg",
                    is_open=False
                ),

                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Plot Correlations")),
                        dbc.ModalBody("""This button plots a correlation matrix 
                        (https://www.displayr.com/what-is-a-correlation-matrix/) between all the features with
                        a threshold slider where everything below the threshold is set to 0."""),
                    ],
                    id="A_corr",
                    size="lg",
                    is_open=False
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Plot Feature Distribution")),
                        dbc.ModalBody("""This dropdown menu allows you to select a feature and generate a violinplot
                         (https://towardsdatascience.com/violin-plots-explained-fb1d115e023d) for the feature and each
                         subgroup in the feature."""),
                    ],
                    id="A_feature",
                    size="lg",
                    is_open=False
                )
            ]),
            html.Div(id='local', children=[
                html.P(id='local_text', children='local explainers',style={'display':'inline-block','margin-right':'10px'}),

                html.Div(id='sim_div',children=[
                dcc.Dropdown(['Normalized', 'Standard'], placeholder='Most Similar', id='button_sim',
                             style={'margin-right': '3px', 'display': 'inline-block', 'width': '350px',
                                    'height': '30px', 'margin-bottom': '-10px', 'color': '#000000'}),
                html.Button('?', id='Q_sim', n_clicks=0,
                            style={'margin-right': '3px', 'border-radius': '50%', "font-weight": "bold"}),
                ]),
                html.Div(id='lime_div',children=[
                html.Button('Perform LIME', id='button_LIME', n_clicks=0, style={'margin-right': '3px'}),
                html.Button('?', id='Q_lime', n_clicks=0,
                            style={'margin-right': '3px', 'border-radius': '50%', "font-weight": "bold"}),
                ]),
                html.Div(id='shap_div',children=[
                html.Button('Perform SHAP', id='button_SHAP', n_clicks=0, style={'margin-right': '3px'}),
                html.Button('?', id='Q_shap', n_clicks=0,
                            style={'margin-right': '3px', 'border-radius': '50%', "font-weight": "bold"}),
                ]),
                html.Div(id='grid_div',children=[
                html.Button('Run Grid Search', id='button_counterexample_run', n_clicks=0,
                            style={'margin-right': '3px'}),
                html.Button('?', id='Q_grid', n_clicks=0,
                            style={'margin-right': '3px', 'border-radius': '50%', "font-weight": "bold"}),
                ]),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Grid Search")),
                        dbc.ModalBody("""Runs in two stages. In stage 1 you select 2-5 different features using the
                        checkboxes that appear after clicking the button. For each of these features you must also 
                        include a range to search in either percentages of the total range (a value between 0 and 1)
                        or an absolute amount of steps (an integer value). In stage 2 you press the button again and
                        a grid search will run over each combination of feature values within the range. It then 
                        prints a heatmap for each combination of two factors where the color indicates the fraction of
                        the two values combined with the other features will result in a good result."""),
                    ],
                    id="A_grid",
                    size="lg",
                    is_open=False
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Lime")),
                        dbc.ModalBody("""Performs a LIME analysis 
                        (https://towardsdatascience.com/understanding-model-predictions-with-lime-a582fdff3a3b)
                         using the client's data """),
                    ],
                    id="A_lime",
                    size="lg",
                    is_open=False
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Shap")),
                        dbc.ModalBody("""Performs a SHAP analysis 
                        (https://www.vantage-ai.com/blog/burning-down-the-black-box-of-ml-using-shap)
                         using the client's data"""),
                    ],
                    id="A_shap",
                    size="lg",
                    is_open=False
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Similar")),
                        dbc.ModalBody("""Plots a bargraph with the current client's data and the most similar other data
                                      point in the dataset"""),
                    ],
                    id="A_sim",
                    size="lg",
                    is_open=False
                ),
            ])

        ]),

        # store information of current client
        dcc.Store(id='store_person'),
        # store a counter
        html.Button(id='tally', n_clicks=0, style={'display':'none'}),


        # store plots and only show last one
        dcc.Store(id='heatmaps_plt', data=[None, -10]),
        dcc.Store(id='LIME_plt', data=[None, -10]),
        dcc.Store(id='SHAP_plt', data=[None, -10]),
        dcc.Store(id='LDA_plt', data=[None, -10]),
        dcc.Store(id='Corr_plt', data=[None, -10]),
        dcc.Store(id='Val_plt', data=[None, -10]),
        dcc.Store(id='sim_plt', data=[None, -10]),
        dcc.Store(id='lime_plt', data=[None, -10]),
        dcc.Store(id='shap_plt', data=[None, -10]),
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
        Output('current_eval', 'children'),
        Input('store_person', 'data'),
    )
    def GetPersonEval(current):
        prediction = model.predict(current)
        ci = CI(current, prediction, model)
        return f'Output: {prediction} with 95% confidence interval: {ci}'


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
                              style={'display': 'block', 'height': '21px', 'width': '140px',
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
        Output('heatmaps_plt','data'),
        Output('rangeSearchChecklist', 'inputStyle'),
        Output('button_counterexample_run', 'style'),
        Output('button_counterexample_run', 'children'),
        Output('button_counterexample_run', 'n_clicks'),
        Output('rangeSearchChecklist', 'value'),
        Output('toptext', 'children'),
        Input('store_person','data'),
        Input(component_id="button_counterexample_run",component_property='n_clicks'),
        Input('rangeSearchChecklist', 'value'),
        Input('percentages', 'children'),
        Input('tally','n_clicks')
    )
    def counterExampleSearch(current, button, checklist, children, tally, ):
        if button == 1:
            return [0, tally],\
                   {'display': 'inline-block'},\
                   {'color': 'gold', 'margin-right': '3px'},\
                   'select columns', \
                   1,\
                   dash.no_update, \
                   ["Fill in between 2-5 items to check the range, all original values and % of total range to check"]

        if button != 2:
            raise exceptions.PreventUpdate

        else:
            prediction = model.predict(current)
            ci = CI(current, prediction, model)


            if len(checklist) > 5 or len(checklist) < 2:
                return [0, tally], {'display':'none'}, \
                       {'color': '#FFFFFF', 'margin-right': '3px'}, 'Run Grid Search', dash.no_update, [], []

            #Get all percentage ranges
            percentages = []
            for i in range(len(children)):
                    percentages.append(children[i]['props']['children'][0]['props']['value'])

            # create a grid with all the data for a gridsearch
            if len(np.array(percentages)[np.array(percentages)<0]) > 0:
                #No negative ranges
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
            return [html.Div(heatmaps), tally+1], {'display': 'none'}, \
                   {'color': '#FFFFFF', 'margin-right': '3px'}, 'Run Grid Search', 0, [], []

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
        n_df = ldfa_df.copy()
        n_df.loc[len(n_df.index)] = {'LD1': person_trans[0],
                                           'LD2': person_trans[1],
                                           'LD3': person_trans[2],
                                           'size': 10,
                                           'labels': 'Client'}

        labels = {
            str(i): f"LD {i + 1} ({var:.1f}%)"
            for i, var in enumerate(lda_model.explained_variance_ratio_ * 100)
        }
        fig_lda = px.scatter_matrix(
            n_df,
            labels=labels,
            size='size',
            size_max=5,
            dimensions=lda_dims,
            color='labels',
        )
        fig_lda.update_traces(diagonal_visible=False)
        fig_lda.update_traces(marker=dict(line=dict(width=0)))

        if checklist:
            fig_lda.for_each_trace(
                lambda trace: trace.update(visible='legendonly') if trace.name in checklist else ()
            )

        return [html.Div(dcc.Graph(figure=fig_lda,style={'width':'1600px','height':'800px'})), tally+1], 0

    @app.callback(
        Output('Corr_plt', 'data'),
        Output('button_corr', 'n_clicks'),
        Input('button_corr', 'n_clicks'),
        Input('tally', 'n_clicks'),

    )
    def plot_corr_matrix(button, tally):
        if button == 0:
            raise exceptions.PreventUpdate
        return [plot_correlations(), tally+1], 0

    @app.callback(
        Output('Val_plt', 'data'),
        Output('dd_vals', 'value'),
        Input('dd_vals', 'value'),
        Input('tally', 'n_clicks')
    )
    def plot_vals(vals, tally):
        if not vals:
            raise exceptions.PreventUpdate
        else:
            return [plot_violins(vals), tally+1], []

    @app.callback(
        Output('sim_plt', 'data'),
        Output('button_sim', 'value'),
        Input('store_person', 'data'),
        Input('button_sim', 'value'),
        Input('tally', 'n_clicks'),
    )
    def plot_similar(person, button, tally):
        if not button:
            raise exceptions.PreventUpdate
        if button != 'Normalized':
            return [plot_most_similar(person, normalize=False), tally+1], 0
        return [plot_most_similar(person), tally+1], 0

    @app.callback(
        Output('lime_plt','data'),
        Output('button_LIME','n_clicks'),
        Input('store_person', 'data'),
        Input('button_LIME', 'n_clicks'),
        Input('tally','n_clicks')
    )
    def plot_lime(person, button, tally):
        if button == 0:
            raise exceptions.PreventUpdate
        return [plot_Lime(person), tally+1], 0

    @app.callback(
        Output('shap_plt','data'),
        Output('button_SHAP','n_clicks'),
        Input('store_person', 'data'),
        Input('button_SHAP', 'n_clicks'),
        Input('tally','n_clicks')
    )
    def plot_shap(person, button, tally):
        if button == 0:
            raise exceptions.PreventUpdate
        return [plot_Shap(person), tally+1], 0

    @app.callback(
        Output('tally', 'n_clicks'),
        Output('plot', 'children'),
        Input('heatmaps_plt', 'data'),
        Input('LIME_plt', 'data'),
        Input('SHAP_plt', 'data'),
        Input('LDA_plt', 'data'),
        Input('Corr_plt', 'data'),
        Input('Val_plt', 'data'),
        Input('sim_plt', 'data'),
        Input('lime_plt', 'data'),
        Input('shap_plt','data')
    )
    def ShowPlot(heatmap, LIME, SHAP, LDA, corr, vals, sim, lime, shap):

        curr_tally = -10
        curr_plot = None
        for plot, tally in [heatmap, LIME, SHAP, LDA, corr, vals, sim, lime, shap]:
            if tally > curr_tally:
                curr_tally = tally
                curr_plot = plot

        return curr_tally, curr_plot


    @app.callback(
        Output("A_LDA", "is_open"),
        Input("Q_lda", "n_clicks"),
        State("A_LDA", "is_open"),
    )
    def toggle_modal(n1, is_open):
        if n1:
            return not is_open
        return is_open
    @app.callback(
        Output("A_corr", "is_open"),
        Input("Q_corr", "n_clicks"),
        State("A_corr", "is_open"),
    )
    def toggle_modal(n1, is_open):
        if n1:
            return not is_open
        return is_open

    @app.callback(
        Output("A_feature", "is_open"),
        Input("Q_feature", "n_clicks"),
        State("A_feature", "is_open"),
    )
    def toggle_modal(n1, is_open):
        if n1:
            return not is_open
        return is_open

    @app.callback(
        Output("A_grid", "is_open"),
        Input("Q_grid", "n_clicks"),
        State("A_grid", "is_open"),
    )
    def toggle_modal(n1, is_open):
        if n1:
            return not is_open
        return is_open

    @app.callback(
        Output("A_lime", "is_open"),
        Input("Q_lime", "n_clicks"),
        State("A_lime", "is_open"),
    )
    def toggle_modal(n1, is_open):
        if n1:
            return not is_open
        return is_open

    @app.callback(
        Output("A_shap", "is_open"),
        Input("Q_shap", "n_clicks"),
        State("A_shap", "is_open"),
    )
    def toggle_modal(n1, is_open):
        if n1:
            return not is_open
        return is_open

    @app.callback(
        Output("A_sim", "is_open"),
        Input("Q_sim", "n_clicks"),
        State("A_sim", "is_open"),
    )
    def toggle_modal(n1, is_open):
        if n1:
            return not is_open
        return is_open

    app.run_server(debug=True)
