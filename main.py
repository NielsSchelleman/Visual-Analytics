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
import Lime_shap
import base64
from sklearn.preprocessing import Normalizer
from sklearn.calibration import CalibratedClassifierCV

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

def giveRanges():
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


def getRanges(percentages, current_vals):
    # Indices of chosen variables
    non_zero_idx = np.nonzero(np.array(percentages))

    # Keep only variables which ranges we want to change
    variable_list = np.array(rangeSearchChecklist())[non_zero_idx]
    current_vals = np.array(current_vals).flatten()[non_zero_idx]
    percentages = np.array(percentages)[non_zero_idx]

    # acceptable search spaces, assume you cannot go from some value to any special value eg. 26 --> -9
    ranges = giveRanges()
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
    # just gets the names of the features
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
    """Creates a grid based on the current person and entered ranges for the gridsearch

    current: the current person
    ranges: the range to check, entered in the gui
    checklist: all the selected columns

    returns: a grid of point combinations to search in grid search"""
    gridbase = []
    # counts checks if the correct number of ranges were entered (2 through 5)
    counts = []
    # create a range for each column
    for nr, (val, name, valrange) in enumerate(zip(current[0], ranges.keys(), ranges.values())):
        if name in checklist:
            counts.append(nr)
            gridbase.append(valrange)
        else:
            gridbase.append(val)
    # create a grid of all combinations of the ranges
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
    scaler = Normalizer()
    dropped = features.drop('RiskPerformance', axis=1)
    scaler.fit(dropped)
    #Select counter group
    if same_group:
        opposing_group = features[features['RiskPerformance'] == prediction[0]].reset_index(drop=True).drop('RiskPerformance', axis=1)
    else:
        opposing_group = features[features['RiskPerformance'] != 'Bad'].reset_index(drop=True).drop('RiskPerformance', axis=1)

    scaled_opposing_group = scaler.transform(opposing_group)
    scaled_opposing_group = pd.DataFrame(scaled_opposing_group)
    scaled_current = scaler.transform(current)
    dist = np.linalg.norm(scaled_opposing_group - scaled_current, axis=1)
    idx_min = np.argmin(dist)


    if not normalize:
        #Show original values
        current = current
        most_similar = np.array(opposing_group.loc[idx_min]).reshape(1, -1)
        title = "Comparison to most similar person in group with loan accepted"
        mini = min(current[0]) - 1
        maxi = max(current[0]) + 1
    else:
        most_similar = np.array(scaled_opposing_group.loc[idx_min]).reshape(1, -1)
        current = scaled_current
        title = "Normalized comparison to most similar person in group with loan accepted"
        mini = -1 if (np.amin(current) < 0) else 0
        maxi = 1

    #else show normalized values
    fig = go.Figure(layout=dict(xaxis_title='Features', yaxis_title='Values', title=title),
    data=[go.Bar(name = 'Me', x=opposing_group.columns, y=current[0]),
                        go.Bar(name='Other', x=opposing_group.columns, y=most_similar[0])])
    fig.update_layout(barmode='group')

    if mini<0 and maxi<0:
        fig.update_yaxes(range=[mini, 0])
    elif mini<0 and maxi>0:
        fig.update_yaxes(range=[mini, maxi])
    else:
        fig.update_yaxes(range=[0, maxi])
    return dcc.Graph(figure=fig, style={'width': '1000px', 'display': 'inline-block'})

def plot_heatmaps(counts, data, ranges):
    """Plots the heatmaps generated by the gridsearch"""
    axes = list(combinations(counts, 2))
    heatmaps = []
    # plot a heatmap for each combination of two features.
    for combo in axes:
        tempdata = data[[combo[0], combo[1], 'num_outcome']]
        heatmapdata = tempdata.groupby(list(combo)).mean().unstack()
        axis1 = list(zip(*heatmapdata.columns))[1]
        axis2 = list(heatmapdata.index)
        #generates the heatmap
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
        # keep the figure square as to not give the impression of one axis being more important than another
        fig.update_layout(width=900,height=900)
        heatmaps.append(dcc.Graph(figure=fig))
    return heatmaps

def prep_lda(features):
    """Performs LDA on the features"""
    # get features
    X = features.drop('RiskPerformance', axis=1)
    y = features['RiskPerformance']
    y_hat = model.predict(X)

    lda = LDA(n_components=3)

    # get a feature showing the true positives, negatives and type1, type2 errors.
    y2 = [i if i == j else 'type1' if i == 'Bad' else 'type2' for i, j in zip(y, y_hat)]
    dims = ['LD1', 'LD2', 'LD3']
    components = lda.fit_transform(X, y2)
    comp_df = pd.DataFrame(components, columns=dims)
    comp_df['size'] = 1
    comp_df['labels'] = y2
    return comp_df, lda, dims

def plot_correlations():
    """plots the correlations of the features including a threshold."""
    X = features.drop('RiskPerformance', axis=1).corr()
    dfs = []
    # create a frame for each possible value for the threshold so that they can be shown on the fly
    for i in np.linspace(0, 1, 21):
        d = X.copy()
        for col in X.keys():
            d[col] = d[col].where(d[col] >= i, other=0)
        dfs.append(d)
    # generate the frames.
    frames = [
        go.Frame(data=go.Heatmap(z=df.values, x=df.columns, y=df.index, colorscale='Blues'), name=i / 20)
        for i, df in enumerate(dfs)
    ]
    # create the figure using the frames
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
    # set width to 1000 to avoid it becoming way too large
    return dcc.Graph(figure=corrmap,style={'width': '1000px', 'display': 'inline-block'})

def plot_violins(column):
    """Takes a feature and creates a violinplot"""
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
    app.layout = html.Div([
        # the bar on the top
        html.Div(id='topbar', children=[
            html.P(id='info', children="Please first enter the client's data  |"),
            html.Div(id='toptext')
            ]),

        # the names of all the features on the left, including the boxes that appear when you select the grid search.
        html.Div([
                dcc.Checklist(options=rangeSearchChecklist(),
                              id='rangeSearchChecklist',
                              inputStyle={'display': 'none'},
                              labelStyle={'display': 'block', 'height': '22px', 'width': '300px'})],
                 style={'width': '300px', 'display': 'inline-block'}),

        # the inputs where you can fill in the values for each feature
        html.Div([j for i in variable_values_ids for j in [html.Div([dcc.Input(id=i, value=-9, type='number',
                                                                     style={'height': '21px',
                                                                            'width': '140px'})],
                                                                    style={'height': '22px',
                                                                           'width': '150px'})]],
                 style={'width': '160px', 'display': 'inline-block'}),


        # the inputs that appear whenever you select the grid search that allows you to fill in the ranges
        html.Div(id='percentages',style={'width': '160px', 'display': 'inline-block'}),

        # the global shap plot
        html.Img(id='shapsum', src='data:image/png;base64,{}'.format(plot_Shap_Summary()),
                 style={'width':'625px','margin-left':'100px', 'margin-top':'-490px'}),
        html.Button('?', id='Q_shap_sum', n_clicks=0,
                    style={'margin-right': '3px', 'border-radius': '50%', "font-weight": "bold"}),
        # the text that shows the current evaluation and confidence interval
        html.Div(id='current_eval'),
        html.Button('?', id='Q_eval', n_clicks=0,
                    style={'margin-right': '3px', 'border-radius': '50%', "font-weight": "bold"}),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Model confidence")),
                dbc.ModalBody("""This line shows the model output for the given customer values. Aligned with this output the model returns
                probabilities belonging to the chance that this customer belongs to that class. Higher probability means that the model
                is more certain that this customer belongs to that group.
             """),
            ],
            id="A_eval",
            size="lg",
            is_open=False
        ),
        html.Div(id='in-between-counterexample', style={'display': 'none'}),

        # the big navbar with the buttons on it
        html.Div(id='searchbar', children=[
            # the top row of the navbar
            html.Div(id='global', children=[
                # text box
                html.P(id='global_text', children='global explainers',style={'display':'inline-block','margin-right':'10px'}),
                # the first button for generating feature selection violin plots
                html.Div(id='dd_vals_div',children=[
                dcc.Dropdown(rangeSearchChecklist(), placeholder="See feature distribution",
                             id='dd_vals', style={'margin-right': '3px', 'display': 'inline-block', 'width': '350px',
                                                  'height': '30px', 'margin-bottom': '-10px', 'color': '#000000'}),
                # these are the questionmarks that give the info pop-ups
                html.Button('?', id='Q_feature', n_clicks=0,
                            style={'margin-right': '3px', "font-weight": "bold"}),
                ]),
                # the second button for generating the correlation matrix
                html.Div(id='corr_div',children=[
                html.Button('Correlation Matrix', id='button_corr', n_clicks=0, style={'margin-right': '3px'}),
                html.Button('?', id='Q_corr', n_clicks=0,
                            style={'margin-right': '3px', 'border-radius': '50%', "font-weight": "bold"}),
                ]),
                # the third button for generating the LDA plot
                html.Div(id='lda_div', children=[
                html.Button('Show Interactive LDA', id='button_LDA', n_clicks=0, style={'margin-right': '3px'}),
                html.Button('?', id='Q_lda', n_clicks=0,
                            style={'margin-right': '3px', 'border-radius': '50%', "font-weight": "bold"}),
                ]),
                # code for the pop-ups you see when you press any of the ?'s
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
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Shap summary")),
                        dbc.ModalBody("""In this plot you can see the impact on the model output of each variable.
                    A variable that has points far away from the middle has more impact on the output of the model.
                    Lower values are plotted as blue, whereas higher values are plotted as red.
                    This should help you get an understanding of which variables have a big impact when they change.
                    
            """),
                    ],
                    id="A_shap_sum",
                    size="lg",
                    is_open=False
                )
            ]),
            # bottom row of the navbar
            html.Div(id='local', children=[
                # text box
                html.P(id='local_text', children='local explainers',style={'display':'inline-block','margin-right':'10px'}),
                # the first button for generating the most similar person
                html.Div(id='sim_div',children=[
                dcc.Dropdown(['Normalized', 'Standard'], placeholder='Most Similar', id='button_sim',
                             style={'margin-right': '3px', 'display': 'inline-block', 'width': '350px',
                                    'height': '30px', 'margin-bottom': '-10px', 'color': '#000000'}),
                html.Button('?', id='Q_sim', n_clicks=0,
                            style={'margin-right': '3px', 'border-radius': '50%', "font-weight": "bold"}),
                ]),
                # the second button for generating the LIME plot
                html.Div(id='lime_div',children=[
                html.Button('Perform LIME', id='button_LIME', n_clicks=0, style={'margin-right': '3px'}),
                html.Button('?', id='Q_lime', n_clicks=0,
                            style={'margin-right': '3px', 'border-radius': '50%', "font-weight": "bold"}),
                ]),
                # the third button for generating the SHAP plot
                html.Div(id='shap_div',children=[
                html.Button('Perform SHAP', id='button_SHAP', n_clicks=0, style={'margin-right': '3px'}),
                html.Button('?', id='Q_shap', n_clicks=0,
                            style={'margin-right': '3px', 'border-radius': '50%', "font-weight": "bold"}),
                ]),

                # the fourth button for generating a partial dependence plot on a feature
                html.Div(id='pdp_div', children=[
                    dcc.Dropdown(rangeSearchChecklist(), placeholder="Show Feature impact",
                                 id='pdp_vals',
                                 style={'margin-right': '3px', 'display': 'inline-block', 'width': '350px',
                                        'height': '30px', 'margin-bottom': '-10px', 'color': '#000000'}),
                    html.Button('?', id='Q_pdp', n_clicks=0,
                                style={'margin-right': '3px', 'border-radius': '50%', "font-weight": "bold"}),
                ]),


                # the fifth button for generating the grid search heatmaps
                html.Div(id='grid_div',children=[
                html.Button('Run Grid Search', id='button_counterexample_run', n_clicks=0,
                            style={'margin-right': '3px'}),
                html.Button('?', id='Q_grid', n_clicks=0,
                            style={'margin-right': '3px', 'border-radius': '50%', "font-weight": "bold"}),
                ]),
                # code for the pop-ups you see when you press any of the ?'s
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
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Feature Impact")),
                        dbc.ModalBody("""For the selected feature, this shows a partial dependence plot of the feature
                        https://christophm.github.io/interpretable-ml-book/pdp.html with respect to the confidence of
                        the model"""),
                    ],
                    id="A_pdp",
                    size="lg",
                    is_open=False
                ),
            ])

        ]),

        # store information of current client
        dcc.Store(id='store_person'),
        # store a counter
        html.Button(id='tally', n_clicks=0, style={'display':'none'}),


        # store plots and a tally to keep track of the last plot the person requested to see
        dcc.Store(id='heatmaps_plt', data=[None, -10]),
        dcc.Store(id='LIME_plt', data=[None, -10]),
        dcc.Store(id='SHAP_plt', data=[None, -10]),
        dcc.Store(id='LDA_plt', data=[None, -10]),
        dcc.Store(id='Corr_plt', data=[None, -10]),
        dcc.Store(id='Val_plt', data=[None, -10]),
        dcc.Store(id='sim_plt', data=[None, -10]),
        dcc.Store(id='lime_plt', data=[None, -10]),
        dcc.Store(id='shap_plt', data=[None, -10]),
        dcc.Store(id='pd_plt', data=[None, -10]),

        #plot into here
        html.Div(id='misc_persist', children=[
            dcc.Checklist(id='lda_checklist',
                          options=['Bad','Good','Type1','Type2','Client'],
                          style={'display': 'none'})
        ]),
        html.Div(id='plot'),

    ])

    # callback that generates the model prediction and corresponding confidence interval for the current person.
    @app.callback(
        Output('current_eval', 'children'),
        Input('store_person', 'data'),
    )
    def GetPersonEval(current):
        prediction = model.predict(current)
        calibrated_model = CalibratedClassifierCV(base_estimator=model, cv='prefit')
        y = features[labelDimension]
        X = features.drop(labelDimension, axis=1)

        calibrated_model.fit(X, y)

        calibrated_prop = calibrated_model.predict_proba(current)
        return f'Output: {prediction} with class probabilities: Bad: {round(calibrated_prop[0][0], 3)}, Good: {round(calibrated_prop[0][1], 3)}'


    # callback that generates the input boxes for when you need to fill in the ranges after pressing the grid search.
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

    # callback that stores the data for the person inside a container
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

    # callback that creates the heatmaps for the grid search
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
        # on the first press allow a person to select features and change the button to indicate it was pressed
        if button == 1:
            return [0, tally],\
                   {'display': 'inline-block'},\
                   {'color': 'gold', 'margin-right': '3px'},\
                   'select columns', \
                   1,\
                   dash.no_update, \
                   ["Fill in between 2-5 items to check the range, all original values and % of total range to check"]
        # if the button is not pressed or spam pressed don't do anything
        if button != 2:
            raise exceptions.PreventUpdate
        # check if the right amount of features has been pressed otherwise do nothing and reset to initial state
        else:
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

            # plot the heatmaps and show it, also return the button to the initial state
            heatmaps = plot_heatmaps(counts, newdata, ranges)
            return [html.Div(heatmaps), tally+1], {'display': 'none'}, \
                   {'color': '#FFFFFF', 'margin-right': '3px'}, 'Run Grid Search', 0, [], []

    # callback for the LDA plot
    @app.callback(
        Output('LDA_plt','data'),
        Output('button_LDA', 'n_clicks'),
        Input('store_person', 'data'),
        Input('button_LDA', 'n_clicks'),
        Input('tally', 'n_clicks'),
        Input('lda_checklist', 'value')
    )
    def LDA_plotting(person, button, tally, checklist):
        # if the button has not been pressed, do nothing
        if button == 0:
            raise exceptions.PreventUpdate

        # use the LDA model to find the position of the person in the LDA space
        person_trans = lda_model.transform(pd.DataFrame(np.array(person[0]).reshape(1, 23), columns=features.keys()[1:]))[0]
        n_df = ldfa_df.copy()
        # create a new trace for the person
        n_df.loc[len(n_df.index)] = {'LD1': person_trans[0],
                                           'LD2': person_trans[1],
                                           'LD3': person_trans[2],
                                           'size': 10,
                                           'labels': 'Client'}
        # plot the figure for the lda model
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

    # callback for plotting the correlations
    @app.callback(
        Output('Corr_plt', 'data'),
        Output('button_corr', 'n_clicks'),
        Input('button_corr', 'n_clicks'),
        Input('tally', 'n_clicks'),

    )
    def plot_corr_matrix(button, tally):
        # plot the correlations if the button has been pressed
        if button == 0:
            raise exceptions.PreventUpdate
        return [plot_correlations(), tally+1], 0

    # callback for the violinplots for each feature
    @app.callback(
        Output('Val_plt', 'data'),
        Output('dd_vals', 'value'),
        Input('dd_vals', 'value'),
        Input('tally', 'n_clicks')
    )
    def plot_vals(vals, tally):
        # plot the violinplot for a feature if a feature was selected
        if not vals:
            raise exceptions.PreventUpdate
        else:
            return [plot_violins(vals), tally+1], []

    # callback for the similarity plot
    @app.callback(
        Output('sim_plt', 'data'),
        Output('button_sim', 'value'),
        Input('store_person', 'data'),
        Input('button_sim', 'value'),
        Input('tally', 'n_clicks'),
    )
    def plot_similar(person, button, tally):
        # plot the person along with the most similar person when a value is selected
        if not button:
            raise exceptions.PreventUpdate
        # either plot the scaled variant or not depending on what the user selects
        if button != 'Normalized':
            return [plot_most_similar(person, normalize=False), tally+1], 0
        return [plot_most_similar(person), tally+1], 0

    # callback for the LIME plot
    @app.callback(
        Output('lime_plt','data'),
        Output('button_LIME','n_clicks'),
        Input('store_person', 'data'),
        Input('button_LIME', 'n_clicks'),
        Input('tally','n_clicks')
    )
    def plot_lime(person, button, tally):
        # plot the lime plot if the button has been pressed
        if button == 0:
            raise exceptions.PreventUpdate
        return [plot_Lime(person), tally+1], 0

    # callback for the SHAP plot
    @app.callback(
        Output('shap_plt','data'),
        Output('button_SHAP','n_clicks'),
        Input('store_person', 'data'),
        Input('button_SHAP', 'n_clicks'),
        Input('tally','n_clicks')
    )
    def plot_shap(person, button, tally):
        # plot the shap plot if the button has been pressed
        if button == 0:
            raise exceptions.PreventUpdate
        return [plot_Shap(person), tally+1], 0

    # callback for the partial dependence plot
    @app.callback(
        Output('pd_plt', 'data'),
        Output('pdp_vals', 'value'),
        Input('store_person', 'data'),
        Input('pdp_vals', 'value'),
        Input('tally', 'n_clicks'),
    )
    def plot_pdp(person, button, tally):
        # plot the person along with the most similar person when a value is selected
        if not button:
            raise exceptions.PreventUpdate
        # either plot the scaled variant or not depending on what the user selects
        calibrated_model = CalibratedClassifierCV(base_estimator=model, cv='prefit')
        y = features[labelDimension]
        X = features.drop(labelDimension, axis=1)
        calibrated_model.fit(X, y)
        name = [i for i,feature in enumerate(rangeSearchChecklist()) if feature == button][0]
        ax1 = giveRanges()[button]
        biglist = []
        for i in ax1:
            p2 = person[0].copy()
            p2[name] = i
            biglist.append(p2)
        calibrated_prop = calibrated_model.predict_proba(biglist)
        pdp_frame = pd.DataFrame(calibrated_prop,columns=['confidence bad','confidence good'])
        pdp_frame = pdp_frame.melt()
        pdp_frame[button] = list(ax1) * 2
        pdp_frame = pdp_frame.rename(columns={'value':'confidence','variable':'legend'})
        fig = px.line(pdp_frame, x=button, y="confidence", color='legend')
        return [dcc.Graph(id='pdp_graph',figure= fig), tally+1],[]




    # callback for only showing the plot corresponding to the last button the user pressed
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
        Input('shap_plt','data'),
        Input('pd_plt','data')
    )
    def ShowPlot(heatmap, LIME, SHAP, LDA, corr, vals, sim, lime, shap, pdp):

        curr_tally = -10
        curr_plot = None
        # check for each plot stored in one of the stores if its tally value is the largest, if it is, plot that plot
        for plot, tally in [heatmap, LIME, SHAP, LDA, corr, vals, sim, lime, shap, pdp]:
            if tally > curr_tally:
                curr_tally = tally
                curr_plot = plot

        return curr_tally, curr_plot

    # callbacks for the pop-ups
    # when the button is pressed, show the pop-up
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

    @app.callback(
        Output("A_shap_sum", "is_open"),
        Input("Q_shap_sum", "n_clicks"),
        State("A_shap_sum", "is_open"),
    )
    def toggle_modal(n1, is_open):
        if n1:
            return not is_open
        return is_open

    @app.callback(
        Output("A_eval", "is_open"),
        Input("Q_eval", "n_clicks"),
        State("A_eval", "is_open"),
    )
    def toggle_modal(n1, is_open):
        if n1:
            return not is_open
        return is_open

    @app.callback(
        Output("A_pdp", "is_open"),
        Input("Q_pdp", "n_clicks"),
        State("A_pdp", "is_open"),
    )
    def toggle_modal(n1, is_open):
        print(n1)
        if n1:
            return not is_open
        return is_open

    app.run_server(debug=True)
