import pickle
from lime import lime_tabular
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def get_lime_model(data):
    """"This function receives a set of data and builds a lime model on it.
    This lime model is returned"""
    try: # if we already made the model, we load it instead of creating it
        lime_explainer = pickle.load(open('lime.sav', 'rb'))
    except:
        try: # if the target variable is still in the data, we try to drop it.
            features = data.drop(['RiskPerformance'], axis=1)
        except:
            features = data

        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(features),
            feature_names=features.columns,
            class_names=['Bad', 'Good'],
            mode='classification')
        pickle.dump(lime_explainer, open('lime.sav', 'wb') )
    return lime_explainer

def lime_explain(limemodel, model, input_values):
    """
    :param limemodel: a model created by get_lime_model()
    :param model: a model created by get_model()
    :param input_values: the values created by the input on the dash interface
    :return: a lime_explain.jpg is created, which can be shown in the dash interface
    """

    input_instance = np.array(input_values[0])  # reformat the input values so we can use them
    lime_exp = limemodel.explain_instance(
        num_features=30,
        data_row=input_instance,
        predict_fn=model.predict_proba
    )
    limeplot = lime_exp.as_pyplot_figure()
    limeplot.subplots_adjust(left=0.15)
    limeplot.set_size_inches(23, 10)
    limeplot.savefig('lime_explain.jpg')
    plt.clf()  # clear the figure so it won't interfere later
    print('lime explained')

def get_shap_model(model):
    """"This function takes a random forest model and returns a shap model"""
    try:  # we try to load the model if we already created it before.
        shap_model = pickle.load(open('shap.sav', 'rb'))
    except:  # if it hasn't been created before, we create it and save it.
        shap_model = shap.TreeExplainer(model)
        pickle.dump(shap_model, open('shap.sav', 'wb'))
    return shap_model

def get_shap_values(shap_explainer, features):
    """"this function creates shap values for an entire dataset, which takes a long while"""
    try:  # we load the values if they were already calculated before.
        shap_values_all = pickle.load(open("shap.pikl", "rb"))
    except:  # if it hasn't been calculated before, we calculate them and save it in a file.
        shap_values_all = shap_explainer.shap_values(features)
        pickle.dump(shap_values_all, open('shap.pikl', 'wb'))
    return shap_values_all

def calculate_shap_value(explainer, input_values):
    """"this function is for calculating a single shap value,
     using a shap explainer and the input values for a client"""
    input_instance = np.array(input_values[0])
    shapvalue = explainer.shap_values(input_instance)
    return shapvalue

def shap_summary_plot(shap_values, features):
    """"This function uses the list of all shap values and the features to create the shap summary plot"""
    shap.summary_plot(shap_values, features, plot_type='violin', show=False)
    plt.savefig('shap_summary.png')

class OG_explainer():
    """"Because the shap package is poorly built, we created this class to be an old version of the explainer,
    which can be used by shap.plots.waterfall"""
    def __init__(self, explainer, shap_values, data, featurenames=None):
        self.base_values = explainer.expected_value.mean()
        self.data = data
        self.values = shap_values
        if type(data) == pd.core.series.Series:
            self.feature_names = data._index
        # elif type(data) == pd.core.series.DataFrame: #assuming it is a dataframe
        #     self.feature_names = data.columns
        else:
            self.feature_names = featurenames

def shap_waterfall_plot(explainer, shap_values, data, featurenames):
    """"This function takes the different parts of the shap explainer.
    Then it plots the shap values in a waterfall plot, saves it so it can be loaded in the dash interface.
    """
    inputvalues = np.array(data[0])  # reformat the input values so we can use them
    thing = OG_explainer(explainer, shap_values[0], inputvalues, featurenames)
    # Use the OG_explainer object to transform all the inputs so it is usable by the waterfall function
    shap.plots.waterfall(thing, show=False)
    plt.gcf().set_size_inches(30, 10)
    plt.savefig('shap_waterfall.png')
    plt.clf()  # clear the figure so it won't interfere later
    print('shap explained')


