import pickle
from lime import lime_tabular
import numpy as np
import pandas as pd
import shap


def get_model():
    try:
        raise the_roof # purely to check if the except statement works, remove later
        model = pickle.load(open('rf_mod.sav', 'rb'))
    except:
        from Model import buildModel

        # load in the dataset
        features = pd.read_csv('heloc_dataset_v1.csv')

        # the columns that stores the labels
        labelDimension = "RiskPerformance"

        # build a random forest classifier
        model = buildModel(features, labelDimension)

        pickle.dump(model, open('rf_mod.sav', 'wb')) #todo check if this works
    return model

def get_lime_model(data):
    try:
        lime_explainer = pickle.load(open('lime.sav', 'rb')) #todo check if this works
    except:
        features = data.drop(['RiskPerformance'], axis=1)
        # y_lime = features['RiskPerformance']

        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(features),
            feature_names=features.columns,
            class_names=['Bad', 'Good'],
            mode='classification')
        pickle.dump(lime_explainer, open('lime.sav', 'wb') ) # todo check if this works
    return lime_explainer

def lime_explain(model, input_instance):
    lime_exp = model.explain_instance(
        num_features=30,
        data_row=input_instance,
        predict_fn=model.predict_proba
    )
    lime_exp.save_to_file('lime_explain.html',show_table=True)
    lime_exp.as_pyplot_figure().savefig('lime_explain.png')

def get_shap_model(model):
    try:
        shap_model = pickle.load(open('shap.sav'), 'rb') #todo check if this works
    except:
        shap_model = shap.TreeExplainer(model)
        pickle.dump(shap_model, open('shap.sav', 'wb')) #todo check if this works
    return shap_model

def get_shap_values(shap_explainer, features):
    try:
        shap_values_all = pickle.load(open("shap.pikl", "rb"))
    except:
        shap_values_all = shap_explainer.shap_values(features)
        pickle.dump(shap_values_all, open('shap.pikl', 'wb'))
    return shap_values_all

def calculate_shap_value(explainer, data):
    """"this function is for calculating a single shap value"""
    shapvalue = explainer.shap_values(data)
    return shapvalue

def shap_summary_plot(shap_values, features):
    shap.summary_plot(shap_values, features, plot_type='violin', show=False)

class OG_explainer():
    def __init__(self, explainer, shap_values, data):
        self.base_values = explainer.expected_value
        self.expected_value = explainer.expected_value[0]
        self.data = data
        self.values = shap_values
        if type(data) == pd.core.series.Series:
            self.feature_names = data._index
        else: #assuming it is a dataframe
            self.feature_names = data.columns

def shap_waterfall_plot(explainer, shap_values, data):
    thing = OG_explainer(explainer, shap_values[0], data)
    shap.plots.waterfall(thing)



