import pickle
from lime import lime_tabular
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def get_lime_model(data):
    try:
        lime_explainer = pickle.load(open('lime.sav', 'rb'))
    except:
        try:
            features = data.drop(['RiskPerformance'], axis=1)
        except:
            features = data
        # y_lime = features['RiskPerformance']

        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(features),
            feature_names=features.columns,
            class_names=['Bad', 'Good'],
            mode='classification')
        # pickle.dump(lime_explainer, open('lime.sav', 'wb') )
    return lime_explainer

def lime_explain(limemodel, model, input_values):
    input_instance = np.array(input_values[0])
    lime_exp = limemodel.explain_instance(
        num_features=30,
        data_row=input_instance,
        predict_fn=model.predict_proba
    )
    lime_exp.as_pyplot_figure().savefig('lime_explain.jpg')
    print('lime explained')

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

def calculate_shap_value(explainer, input_values):
    """"this function is for calculating a single shap value"""
    input_instance = np.array(input_values[0])
    shapvalue = explainer.shap_values(input_instance)
    print('shap value calculated')
    return shapvalue

def shap_summary_plot(shap_values, features):
    shap.summary_plot(shap_values, features, plot_type='violin', show=False)
    plt.savefig('shap_summary.png')

class OG_explainer():
    def __init__(self, explainer, shap_values, data):
        self.base_values = explainer.expected_value.mean()
        self.data = data
        self.values = shap_values
        if type(data) == pd.core.series.Series:
            self.feature_names = data._index
        else: #assuming it is a dataframe
            self.feature_names = data.columns

def shap_waterfall_plot(explainer, shap_values, data):
    thing = OG_explainer(explainer, shap_values[0], data)
    shap.plots.waterfall(thing)
    plt.savefig('shap_waterfall.png')
    print('shap explained')

