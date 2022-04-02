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
            pass

        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(features),
            feature_names=features.columns,
            class_names=['Bad', 'Good'],
            mode='classification')
        pickle.dump(lime_explainer, open('lime.sav', 'wb') )
    return lime_explainer

def lime_explain(model, input_instance):
    lime_exp = model.explain_instance(
        num_features=30,
        data_row=input_instance,
        predict_fn=model.predict_proba
    )
    lime_exp.save_to_file('lime_explain.html', show_table=True)
    lime_exp.as_pyplot_figure().savefig('lime_explain.png')

def get_shap_model(model):
    try:
        shap_model = pickle.load(open('shap.sav', 'rb'))
    except:
        shap_model = shap.TreeExplainer(model)
        pickle.dump(shap_model, open('shap.sav', 'wb'))
    return shap_model

def get_shap_values(shap_explainer, features):
    try:
        shap_values_all = pickle.load(open("shap.sav", "rb"))
    except:
        shap_values_all = shap_explainer.shap_values(features)
        pickle.dump(shap_values_all, open('shap.sav', 'wb'))
    return shap_values_all

def calculate_shap_value(explainer, data):
    """"this function is for calculating a single shap value"""
    shapvalue = explainer.shap_values(data)
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



