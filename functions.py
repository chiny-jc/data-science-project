''' -------------------------------------- FUNCTIONS -------------------------------------- '''
import re
import numpy as np


import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string 

from re import search 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

from pdpbox import pdp, get_dataset, info_plots
from sklearn.inspection import permutation_importance
from time import time
from rfpimp import *
import shap


def add_unique_elements(list_to_iterate, set_to_add):
    for element in list_to_iterate:
        set_to_add.add(element)
        
def regex_lookup(column, regex_pattern, match_only=True):
    matches = []
    idx = []
    for i in range(len(column) - 1):
        match = re.search(regex_pattern, column.iloc[i])
        if match != None:
            matches.append(match.group(0))
            idx.append(i)
    
    if match_only:
        return matches
    else:
        return column.iloc[idx]
    
def calculate_distance_between_coordinates(first_coordinates, second_coordinates):
    meters_per_coordinate_degree = 111.139
    latitude_difference_in_km = abs(first_coordinates[0] - second_coordinates[0]) * meters_per_coordinate_degree
    longitude_difference_in_km = abs(first_coordinates[1] - second_coordinates[1]) * meters_per_coordinate_degree
    linear_distance = np.sqrt(latitude_difference_in_km ** 2 + longitude_difference_in_km ** 2)
    return linear_distance


def text_cleaner(text):
    text = text.replace(".", "")
    text = text.replace("'", "")
    text = text.replace(":", "")
    text = text.replace(";", "")
    text = text.replace("?", "")
    text = text.replace("!", "")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\"", "")
    text = text.replace("-", " ")
    text = text.replace("=", "")
    text = text.replace("1", "")
    text = text.replace("2", "")
    text = text.replace("3", "")
    text = text.replace("4", "")
    text = text.replace("5", "")
    text = text.replace("6", "")
    text = text.replace("7", "")
    text = text.replace("8", "")
    text = text.replace("9", "")
    text = text.replace("0", "")
    text = text.replace("%", "")
    text = text.replace("<", "")
    text = text.replace(">", "")
    text = text.replace("&", "")
    text = text.replace("§", "")
    text = text.replace("/", "")
    text = text.replace("+", "")
    text = text.replace("*", "")
    text = text.replace("#", "")
    text = text.replace("br", "")
    

    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    return text.lower()


'''------------------------Interpretable Machine Learning--------------------------'''

def ploting_pdp (f):
    ''' Function for ploting PDP '''
    pdp_surv = pdp.pdp_isolate(model=rf, dataset=X_train, model_features=X_train.columns, feature=f, cust_grid_points=None)
    pdp.pdp_plot(pdp_surv, 'price')
    plt.show()


def two_dim_pdp(f):
    ''' Function for plotting a two dimension PDP'''
    inter= pdp.pdp_interact(model=rf, dataset=X_train, model_features=X_train.columns, features=f)
    pdp.pdp_interact_plot(pdp_interact_out=inter, feature_names=f, plot_type='grid')   
    plt.show()


def shap_summary(model, train_set):
    ''' Summary plot of SHAP Values'''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_set)
    res = shap.summary_plot(shap_values, train_set)
    return res


def dependency_contribution(f, i):
    ''' Dependency contribution plots '''
    explainer = shap.TreeExplainer(rf)  
    shap_values = explainer.shap_values(X_train)  
    res = shap.dependence_plot(f, shap_values[i], X_train)
    return res



def forceplots(row_to_show, interest_class):
    ''' Functions for forceplots '''
    data_for_prediction = X_train.iloc[row_to_show]
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
    rf.predict_proba(data_for_prediction_array)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(data_for_prediction)
    shap.initjs()
    res = shap.force_plot(explainer.expected_value[interest_class], shap_values[interest_class], data_for_prediction)
    return res



'''-------------------------- Selecting the Top Correlations ------------------------------'''

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
