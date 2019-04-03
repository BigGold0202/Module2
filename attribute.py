import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.util import mark_negation
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import re
import ast
import sys
from sklearn.feature_selection import SelectKBest, chi2
from nltk.tokenize import RegexpTokenizer
from multi_cleaning import sent_tokenize
import string
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import scale, Normalizer


busi_data = pd.read_json(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\business_train.json', orient='records', lines=True)
brun_rev = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\brun_review.csv')


def slice_dict(x):
    out = {}
    if x is None:
        out = None
    else:
        for key, value in x.items():
            if type(value) is str:
                value = ast.literal_eval(value)
            if type(value) is dict:
                out = dict(out, **slice_dict(value))
            else:
                out[key] = value
    return out


def sum_dict(x):
    out = {}
    if x is None:
        out = None
    else:
        for key, value in x.items():
            if type(value) is str:
                value = ast.literal_eval(value)
            if type(value) is dict:
                value = sum(list(value.values()))
            out[key] = value
    return out


def atb_process(x, method):
    if method == 'slice':
        atb_slice = [pd.DataFrame(slice_dict(i), index=[0]) for i in x]
        atb_slice_df = pd.concat(atb_slice, sort='False')
        slice_stat = pd.Series([np.mean(atb_slice_df.iloc[:, i].notnull()) for i in range(atb_slice_df.shape[1])])
        slice_stat.index = atb_slice_df.columns
        slice_stat.sort_values(ascending=False, inplace=True)
        return atb_slice_df, slice_stat
    if method == 'sum':
        atb_sum = [pd.DataFrame(sum_dict(i), index=[0]) for i in x]
        atb_sum_df = pd.concat(atb_sum, sort='False')
        sum_stat = pd.Series([np.mean(atb_sum_df.iloc[:, i].notnull()) for i in range(atb_sum_df.shape[1])])
        sum_stat.index = atb_sum_df.columns
        sum_stat.sort_values(ascending=False, inplace=True)
        return atb_sum_df, sum_stat


bus_sp = []
for i in range(busi_data.shape[0]):
    if busi_data.categories[i] is not None:
        bus_sp.append(nltk.word_tokenize(busi_data.categories[i]))
    else:
        bus_sp.append('0')

brun_id = [i for i in range(len(bus_sp)) if 'Brunch' in bus_sp[i]]  # 160796
brun_data = busi_data.iloc[brun_id, ]

brun_slice, brun_slice_stat = atb_process(brun_data.attributes, 'slice')
brun_slice['business_id'] = brun_data.business_id.values

brun_sum, brun_sum_stat = atb_process(brun_data.attributes, 'sum')

brun_star = brun_rev.groupby(['business_id'], as_index=False).agg('mean')
brun_slice = brun_slice.merge(brun_star, how='right')


# =============rf selection======================================================
def rf_select(x, y, word, n_estimators=100):
    sel = RandomForestRegressor(n_estimators=n_estimators, verbose=2, n_jobs=4)
    sel.fit(x, y)

    imp_sc = sel.feature_importances_
    ind = np.argsort(-1 * imp_sc)
    imp_sc = imp_sc[ind]
    word = word[ind]
    res = pd.DataFrame({'word': word, 'score': imp_sc})
    return res


le = LabelEncoder()
atb_x = brun_slice.drop(['business_id', 'stars'], axis=1)
atb_x.fillna(np.nan, inplace=True)

for i in range(atb_x.shape[1]):
    atb_x.iloc[:, i] = atb_x.iloc[:, i].astype(str)
    atb_x.iloc[:, i] = le.fit_transform(atb_x.iloc[:, i])


atb_rf = rf_select(atb_x, brun_slice.stars, atb_x.columns, 100)

