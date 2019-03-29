# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:42:52 2019

@author: Dell
"""

import json
import pandas as pd
import nltk
import re
from pandas.core.frame import DataFrame
bus_data = pd.read_json(r'C:\Frank Zhou\UM-MADISON\4\628\Module2\business_train.json', lines=True, orient='records')
review_data = pd.read_json(r'C:\Frank Zhou\UM-MADISON\4\628\Module2\review_train.json', lines=True, orient='records')


brun_id = [i for i in range(len(bus_sp)) if 'Brunch' in bus_sp[i]]
brun_data = bus_data.iloc[brun_id, ].reset_index(drop=True)
brun_col = [i for i in range(review_data.shape[0]) if review_data.business_id[i] in brun_id]
brun_review = review_data.iloc[brun_col, ].reset_index(drop=True)


rest_id = [i for i in range(len(bus_sp)) if 'Restaurants' in bus_sp[i]]
rest_data = bus_data.iloc[rest_id, ].reset_index(drop=True)
rest_col = [i for i in range(review_data.shape[0]) if review_data.business_id[i] in rest_id]
rest_review = review_data.iloc[rest_col, ].reset_index(drop=True)


punc_use = ['!','!!','!!!',':)','!!!!','?!']

def keep_punc(review):
    punc_kept = []
    temp = re.sub('[a-zA-Z0123456789]', ' ', review)
    temp = re.sub('\s+',' ',temp)
    for w in temp.split(" "):
        if w in punc_use:
            punc_kept.append(w)
    return punc_kept
punc_kept = review_data.text.apply(keep_punc)

def process(text):
    temp = re.sub("[^a-zA-Z.]",' ',text)
    temp = re.sub('\s+',' ',temp)
    temp = temp.lower()
    temp = temp.split()
    return temp
text_step1 = review_data.text.apply(process)

text_step1.append(punc_kept)

