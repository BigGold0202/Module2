# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:42:52 2019

@author: Dell
"""

import json
import pandas as pd
import nltk
import re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.util import mark_negation
from pandas.core.frame import DataFrame
bus_data = pd.read_json(r'C:\Frank Zhou\UM-MADISON\4\628\Module2\business_train.json', lines=True, orient='records')
review_data = pd.read_json(r'C:\Frank Zhou\UM-MADISON\4\628\Module2\review_train.json', lines=True, orient='records')
'''
def process(text):
    text = text.lower()
    temp = re.sub("\,",'.',text)
    temp = re.findall('[a-zA-Z]+|:\)|\.\.\.+|[!]+|\!\?|\.',temp)
    return temp
review_sp = review_data.text.apply(process)
'''

def wordnet_pos(x):
    if x.startswith('V'):
        return wordnet.VERB
    else:
        return wordnet.NOUN


def sent_tokenize(x):   # have trouble with double negation, input a df
    '''
    stopword = set(stopwords.words('english')) - {'he', 'him', 'his', 'himself',
                                                  'she', 'her', "she's", 'her', 'hers', 'herself',
                                                'they', 'them', 'their', 'theirs', 'themselves'}
    ''' 
    lmtzer = WordNetLemmatizer()
    x = x.lower()
    temp = re.sub("\,",'.',x)
    word = re.findall('[a-zA-Z]+|:\)|\.\.\.+|[!]+|\!\?|\.',temp)
    word = mark_negation(word)
    #word = [i for i in word if i not in stopword]
    word_tag = nltk.pos_tag(word)
    lmt_word = [lmtzer.lemmatize(i_pair[0], pos=wordnet_pos(i_pair[1])) for i_pair in word_tag]
    return lmt_word

review_sp = review_data.text.apply(sent_tokenize)


brun_id = [i for i in range(len(bus_sp)) if 'Brunch' in bus_sp[i]]
brun_data = bus_data.iloc[brun_id, ].reset_index(drop=True)
brun_col = [i for i in range(review_data.shape[0]) if review_data.business_id[i] in brun_id]
brun_review = review_data.iloc[brun_col, ].reset_index(drop=True)
brun_star = brun_review['stars'].groupby(brun_review['business_id']).mean()

rest_id = [i for i in range(len(bus_sp)) if 'Restaurants' in bus_sp[i]]
rest_data = bus_data.iloc[rest_id, ].reset_index(drop=True)
rest_col = [i for i in range(review_data.shape[0]) if review_data.business_id[i] in rest_id]
rest_review = review_data.iloc[rest_col, ].reset_index(drop=True)
rest_star = rest_review['stars'].groupby(brun_review['business_id']).mean()



