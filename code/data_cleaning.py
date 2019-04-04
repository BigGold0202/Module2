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

#Calculate average star rates for brunch restaurants and all restaurants
def devide(text):
    text = str(text)
    temp = re.sub("[^a-zA-Z.]",' ',text)
    temp = re.sub('\s+',' ',temp)
    temp = temp.lower()
    temp = temp.split()
    return temp
bus_categories = bus_data.categories.apply(devide)

brun_id = [bus_data.business_id[i] for i in range(len(bus_categories)) if 'brunch' in bus_categories[i]]
brunch_col = [i for i in range(len(bus_categories)) if 'brunch' in bus_categories[i]]
brun_data = bus_data.iloc[brunch_col, ].reset_index(drop=True)
brun_col = [i for i in range(review_data.shape[0]) if review_data.business_id[i] in brun_id]
brun_review = review_data.iloc[brun_col, ].reset_index(drop=True)#pick all brunch restaurants reviews
brun_star = brun_review['stars'].groupby(brun_review['business_id']).mean()

rest_id = [bus_data.business_id[i] for i in range(len(bus_categories)) if 'restaurants' in bus_sp[i]]
restaurant_col = = [i for i in range(len(bus_categories)) if 'restaurants' in bus_categories[i]]
rest_data = bus_data.iloc[rest_col, ].reset_index(drop=True)#pick all restaurants
rest_col = [i for i in range(review_data.shape[0]) if review_data.business_id[i] in rest_id]
rest_review = review_data.iloc[rest_col, ].reset_index(drop=True)#pick all restaurants reviews
rest_star = rest_review['stars'].groupby(brun_review['business_id']).mean()


#Select all the punctuations in the review
def check_punc(review):
    punc_temp = []
    temp_text = re.sub('[a-zA-Z0123456789]', ' ', review)#delete all the letters and numbers
    punc = re.sub('\s+',' ',temp_text)
    for p in punc.split(" "):
        if len(p) >= 1:
            punc_temp.append(p)
    return punc_temp
punc_process = review_data.text.apply(check_punc)

punc=[]
for p in punc_process:
    punc.extend(p)

punc = Counter(punc)
punc = dict(punc)
sort_punc = sorted(punc.items(), key=lambda d:d[1], reverse = True)#Count the occurence number for all punctuations

punc_use = ['!','!!','!!!',':)','!!!!','?!','.']

#Keep all "!"s; ":)"; "?!"; "."; "..." in the reviews
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
    
    stopword = set(stopwords.words('english')) - {'he', 'him', 'his', 'himself',
                                                  'she', 'her', "she's", 'her', 'hers', 'herself',
                                                'they', 'them', 'their', 'theirs', 'themselves'}
    
    lmtzer = WordNetLemmatizer()
    x = x.lower()
    temp = re.sub("\,",'.',x)
    word = re.findall('[a-zA-Z]+|:\)|\.\.\.+|[!]+|\!\?|\.',temp)
    word = mark_negation(word)
    word = [i for i in word if i not in stopword]
    word_tag = nltk.pos_tag(word)
    lmt_word = [lmtzer.lemmatize(i_pair[0], pos=wordnet_pos(i_pair[1])) for i_pair in word_tag]
    return lmt_word

review_sp = review_data.text.apply(sent_tokenize)

