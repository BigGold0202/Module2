# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 15:37:06 2019

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
import numpy as np

bus_data = pd.read_json(r'C:\Frank Zhou\UM-MADISON\4\628\Module2\business_train.json', lines=True, orient='records')
review_data = pd.read_csv(r'C:\Frank Zhou\UM-MADISON\4\628\Module2\rev_clean.csv')

def process(text):
    text = str(text)
    temp = re.sub("[^a-zA-Z.]",' ',text)
    temp = re.sub('\s+',' ',temp)
    temp = temp.lower()
    temp = temp.split()
    return temp
bus_categories = bus_data.categories.apply(process)

brun_id = [bus_data.business_id[i] for i in range(len(bus_categories)) if 'brunch' in bus_categories[i]]
brunch_col = [i for i in range(len(bus_categories)) if 'brunch' in bus_categories[i]]
brun_data = bus_data.iloc[brunch_col, ].reset_index(drop=True)
brun_col = [i for i in range(review_data.shape[0]) if review_data.business_id[i] in brun_id]
brun_review = review_data.iloc[brun_col, ].reset_index(drop=True)

def check_word(text):
    new_review = []
    text = str(text)
    text = re.sub("[:!]",".",text)
    text = text.split(".")
    for sentence in text:
        if word in sentence:
            temp = sentence.split(" ")
            for w in temp:
                if w == word:
                    new_review.append(sentence)
    return new_review

word_list = ['sandwich','fry','cheese','salad','pancake','burger','bacon','potato','waffle','sauce','bread','dessert','steak','cream','taco','meat','cake','crepe','beef','benedict']
for word in word_list:
    brun_review_id = brun_review.business_id
    brun_review_id = list(brun_review_id)
    word_review = brun_review.text.apply(check_word)
    word_review = list(word_review)

    brun_word_review = [brun_review_id,word_review]
    brun_word_review = DataFrame(brun_word_review)
    brun_word_review = brun_word_review.T
    brun_word_review.rename(columns={0:'busi_id',1:'review'},inplace=True)

    review_temp = brun_word_review

    dele = []
    for i in range(review_temp.shape[0]):
        if review_temp.review[i] == []:
            dele.append(i)

    brunch_review = review_temp.drop(index=dele).reset_index(drop=True)

    brunch_id = brunch_review.busi_id
    brunch_id = list(brunch_id)
    brunch_id = set(brunch_id)
    brunch_id = list(brunch_id)

    lst = ['i' for n in range(len(brunch_id))]
    word_id_review = DataFrame([brunch_id,lst])
    word_id_review = word_id_review.T
    word_id_review.rename(columns={0:'business_id',1:'review'},inplace=True)

    j = 0
    for id_temp in brunch_id:
        id_word_review = []
        id_col = [i for i in range(brunch_review.shape[0]) if brunch_review.busi_id[i] == id_temp]
        id_review = brunch_review.iloc[id_col, ].reset_index(drop=True)
        for r in id_review.review:
            id_word_review.append(r)
            word_id_review.review[j] = id_word_review
            j = j + 1

    word_id_review.to_csv(r"C:\Frank Zhou\UM-MADISON\4\628\Module2\data\word" + word + '.csv')

