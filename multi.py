import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.util import mark_negation
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from multiprocessing import Pool
import re

toy_sample = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\toy.csv')


def wordnet_pos(x):
    if x.startswith('V'):
        return wordnet.VERB
    elif x.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def sent_lmt(x):   # have trouble with double negation
    x.text = re.sub(',', '.', x.text)
    lmtzer = WordNetLemmatizer()
    word_tag = nltk.pos_tag(nltk.word_tokenize(x.text))
    lmt_word = [lmtzer.lemmatize(i_pair[0], pos=wordnet_pos(i_pair[1])) for i_pair in word_tag]
    lmt_word = mark_negation(lmt_word)
    sent = ' '.join(lmt_word)
    x.text = sent
    return x


def multi_rev(data):
    for i in range(data.shape[0]):
        data.text.iloc[i] = sent_lmt(data.text.iloc[i])
    return data


num_partitions = 10 # number of partitions to split dataframe
num_cores = 10 # number of cores on your machine


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


parallelize_dataframe(toy_sample, multi_rev).to_csv()
