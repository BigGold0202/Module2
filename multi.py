import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.util import mark_negation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from multiprocessing import Pool
import re
import time

rev_data = pd.read_json(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\review_sample.json', lines=True, orient='records')


def wordnet_pos(x):
    if x.startswith('V'):
        return wordnet.VERB
    else:
        return wordnet.NOUN


def sent_lmt(x):   # have trouble with double negation, input a df
    x = x.lower()
    x = re.sub(',', '.', x)
    lmtzer = WordNetLemmatizer()
    word_tag = nltk.pos_tag(nltk.word_tokenize(x))
    lmt_word = [lmtzer.lemmatize(i_pair[0], pos=wordnet_pos(i_pair[1])) for i_pair in word_tag]
    lmt_word = mark_negation(lmt_word)
    sent = ' '.join(lmt_word)
    x = sent
    return x


def multi_rev(data):
    data.text = data.text.apply(sent_lmt)
    return data


num_cores = 10  # number of cores on your machine


def parallelize_dataframe(df, func):
    df = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df))
    pool.close()
    pool.join()
    return df


if __name__ == '__main__':
    print('start')
    start = time.time()
    rev_post = parallelize_dataframe(rev_data, multi_rev)
    end = time.time()
    print('done')
    print(end - start)
    rev_post.to_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\sample_post.csv', index=False)
