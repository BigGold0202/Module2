import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.util import mark_negation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from multiprocessing import Pool
import re
import time
from nltk.corpus import stopwords


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
    # tokenizer = RegexpTokenizer(r'\w+')
    x = x.lower()
    x = re.sub(',', '.', x)
    # word = tokenizer.tokenize(x)
    word = nltk.word_tokenize(x)
    word = mark_negation(word)
    word = [i for i in word if i not in stopword]
    word_tag = nltk.pos_tag(word)
    lmt_word = [lmtzer.lemmatize(i_pair[0], pos=wordnet_pos(i_pair[1])) for i_pair in word_tag]
    lmt_word = " ".join(lmt_word)
    return lmt_word


def multi_rev(data):
    data.text = data.text.apply(sent_tokenize)
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
    rev_data = pd.read_json(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\review_sample.json', lines=True,
                            orient='records')
    print('done reading ')

    print('start cleaning ')
    start = time.time()
    rev_data = parallelize_dataframe(rev_data, multi_rev)
    end = time.time()
    print('done')
    print(end - start)
    rev_data.to_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\sample_cleancsv', index=False)
