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
from multiprocessing import Pool
import re

# =====read data===============================================
# bdata = open(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\business_train.json').read()
busi_data = pd.read_json(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\business_train.json', orient='records', lines=True)
# busi_data.columns: Index(['attributes', 'business_id', 'categories', 'city', 'hours', 'is_open',
#                           'latitude', 'longitude', 'name', 'postal_code', 'state'], dtype='object')
rev_data = pd.read_json(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\review_train.json', lines=True, orient='records')
# rev_data.columns: Index(['business_id', 'date', 'stars', 'text'], dtype='object')

# check dist.
plt.hist(rev_data.stars)
set(rev_data.stars)
# three labels check
a = rev_data.text.loc[rev_data.stars.isin([4, 5]), ].head(10)
b = rev_data.text.loc[rev_data.stars.isin([3]), ].head(10)
c = rev_data.text.loc[rev_data.stars.isin([1, 2]), ].head(10)

c_tk = [nltk.word_tokenize(rev) for rev in c]
c_tk = [mark_negation(rev) for rev in c_tk]


# =====lemmatizer=============================================================
def wordnet_pos(x):
    if x.startswith('V'):
        return wordnet.VERB
    elif x.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def sent_lmt(x):   # have trouble with double negation
    x = re.sub(',', '.', x)
    lmtzer = WordNetLemmatizer()
    word_tag = nltk.pos_tag(nltk.word_tokenize(x))
    lmt_word = [lmtzer.lemmatize(i_pair[0], pos=wordnet_pos(i_pair[1])) for i_pair in word_tag]
    lmt_word = mark_negation(lmt_word)
    sent = ' '.join(lmt_word)
    return sent


for i in range(rev_data.shape[0]):
    rev_data.text.iloc[i] = sent_lmt(rev_data.text.iloc[i])
    if i % 100 == 0:
        print('%d / %d' % (i, rev_data.shape[0]))

toy_sample = rev_data.iloc[0:10, ].copy()
toy_sample.to_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\toy.csv', index=False)
for i in range(10):
    toy_sample.text.iloc[i] = sent_lmt(toy_sample.text.iloc[i])

# =====mark negation================================================================

rev_mkn = [mark_negation(nltk.word_tokenize(i)) for i in rev_data.text]
rev_mkn = [" ".join(rev) for rev in rev_mkn]
# ======find burnch review===============================================================
# bus_sp = [busi_data.categories[i].split(", ") for i in range(busi_data.shape[0])
# if busi_data.categories[i] is not None]
bus_sp = []
for i in range(busi_data.shape[0]):
    if busi_data.categories[i] is not None:
        bus_sp.append(nltk.word_tokenize(busi_data.categories[i]))
    else:
        bus_sp.append('0')

# bus_sp = [nltk.word_tokenize(busi_data.categories[i]) for i in range(busi_data.shape[0])
# if busi_data.categories[i] is not None]
brun_id = [i for i in range(len(bus_sp)) if 'Brunch' in bus_sp[i]] # 160796
brun_data = busi_data.iloc[brun_id, ]
brun_rev = rev_data.loc[rev_data.business_id.isin(brun_data.business_id)].reset_index(drop=True)


# =====use business data===============================================================
# compare difference in attributes
brun_sc = [np.mean(brun_rev.stars[brun_rev.business_id == i]) for i in brun_data.business_id]
brun_ct = [np.sum(brun_rev.business_id == i) for i in brun_data.business_id]
# =====clean daata==============================================================
