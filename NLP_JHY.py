import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.util import mark_negation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# =====read data===============================================
# bdata = open(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\business_train.json').read()
busi_data = pd.read_json(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\business_train.json', orient='records', lines=True)
# busi_data.columns: Index(['attributes', 'business_id', 'categories', 'city', 'hours', 'is_open',
#                           'latitude', 'longitude', 'name', 'postal_code', 'state'], dtype='object')
rev_data = pd.read_json(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\review_sample.json', lines=True, orient='records')
# rev_data.columns: Index(['business_id', 'date', 'stars', 'text'], dtype='object')
bus_set = set(busi_data.name)

# select starbucks
stbk = busi_data.loc[busi_data.name == 'Starbucks', ].reset_index(drop=True)
stbk_rev = rev_data.loc[rev_data.business_id.isin(stbk.business_id)].reset_index(drop=True)
# check dist.
plt.hist(rev_data.stars)
set(rev_data.stars)
# three labels check
a = rev_data.text.loc[rev_data.stars.isin([4, 5]), ].head(10)
b = rev_data.text.loc[rev_data.stars.isin([3]), ].head(10)
c = rev_data.text.loc[rev_data.stars.isin([1, 2]), ].head(10)

c_tk = [nltk.word_tokenize(rev) for rev in c]
c_tk = [mark_negation(rev) for rev in c_tk]

# ====relabel to three===========================================
rev_data['label'] = [0] * len(rev_data)
for i in range(len(rev_data)):
    if rev_data.stars[i] in [4,5]:
        rev_data.label[i] = 2
    elif rev_data.stars[i] in [4,5]:
        rev_data.label[i] = 1

# hard to derive two classes out of one class...

# ====use 5 labels=================================================
tfvec1 = TfidfVectorizer()
tfvec1_3 = TfidfVectorizer(ngram_range=(1, 3))

# mark negation
rev_mkn = [mark_negation(nltk.word_tokenize(i)) for i in rev_data.text]

tfvec.fit()