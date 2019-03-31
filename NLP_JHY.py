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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import scale, Normalizer

# =====read data===============================================
# bdata = open(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\business_train.json').read()
busi_data = pd.read_json(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\business_train.json', orient='records', lines=True)
# busi_data.columns: Index(['attributes', 'business_id', 'categories', 'city', 'hours', 'is_open',
#                           'latitude', 'longitude', 'name', 'postal_code', 'state'], dtype='object')
rev_data = pd.read_json(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\review_sample.json', lines=True, orient='records')
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


# ==========official=======================================================
def wordnet_pos(x):
    if x.startswith('V'):
        return wordnet.VERB
    else:
        return wordnet.NOUN


stopword = set(stopwords.words('english')) - {'he', 'him', 'his', 'himself',
                                              'she', 'her', "she's", 'her', 'hers', 'herself',
                                              'they', 'them', 'their', 'theirs', 'themselves'}
lmtzer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')


def sent_lmt(x):   # have trouble with double negation, input a df
    x = x.lower()
    x = re.sub(',', '.', x)
    word = tokenizer.tokenize(x)
    word = [i for i in word if i not in stopword]
    word_tag = nltk.pos_tag(word)
    lmt_word = [lmtzer.lemmatize(i_pair[0], pos=wordnet_pos(i_pair[1])) for i_pair in word_tag]
    lmt_word = mark_negation(lmt_word)
    return lmt_word


x = "he doesn't like it, but he likes that. he doesn't like it anyway"
# ============================================================================================
for i in range(rev_data.shape[0]):
    rev_data.text.iloc[i] = sent_lmt(rev_data.text.iloc[i])
    if i % 100 == 0:
        print('%d / %d' % (i, rev_data.shape[0]))

toy_sample = rev_data.iloc[0:10000, ].copy()
toy_sample.to_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\toy.csv', index=False)
x = toy_sample.text.iloc[0]


# =====mark negation================================================================
# ======find burnch review===============================================================
bus_sp = []
for i in range(busi_data.shape[0]):
    if busi_data.categories[i] is not None:
        bus_sp.append(nltk.word_tokenize(busi_data.categories[i]))
    else:
        bus_sp.append('0')


brun_id = [i for i in range(len(bus_sp)) if 'Brunch' in bus_sp[i]]  # 160796
brun_data = busi_data.iloc[brun_id, ]
brun_rev = rev_data.loc[rev_data.business_id.isin(brun_data.business_id)].reset_index(drop=True)


# =====use business data===============================================================
# compare difference in attributes
brun_sc = [np.mean(brun_rev.stars[brun_rev.business_id == i]) for i in brun_data.business_id]
brun_ct = [np.sum(brun_rev.business_id == i) for i in brun_data.business_id]

# =====attribute clean daata==============================================================
atb = busi_data.attributes
a = [dict(key=value) for key, value in atb[1].items()]


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


x = atb[10]
key, value = list(x.items())[3]
x0 = slice_dict(x)
x1 = sum_dict(x)
[print(sys.getsizeof(i)) for i in [x, x0, x1]]


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


brun_slice, brun_slice_stat = atb_process(brun_data.attributes, 'slice')
brun_sum, brun_sum_stat = atb_process(brun_data.attributes, 'sum')


# =====Selcetion================================================================
def split_token(x):
    return x.split(' ')


rev_data = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\sample_clean.csv', nrows=10000)
counts = vec.fit_transform(rev_data.text)
counts_sum = np.array(counts.sum(axis=0))[0]
counts = counts[:, counts_sum >= np.quantile(counts_sum, 0.8)]  # np.quantile(counts_sum, 0.8) = 5

counts_01 = counts
counts_ind = counts.nonzero()
counts_01[counts_ind] = 1


counts_columns = np.array(vec.get_feature_names())[counts_sum >= np.quantile(counts_sum, 0.8)]
star = rev_data.stars

chi_select = SelectKBest(chi2, k=200)
chi_count = chi_select.fit_transform(counts, star)
chi_columns = counts_columns[chi_select.get_support()]

chi_select, chi_p = chi2(counts, star)
chi_res = pd.DataFrame({'word': counts_columns, 'score': chi_select, 'p-value': chi_p})
chi_res = chi_res.sort_values(by='score', ascending=False).iloc[:200, :]

minfo = mutual_info_classif(counts, star, discrete_features=True)
minfo_res = pd.DataFrame({'word': counts_columns, 'minfo': minfo})
minfo_res = minfo_res.sort_values(by='minfo', ascending=False).iloc[:200, :]

rf = RandomForestClassifier()
rf.fit(counts, star)
rf_res = pd.DataFrame({'word': counts_columns, 'rf_score': rf.feature_importances_})
rf_res = rf_res.sort_values(by='rf_score', ascending=False).iloc[:200, :]

# =====bi gram=============================================================================
vec2 = CountVectorizer(ngram_range=(2, 2), max_features=2000)
bicounts = vec2.fit_transform(rev_data.text)
bi_column = vec2.get_feature_names()
bi_sum = np.array(bicounts.sum(axis=0))[0]
bi_res = pd.DataFrame({'word': bi_column, 'counts': bi_sum})
bi_res.sort_values(by='counts', inplace=True, ascending=False)


# =====test/train counts==============================================================================
rev_test = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\test_clean.csv')
rev_test.text[rev_test.text.isna()] = 'na'

rev_train = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\rev_clean.csv', low_memory=False)
rev_train.text[rev_train.text.isna()] = 'na'
star = rev_train.stars

vec = CountVectorizer(tokenizer=split_token, stop_words=['.', 'not'])
test_ct = vec.fit_transform(rev_test.text)
test_ct_word = vec.get_feature_names()
test_ct_sum = np.array(test_ct.mean(axis=0))[0]
# plt.hist(test_ct_sum[(0.001> test_ct_sum) & (test_ct_sum>0.0001)])
test_ct_res = pd.DataFrame({'word': test_ct_word, 'count_sum': test_ct_sum})
test_ct_res.sort_values(by='count_sum', ascending=False, inplace=True)
# np.quantile(test_count_sum, [0.85, 0.9, 0.95, 0.96, 0.975]) = 7, 14, 53, 80, 191
# ind = test_ct_sum > np.quantile(test_ct_sum, n)

ind = test_ct_sum >= 0.0005
test_ct = test_ct[:, ind]
test_ct_word = np.array(test_ct_word)[ind]
np.savetxt('test_ct_word.csv', test_ct_word, header='word', fmt='%s')
del test_ct_sum

test_ct_word = np.array(pd.read_csv('test_ct_word.csv')).reshape(-1)
train_ct = vec.fit_transform(rev_train.text)
train_ct_word = vec.get_feature_names()

tfi = TfidfVectorizer(tokenizer=split_token, stop_words=['.', 'not'])
train_tfi = tfi.fit_transform(rev_train.text)
train_tfi_word = tfi.get_feature_names()

del rev_train

test_word = np.array(list(set(test_ct_word) & set(train_ct_word)))

ind = np.isin(train_ct_word, test_word)
train_ct = train_ct[:, ind]
train_ct_word = np.array(train_ct_word)[ind]


# ===============test/train tfidf===========================================================
test_tfi = tfi.fit_transform(rev_test.text)
test_tfi_word = tfi.get_feature_names()
ind = np.isin(test_tfi_word, test_word)
test_tfi = test_tfi[:, ind]
test_tfi_word = np.array(test_tfi_word)[ind]


ind = np.isin(train_tfi_word, test_word)
train_tfi = train_tfi[:, ind]
train_tfi_word = np.array(train_tfi_word)[ind]


# ======training selection==========================================================================
def var_selection(stat, word, n):

    chi_select, chi_p = chi2(stat, star)
    chi_res = pd.DataFrame({'word': word, 'score': chi_select, 'p-value': chi_p})
    chi_res = chi_res.sort_values(by='score', ascending=False).iloc[:n, :]
    print('done chi')

    '''minfo = mutual_info_classif(stat, star, discrete_features=True)
    minfo_res = pd.DataFrame({'word': word, 'minfo': minfo})
    minfo_res = minfo_res.sort_values(by='minfo', ascending=False).iloc[:n, :]
    print('done minfo')

    rf = RandomForestClassifier()
    rf.fit(stat, star)
    rf_res = pd.DataFrame({'word': word, 'rf_score': rf.feature_importances_})
    rf_res = rf_res.sort_values(by='rf_score', ascending=False).iloc[:n, :]
    print('done rf')'''

    return chi_res  # , minfo_res, rf_res


train_ct_chi = var_selection(train_ct, train_ct_word, 2000)
final_word = train_ct_chi.word

ind = np.isin(train_ct_word, final_word)
train_ct = train_ct[:, ind]
train_ct_word = train_ct_word[ind]

ind = np.isin(train_tfi_word, final_word)
train_tfi = train_tfi[:, ind]
train_tfi_word = train_tfi_word[ind]

# =====Modeling===========================================================================
xtrain_ct, xvalid_ct, ytrain, yvalid = train_test_split(train_ct, star,
                                                        stratify=star,
                                                        random_state=960724,
                                                        test_size=0.1, shuffle=True)

xtrain_tfi, xvalid_tfi, ytrain, yvalid = train_test_split(train_tfi, star,
                                                          stratify=star,
                                                          random_state=960724,
                                                          test_size=0.1, shuffle=True)

xtrain_ct = Normalizer().fit_transform(xtrain_ct)
clf = LogisticRegression(multi_class='multinomial', solver='sag',
                         random_state=960724, verbose=1).fit(xtrain_ct, ytrain)
ct_pred = clf.predict(Normalizer().fit_transform(xvalid_ct))
np.sqrt(np.mean((ct_pred.astype(int) - yvalid.astype(int)) ^ 2))

