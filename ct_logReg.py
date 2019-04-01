import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import scale, Normalizer
import lightgbm as lgb


def split_token(x):
    return x.split(' ')


def var_selection(stat, word, n):
    chi_select, chi_p = chi2(stat, star)
    chi_res = pd.DataFrame({'word': word, 'score': chi_select, 'p-value': chi_p})
    chi_res = chi_res.sort_values(by='score', ascending=False).iloc[:n, :]
    print('done chi')
    return chi_res  # , minfo_res, rf_res


rev_test = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\test_clean.csv')
rev_test.text[rev_test.text.isna()] = 'na'
id = rev_test.KaggleID

rev_train = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\rev_clean.csv', low_memory=False)
star = rev_train.stars
rev_train.text[rev_train.text.isna()] = 'na'

print('done reading')
test_word = np.array(pd.read_csv('test_ct_word.csv')).reshape(-1)

vec = CountVectorizer(tokenizer=split_token, stop_words=['.', 'not'])
train_ct = vec.fit_transform(rev_train.text)
train_ct_word = vec.get_feature_names()

test_ct = vec.fit_transform(rev_test.text)
test_ct_word = vec.get_feature_names()

del rev_test
del rev_train

test_word = np.array(list(set(test_word) & set(train_ct_word)))

ind = np.isin(train_ct_word, test_word)
train_ct = train_ct[:, ind]
train_ct_word = np.array(train_ct_word)[ind]

train_ct_chi = var_selection(train_ct, train_ct_word, 2000)
final_word = train_ct_chi.word

ind = np.isin(train_ct_word, final_word)
train_ct = train_ct[:, ind]
train_ct_word = train_ct_word[ind]
print('done selection')

ind = np.isin(test_ct_word, final_word)
test_ct = test_ct[:, ind]

# =======logReg===========================================================
xtrain_ct, xvalid_ct, ytrain, yvalid = train_test_split(train_ct, star,
                                                        stratify=star,
                                                        random_state=960724,
                                                        test_size=0.1, shuffle=True)

xtrain_ct = Normalizer().fit_transform(xtrain_ct)
clf = LogisticRegression(multi_class='multinomial', solver='sag',
                         random_state=960724, verbose=1).fit(xtrain_ct, ytrain)

ct_pred = clf.predict(Normalizer().fit_transform(test_ct))

res = pd.DataFrame({'ID':id, 'Expected':ct_pred})
res.to_csv('outcome.csv', index=False)

# ===============LGBM========================================================
star = star.astype(int) - 1

xtrain_ct, xvalid_ct, ytrain, yvalid = train_test_split(train_ct, star,
                                                        stratify=star,
                                                        random_state=960724,
                                                        test_size=0.2, shuffle=True)

lg_train = lgb.Dataset(xtrain_ct.astype(np.float64), label=ytrain)
lg_val = lg_train.create_valid(xvalid_ct.astype(np.float64), label=yvalid)
lg_test = lgb.Dataset(test_ct)

param = {'objective': 'multiclass', 'task': 'train', 'num_threads': 4, 'seed': 960724,
         'early_stopping_round': 50, 'verbosity': 1, 'num_class': 5,
         'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': -1,
         'feature_fraction': 0.8, 'bagging_fraction': 0.8}

lg_model = lgb.train(params=param, train_set=lg_train, valid_sets=lg_val, num_boost_round=200)

lg_pred = lg_model.predict(test_ct.astype(np.float64))

lg_res = [np.argsort(-1 * lg_pred[i, :], )[0] + 1 for i in range(lg_pred.shape[0])]
lg_res = pd.DataFrame({'ID': id, 'Expected': lg_res})

lg_res.to_csv('lg.csv', index=False)


# ==============tfidf=========================================================================
tfi = TfidfVectorizer(tokenizer=split_token, stop_words=['.', 'not'])

train_tfi = tfi.fit_transform(rev_train.text)
train_tfi_word = tfi.get_feature_names()
ind = np.isin(train_tfi_word, final_word)
train_tfi = train_tfi[:, ind]
train_tfi_word = np.array(train_tfi_word)[ind]

test_tfi = tfi.fit_transform(rev_test.text)
test_tfi_word = tfi.get_feature_names()
ind = np.isin(test_tfi_word, final_word)
test_tfi = test_tfi[:, ind]
test_tfi_word = np.array(test_tfi_word)[ind]

train_tfi = Normalizer().fit_transform(train_tfi)
test_tfi = Normalizer().fit_transform(test_tfi)

del rev_test, rev_train
clf = LogisticRegression(multi_class='multinomial', solver='sag',
                         random_state=960724, verbose=1).fit(train_tfi, star)


