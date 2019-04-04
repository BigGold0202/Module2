import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import scale, Normalizer
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import matplotlib.pyplot as plt


def split_token(x):
    return x.split(' ')


def var_selection(stat, star, word, n):
    chi_select, chi_p = chi2(stat, star)
    chi_res = pd.DataFrame({'word': word, 'score': chi_select, 'p-value': chi_p})
    res = chi_res.sort_values(by='score', ascending=False).iloc[:n, :]
    return res


def rf_select(stat, star, word, n_estimators=100):
    sel = RandomForestClassifier(n_estimators=n_estimators, verbose=2, n_jobs=4)
    sel.fit(stat, star)

    imp_sc = sel.feature_importances_
    ind = np.argsort(-1 * imp_sc)
    imp_sc = imp_sc[ind]
    word = word[ind]
    res = pd.DataFrame({'word': word, 'score': imp_sc})
    return res


def plot_rf(res, n, name):
    imp_sc = res.score
    word = res.word
    plt.title('Feature Importances' + name)
    plt.barh(range(n), imp_sc[::-1], color='b', align='center')
    plt.yticks(range(n), word[::-1])
    plt.xlabel('Relative Importance')
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def data_prep(word_list, n, mode):
    vec = CountVectorizer(tokenizer=split_token, stop_words=['.'])
    tfi = TfidfVectorizer(tokenizer=split_token, stop_words=['.'])

    if mode == 'test':
        rev_test = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\test_clean.csv')
        rev_test.text[rev_test.text.isna()] = 'na'
        print('done reading')
        other = rev_test.KaggleID
        test_ct = vec.fit_transform(rev_test.text)
        print('done ct')

        test_ct_word = np.array(vec.get_feature_names())
        ind = np.isin(test_ct_word, word_list)
        test_ct = test_ct[:, ind]
        test_ct_word = test_ct_word[ind]

        test_tfi = tfi.fit_transform(rev_test.text)
        print('done tfi')

        test_tfi_word = np.array(tfi.get_feature_names())
        ind = np.isin(test_tfi_word, final_word)
        test_tfi = test_tfi[:, ind]
        test_tfi_word = test_tfi_word[ind]

        return test_ct, test_ct_word, test_tfi, test_tfi_word, other
    elif mode == 'train':
        rev_train = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\rev_clean.csv', low_memory=False,
                                nrows=n)
        other = rev_train.stars
        rev_train.text[rev_train.text.isna()] = 'na'
        print('done reading')

        train_ct = vec.fit_transform(rev_train.text)
        print('done ct')

        train_ct_word = np.array(vec.get_feature_names())

        ind = np.isin(train_ct_word, word_list)
        train_ct = train_ct[:, ind]
        train_ct_word = train_ct_word[ind]

        train_tfi = tfi.fit_transform(rev_train.text)
        print('done tfi')

        train_tfi_word = np.array(tfi.get_feature_names())

        ind = np.isin(train_tfi_word, final_word)
        train_tfi = train_tfi[:, ind]
        train_tfi_word = train_tfi_word[ind]
        return train_ct, train_ct_word, train_tfi, train_tfi_word, other


rev_test = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\test_clean.csv')
rev_test.text[rev_test.text.isna()] = 'na'

rev_train = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\rev_clean.csv', low_memory=False)
rev_train.text[rev_train.text.isna()] = 'na'

print('done reading')
test_word = np.array(pd.read_csv('test_ct_word.csv')).reshape(-1)

vec = CountVectorizer(tokenizer=split_token, stop_words=['.', 'not'])
train_ct = vec.fit_transform(rev_train.text)
train_ct_word = vec.get_feature_names()


del rev_test, rev_train

test_word = np.array(list(set(test_word) & set(train_ct_word)))

ind = np.isin(train_ct_word, test_word)
train_ct = train_ct[:, ind]
train_ct_word = np.array(train_ct_word)[ind]

train_ct_chi = var_selection(train_ct, train_ct_word, 2000)
final_word = train_ct_chi.word

np.savetxt('final_word.csv', final_word, header='word', fmt='%s')
# =========begin again====================================================================================
# ============read==============================================================================
final_word = np.array(pd.read_csv('rf_word.csv', header=None)).reshape(-1)

rev_test = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\test_clean.csv')
rev_test.text[rev_test.text.isna()] = 'na'
kid = rev_test.KaggleID

rev_train = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\rev_clean.csv', low_memory=False)
star = rev_train.stars
rev_train.text[rev_train.text.isna()] = 'na'

# ==============count====================================================================
vec = CountVectorizer(tokenizer=split_token, stop_words=['.', 'not'])
train_ct = vec.fit_transform(rev_train.text)
train_ct_word = np.array(vec.get_feature_names())

test_ct = vec.fit_transform(rev_test.text)
test_ct_word = np.array(vec.get_feature_names())

ind = np.isin(train_ct_word, final_word)
train_ct = train_ct[:, ind]
train_ct_word = train_ct_word[ind]

ind = np.isin(test_ct_word, final_word)
test_ct = test_ct[:, ind]
test_ct_word = test_ct_word[ind]

# ====================tfidf===============================================
tfi = TfidfVectorizer(tokenizer=split_token, stop_words=['.', 'not'])

train_tfi = tfi.fit_transform(rev_train.text)
train_tfi_word = np.array(tfi.get_feature_names())

ind = np.isin(train_tfi_word, final_word)
train_tfi = train_tfi[:, ind]
train_tfi_word = train_tfi_word[ind]

test_tfi = tfi.fit_transform(rev_test.text)
test_tfi_word = np.array(tfi.get_feature_names())
ind = np.isin(test_tfi_word, final_word)
test_tfi = test_tfi[:, ind]
test_tfi_word = test_tfi_word[ind]

# ========Normalization====================================================
del rev_train, rev_test
train_tfi_N = Normalizer().fit_transform(train_tfi)
test_tfi_N = Normalizer().fit_transform(test_tfi)

# ============rf selection=====================================================
test_word = np.array(pd.read_csv('test_ct_word.csv')).reshape(-1)

ind = np.isin(train_ct_word, test_word)
train_ct = train_ct[:, ind]
train_ct_word = np.array(train_ct_word)[ind]

# ============================tfidf=======================================
ind = np.isin(train_tfi_word, test_word)
train_tfi = train_tfi[:, ind]
train_tfi_word = np.array(train_tfi_word)[ind]

sel = RandomForestClassifier(n_estimators=100, verbose=2, n_jobs=4)
sel.fit(train_tfi, star)

n = 50

imp_sc = sel.feature_importances_
ind = np.argsort(-1 * imp_sc)
word = train_tfi_word[ind][:n]
imp_sc = imp_sc[ind][:n]

plt.title('Feature Importances')
plt.barh(range(n), imp_sc[::-1], color='b', align='center')
plt.yticks(range(n), word[::-1])
plt.xlabel('Relative Importance')

rf_res = rf_select(train_tfi, star, train_tfi_word)
rf_res0 = rf_select(train_tfi, star, train_tfi_word, 10)
len(set(rf_res.word[:2000]) & set(rf_res0.word[:2000]))
rf_res_N = rf_select(train_tfi_N, star, train_tfi_word)
len(set(rf_res.word[:2000]) & set(rf_res_N.word[:2000]))

plot_learning_curve(RandomForestClassifier(n_estimators=100, verbose=2), 'full rf', cv=4, n_jobs=4, X=train_tfi_N, y=star)

rf_word = rf_res_N.word[:2000]
rf_word.to_csv('rf_word.csv', index=False, header=False)


# =======logReg===========================================================
xtrain_ct, xvalid_ct, ytrain, yvalid = train_test_split(train_ct, star,
                                                        stratify=star,
                                                        random_state=960724,
                                                        test_size=0.1, shuffle=True)

xtrain_ct = Normalizer().fit_transform(xtrain_ct)
clf = LogisticRegression(multi_class='multinomial', solver='sag',
                         random_state=960724, verbose=1).fit(xtrain_ct, ytrain)

ct_pred = clf.predict(Normalizer().fit_transform(test_ct))

res = pd.DataFrame({'ID': kid, 'Expected': ct_pred})
res.to_csv('outcome.csv', index=False)

# ===============LGBM count========================================================
star = star.astype(int) - 1

xtrain_ct, xvalid_ct, ytrain, yvalid = train_test_split(train_ct, star,
                                                        stratify=star,
                                                        random_state=960724,
                                                        test_size=0.2, shuffle=True)

lg_train = lgb.Dataset(xtrain_ct.astype(np.float64), label=ytrain)
lg_val = lg_train.create_valid(xvalid_ct.astype(np.float64), label=yvalid)

param = {'objective': 'multiclass', 'task': 'train', 'num_threads': 4, 'seed': 960724,
         'early_stopping_round': 50, 'verbosity': 1, 'num_class': 5,
         'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': -1,
         'feature_fraction': 0.8, 'bagging_fraction': 0.8}

lg_model = lgb.train(params=param, train_set=lg_train, valid_sets=lg_val, num_boost_round=200)

model_list = []
for i in [50, 64, 80]:
    print(i)
    param_tmp = {'objective': 'multiclass', 'task': 'train', 'num_threads': 4, 'seed': 960724,
                 'early_stopping_round': 50, 'verbosity': 1, 'num_class': 5,
                 'learning_rate': 0.1, 'num_leaves': i, 'max_depth': -1,
                 'feature_fraction': 1, 'bagging_fraction': 1}
    tmp_model = lgb.train(params=param_tmp, train_set=lg_train, valid_sets=lg_val, num_boost_round=200)
    model_list.append(tmp_model)

lg_pred = lg_model.predict(test_ct.astype(np.float64))

lg_res = [np.argsort(-1 * lg_pred[i, :], )[0] + 1 for i in range(lg_pred.shape[0])]
lg_res = pd.DataFrame({'ID': kid, 'Expected': lg_res})

lg_res.to_csv('lg.csv', index=False)


# ============logReg==tfidf=========================================================================
xtrain_tfi, xvalid_tfi, ytrain, yvalid = train_test_split(train_tfi_N, star,
                                                          stratify=star,
                                                          random_state=960724,
                                                          test_size=0.2, shuffle=True)

clf = LogisticRegression(multi_class='multinomial', solver='sag',
                         random_state=960724, verbose=1)
clf.fit(xtrain_tfi, ytrain)

plot_learning_curve(clf, 'logReg', cv=3, n_jobs=5, X=train_tfi_N, y=star)

param = {'C': [0.8, 1, 1.2]}
gsearch1 = GridSearchCV(estimator=clf,
                        param_grid=param, scoring='accuracy', cv=5, n_jobs=5)
gsearch1.fit(xtrain_tfi, ytrain)

print(gsearch1.best_params_)
print(gsearch1.best_score_)

tfi_pred = clf.predict(xvalid_tfi)
print(accuracy_score(yvalid, tfi_pred))  # 0.669

tfi_pred = clf.fit(train_tfi_N, star).predict(test_tfi_N)


res = pd.DataFrame({'ID': kid, 'Expected': tfi_pred})
res.to_csv('tfi_outcome.csv', index=False)

# ========LGBM tfidf======================================================
xtrain_tfi, xvalid_tfi, ytrain, yvalid = train_test_split(train_tfi, star,
                                                          stratify=star,
                                                          random_state=960724,
                                                          test_size=0.2, shuffle=True)

lg_train = lgb.Dataset(xtrain_tfi, label=ytrain)
lg_val = lg_train.create_valid(xvalid_tfi, label=yvalid)

model_list = []
for i in [60, 80]:
    print(i)
    param_tmp = {'objective': 'multiclass', 'task': 'train', 'num_threads': 4,
                 'early_stopping_round': 50, 'verbosity': 1, 'num_class': 5, 'max_bin': 125,
                 'learning_rate': 0.1, 'num_leaves': i, 'max_depth': -1, 'min_data_in_leaf': 500,
                 'feature_fraction': 1, 'bagging_fraction': 1}
    tmp_model = lgb.train(params=param_tmp, train_set=lg_train, valid_sets=lg_val, num_boost_round=200)
    model_list.append(tmp_model)

# ===========tfidf Random Forest=====================================================
xtrain_tfi, xvalid_tfi, ytrain, yvalid = train_test_split(train_tfi_N, star,
                                                          stratify=star,
                                                          random_state=960724,
                                                          test_size=0.2, shuffle=True)

rf = RandomForestClassifier(n_estimators=50, n_jobs=None, verbose=2)  #100: 0.6187,170:0.61955, 200:0.61715
rf.fit(train_tfi_N, star)
rf_pred = rf.predict(test_tfi_N)
rf_res = pd.DataFrame({'ID': kid, 'Expected': rf_pred})

rf_res.to_csv('rf.csv', index=False)

plot_learning_curve(rf, 'randomF', cv=5, n_jobs=5, X=xtrain_tfi, y=ytrain)


rf_pred = rf.predict(xvalid_tfi)
print(accuracy_score(yvalid, rf_pred))

param = {'n_estimators': range(170, 300, 30)}
gsearch1 = GridSearchCV(estimator=RandomForestClassifier(verbose=2, n_jobs=4),
                        param_grid=param, scoring='accuracy', cv=2, n_jobs=1)

gsearch1.fit(xtrain_tfi, ytrain)

