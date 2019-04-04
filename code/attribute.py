import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf


busi_data = pd.read_json(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\business_train.json', orient='records', lines=True)
brun_rev = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\brun_review.csv')


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


def plot_rf(res, n, name):
    imp_sc = res.score
    word = res.word
    plt.title('Feature Importances' + name)
    plt.barh(range(n), imp_sc[:n][::-1], color='b', align='center')
    plt.yticks(range(n), word[:n][::-1])
    plt.xlabel('Relative Importance')
    plt.show()


# select brunch attributes
bus_sp = []
for i in range(busi_data.shape[0]):
    if busi_data.categories[i] is not None:
        bus_sp.append(nltk.word_tokenize(busi_data.categories[i]))
    else:
        bus_sp.append('0')

brun_id = [i for i in range(len(bus_sp)) if 'Brunch' in bus_sp[i]]  # 160796
brun_data = busi_data.iloc[brun_id, ]

# process attributes and statistics shows how many NA
brun_slice, brun_slice_stat = atb_process(brun_data.attributes, 'slice')
brun_slice['business_id'] = brun_data.business_id.values
brun_slice = brun_slice.drop(['BusinessParking', 'BestNights', 'GoodForMeal', 'Ambience', 'Music'], axis=1)
brun_slice.to_csv('brun_slice.csv', index=False)


brun_star = brun_rev.iloc[:, 0:2]
brun_slice = brun_slice.merge(brun_star, how='right')
brun_slice.to_csv('brun_slice_mg.csv', index=False)


# =============rf selection======================================================
def rf_select(x, y, word, n_estimators=100):
    sel = RandomForestClassifier(n_estimators=n_estimators, verbose=2, n_jobs=4)
    sel.fit(x, y)

    imp_sc = sel.feature_importances_
    ind = np.argsort(-1 * imp_sc)
    imp_sc = imp_sc[ind]
    word = word[ind]
    res = pd.DataFrame({'word': word, 'score': imp_sc})
    return res


le = LabelEncoder()
atb_x = brun_slice.drop(['business_id', 'stars'], axis=1)
atb_x.fillna(np.nan, inplace=True)

for i in range(atb_x.shape[1]):
    atb_x.iloc[:, i] = atb_x.iloc[:, i].astype(str)
    atb_x.iloc[:, i] = le.fit_transform(atb_x.iloc[:, i])

atb_rf = rf_select(atb_x, brun_slice.stars, atb_x.columns, 100)
plot_rf(atb_rf, 15, ' Brunch')
atb_rf.word.to_csv(r'D:\OneDrive - UW-Madison\Module2\brunch_vi.csv', index=False, header=False)

atb = brun_slice.drop(['business_id'], axis=1)


atb_score = []
for i in range(20):
    gp = atb.loc[:, [atb_rf.word[i], 'stars']].groupby(atb_rf.word[i], as_index=False)
    tmp = gp.mean()
    tmp['count'] = gp.count().stars
    atb_score.append(tmp)
    tmp.to_csv(r'D:\OneDrive - UW-Madison\Module2\Attr' + atb_rf.word[i] + '.csv')

# =======all restaurant===============================================================
rev_star = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\rev_clean.csv', usecols=[0, 2])

rest_id = [i for i in range(len(bus_sp)) if 'Restaurants' in bus_sp[i]]  # 160796
rest_data = busi_data.iloc[rest_id, ]

rev_star = rev_star.loc[rev_star.business_id.isin(rest_data.business_id)].reset_index(drop=True)

rest_slice0, rest_slice_stat = atb_process(rest_data.attributes, 'slice')
rest_slice0['business_id'] = rest_data.business_id.values
rest_slice0 = rest_slice0.drop(['BusinessParking', 'BestNights', 'GoodForMeal', 'Ambience', 'Music'], axis=1)
rest_slice = rest_slice0.merge(rev_star, how='right')
rest_slice0.to_csv('rest_slice.csv', index=False)
rest_slice.to_csv('rest_slice_me.csv', index=False)

atb_rex = rest_slice.drop(['business_id', 'stars'], axis=1)
atb_rex.fillna(np.nan, inplace=True)

for i in range(atb_rex.shape[1]):
    print(i)
    atb_rex.iloc[:, i] = atb_rex.iloc[:, i].astype(str)
    atb_rex.iloc[:, i] = le.fit_transform(atb_rex.iloc[:, i])

atb_rf_res = rf_select(atb_rex, rest_slice.stars, atb_rex.columns, 100)
plot_rf(atb_rf_res, 15, ' Restaurant')

atb_rf_res.word.to_csv(r'D:\OneDrive - UW-Madison\Module2\rest_vi.csv', header=False, index=False)

atb_res = rest_slice.drop(['business_id'], axis=1)


atb_score_res = []
for i in range(20):
    gp = atb_res.loc[:, [atb_rf_res.word[i], 'stars']].groupby(atb_rf_res.word[i], as_index=False)
    tmp = gp.mean()
    tmp['count'] = gp.count().stars
    atb_score_res.append(tmp)

# ======one hot encode=========TRASH==========================================
atb_one = brun_slice.drop(['business_id', 'stars'], axis=1)
atb_one.fillna(np.nan, inplace=True)

for i in range(atb_one.shape[1]):
    atb_one.iloc[:, i] = atb_one.iloc[:, i].astype(str)

ohe = OneHotEncoder(handle_unknown='error')
tmp = brun_slice.loc[:, ["WiFi", 'stars']].dropna()
atb_one = pd.get_dummies(tmp.iloc[:, 0])

atb_y = tmp.stars

reg = LinearRegression().fit(atb_one, atb_y)


reg = smf.OLS(atb_y, atb_one).fit()


