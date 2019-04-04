import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

word_list = ['beer','friendly','pretty','server','waitress','waiter','manager','kid',
             'atmosphere','location','walk','check','selection','free',
             'patio','diner','decor','kitchen','party','din','drive','music','counter',
             'vegan','reservation','busy','quick','attentive','clean','cold','yummy',
             'warm','slow','dry','crowd','chicken','egg','fry','sandwich',
             'cheese','salad','pancake','burger','potato','waffle','sauce','bread',
             'dessert','steak','cream','taco','meat','cake','crepe','beef','benedict',
             'coffee','chocolate','soup','wine','water','juice','cocktail']

# run bigram and trigram statistics
for i in word_list:
    print(i)
    rev_text = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\keyword\word_' + i + '_sep.csv')
    rev_text = rev_text.review
    rev_text = rev_text.apply(lambda x: " ".join(ast.literal_eval(x)))
    # rev_text = rev_text.apply(lambda x: x.split(' '))

    vec = CountVectorizer(tokenizer=lambda x: x.split(' '), ngram_range=(2, 2), stop_words=['.'])
    rev_ct2 = vec.fit_transform(rev_text)
    rev_ct2 = np.array(rev_ct2.sum(axis=0))[0]
    ct2_word = vec.get_feature_names()
    ct2_res = pd.DataFrame({'word': ct2_word, 'count_sum': rev_ct2}).sort_values(by='count_sum',
                                                                                 ascending=False).reset_index(drop=True)
    ct2_res.to_csv(r'D:\OneDrive - UW-Madison\Module2\keyword\test\food\ct2_res_' + i + '.csv', index=False)

    vec = CountVectorizer(tokenizer=lambda x: x.split(' '), ngram_range=(3, 3), stop_words=['.'])
    rev_ct3 = vec.fit_transform(rev_text)
    rev_ct3 = np.array(rev_ct3.sum(axis=0))[0]
    ct3_word = vec.get_feature_names()
    ct3_res = pd.DataFrame({'word': ct3_word, 'count_sum': rev_ct3}).sort_values(by='count_sum',
                                                                                 ascending=False).reset_index(drop=True)
    ct3_res.to_csv(r'D:\OneDrive - UW-Madison\Module2\keyword\test\food\ct3_res_' + i + '.csv', index=False)

    tfi = TfidfVectorizer(tokenizer=lambda x: x.split(' '), ngram_range=(2, 2), stop_words=['.'])
    rev_tfi2 = tfi.fit_transform(rev_text)
    rev_tfi2 = np.array(rev_tfi2.sum(axis=0))[0]
    tfi2_word = tfi.get_feature_names()
    tfi2_res = pd.DataFrame({'word': tfi2_word, 'tfi_sum': rev_tfi2}).sort_values(by='tfi_sum',
                                                                                  ascending=False).reset_index(drop=True)
    tfi2_res.to_csv(r'D:\OneDrive - UW-Madison\Module2\keyword\test\food\tfi2_res_' + i + '.csv', index=False)

    tfi = TfidfVectorizer(tokenizer=lambda x: x.split(' '), ngram_range=(3, 3), stop_words=['.'])
    rev_tfi3 = tfi.fit_transform(rev_text)
    rev_tfi3 = np.array(rev_tfi3.sum(axis=0))[0]
    tfi3_word = tfi.get_feature_names()
    tfi3_res = pd.DataFrame({'word': tfi3_word, 'tfi_sum': rev_tfi3}).sort_values(by='tfi_sum',
                                                                                  ascending=False).reset_index(drop=True)
    tfi3_res.to_csv(r'D:\OneDrive - UW-Madison\Module2\keyword\test\food\tfi3_res_' + i + '.csv', index=False)


