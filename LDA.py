import gensim
from gensim.utils import simple_preprocess
import pandas as pd
import numpy as np
from multi_cleaning import sent_tokenize
from gensim.models.coherencemodel import CoherenceModel
import re
import ast

rev_data = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\brun_review.csv')
rev_text = rev_data.text
rev_text[rev_text.isna()] = 'na'
rev_text = rev_text.apply(lambda x: x.split(' '))

rec_dict = gensim.corpora.Dictionary(rev_text)
rec_dict.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
rec_corpus = [rec_dict.doc2bow(doc) for doc in rev_text]

lda_model = gensim.models.LdaModel(rec_corpus, num_topics=5, id2word=rec_dict, passes=2)

cm_score = []
for i in [5, 10]:
    lda_model0 = gensim.models.LdaModel(rec_corpus, num_topics=i, iterations=100,
                                        id2word=rec_dict, passes=2)
    cm = CoherenceModel(model=lda_model0, corpus=rec_corpus, coherence='u_mass')
    cm_score.append(cm.get_coherence())

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


def lda_ana(rev_text, n, iterations=100):
    # rev_text[rev_text.isna()] = 'na'
    rev_text = rev_text.apply(lambda x: " ".join(ast.literal_eval(x)))
    rev_text = rev_text.apply(lambda x: x.split(' '))

    rec_dict = gensim.corpora.Dictionary(rev_text)
    rec_dict.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    rec_corpus = [rec_dict.doc2bow(doc) for doc in rev_text]

    lda_model = gensim.models.LdaModel(rec_corpus, num_topics=n, id2word=rec_dict, passes=2, iterations=iterations)
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
    return lda_model, rec_corpus


def lda_score(model, rec_corpus):
    cm = CoherenceModel(model=model, corpus=rec_corpus, coherence='u_mass')
    return cm.get_coherence()


word_list = ['chicken','egg','fry','sandwich','cheese','salad','pancake',
             'burger','potato','waffle','sauce','bread','dessert','steak',
             'cream','taco','meat','cake','crepe','beef','benedict','hamburger',
             'coffee','chocolate','soup','wine','cheese','water','juice','cocktail']

for i in word_list:
    tmp = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\keyword\word_' + i + '_sep.csv')
    tmp = tmp.review
    tmp_lda, tmp_cp = lda_ana(tmp, 3, 150)
    lda_score(tmp_lda, tmp_cp)

tmp = pd.read_csv(r'D:\OneDrive - UW-Madison\Module2\keyword\word_' + 'beef' + '.csv')
tmp = tmp.review

tmp_lda, tmp_cp = lda_ana(tmp, 3, 150)
lda_score(tmp_lda, tmp_cp)


