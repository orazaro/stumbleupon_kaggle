#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Model 12 ==  Model 06 with new driver 
"""

import sys, random, pickle, copy, json, logging
import datetime, csv, gzip, joblib

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.base import clone
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from kgml.nltk_preprocessing import preprocess_pipeline
from kgml.rfecv import RFECVp
from kgml.feature_extraction import select_topwords, extract_words

from project import Project
import prepare
from prepare import BPobj
        
random_state = 124
logger = logging.getLogger(__name__)

class Transformer(BaseEstimator, TransformerMixin):
  """
    Transformer for model04
  """
  def __init__(self, names=None, use_best=False, use_bp=True, use_stats=True, use_table=True, 
        X_all=None, BP=None):
    self.names = names
    self.use_best = use_best
    self.use_bp = use_bp
    self.use_stats = use_stats
    self.use_table = use_table
    self.X_all = X_all
    self.BP = BP

  def fit(self, Xmask, y):
    #print "X_df.columns:", self.names
    self.X_all = np.asarray(self.X_all)
    y = np.asarray(y)
    X_df = self.X_all[Xmask,:]

    self.keywords = ['title','body','url']
    self.params = {kw:{'bgram':True,'minc':5,'nmax':100} for kw in self.keywords}
    self.params['title'] = {'bgram':True,'minc':10,'nmax':256}
    self.params['body'] = {'bgram':True,'minc':10,'nmax':1028}
    self.params['url'] = {'bgram':True,'minc':10,'nmax':256}

    if self.use_best:
        self.topwords, wordlists = self.best_boilerplate(X_df, y)
    else:
        wordlists = None
   
    if self.use_table or self.use_best:
        X_dicts = self.make_dicts(X_df, wordlists)
        self.vectorizer = DictVectorizer()
        self.vectorizer.fit(X_dicts)

        X = self.vectorizer.transform(X_dicts)
        self.scaler = StandardScaler()
        #self.scaler.fit(X.todense())

    return self
  
  def transform(self, Xmask):
    X_df = self.X_all[Xmask,:]
    logger.debug("transform X_df:%s",X_df.shape)
    
    if self.use_best:
        wordlists = self.get_boilerplate(X_df)
    else:
        wordlists = None

    if self.use_table or self.use_best:
        X_dicts = self.make_dicts(X_df, wordlists)
        #print "X_dicts:",len(X_dicts)
        X = self.vectorizer.transform(X_dicts)
        #print "=> X:",X.shape #,"features:",self.vectorizer.get_feature_names()
        #X = self.scaler.transform(X.todense())
        X = sparse.csr_matrix(X)

    if self.use_bp:
        BP = self.BP[Xmask,:]
        if self.use_table or self.use_best:
            X = sparse.hstack((X,BP)).tocsr()
        else:
            X = BP
        #print "X+BP:",X.shape 

    X = X.todense()
    return X
 
  def make_dicts(self, X_df, wordlists):
    N = X_df.shape[0]
    X_dicts = []
    for i in range(N):
        d = {}
        if wordlists:
            for kw in self.keywords:
                words = wordlists[kw][i]
                for (key,val) in words.items():
                    f = '%s_%s' % (kw,key)
                    d[f] = val
        if 1:
            # from X_df
            row = X_df[i,:]
            if not pd.isnull(row[3]):
                f = 'ac_%s' % row[3]
                d[f] = float(row[4])
            else:
                f = 'ac_%s' % 'unknown'
                d[f] = float(0.0)
        if self.use_stats:
            for j,name in enumerate(self.names):
                if j < 5:
                    pass
                elif j in [17,20]:
                    if pd.isnull(row[j]):
                        d[name] = 0
                    elif row[j]:
                        d[name] = 1
                    else:
                        d[name] = -1
                elif not pd.isnull(row[j]):
                    f = name
                    d[f] = row[j]
                else:
                    raise ValueError("bad row:",row)
        """
        n_components = 128
        for j in range(n_components):
            f = "bp_%03d" % j
            d[f] = BP[start+i,j]
        if i < 2:
            print "i:",i,"dicts:",d
        """
        X_dicts.append(d)
    return X_dicts

  def get_boilerplate(self, X_df):
    """
        get data from boilerplate using top words (transform)
    """
    keywords = self.keywords
    N = X_df.shape[0]
    D = {kw:[] for kw in keywords}
    for i in range(N):
        observation = X_df[i,2]
        d = json.loads(observation)
        for kw in keywords:
            if kw in d and d[kw]:
                D[kw].append(d[kw])
            else:
                D[kw].append('')
    
    wordlists = {}
    for kw in keywords:
        use_bigram = self.params[kw]['bgram']
        voc = self.topwords[kw]
        dlist = D[kw]
        wordlist = []
        for i in range(N):
            words = extract_words(dlist[i], vocab=voc,
                use_bigrams=use_bigram)
            wordlist.append(words)
        wordlists[kw] = wordlist
    return wordlists
  
  def best_boilerplate(self, X_df, y):
    """
        select top words from boilerplate (fit)
    """
    keywords = self.keywords
    N = X_df.shape[0]
    D = {kw:[] for kw in keywords}
    for i in range(N):
        observation = X_df[i,2]
        d = json.loads(observation)
        for kw in keywords:
            if kw in d and d[kw]:
                D[kw].append(d[kw])
            else:
                D[kw].append('')
    
    wordlists = {}
    topwords = {} 
    for kw in keywords:
        top1, wordlists[kw] = select_topwords(
            D[kw], 
            y, 
            use_bigrams=self.params[kw]['bgram'],
            mincount=self.params[kw]['minc'],
            nmax=self.params[kw]['nmax']
            )
        topwords[kw] = top1
        print >>sys.stderr,"=TOP= %s(%d)"%(kw,len(top1)),','.join(top1)
    return topwords, wordlists
  
from model import Model

class Model12(Model):
  def __init__(self):
    pass
  def fit(self, Xmask, y):
    pr = prepare.Prepare_0(model=12, n_components=128, min_df=3, preproc=0, 
        use_svd=True, tfidf=2, stemmer=0)
    (X_all_df,_,BP,params) = pr.load_transform(update=False)
    names = list(X_all_df.columns)
    X_all = np.asarray(X_all_df)
    self.X_all, self.names = X_all, names

    clf = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=random_state)

    self.rd = Pipeline([
        ("trans", Transformer(names=self.names, X_all=X_all, BP=BP)),
        #("scaler",StandardScaler(with_mean=False)), 
        ("scaler",StandardScaler(with_mean=True)), 
        ("est", clf)
        ])

    self.rd.fit(Xmask,np.asarray(y))
    return self
  def predict_proba(self, Xmask):
    return self.rd.predict_proba(Xmask)
  def predict(self, Xmask):
    return self.rd.predict(Xmask)
  
  def starter(self):
    print "Model12 starter"
    self.fit(np.arange(100),np.arange(100))

def main(submit=0):
    y, colnames, n_train, n_test, n_all = prepare.Prepare_0().load_y_colnames()
    X_all = np.arange(n_all)
    X = X_all[:n_train]
    
    rd = Model12()
    rd.starter()
    
    if not submit:
        cv_run(rd, X, y)
        return
    else:
        print "Prepare submission.."

    print "training on full data"
    rd.fit(X_all[:n_train],y)
    Xtest = X_all[n_train:]
    pred = rd.predict_proba(Xtest)[:,1]
    import submit
    submit.do_submit(pred)

def cv_run(rd, X, y):
    print "X:",X.shape,"y:",y.shape
    n_cv = 16
    #cv1 = cross_validation.KFold(len(y), n_folds=n_cv, random_state=random_state)
    cv1 = cross_validation.StratifiedKFold(y, n_folds=n_cv)
    scores = cross_validation.cross_val_score(rd, X, y, cv=cv1, 
        scoring='roc_auc', 
        #scoring=make_scorer(roc_auc_score), 
        n_jobs=-1, verbose=1)
    print "scores:",scores
    print "%d Fold CV Score: %.6f +- %.4f" % (n_cv, np.mean(scores), 2*np.std(scores),)

def test():
    y, colnames, n_train, n_test, n_all = prepare.Prepare_0().load_y_colnames()
    X_all = np.arange(n_all)
    X = X_all[:200]
    y = y[:200]
    rd = Model12()
    rd.starter()
    cv_run(rd, X, y)
    
    print "tests ok"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare.')
    parser.add_argument('cmd', nargs='?', default='main')
    parser.add_argument('-update', default='0')    
    parser.add_argument('-rs', default=None)
    parser.add_argument('-submit', default='0')
    args = parser.parse_args()
    print >>sys.stderr,args 
  
    if args.rs:
        random_state = int(args.rs)
    if args.submit and int(args.submit):
        random_state = 1961
        np.random_state = 1961
    if random_state:
        print "random_state:", random_state
        random.seed(random_state)
        np.random.seed(random_state)
    
    if args.cmd == 'test':
        test()
    elif args.cmd == 'main':
        main(submit=int(args.submit))
    else:
        raise ValueError("bad cmd")

