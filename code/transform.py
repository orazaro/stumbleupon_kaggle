#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   Transformer
"""

import sys, random, pickle, copy, json
import datetime, csv, gzip, joblib
import logging

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
from kgml.classifier import LassoCV_proba,RidgeCV_proba,KNeighborsClassifier_proba

from project import Project
import prepare
from prepare import BPobj
        
random_state = 1
logger = logging.getLogger(__name__)

class Transformer(BaseEstimator, TransformerMixin):
  """
    Transformer for model04
  """
  def __init__(self, names=None, use_best=False, use_bp=True, use_stats=False, use_table=False, 
        bst_bgram=1,bst_minc=10,bst_title=256,bst_body=1028,bst_url=256,
        predict_bp=False, bp_clfs=['LR'],
        X_all=None, BP=None, verbose=0):
    self.names = names
    self.use_best = use_best
    self.use_bp = use_bp
    self.use_stats = use_stats
    self.use_table = use_table
    
    self.predict_bp = predict_bp
    self.bp_clfs = bp_clfs
    
    self.X_all = X_all
    self.BP = BP
    self.verbose = verbose

    self.bst_bgram = bst_bgram
    self.bst_minc = bst_minc
    self.bst_title = bst_title
    self.bst_body = bst_body
    self.bst_url = bst_url


  def fit(self, Xmask, y):
    self.deblog = logger.info if self.verbose>0 else logger.debug 
    #logger.debug("X_df.columns:%s", self.names)
    self.X_all = np.asarray(self.X_all)
    y = np.asarray(y)
    X_df = self.X_all[Xmask,:]

    self.keywords = ['title','body','url']
    self.params = {kw:{'bgram':self.bst_bgram,'minc':self.bst_minc,'nmax':100} for kw in self.keywords}
    self.params['title'] = {'bgram':self.bst_bgram,'minc':self.bst_minc,'nmax':self.bst_title}
    self.params['body'] = {'bgram':self.bst_bgram,'minc':self.bst_minc,'nmax':self.bst_body}
    self.params['url'] = {'bgram':self.bst_bgram,'minc':self.bst_minc,'nmax':self.bst_url}

    if self.use_best:
        logger.debug("Select best bgram=%d minc=%d title=%d body=%d url=%d",self.bst_bgram,self.bst_minc,
            self.bst_title,self.bst_body,self.bst_url)
        self.topwords, wordlists = self.best_boilerplate(X_df, y)
    else:
        wordlists = None

    if self.predict_bp:
        BP = self.BP[Xmask,:]
        clfs = dict( 
            LR = lm.LogisticRegression(penalty='l2', dual=True, tol=0.00001, 
                                 C=1, fit_intercept=True, intercept_scaling=1.0, 
                                 class_weight=None, random_state=random_state),
            RCVp1 = RidgeCV_proba(alphas=np.linspace(0,10), cv=4),
            RCVp2 = RidgeCV_proba(alphas=[0,1,1.8,3], cv=4),
            LCVp1 = LassoCV_proba(alphas=None, cv=4),
            LCVp2 = LassoCV_proba(alphas=[0.000111139489611,9.56158879464e-05], cv=4),
            GNB = GaussianNB(),
            MNB = MultinomialNB(alpha=0.8),
            BNB = BernoulliNB(alpha=1, binarize=0.01),
            BNB_28 = BernoulliNB(alpha=0.5, binarize=0.01),
            MNB_28 = MultinomialNB(alpha=0.03),
            LR_28  = lm.LogisticRegression(penalty='l2', dual=True, tol=0.00001, C=17, fit_intercept=True, 
                intercept_scaling=1.0, class_weight=None, random_state=random_state),
        )
        self.bp_estimators = [clfs[c] for c in self.bp_clfs]
        for est in self.bp_estimators:
            est.fit(BP,y)

    if self.use_table or self.use_best:
        X_dicts = self.make_dicts(X_df, wordlists)
        self.vectorizer = DictVectorizer()
        self.vectorizer.fit(X_dicts)

        X = self.vectorizer.transform(X_dicts)
        self.scaler = StandardScaler()
        self.scaler.fit(X.todense())

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
        X = self.vectorizer.transform(X_dicts)
        logger.debug("=> X:%s",X.shape) #,"features:",self.vectorizer.get_feature_names()
        X = self.scaler.transform(X.todense())
        X = sparse.csr_matrix(X)

    if self.use_bp:
        BP = self.BP[Xmask,:]
        if self.use_table or self.use_best:
            X = sparse.hstack((X,BP)).tocsr()
        else:
            X = BP
        logger.debug("X+BP:%s",X.shape)

    if self.predict_bp:
        BP = self.BP[Xmask,:]
        Y = []
        #logger.debug("self.bp_estimators:%s",self.bp_estimators)
        for est in self.bp_estimators:
            Y.append( est.predict_proba(BP)[:,1] )
        Y = np.vstack(Y).T
        logger.debug("X:%s Y:%s",X.shape,Y.shape)
        X = sparse.hstack((X,Y)).tocsr()

    if not self.use_bp or X.shape[1] < 256:
        X = X.todense()
    logger.info('transform result X=%s',X.shape)
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
        #if i < 2: logger.debug("i=%s dicts=%s",i,d)
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
        #logging.debug("kw=%s top1=%s wordlists=%s",kw,top1, wordlists[kw])
        topwords[kw] = top1
        self.deblog("=TOP= %s(%d) %s",kw,len(top1),','.join(top1))
    return topwords, wordlists
  

def test():
    print "tests ok"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Transform.')
    parser.add_argument('cmd', nargs='?', default='test')
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
    else:
        raise ValueError("bad cmd")

