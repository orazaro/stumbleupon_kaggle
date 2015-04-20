#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Naive Bayes
"""

import sys, random, pickle, copy, json
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
from transform import Transformer
        
random_state = 1

from model import Model

class Model10(Model):
  def __init__(self):
    pass
  def fit(self, Xmask, y):
    pr = prepare.Prepare_0(model=10, preproc=1, min_df=1, use_svd=False, tfidf=2,
        stemmer=0)
    (X_all_df,_,BP,params) = pr.load_transform(update=False)
    names = list(X_all_df.columns)
    X_all = np.asarray(X_all_df)
    self.X_all, self.names = X_all, names

    clf0 = GaussianNB()
    clf1 = MultinomialNB(alpha=0.8)
    clf2 = BernoulliNB(alpha=1, binarize=0.01)

    clf = clf1
    self.rd = Pipeline([
        ("trans", Transformer(names=self.names, X_all=X_all, BP=BP)),
        #("scaler",StandardScaler(with_mean=False)), 
        ("est", clf)
        ])

    self.rd.fit(Xmask,np.asarray(y))
    return self
  def predict_proba(self, Xmask):
    return self.rd.predict_proba(Xmask)
  def predict(self, Xmask):
    return self.rd.predict(Xmask)
  
  def starter(self):
    print "Model10 starter"
    self.fit(np.arange(100),np.arange(100))

def main(submit=0):
    y, colnames, n_train, n_test, n_all = prepare.Prepare_0().load_y_colnames()
    X_all = np.arange(n_all)
    X = X_all[:n_train]
    
    rd = Model10()
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
    rd = Model10()
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

