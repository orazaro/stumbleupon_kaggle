#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Model 14: LogisticRegression with SVD and preproc (analog model02)
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
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
from kgml.model_selection import cv_run

from project import Project
import prepare
from prepare import BPobj
from transform import Transformer
from model import Model
        
random_state = 1

class Model14(Model):
  def __init__(self):
    pass
  def fit(self, Xmask, y):
    pr = prepare.Prepare_0(model=14, n_components=512, preproc=1, min_df=1, use_svd=True, tfidf=2,
        stemmer=0)
    (X_all_df,_,BP,params) = pr.load_transform(update=False)
    names = list(X_all_df.columns)
    X_all = np.asarray(X_all_df)
    self.X_all, self.names = X_all, names

    clf1 = lm.LogisticRegression(penalty='l2', dual=True, tol=0.00001, 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=random_state)

    class LassoCV_proba(lm.LassoCV):
        def predict_proba(self,X):
            print 'alpha_:',self.alpha_
            y = self.predict(X)
            y = 1./(1+np.exp(-(y-0.5)))
            return np.vstack((1-y,y)).T
    class RidgeCV_proba(lm.RidgeCV):
        def predict_proba(self,X):
            print 'alpha_:',self.alpha_
            y = self.predict(X)
            if 0:
                y_min,y_max = y.min(),y.max()
                if y_max>y_min:
                    y = (y-y_min)/(y_max-y_min)
            else:
                y = 1./(1+np.exp(-(y-0.5)))
            return np.vstack((1-y,y)).T

    clf2 = RidgeCV_proba(alphas=np.linspace(0,10), cv=4)
    clf3 = LassoCV_proba(alphas=None, cv=4)
    clf4 = svm.SVR(C=3,kernel='linear')
    
    clf = clf1

    self.rd = Pipeline([
        ("trans", Transformer(names=self.names, X_all=X_all, BP=BP)),
        #("scaler",StandardScaler(with_mean=False)), 
        #("filter",lm.LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=random_state)),
        ("est", clf)
        ])

    self.rd.fit(Xmask,np.asarray(y))
    return self
  def predict_proba(self, Xmask):
    return self.rd.predict_proba(Xmask)
  def predict(self, Xmask):
    return self.rd.predict(Xmask)
  
  def starter(self):
    print "Model14 starter"
    try:
        self.fit(np.arange(100),np.arange(100))
    except:
        print "starter err"

def main(submit=0):
    y, colnames, n_train, n_test, n_all = prepare.Prepare_0().load_y_colnames()
    X_all = np.arange(n_all)
    X = X_all[:n_train]
    
    rd = Model14()
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


def test():
    y, colnames, n_train, n_test, n_all = prepare.Prepare_0().load_y_colnames()
    X_all = np.arange(n_all)
    X = X_all[:200]
    y = y[:200]
    rd = Model14()
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

