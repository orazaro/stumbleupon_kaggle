#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model03 (copy of model05.py (copy from model02) )
"""

import sys, random, pickle, copy
import numpy as np
import datetime, csv, gzip
import pandas as pd

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
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.base import clone

from kgml.nltk_preprocessing import preprocess_pipeline
from kgml.rfecv import RFECVp

from project import Project
import prepare
from prepare import BPobj
        
random_state = 1

class Transformer(BaseEstimator, TransformerMixin):
  """
    Transformer for model03
  """
  def __init__(self):
    pass

  def fit(self, X_df, y):
    return self
  
  def transform(self, X_df):
    BP = self._transform_bp(X_df)
    return BP
  
  def _transform_bp(self, X_df):
    print "transforming data X_df:", X_df.shape
    BP = np.asarray(X_df)[:,2]
    BP = prepare.BP_tfv_transform(BP)
    
    print "transforming svd BP:", BP.shape,
    BP = prepare.BP_svd_transform(BP)
    print "=>", BP.shape
    return BP

def main(submit=0):
    Xall_df,y = prepare.Prepare_0().load(preproc=0, update=False)
    #Xall_df,y = Xall_df.iloc[:500,:],y[:300]
    lentrain = len(y)
    Xtrain_df = Xall_df.iloc[:lentrain,:]

    clf1 = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=random_state)
    clf2 = RandomForestClassifier(n_estimators=200, max_depth=24,
            n_jobs=-1, random_state=random_state, verbose=0)

    clf3 = GradientBoostingClassifier(n_estimators=42, max_depth=24,
            random_state=random_state, verbose=2, subsample=0.9)

    clf4 = svm.SVC(probability=True)

    clf5 = KNeighborsClassifier(n_neighbors=5)

    clf6 = SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
           fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
           loss='hinge', n_iter=50, n_jobs=1, penalty='elasticnet', power_t=0.5,
           random_state=random_state, rho=None, shuffle=False, verbose=0,
           warm_start=False)

    clf = clf1
   
    if 0:
        selector = RFECVp(clf,clf, step=10, cv=4, scoring="roc_auc", verbose=2)
        selector = selector.fit( Transformer().fit_transform(Xtrain_df, y), y)
        clf = selector

    rd = Pipeline([
        ("trans", Transformer()),
        #("selector", SelectPercentile(chi2, percentile=90)),
        #("selector", SelectPercentile(f_classif, percentile=50)),
        #("selector", lm.RandomizedLogisticRegression(C=1, random_state=random_state, verbose=1)),
        #("pca", PCA(n_components='mle')),
        #("pca", PCA(n_components=500)),
        #("svd", TruncatedSVD(n_components=200, random_state=random_state )),
        #("lasso",svm.LinearSVC(C=0.5, penalty="l1", dual=False)),
        ("est", clf)
        ])

    
    if not submit:
        cv_run(rd, Xtrain_df, y)
        return
    else:
        print "Prepare submission.."

    print "training on full data"
    rd.fit(Xtrain_df,y)
    Xtest_df = Xall_df.iloc[lentrain:,:]
    pred = rd.predict_proba(Xtest_df)[:,1]
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
    Xall_df,y = prepare.Prepare_0().load()
    Xall_df,y = Xall_df.iloc[:400,:],y[:200]
    lentrain = len(y)
    
    clf1 = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=random_state)
    clf5 = KNeighborsClassifier(n_neighbors=5)
    
    clf= clf5

    rd = Pipeline([
        ("trans", Transformer()),
        ("est", clf)
        ])
    cv_run(rd, Xall_df.iloc[:lentrain,:], y)
    
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

