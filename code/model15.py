#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Models Second Order
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
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
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
from kgml.classifier import LassoCV_proba,RidgeCV_proba,KNeighborsClassifier_proba

from project import Project
import prepare
from prepare import BPobj

from model import Model
random_state = 1
logger = logging.getLogger(__name__)


class ModelTwo(Model):
  def __init__(self, use_best=False, use_bp=False, use_stats=True, use_table=True, 
    predict_bp=True, clf = 'LR', bp_clfs = ['LR'], use_scaler=0):
    super(ModelTwo,self).__init__(use_best=use_best, use_bp=use_bp, use_stats=use_stats, 
        use_table=use_table, predict_bp=predict_bp, clf=clf, bp_clfs=bp_clfs, use_scaler=use_scaler)

  def _get_featureset(self):
    return prepare.Prepare_0(model=14, n_components=512, preproc=1, min_df=1, 
        use_svd=True, tfidf=2, stemmer=0)

  def _get_clf(self, clf):
    clfs = dict(
        LR = lm.LogisticRegression(penalty='l2', dual=True, tol=0.00001, 
                                 C=1, fit_intercept=True, intercept_scaling=1.0, 
                                 class_weight=None, random_state=random_state),
        RCV = RidgeCV_proba(alphas=np.linspace(0,10), cv=4),
        LCVp = LassoCV_proba(alphas=None, cv=4),
        SVR = svm.SVR(C=3,kernel='linear'),
        KNCp = KNeighborsClassifier_proba(n_neighbors=64, weights='distance'),
        RFC = RandomForestClassifier(n_estimators=500, max_depth=12,
                n_jobs=-1, random_state=random_state, verbose=0),
        GBC = GradientBoostingClassifier(n_estimators=100, max_depth=1,
                random_state=random_state, verbose=0, subsample=0.95),
        SVC = svm.SVC(kernel='rbf',C=5,gamma=0.010, probability=True,random_state=random_state),
        SGDC  = SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
               fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
               loss='hinge', n_iter=50, n_jobs=1, penalty='elasticnet', power_t=0.5,
               random_state=random_state, rho=None, shuffle=False, verbose=0,
               warm_start=False),
        ABC_DTC = AdaBoostClassifier(
                    algorithm='SAMME.R',
                    base_estimator=DecisionTreeClassifier(compute_importances=None, criterion='gini', max_depth=1, max_features=1.0, min_density=None, min_samples_leaf=1, min_samples_split=2, random_state=random_state, splitter='best'),
                    learning_rate=0.1, 
                    n_estimators=130, 
                    random_state=random_state),
        ABC_ETC = AdaBoostClassifier(
                    algorithm='SAMME.R',
                    base_estimator=ExtraTreeClassifier(compute_importances=None, criterion='gini', max_depth=1, max_features=1.0, min_density=None, min_samples_leaf=1, min_samples_split=2, random_state=random_state, splitter='best'),
                    learning_rate=0.1, 
                    n_estimators=130, 
                    random_state=random_state),
    )
    return clfs[clf]
    

class Model15(ModelTwo):
  def __init__(self):
    super(Model15,self).__init__(clf='LR', use_scaler=2)

class Model17(ModelTwo):
  def __init__(self):
    super(Model17,self).__init__(clf='RFC')

class Model18(ModelTwo):
  def __init__(self):
    super(Model18,self).__init__(clf='GBC')

class Model19(ModelTwo):
  def __init__(self):
    super(Model19,self).__init__(clf='ABC_DTC')

class Model20(ModelTwo):
  def __init__(self):
    super(Model20,self).__init__(clf='ABC_DTC',bp_clfs=['LR','BNB'])


def test():
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
    else:
        raise ValueError("bad cmd")

