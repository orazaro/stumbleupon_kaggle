#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Model Class
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
from transform import Transformer
        
random_state = 1
logger = logging.getLogger(__name__)

class Model(BaseEstimator):
  def __init__(self, use_best=False, use_bp=True, use_stats=False, use_table=False, 
    bst_bgram=1,bst_minc=10,bst_title=256,bst_body=1028,bst_url=256,
    predict_bp=False, clf='LR', bp_clfs=None, use_scaler=0):
    
    self.use_best = use_best
    self.use_bp = use_bp
    self.use_stats = use_stats
    self.use_table = use_table
    self.predict_bp = predict_bp
    self.clf = clf
    self.bp_clfs = bp_clfs
    self.use_scaler = use_scaler
    
    self.bst_bgram = bst_bgram
    self.bst_minc = bst_minc
    self.bst_title = bst_title
    self.bst_body = bst_body
    self.bst_url = bst_url

  def fit(self, Xmask, y):
    pr = self._get_featureset()
    (X_all_df,_,BP,params) = pr.load_transform(update=False)
    names = list(X_all_df.columns)
    X_all = np.asarray(X_all_df)
    self.X_all, self.names = X_all, names

    logger.debug('Fit: use_stats=%s,use_table=%s,predict_bp=%s,use_scaler=%s',
        self.use_stats,self.use_table,self.predict_bp,self.use_scaler)
    logger.debug('Fit: bst_bgram=%s,bst_minc=%s,bst_title=%s,bst_body=%s,bst_url=%s',
        self.bst_bgram,self.bst_minc,self.bst_title,self.bst_body,self.bst_url)
    
    PipelineList = []
    PipelineList.append( ("trans", Transformer(names=self.names, use_best=self.use_best, 
                    use_bp=self.use_bp, use_stats=self.use_stats, use_table=self.use_table, 
                    bst_bgram=self.bst_bgram,bst_minc=self.bst_minc,bst_title=self.bst_title,
                    bst_body=self.bst_body,bst_url=self.bst_url,
                    predict_bp=self.predict_bp, bp_clfs=self.bp_clfs, X_all=X_all, BP=BP)) )
    if self.use_scaler > 0:
        PipelineList.append( ("scaler",StandardScaler(with_mean=(self.use_scaler>1))) )
    self._pipeline_append(PipelineList)
    PipelineList.append( ("est", self._get_clf(self.clf)) )
    
    self.rd = Pipeline(PipelineList)
    logger.debug('Pipline: %s',[(k,v.__class__.__name__) for k,v in PipelineList])
    logger.debug("Pipeline.estimator=%s",dict(PipelineList)['est'])

    self.rd.fit(Xmask,np.asarray(y))
    return self

  def _pipeline_append(self, pipelineList):
    "append any transformers into current pipeline"
    pass

  @staticmethod
  def scale_proba(y):
    # wrong implementation!!! moves 0.5 marker
    y_min,y_max = y.min(),y.max()
    y1 = (y-y_min)/(y_max-y_min) if y_max>y_min else y
    return y1
  def predict_proba(self, Xmask):
    #return self.scale_proba( self.rd.predict_proba(Xmask) )
    return self.rd.predict_proba(Xmask)
  def predict(self, Xmask):
    return self.rd.predict(Xmask)
  
  def starter(self):
    logger.info("%s starter", self.__class__.__name__)
    try:
        self.fit(np.arange(100),np.arange(100))
        pass
    except:
        logger.error("starter err")
  
  @classmethod
  def get_name(cls):
    return cls.__name__
  def get_param_grid(self, randomized=True):
    "get param_grid for this model to use with GridSearch"
    return []
  def _get_featureset(self):
    raise NotImplementedError('virtual function')
  def _get_clf(self, clf):
    raise NotImplementedError('virtual function')
    clfs = dict(
        GNB = GaussianNB(),
        MNB = MultinomialNB(alpha=self.alpha),
        BNB = BernoulliNB(alpha=.9, binarize=1),

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
                random_state=random_state, verbose=2, subsample=0.95),
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

