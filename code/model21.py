#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Models with Naive Bayes
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


param_grid_NB = {
     'alpha':[0.01, 0.1, 0.8, 1, 2],
     'binarize':[0,0.1,1],
     'fit_area': ['test'],
     'stemmer': [0,1,2,3],
     'ngram_max': [1,2],
     'max_df': [0.01,0.03,0.1,1.0],
     'max_features': [300000,900000],
     'binary': [False,True],
     'min_df': [1,2,3],
     'token_min': [1,2,3,5,7],
     'tfidf': [0,2],
     'preproc': [0,1],
     'do_remove_stopwords': [False,True],
     'use_svd': [False],
     'n_components': [256],
     'use_idf': [0,1],
     'smooth_idf': [0,1],
     'sublinear_tf': [0,1],
     'norm': [None,'l1','l2'],
    }

class Model21(Model):
  def __init__(self, use_best=False, use_bp=True, use_stats=False, use_table=False, clf = 'MNB', 
    alpha=0.03, binarize=0, use_svd=False, tfidf=2, fit_area='test', preproc=1, min_df=1,
    stemmer=1, ngram_max=1, max_df=0.1, binary=False, max_features=9000000, n_components=256,
    use_idf=1, smooth_idf=1, sublinear_tf=0, norm='l1', token_min=3,
    do_remove_stopwords = False,
    ):
    super(Model21,self).__init__(use_best=use_best, use_bp=use_bp, use_stats=use_stats, 
        use_table=use_table, clf=clf)
    self.alpha = alpha
    self.binarize = binarize
    self.use_svd = use_svd
    self.tfidf = tfidf
    self.fit_area = fit_area
    self.preproc = preproc
    self.min_df = min_df
    self.stemmer = stemmer
    self.ngram_max = ngram_max
    self.max_df = max_df
    self.binary = binary
    self.max_features = max_features
    self.n_components = n_components
    self.use_idf=use_idf
    self.smooth_idf=smooth_idf
    self.sublinear_tf=sublinear_tf
    self.norm=norm
    self.token_min = token_min
    self.do_remove_stopwords = do_remove_stopwords

  def _get_featureset(self):
    extra_par = dict(ngram_max=self.ngram_max,max_df=self.max_df,binary=self.binary,
        max_features=self.max_features,use_idf=self.use_idf, smooth_idf=self.smooth_idf, 
        sublinear_tf=self.sublinear_tf, norm=self.norm, token_min=self.token_min,
        do_remove_stopwords=self.do_remove_stopwords)
    extra_js = json.dumps(extra_par)
    return prepare.Prepare_0(n_components=self.n_components, preproc=self.preproc, 
        min_df=self.min_df, use_svd=self.use_svd, tfidf=self.tfidf, stemmer=self.stemmer, 
        fit_area=self.fit_area, extra=extra_js)

  def _get_clf(self, clf):
    return MultinomialNB(alpha=self.alpha)
  def get_param_grid(self, randomized=True):
    "get param_grid for this model to use with GridSearch"
    if randomized:
        param_grid = param_grid_NB
    else:
        param_grid = {
             'alpha':[0.001, 0.003, 0.006, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.5, 0.8, 1, 2, 3],
             #'alpha':[0.01, 0.05, 0.1, 0.5, 0.6, 0.7, 0.8, 1, 1.5, 2],
             #'binarize':[0,0.0001,0.001,0.01,1],
             #'stemmer': [0,1,2,3],
             #'ngram_max': [1,2],
             #'max_df': [0.01,0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
             #'max_df': [0.01,0.02,0.03,0.04,0.05,0.1,0.2],
             #'binary': [False,True],
             #'min_df': [1,2,3],
             #'use_svd': [False,True],
             #'tfidf': [0],
             #'preproc': [0],
             #'fit_area': ['test','all'],
            }
    return param_grid
    
class Model22(Model21):
  def __init__(self, 
    alpha=0.8, binarize=0.1, use_svd=False, tfidf=2, fit_area='test', preproc=1, min_df=3,
    stemmer=1, ngram_max=1, max_df=0.1, binary=False, max_features=900000, n_components=256,
    use_idf=1, smooth_idf=0, sublinear_tf=1, norm='l2', token_min=1,
    do_remove_stopwords = True,
    ):
    super(Model22,self).__init__(clf='BNB',
    alpha=alpha, binarize=binarize, use_svd=use_svd, tfidf=tfidf, fit_area=fit_area, preproc=preproc, 
    min_df=min_df, stemmer=stemmer, ngram_max=ngram_max, max_df=max_df, binary=binary, 
    max_features=max_features, n_components=n_components, use_idf=use_idf, smooth_idf=smooth_idf, 
    sublinear_tf=sublinear_tf, norm=norm, token_min=token_min,
    do_remove_stopwords = do_remove_stopwords,)

  def _get_clf(self, clf):
    clf = BernoulliNB(alpha=self.alpha, binarize=self.binarize)
    return clf
  def get_param_grid(self, randomized=True):
    "get param_grid for this model to use with GridSearch"
    if randomized:
        param_grid = param_grid_NB
    else:
        param_grid = {
             'alpha':[0.001, 0.003, 0.006, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.5, 0.8, 1, 2, 3],
             #'alpha':[0.01, 0.05, 0.1, 0.5, 0.6, 0.7, 0.8, 1, 1.5, 2],
             'binarize':[0,0.0001,0.001,0.01,0.1,1],
             #'stemmer': [0,1,2,3],
             #'ngram_max': [1,2],
             #'max_df': [0.01,0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
             #'max_df': [0.01,0.02,0.03,0.04,0.05,0.1,0.2],
             #'binary': [False,True],
             #'min_df': [1,2,3],
             #'use_svd': [False,True],
             #'tfidf': [0],
             #'preproc': [0],
             #'fit_area': ['test','all'],
            }
    #param_grid = [{'alpha': [0.04],'binarize':[0,0.001]}]
    return param_grid

class Model23(Model21):
  def __init__(self, 
    alpha=2, binarize=0, use_svd=False, tfidf=2, fit_area='test', preproc=0, min_df=3,
    stemmer=1, ngram_max=2, max_df=0.1, binary=True, max_features=300000, n_components=256,
    use_idf=1, smooth_idf=0, sublinear_tf=1, norm='l2', token_min=1,
    do_remove_stopwords = False,
    C=0.7
    ):
    super(Model23,self).__init__(clf='LR',
    alpha=alpha, binarize=binarize, use_svd=use_svd, tfidf=tfidf, fit_area=fit_area, preproc=preproc, 
    min_df=min_df, stemmer=stemmer, ngram_max=ngram_max, max_df=max_df, binary=binary, 
    max_features=max_features, n_components=n_components, use_idf=use_idf, smooth_idf=smooth_idf, 
    sublinear_tf=sublinear_tf, norm=norm, token_min=token_min,
    do_remove_stopwords = do_remove_stopwords,)
    self.C = C

  def _get_clf(self, clf):
    clf =lm.LogisticRegression(penalty='l2', dual=True, tol=0.00001, C=self.C, fit_intercept=True, 
        intercept_scaling=1.0, class_weight=None, random_state=random_state)
    return clf
  def get_param_grid(self, randomized=True):
    "get param_grid for this model to use with GridSearch"
    if randomized:
        param_grid = param_grid_NB
    else:
        param_grid = {
             'C':[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1,3],
            }
    return param_grid

class Model28(Model21):
  def __init__(self, use_best=False, use_bp=False, use_stats=True, use_table=True, clf='RFC',
    alpha=0.03, binarize=0.01, C=17,
    predict_bp=True, bp_clfs = ['MNB_28','BNB_28','LR_28'], use_scaler=0,
    max_depth=12,
    ):
    super(Model28,self).__init__(use_best=use_best, use_bp=use_bp, use_stats=use_stats, 
        use_table=use_table, clf=clf)
    self.alpha = alpha
    self.binarize = binarize
    self.C = C
    self.predict_bp = predict_bp
    self.bp_clfs =  bp_clfs
    self.use_scaler = use_scaler
    self.max_depth = max_depth

  def _get_clf(self, clf):
    MNB_28 = MultinomialNB(alpha=self.alpha)
    BNB_28 = BernoulliNB(alpha=self.alpha, binarize=self.binarize)
    LR_28  = lm.LogisticRegression(penalty='l2', dual=True, tol=0.00001, C=self.C, fit_intercept=True, 
        intercept_scaling=1.0, class_weight=None, random_state=random_state)
    RFC = RandomForestClassifier(n_estimators=500, max_depth=self.max_depth,
            n_jobs=-1, random_state=random_state, verbose=0)
    return RFC
  def get_param_grid(self, randomized=True):
    "get param_grid for this model to use with GridSearch"
    if randomized:
        raise NotImplementedError('no randomized for this class')
    else:
        param_grid = {
             #'alpha':[0.001, 0.003, 0.006, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.5, 0.8, 1, 2, 3],
             #'binarize':[0,0.0001,0.001,0.01,1],
             #'C':[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1,3,5,7,9],
             #'C':[12,13,14,15,16,17,18,19,20],
             'max_depth':[4,5,6,7,8]
            }
    return param_grid

class Model29(Model28):
  "Model28 with best"
  def __init__(self, use_best=True,
    bst_bgram=1,bst_minc=1,bst_title=0,bst_body=32,bst_url=2,
    sel_percentile=95,
    max_depth=5,
    ):
    
    super(Model29,self).__init__(use_best=True)
    
    self.bst_bgram = bst_bgram
    self.bst_minc = bst_minc
    self.bst_title = bst_title
    self.bst_body = bst_body
    self.bst_url = bst_url
    
    self.sel_percentile = sel_percentile

    self.max_depth = max_depth
  
  def _pipeline_append(self, pipelineList):
    "append any transformers into current pipeline"
    if self.sel_percentile:
        pipelineList.append( ('percentile',SelectPercentile(score_func=f_classif,
            percentile=self.sel_percentile)) )

  def get_param_grid(self, randomized=True):
    "get param_grid for this model to use with GridSearch"
    if randomized:
        param_grid = {
            'bst_bgram':[1],
            'bst_minc':[1],
            'bst_title':[0],
            'bst_body':[64],
            'bst_url':[0],
            'max_depth':[12],
            'sel_percentile':[95],
            }
    else:
        param_grid = {
            'max_depth':[4,5,6],
            #'sel_percentile':[93,95,97],
            }
    return param_grid

class Model30(Model28):
  def __init__(self,
    sel_percentile=95,
    max_depth=1,
    learning_rate=0.1, n_estimators=10,
    ):
    super(Model30,self).__init__()

    self.sel_percentile = sel_percentile
    self.max_depth = max_depth
    self.learning_rate = learning_rate
    self.n_estimators = n_estimators

  def _get_clf(self, clf):
    
    clf = AdaBoostClassifier(
                algorithm='SAMME.R',
                base_estimator=DecisionTreeClassifier(compute_importances=None, criterion='gini', max_depth=self.max_depth, max_features=1.0, min_density=None, min_samples_leaf=1, min_samples_split=2, random_state=random_state, splitter='best'),
                learning_rate=self.learning_rate, 
                n_estimators=self.n_estimators, 
                random_state=random_state)
    return clf
  
  def get_param_grid(self, randomized=True):
    "get param_grid for this model to use with GridSearch"
    if randomized:
        param_grid = {
            'max_depth':[1,2],
            'learning_rate':[0.1,0.3],
            'n_estimators':[10,50,100],
            }
    else:
        param_grid = {
            'max_depth':[1,2,3],
            #'learning_rate':[0.9,1.0],
            #'sel_percentile':[0,91,93,95,97]
            #'n_estimators':[20,21,22,23,24,25,26,27,28,29,30],
            #'n_estimators':[30,60,100]
            }
    return param_grid


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

