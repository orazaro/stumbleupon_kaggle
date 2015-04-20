#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Models with Linear Models and RF
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
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier)
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
from kgml.base import check_n_jobs

from project import Project
import prepare
from prepare import BPobj

from model import Model

random_state = 1
logger = logging.getLogger(__name__)

param_grid_svd = {
     'C':[0.5, 0.7, 1, 1.5],
     'alpha':[2],
     'binarize':[0.1],
     'fit_area': ['test'],
     'stemmer': [1,3],
     'ngram_max': [1,2],
     'max_df': [0.01,0.1,0.2,0.7,1.0],
     'max_features': [600000],
     'binary': [False,True],
     'min_df': [1,2,3],
     'token_min': [1,2,3],
     'tfidf': [2],
     'preproc': [1],
     'do_remove_stopwords': [False,True],
     'use_svd': [True],
     'n_components': [32,512],
     'use_idf': [1],
     'smooth_idf': [0,1],
     'sublinear_tf': [0,1],
     'norm': ['l1','l2'],
    }

class Model16(Model):
  def __init__(self, use_best=False, use_bp=True, use_stats=False, use_table=False, clf = 'LR',
    alpha=2, binarize=0.1, use_svd=True, tfidf=2, fit_area='test', preproc=1, min_df=3,
    stemmer=1, ngram_max=2, max_df=0.7, binary=False, max_features=600000, n_components=512,
    C = 0.9,
    use_idf=1, smooth_idf=0, sublinear_tf=1, norm='l2', token_min=1,
    do_remove_stopwords = False,
    ):

    super(Model16,self).__init__(use_best=use_best, use_bp=use_bp, use_stats=use_stats, 
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
    self.C = C
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
    return prepare.Prepare_0(model=16, n_components=self.n_components, preproc=self.preproc, 
        min_df=self.min_df, use_svd=self.use_svd, tfidf=self.tfidf, stemmer=self.stemmer, 
        fit_area=self.fit_area, extra=extra_js)
  def _get_clf(self, clf):
    clf =lm.LogisticRegression(penalty='l2', dual=True, tol=0.00001, C=self.C, fit_intercept=True, 
        intercept_scaling=1.0, class_weight=None, random_state=random_state)
    return clf
  def get_param_grid(self, randomized=True):
    "get param_grid for this model to use with GridSearch"
    if randomized:
        param_grid = param_grid_svd
    else:
        param_grid = {
            #'n_components':[256,512]
            #'min_df':[3,5]
            #'ngram_max': [1,2],
            #'smooth_idf': [0,1],
            #'binary': [False,True],
            #'do_remove_stopwords': [False,True],
            #'max_df': [0.5,0.7,1.0],
            'C':[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,3],
            #'C':[0.7,0.8,0.9,1.0],
            }
    return param_grid

class Model24(Model16):
  def __init__(self, use_best=True,
    bst_bgram=1,bst_minc=2,bst_title=8,bst_body=16,bst_url=8,
    use_bp=True, use_stats=False, use_table=True, n_components=16, 
    C = 5,
    ):
    super(Model24,self).__init__(use_best=use_best, use_bp=use_bp, use_stats=use_stats, 
        use_table=use_table, n_components=n_components)
 
    self.use_best = use_best
    self.use_bp = use_bp
    self.use_stats = use_stats
    self.use_table = use_table
    self.n_components = n_components
    
    self.bst_bgram = bst_bgram
    self.bst_minc = bst_minc
    self.bst_title = bst_title
    self.bst_body = bst_body
    self.bst_url = bst_url
    
    self.C = C
  
  def _get_clf(self, clf):
    clf = lm.LogisticRegression(penalty='l2', dual=True, tol=0.00001, C=self.C, fit_intercept=True, 
            intercept_scaling=1.0, class_weight=None, random_state=random_state)
    return clf

  def get_param_grid(self, randomized=True):
    "get param_grid for this model to use with GridSearch"
    if randomized:
        param_grid = {
            'use_best': [True],
            'bst_bgram':[1],
            'bst_minc':[1,2,3],
            'bst_title':[8],
            'bst_body':[16],
            'bst_url':[8],
            'use_bp': [True],
            'use_stats': [False],
            'use_table': [True],
             'C':[0.8],
             #'n_components':[1,2,4,8,16,32,64,128,256,512]
             'n_components':[16],
            }
    else:
        param_grid = {
             #'C':[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1,3,5],
             'C':[5,7],
            }
    return param_grid

class Model25(Model16):
  def __init__(self, use_best=True,
    bst_bgram=1,bst_minc=1,bst_title=1,bst_body=32,bst_url=1,
    use_bp=True, use_stats=True, use_table=True, n_components=16, 
    max_depth=12,
    ):
    super(Model25,self).__init__(use_best=use_best, use_bp=use_bp, use_stats=use_stats, 
        use_table=use_table, n_components=n_components)
 
    self.use_best = use_best
    self.use_bp = use_bp
    self.use_stats = use_stats
    self.use_table = use_table
    self.n_components = n_components
    
    self.bst_bgram = bst_bgram
    self.bst_minc = bst_minc
    self.bst_title = bst_title
    self.bst_body = bst_body
    self.bst_url = bst_url
    
    self.max_depth = max_depth
  
  def _get_clf(self, clf):
    clf = RandomForestClassifier(n_estimators=500, max_depth=self.max_depth,
                n_jobs=check_n_jobs(-1), random_state=random_state, verbose=0)
    return clf

  def get_param_grid(self, randomized=True):
    "get param_grid for this model to use with GridSearch"
    if randomized:
        param_grid = {
            'use_best': [True],
            'bst_bgram':[1],
            'bst_minc':[1],
            'bst_title':[1],
            'bst_body':[32],
            'bst_url':[1],
            'use_bp': [True],
            'use_stats': [False],
            'use_table': [True],
             'max_depth':[12],
             'n_components':[16],
            }
    else:
        param_grid = {
            'max_depth':[9,10,11,12,13,14],
            }
    return param_grid

class Model26(Model25):
  def __init__(self, max_depth=19):
    super(Model26,self).__init__()
    self.max_depth = max_depth
  
  def _get_clf(self, clf):
    clf = ExtraTreesClassifier(n_estimators=500, max_depth=self.max_depth,
                n_jobs=check_n_jobs(-1), random_state=random_state, verbose=0)
    return clf

  def get_param_grid(self, randomized=True):
    "get param_grid for this model to use with GridSearch"
    if randomized:
        param_grid = {
             'max_depth':[12],
            }
    else:
        param_grid = {
            'max_depth':[18,19,20],
            }
    return param_grid

class Model27(Model16):
  "RFC experimental model"
  def __init__(self, use_best=True,
    bst_bgram=1,bst_minc=1,bst_title=0,bst_body=32,bst_url=2,
    use_bp=True, use_stats=True, use_table=True, n_components=16, 
    max_depth=12,
    pca_components=0,pca_whiten=True,sel_percentile=95,sel_chi2=False,
    ):
    super(Model27,self).__init__(use_best=use_best, use_bp=use_bp, use_stats=use_stats, 
        use_table=use_table, n_components=n_components)
 
    self.use_best = use_best
    self.use_bp = use_bp
    self.use_stats = use_stats
    self.use_table = use_table
    self.n_components = n_components
    
    self.bst_bgram = bst_bgram
    self.bst_minc = bst_minc
    self.bst_title = bst_title
    self.bst_body = bst_body
    self.bst_url = bst_url
    
    self.max_depth = max_depth

    self.pca_components = pca_components
    self.pca_whiten = pca_whiten
    self.sel_percentile = sel_percentile
    self.sel_chi2 = sel_chi2
  
  def _get_clf(self, clf):
    "CHANGE BACK TO 500!!!"
    clf = RandomForestClassifier(n_estimators=500, max_depth=self.max_depth,
                n_jobs=check_n_jobs(-1), random_state=random_state, verbose=0)
    return clf
  
  def _pipeline_append(self, pipelineList):
    "append any transformers into current pipeline"
    if self.sel_percentile:
        pipelineList.append( ('percentile',SelectPercentile(score_func=chi2 if self.sel_chi2 else f_classif,
            percentile=self.sel_percentile)) )
    if self.pca_components:
        pipelineList.append( ('pca',PCA(n_components=self.pca_components, whiten=self.pca_whiten)) )

  def get_param_grid(self, randomized=True):
    "get param_grid for this model to use with GridSearch"
    if randomized:
        param_grid = {
            'use_best': [True],
            'bst_bgram':[1],
            'bst_minc':[1],
            'bst_title':[0],
            'bst_body':[64],
            'bst_url':[0],
            'use_bp': [True],
            'use_stats': [True],
            'use_table': [True],
            'max_depth':[12],
            'n_components':[16],
            'pca_components':[0],
            'pca_whiten':[False],
            'sel_percentile':[95],
            'sel_chi2':[False],
            }
    else:
        param_grid = {
            #'bst_title':[0,1,2],
            #'bst_url':[2,3],
            #'pca_components':[0,60,80,100],
            #'pca_whiten':[False,True],
            #'bst_body':[28,32,36,40],
            #'n_components':[16],
            #'sel_percentile':[93,95,97],
            'max_depth':[11,12,13],
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

