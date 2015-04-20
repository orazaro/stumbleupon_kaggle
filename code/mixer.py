#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Models mixture
"""
from __future__ import division

import sys, random, pickle, copy, json, os, re
import datetime, csv, gzip, joblib
from collections import defaultdict
import logging

import numpy as np
import scipy as sp
import pandas as pd
from scipy import sparse

from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
from sklearn import linear_model
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
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.base import clone
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from kgml.nltk_preprocessing import preprocess_pipeline
from kgml.rfecv import RFECVp
from kgml.feature_extraction import select_topwords, extract_words
from kgml.model_selection import cv_run, find_params, update_params_best_score
from kgml.paulduan_ml import AUCRegressor
from kgml.classifier import LassoCV_proba,RidgeCV_proba,KNeighborsClassifier_proba, MeanClassifier

from project import Project
import prepare
from prepare import BPobj, Prepare
   
logging.basicConfig(
format='[%(asctime)s %(name)s %(levelname)s] %(message)s',
#format="[%(asctime)s] %(levelname)s\t%(message)s",
                    filename="history.log", filemode='a', level=logging.DEBUG,
                    datefmt='%m/%d/%y %H:%M:%S')
#formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s", datefmt='%m/%d/%y %H:%M:%S')
formatter = logging.Formatter('%(levelname)s.%(name)s: %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

#logger = logging.getLogger(__name__)
logger = logging.getLogger('mixer')
 
"""
Models:
02      pre=0;         Tfidf(df=3) SVD(500)      -                LogReg(l2,C=1)            .8775 .88213
09(04)  pre=0;         Tfidf(df=1) SVD(128)      -                LogReg(l2,C=1)            .8767 .88307
10      pre=1,stem=0;  Tfidf(df=1) no SVD        -                MultinomialNB(alpha=0.8)  .8658
11      pre=1,stem=0;  Tfidf(df=1) no SVD        -                LogReg(l2,C=1) 
12(06)  pre=0;         Tfidf(df=3) SVD(128)      tbl+stats+Scale  LogReg(l2,C=1)            .8757 .88226  
13(08)  pre=0;         Tfidf(df=3) SVD(128)      tbl+stats+Scale  LogReg(l1,C=0.05)         .8780 .88083
14(02)  pre=1;         Tfidf(df=1) SVD(512)      -                LogReg(l2,C=1)            .8780  NA

 
"""
from model02 import Model02
from model09 import Model09
from model10 import Model10
from model11 import Model11
from model12 import Model12
from model13 import Model13
from model14 import Model14
from model15 import Model15,Model17,Model18,Model19,Model20
from model16 import Model16,Model24,Model25,Model26,Model27
from model21 import Model21,Model22,Model23,Model28,Model29,Model30

random_state = 1

class ModelStack(BaseEstimator):
  """
  Estimator as a stack of base estimators.
  Predict in two ways:
  - mean of base estimators with possibility of model selection
  - fit_predict on the results of base estimators predictions
  """
  coefs = defaultdict(list)

  def __init__(self, models = (Model02(),Model09()), stack=0, use_vote=0, n_folds_stack=16, gnrl='KNC',
        modsel=0, rfe=0, use_logit=0):
    self.models = models    # list of model type of Model
    self.stack = stack  # to use stack of models
    self.use_vote = use_vote    # use vote instead of mean
    self.n_folds_stack = n_folds_stack  # number of CV folds for stack 
    self.gnrl = gnrl    # abbr of model for generalization
    self.modsel = modsel    # model selection ON/OFF
    self.rfe = rfe    # use RFECVp to select best models
    self.use_logit = use_logit

  @classmethod
  def _get_generalizer(cls, gnrl):
    generalizers = dict(
    MEAN = MeanClassifier(),
    RFC = RandomForestClassifier(n_estimators=500, max_depth=32, n_jobs=-1, random_state=random_state),
    RCV = lm.RidgeCV(alphas=np.linspace(0, 200), cv=100),
    RCVp = RidgeCV_proba(alphas=np.linspace(0, 200), cv=100),
    LCV = lm.LassoCV(),
    LCVp = LassoCV_proba(),
    LSVC = svm.LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=random_state),
    SVC = svm.SVC(C=1.0, kernel='linear', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=random_state),
    LR = lm.LogisticRegression(penalty='l2', dual=True, tol=0.00001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=random_state),
    KNCuniform = KNeighborsClassifier(n_neighbors=1024, weights='uniform'),
    KNC = KNeighborsClassifier(n_neighbors=1024, weights='distance'),
    AUCR = AUCRegressor(),
    ABC_DTC = AdaBoostClassifier(
                algorithm='SAMME.R',
                base_estimator=DecisionTreeClassifier(compute_importances=None, criterion='gini', max_depth=1, max_features=1.0, min_density=None, min_samples_leaf=1, min_samples_split=2, random_state=random_state, splitter='best'),
                learning_rate=0.1, 
                n_estimators=200, 
                random_state=random_state),
    )
    return generalizers[gnrl]

  @classmethod
  def mean_coefs(cls):
    coefs = cls.coefs
    mcoefs = {m:np.mean(coefs[m]) for m in coefs}
    # normalize
    mcoefs_sum = sum(mcoefs.values())
    mcoefs = {m:mcoefs[m]/mcoefs_sum for m in mcoefs}
    return mcoefs

  @staticmethod
  def _str_ident(X_train, y_train, n_folds):
    ident = abs(hash(tuple(list(X_train)+list(y_train)+[n_folds])))
    pref = '%d%d%d%d%d' % tuple(X_train[:5])
    return '%s_%s'%(pref,ident)

  def _get_model_cv_fname(self, model, X_train, y_train, n_folds):
    ident = self._str_ident(X_train, y_train, n_folds)
    fname = "../data/cache/mm_preds_cv_%s_%s.pkl" % (re.sub("[a-z]", '', model.__class__.__name__),ident)
    return fname

  def _get_model_cv_preds(self, model, X_train, y_train):
    """
    Return cross-validation predictions on the training set
    """
    fname = self._get_model_cv_fname(model, X_train, y_train, self.n_folds_stack)
    try:
        logger.debug("trying to load cv_pred from  %s", fname)
        with open(fname,"rb") as f:
            stack_preds = pickle.load(f)
    except IOError:
        logger.debug("not found: %s", fname)
        stack_preds = None

    if stack_preds is None:
        kfold = cross_validation.StratifiedKFold(y_train, self.n_folds_stack)
        stack_preds = []
        indexes_cv = []
        for stage0, stack in kfold:
            model.fit(X_train[stage0], y_train[stage0])
            stack_preds.extend(list(model.predict_proba(
                X_train[stack])[:, 1]))
            indexes_cv.extend(list(stack))
        stack_preds = np.array(stack_preds)[sp.argsort(indexes_cv)]
    
        with open(fname,"wb") as f:
            pickle.dump(stack_preds,f)
    
    if self.use_logit and self.gnrl=='LR':
        logger.debug('transform stack_preds(%s) using logit',stack_preds.shape)
        stack_preds = logit(stack_preds)
    
    return stack_preds
 
  def _find_best_subset(self, y, predictions_list):
    """Finds the combination of models that produce the best AUC."""
    import multiprocessing, itertools
    from functools import partial
    from operator import itemgetter
    from kgml.paulduan_ml import compute_subset_auc
    best_subset_indices = range(len(predictions_list))

    pool = multiprocessing.Pool(processes=4)
    partial_compute_subset_auc = partial(compute_subset_auc,
                                         pred_set=predictions_list, y=y)
    best_auc = 0
    best_n = 0
    best_indices = []

    if len(predictions_list) == 1:
        return [1]

    n_start = int(len(predictions_list)*0.5) if self.modsel == 1 else self.modsel
    n_end = len(predictions_list)+1
    if n_end > 10 and n_end > n_start+9: n_end = n_start+9
    for n in range(n_start, n_end):
        cb = itertools.combinations(range(len(predictions_list)), n)
        combination_results = pool.map(partial_compute_subset_auc, cb)
        #logger.debug('combination_results: %s',combination_results)
        best_subset_auc, best_subset_indices = max(
            combination_results, key=itemgetter(0))
        print "- best subset auc (%d models): %.5f > %s" % (
            n, best_subset_auc, list(best_subset_indices))
        if best_subset_auc > best_auc:
            best_auc = best_subset_auc
            best_n = n
            best_indices = list(best_subset_indices)
    pool.terminate()

    logger.info("best auc: %.5f", best_auc)
    logger.info("best n: %d", best_n)
    logger.info("best indices: %s", best_indices)
    for i, model in enumerate(self.models):
        if i in best_indices:
            logger.info("> model: %s", model.__class__.__name__)
    logger.info("eliminated models: %s",[int(m.__class__.__name__[5:]) for i,m in enumerate(self.models) 
        if i not in best_indices])

    return best_indices

  def fit(self, Xmask, y):
    if self.stack or self.modsel:
        stage0_train = []
        for model in self.models:
            model_stack_preds = self._get_model_cv_preds(model, Xmask, y)
            stage0_train.append(model_stack_preds)
        if self.modsel:
            self.best_subset = self._find_best_subset(y,stage0_train)
            stage0_train = [pred for i,pred in enumerate(stage0_train) 
                if i in self.best_subset]
            self.best_models = [model for i,model in enumerate(self.models)
                if i in self.best_subset]

        X_train = np.array(stage0_train).T
        logger.debug("generalizer.fit Xtrain: %s y: %s", X_train.shape,y.shape)
        clf = self._get_generalizer(self.gnrl)
        if self.rfe:
            #clf2 = self._get_generalizer('RCVp')
            clf2 = self._get_generalizer('LR')
            self.generalizer = RFECVp(clf,clf2, step=1, cv=self.rfe, scoring="roc_auc", verbose=1)
        else:
            self.generalizer = clf
        self.generalizer.fit(X_train,y)
        if self.rfe:
            logger.info('RFE: selected features:')
            support = self.generalizer.get_support()
            for i, model in enumerate(self.models):
                if support[i]:
                    logger.info("> model: %s", model.__class__.__name__)
        
        if not hasattr(self.generalizer,'coef_'):
            if hasattr(self.generalizer,'feature_importances_'):
                self.generalizer.coef_ = self.generalizer.feature_importances_
        if hasattr(self.generalizer,'coef_'):
            logger.info("generalizer coef_:")
            for (c,m) in zip(self.generalizer.coef_.ravel(),self.models):
                ModelStack.coefs[m.__class__.__name__].append(c)
                logger.info("   %.2f %s",c,m.__class__.__name__)

    # fit models to full Xmask
    if self.modsel:
        for rd in self.best_models:
            rd.fit(Xmask,y)
    else:
        for rd in self.models:
            rd.fit(Xmask,y)
    
    return self
 
  def _try_predict_proba(self, rd, Xmask):
    if hasattr(rd,'predict_proba'):
        return rd.predict_proba(Xmask)
    else:
        y = rd.predict(Xmask)
        if 1:
            logger.warning("Predict instead predict_proba => scaled")
            y_min,y_max = y.min(),y.max()
            if y_max>y_min:
                y = (y-y_min)/(y_max-y_min)
        else:
            logger.warning("Predict instead predict_proba => sigmoid, 0.5 as threshold")
            y = 1./(1+np.exp(-(y-0.5)))
        y_proba = np.vstack((1-y,y)).T
        return y_proba

  def predict_proba(self, Xmask):
    pp = []
    if self.modsel:
        for rd in self.best_models:
            pp.append( self._try_predict_proba(rd, Xmask)[:,1] )
    else:
        for rd in self.models:
            pp.append( self._try_predict_proba(rd, Xmask)[:,1] )
    X_predict = np.array(pp).T
    if self.stack:
        return self._try_predict_proba(self.generalizer,X_predict)
    else:
        if self.use_vote:
            X_predict = X_predict>0.5
            mean_preds = np.mean(X_predict, axis=1)
        else:
            mean_preds = np.mean(X_predict, axis=1)
        logger.debug("X_predict: %s mean_preds: %s",X_predict.shape, mean_preds.shape)
        y_proba = np.vstack((1-mean_preds,mean_preds)).T
        return y_proba
  def predict(self, Xmask):
    pp = self.predict_proba(Xmask)
    return pp[:,1]
  def starter(self):
    for model in self.models:
        model.starter()

def logit(X):
    "transform prediction using logit func (inverse of sigmoid)"
    X1 = np.array(X)
    X1[X1>0.9999999]=0.9999999
    X1[X1<0.0000001]=0.0000001
    X1 = -np.log((1-X1)/X1)
    return X1


def main(n_iter, n_folds, smodels, n_jobs=None, stack=0, use_vote=0, gnrl='KNC', 
        modsel=0, rfe=0, psearch=0,
        starter=0, verbose=0, submit=0):
    y, colnames, n_train, n_test, n_all = prepare.Prepare_0().load_y_colnames()
    X_all = np.arange(n_all)

    models = []
    for m in smodels.split('+'):
        models.append( eval('Model%02d()'%int(m)) )
    #models = (Model02(),Model12(),Model10(),) # ***
    logger.debug("models:%s", models)
    X = X_all[:n_train]

    logger.info('Find params for models')
    for model in models:
        model.set_params(**find_params(model, X, y, scoring='roc_auc', n_iter=n_iter,
            n_jobs=n_jobs, random_state=random_state+1, psearch=psearch)
        )

    rd = ModelStack(models,gnrl=gnrl,stack=stack, use_vote=use_vote,
        modsel=modsel,rfe=rfe)
    if starter:
        logger.info('Starters start')
        rd.starter()
   
    if psearch > 1 and len(models)==1:  # update current model best score
        y_pred, scores = cv_run(rd, X, y, n_folds=n_folds, n_iter=n_iter, 
            n_jobs=n_jobs, random_state=random_state+2)
        update_params_best_score(models[0], np.mean(scores))
        return
    elif not submit:
        logger.debug('Cross validation starts')
        y_pred, scores = cv_run(rd, X, y, n_folds=n_folds, n_iter=n_iter, 
            n_jobs=n_jobs, random_state=random_state)
        prepare.Prepare_0().dump_ypred_residuals(y,y_pred)
        if verbose > 1:
            plot_errors(X,y,y_pred)
        if stack:
            logger.info("Mean Coefs: %s", rd.mean_coefs())
        return
    else:
        logger.info("Prepare submission..")

    logger.info("training on full data")
    rd.fit(X_all[:n_train],y)
    Xtest = X_all[n_train:]
    pred = rd.predict_proba(Xtest)[:,1]
    import submit
    submit.do_submit(pred)

def plot_errors(X,y,y_pred):
    import matplotlib.pyplot as plt
    import pylab
    pylab.rcParams['figure.figsize'] = 12, 10
    plt.figure(1)
    nsub = 110
    plt.subplot(nsub+1)
    plt.plot(range(len(y)),y-y_pred,'.')
    plt.title('y');
    plt.ylabel('dy');
    plt.show()


def test():
    print "tests ok"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare.')
    parser.add_argument('cmd', nargs='?', default='main')
    parser.add_argument('-update', default='0')    
    parser.add_argument('-rs', default=None)
    parser.add_argument('-submit', default='0')
    parser.add_argument('-niter', default=4)
    parser.add_argument('-folds', default=16)
    parser.add_argument('-models', default='12+10')
    parser.add_argument('-stack', default=0)
    parser.add_argument('-vote', default=0)
    parser.add_argument('-gnrl', default='KNC')
    parser.add_argument('-starter', default=0)
    parser.add_argument('-modsel', default=0)
    parser.add_argument('-rfe', default=0)
    parser.add_argument('-jobs', default=0)
    parser.add_argument('-psearch', default=0)
    parser.add_argument('-verbose', default=0)
    args = parser.parse_args()
    logger.info("%s",args) 
    logging.basicConfig(level=logging.DEBUG)
  
    if args.rs:
        random_state = int(args.rs)
    if args.submit and int(args.submit):
        random_state = 1961
        np.random_state = 1961
    if random_state:
        logger.info("random_state:%s", random_state)
        random.seed(random_state)
        np.random.seed(random_state)
    
    if args.cmd == 'test':
        test()
    elif args.cmd == 'main':
        main(
            n_iter=int(args.niter), 
            n_folds=int(args.folds), 
            smodels=args.models, 
            stack=int(args.stack), 
            use_vote=int(args.vote), 
            gnrl=args.gnrl,
            starter=int(args.starter),
            modsel=int(args.modsel),
            rfe=int(args.rfe),
            psearch=int(args.psearch),
            n_jobs=int(args.jobs) if args.jobs else None, 
            verbose=int(args.verbose),
            submit=int(args.submit))
    else:
        raise ValueError("bad cmd")

