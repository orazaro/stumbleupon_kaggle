#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    Classifiers
"""

import sys, random
import numpy as np
import logging

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn import metrics

import sklearn.linear_model as lm
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA

from base import check_n_jobs

logger = logging.getLogger(__name__)

from sklearn.ensemble import RandomForestClassifier
import sklearn.linear_model as lm
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)

from sklearn import grid_search

def get_clf(cl,n_jobs=1,random_state=0):
    """ Select clasifier by name
    """
    lm1 = {'C':[0.0001, 0.001, 0.01, 0.1, 0.3, 1, 3, 10]}
    C_range = 10.0 ** np.arange(-5, 3)
    C_range = np.hstack([C_range,[0.3,3]])
    lm2 = dict(C=C_range)
    rf1 = {'max_depth':[2,4,8,16,24,32]}

    if cl=='rf2':
        clf = RandomForestClassifier(n_estimators=200, max_depth=2,
                max_features='auto',
                n_jobs=n_jobs, random_state=random_state, verbose=0)
    elif cl=='rf':
        clf1 = RandomForestClassifier(n_estimators=100, max_depth=2,
                max_features='auto',
                n_jobs=n_jobs, random_state=random_state, verbose=0)
        clf = grid_search.GridSearchCV(clf1, rf1, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='dt':
        from sklearn.tree import DecisionTreeClassifier
        clf1 = DecisionTreeClassifier(max_depth=2, max_features='auto')
        clf = grid_search.GridSearchCV(clf1, rf1, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='lr2':
        clf = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=random_state)

    elif cl=='lr1':
        clf = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0001, 
                             C=1.0, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=random_state)
    elif cl=='lr2g':
        est2 = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
             C=1, fit_intercept=True, intercept_scaling=1.0, 
             class_weight=None, random_state=random_state)
        clf = grid_search.GridSearchCV(est2, lm1, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='lr1g':
        est1 = lm.LogisticRegression(penalty='l1', dual=False, tol=0.0001, 
             C=1, fit_intercept=True, intercept_scaling=1.0, 
             class_weight=None, random_state=random_state)
        clf = grid_search.GridSearchCV(est1, lm1, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='svmL':
        clf = svm.LinearSVC(C=1.0,loss='l2',penalty='l2',dual=True,verbose=0)
    elif cl=='svmL1':
        clf = svm.LinearSVC(C=1.0,loss='l2',penalty='l1',dual=False,verbose=0)
    elif cl=='svmL2':
        clf = svm.LinearSVC(C=1.0,loss='l1',penalty='l2',verbose=0)
    elif cl=='svmL1g':
        #est3 = svm.SVC(kernel='linear',verbose=0)
        est3 = svm.LinearSVC(loss='l2',penalty='l1',dual=False,verbose=0)
        clf = grid_search.GridSearchCV(est3, lm1, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='svmL2g':
        #est3 = svm.SVC(kernel='linear',verbose=0)
        est3 = svm.LinearSVC(loss='l1',penalty='l2',verbose=0)
        clf = grid_search.GridSearchCV(est3, lm1, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='svmRg':
        #C_range = 10.0 ** np.arange(-2, 9)
        #gamma_range = 10.0 ** np.arange(-5, 4)
        C_range = 10.0 ** np.arange(-3, 4)
        gamma_range = 10.0 ** np.arange(-4, 3)
        svm2 = dict(gamma=gamma_range, C=C_range)
        est3 = svm.SVC(kernel='rbf',verbose=0)
        clf = grid_search.GridSearchCV(est3, svm2, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='svmP3':
        svm1 = {'C':[0.001,0.01,0.1,1.0,10],'gamma':[0.1,0.01,0.001,0.0001]}
        svm3 = {'C':[0.001,0.01,0.1,1.0,10],'gamma':[0.1,0.01,0.001,0.0001],
                                                        'coef0':[0,1]}
        est4 = svm.SVC(kernel='poly',degree=3,verbose=0)
        clf = grid_search.GridSearchCV(est4, svm3, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='mnb':
        clf = MultinomialNB(alpha=1.0)
    elif cl=='gnb':
        clf = GaussianNB()
    elif cl=='knn':
        knn1 = {'n_neighbors':2**np.arange(0, 8)}
        clf = grid_search.GridSearchCV(KNeighborsClassifier(), knn1, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='knn100':
        clf = KNeighborsClassifier(n_neighbors=100)
    elif cl=='knn1k':
        clf = KNeighborsClassifier(n_neighbors=1024)
    elif cl=='lda':
        clf = LDA()
    elif cl=='qda':
        clf = QDA()
    elif cl=='gb':
        gb1 = {'max_depth':[1,2,4,8],'n_estimators':[10,20,40,80,160]}
        clf = grid_search.GridSearchCV(
            GradientBoostingClassifier(learning_rate=0.1,
                random_state=random_state,verbose=0,subsample=1.0), 
            gb1, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='rcv':
        clf = RidgeCV_proba()
    elif cl=='lcv':
        clf = LassoCV_proba()
    elif cl=='lr':
        clf = LinearRegression_proba()
    else:
        raise ValueError("bad cl:%s"%cl)

    return clf

class LinearRegression_proba(lm.LinearRegression):
  def predict_proba(self,X):
    y = self.predict(X)
    y = 1./(1+np.exp(-(y-0.5)))
    return np.vstack((1-y,y)).T

class LassoCV_proba(lm.LassoCV):
  def predict_proba(self,X):
    logger.debug('alpha_=%s',self.alpha_)
    y = self.predict(X)
    y = 1./(1+np.exp(-(y-0.5)))
    return np.vstack((1-y,y)).T

class RidgeCV_proba(lm.RidgeCV):
  def predict_proba(self,X):
    logger.debug('alpha_=%s',self.alpha_)
    y = self.predict(X)
    if 0:
        y_min,y_max = y.min(),y.max()
        if y_max>y_min:
            y = (y-y_min)/(y_max-y_min)
    else:
        y = 1./(1+np.exp(-(y-0.5)))
    return np.vstack((1-y,y)).T

class KNeighborsClassifier_proba(KNeighborsClassifier):
  def predict_proba(X):
    y = super(KNeighborsClassifier_proba, self).predict_proba(X)
    y[np.isnan(y)]=0.5
    return y

class ConstClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, c = 0):
    self.c = c
  def fit(self, X, y=None):
    return self
  def predict_proba(self, X):
    X = np.asarray(X)
    y1=np.empty(X.shape[0]); 
    y1.fill(self.c)
    y_proba = np.vstack((1-y1,y1)).T
    return y_proba
  def predict(self, X):
    return self.predict_proba(X)[:,1]

class MeanClassifier(BaseEstimator, ClassifierMixin):
  def fit(self, X, y=None):
    return self
  def predict_proba(self, X):
    X = np.asarray(X)
    y1 = np.mean(X, axis=1)
    y_proba = np.vstack((1-y1,y1)).T
    return y_proba
  def predict(self, X):
    return self.predect_proba()[:,1]
        
class RoundClassifier(BaseEstimator, ClassifierMixin):
  """
    Classifier with rounding classes 
  """
  def __init__(self, est, rup=0, find_cutoff=False):
    self.est = est
    self.rup = rup
    self.find_cutoff = find_cutoff

  def fit(self, X, y):
    from imbalanced import find_best_cutoff,round_smote,round_down,round_up
    if self.rup > 0:
        X1,y1,_ = round_up(X,y) 
    elif self.rup < 0:
        if self.rup < -1:
            X1,y1 = round_smote(X,y) 
        else:
            X1,y1,_ = round_down(X,y) 
    else:
        X1,y1 = X,y
    self.est.fit(X1,y1)
    if self.find_cutoff:
        ypp = self.predict_proba(X)[:,1]
        self.cutoff = find_best_cutoff(y,ypp)
    else:
        self.cutoff = 0.5
    return self

  def predict_proba(self, X):
    X = np.asarray(X)
    if hasattr(self.est,'predict_proba'):
        y_proba = self.est.predict_proba(X)
    else:
        y = self.est.predict(X)
        y_proba = np.vstack((1-y,y)).T
    return y_proba
  def predict(self, X):
    if not self.find_cutoff and hasattr(self.est,'predict'):
        return self.est.predict(X)
    ypp = self.predict_proba(X)[:,1]
    return  np.array(map(int,ypp>self.cutoff))


def test():
    print "tests ok"

if __name__ == '__main__':
    random.seed(1)
    import argparse
    parser = argparse.ArgumentParser(description='Tuner.')
    parser.add_argument('cmd', nargs='?', default='test')
    args = parser.parse_args()
    print >>sys.stderr,args 
  
    if args.cmd == 'test':
        test()
    else:
        raise ValueError("bad cmd")

