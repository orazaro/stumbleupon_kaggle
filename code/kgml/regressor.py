#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    estimators tuner
"""

import sys, random
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm

import sklearn.linear_model as lm
from sklearn import grid_search

from base import check_n_jobs

"add link coef_ to feature_importances_"
for cla in [RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor]:
    if 'coef_' not in dir(cla):
        cla.coef_ = property(lambda self:self.feature_importances_)

def is_rgr(cl):
    """ Check if cl is regressor """
    return cl[0]=='_'

def get_rgr(cl,n_jobs=1,random_state=0):
    """ Select regressor by name
    """
    if cl=='_rf':
        clf1 = RandomForestRegressor(n_estimators=100, max_depth=2,
                max_features='auto',
                n_jobs=n_jobs, random_state=random_state, verbose=0)
        rf1 = {'max_depth':[2,4,8,16,24,32]}
        clf = grid_search.GridSearchCV(clf1, rf1, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='_lr':
        clf = lm.LinearRegression()
    elif cl=='_ridge':
        clf = lm.RidgeCV()
    elif cl=='_lasso':
        clf = lm.LassoCV();
    elif cl=='_svmRg':
        C_range = 10.0 ** np.arange(-3, 4)
        gamma_range = 10.0 ** np.arange(-4, 3)
        svm2 = dict(gamma=gamma_range, C=C_range)
        est3 = svm.SVR(kernel='rbf',verbose=0)
        clf = grid_search.GridSearchCV(est3, svm2, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='_svmP3':
        svm1 = {'C':[0.001,0.01,0.1,1.0,10],'gamma':[0.1,0.01,0.001,0.0001]}
        svm3 = {'C':[0.001,0.01,0.1,1.0,10],'gamma':[0.1,0.01,0.001,0.0001],
                                                        'coef0':[0,1]}
        est4 = svm.SVR(kernel='poly',degree=3,verbose=0)
        clf = grid_search.GridSearchCV(est4, svm3, cv=4, n_jobs=n_jobs, verbose=0)
    elif cl=='_knn':
        knn1 = {'n_neighbors':2**np.arange(0, 8)}
        clf = grid_search.GridSearchCV(KNeighborsRegressor(), knn1, cv=4, n_jobs=n_jobs, verbose=0)
    else:
        raise ValueError("bad cl:%s"%cl)

    return clf
    

class MaeRegressor(BaseEstimator, RegressorMixin):
  """estimator with MAE score"""
  def __init__(self, clf, clftype="", 
        kernel='rbf', degree=3, gamma=0.0, coef0=0.0, tol=0.001, C=1.0,
        alpha=1.0, loss='huber', epsilon=0.1,
        n_estimators=10, max_features='auto', max_depth=None, min_samples_leaf=1,
        subsample=1.0, oob_score=False,
        random_state=None,
        verbose=0,
        ):
    self.clf = clf
    self.clftype = clftype
    self.kernel = kernel
    self.degree = degree
    self.gamma = gamma
    self.coef0 =coef0
    self.tol= tol
    self.C = C
    self.alpha = alpha
    self.loss = loss
    self.epsilon = epsilon
    self.n_estimators = n_estimators
    self.max_features = max_features
    self.max_depth = max_depth
    self.min_samples_leaf = min_samples_leaf
    self.subsample = subsample
    self.oob_score = oob_score
    self.random_state = random_state
    self.verbose = verbose
  def fit(self, X, y):
    if self.clftype == 'svm':
        self.clf.set_params(C=self.C,gamma=self.gamma)
    elif self.clftype == 'lm':
        self.clf.set_params(alpha=self.alpha)
    elif self.clftype == 'sgd':
        self.clf.set_params(alpha=self.alpha,loss=self.loss,epsilon=self.epsilon,
            random_state=self.random_state)
    elif self.clftype == 'rf' or self.clftype == 'ef':
        self.clf.set_params(n_estimators=self.n_estimators, max_features=self.max_features, 
            max_depth=self.max_depth,min_samples_leaf=self.min_samples_leaf,
            oob_score = self.oob_score,
            random_state=self.random_state,
            verbose=self.verbose,n_jobs=check_n_jobs(-1))
    elif self.clftype == 'gb':
        self.clf.set_params(n_estimators=self.n_estimators, max_features=self.max_features, 
            max_depth=self.max_depth,min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
            verbose=self.verbose)
    self.clf.fit(X,y)
    return self
  def predict(self, X):
    return self.clf.predict(X)
  def score(self, X, y):
    y_pred = self.predict(X)
    return -mean_absolute_error(y, y_pred)
  @property
  def coef_(self):
    if 'coef_' in dir(self.clf):
        return self.clf.coef_
    elif 'feature_importances_' in dir(self.clf):
        return self.clf.feature_importances_
    else:
        raise ValueError('no coef_ or feature_importances_')


def scorer_gbr_lad(clf, X, y, verbose=1):
    """Scorer for GradientBoostingRegressor with los='lad' """
    y_pred = clf.predict(X)
    score = -mean_absolute_error(y, y_pred)
    if verbose >0:
        print >>sys.stderr,"Eout=",-score
        if 'staged_predict' in dir(clf):
            if verbose>0: print("Staged predicts (Eout)")
            for i,y_pred in enumerate(clf.staged_predict(X)):
                Eout = mean_absolute_error(y,y_pred)
                if verbose>0: print "tree %3d, test score %f" % (i+1,Eout)
    return score

"""
class GBRegressor(GradientBoostingRegressor):
  def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, min_samples_split=2, min_samples_leaf=1,
                 max_depth=3, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0):

    super(GBRegressor, self).__init__(loss, learning_rate, n_estimators,
                 subsample, min_samples_split, min_samples_leaf,
                 max_depth, init, random_state,
                 max_features, alpha, verbose)
"""



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

