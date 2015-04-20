#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    Model selection 
"""

import random, os, logging, json
import numpy as np
from scipy.stats import randint as sp_randint
from pprint import pformat

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import svm, linear_model
from sklearn import grid_search, cross_validation
from sklearn.base import clone
from sklearn.externals.joblib import Parallel, delayed

logger = logging.getLogger(__name__)

def cv_run(estimator, X, y, scoring='roc_auc', n_folds=16, n_iter=4, n_jobs=None, random_state=None):
    """
        Run cross validation using estimator in two modes:
            - shuffle and select 20% for test, rest for train 
                and run n_iter iterations
            - use n_folds to estimate all samples
    """
    #test_size=1./n_folds
    test_size=0.2
    to_shuffle = n_iter > 0
   
    # problems with Accelerate in MacOs
    if n_jobs is None:
        n_jobs = 1 if os.uname()[0]=='Darwin' else -1

    if to_shuffle:
        logger.info("Shuffled CV run X:%s y:%s",X.shape,y.shape)
        logger.info("n_iter: %d test_size:%.1f%% n_jobs=%d",n_iter,test_size*100,n_jobs)
    else:
        logger.info("CV run X:%s y:%s",X.shape,y.shape)
        logger.info("n_folds: %d n_jobs=%d",n_folds,n_jobs)
    #cv1 = cross_validation.KFold(len(y), n_folds=n_cv, random_state=random_state)
    
    if to_shuffle:
        cv1 = cross_validation.StratifiedShuffleSplit(y, n_iter=n_iter, test_size=test_size, 
                random_state=random_state)
        prefix = "%d Shuffled Iter(test=%.1f%%)" % (n_iter,test_size*100.)
    else:
        cv1 = cross_validation.StratifiedKFold(y, n_folds=n_folds)
        prefix = "%d Fold" % n_folds
   
    if 1:
        y_pred, scores = cross_val_predict_proba(estimator, X, y, scoring=scoring, cv=cv1, 
            n_jobs=n_jobs, verbose=1, fit_params=None, pre_dispatch='2*n_jobs')
    else:
        y_pred = np.zeros(len(y))
        scores = cross_validation.cross_val_score(estimator, X, y, cv=cv1, scoring=scoring, 
            #scoring=make_scorer(roc_auc_score), 
            n_jobs=n_jobs, verbose=1)
    
    logger.info("\nscores:%s",scores)
    logger.info("\n%s CV Score: %.6f +- %.4f",prefix, np.mean(scores), 2*np.std(scores))
    return y_pred, scores

def update_params_best_score(model, score1, params_fname='saved_params.json'):
    saved_params,best_scores = read_saved_params(params_fname)
    best_score = best_scores.get(model.get_name(),None)
    if best_score:
        best_score1 = np.mean([best_score,score1])
    else:
        best_score1 = score1
    logger.info("Update %s best score from %s to %.5f", model.get_name(),best_score,best_score1)
    best_scores[model.get_name()] = best_score1
    saved_params['_best_scores'] = best_scores
    with open(params_fname, 'w') as f:
        json.dump(saved_params, f, indent=4, separators=(',', ': '),
                              ensure_ascii=True, sort_keys=True)
    
def read_saved_params(params_fname): 
    try: 
        with open(params_fname) as f:
            saved_params = json.load(f)
    except IOError:
        saved_params = {}
    
    if '_best_scores' in saved_params:
        best_scores = saved_params['_best_scores']
    else:
        best_score = None
        best_scores = {}
    return saved_params,best_scores
    
def find_params(model, X, y, scoring='roc_auc', n_folds=16, n_iter=4, n_jobs=None, random_state=None,
    rnd_iter = 0,
    psearch=0, params_fname='saved_params.json'):
    """
        Find params for model using it as estimator in two modes:
            - shuffle and select 20% for test, rest for train 
                and run n_iter iterations
            - use n_folds to estimate all samples
        if n_iter > 0: use shuffle CV
        else: use ordinary CV
        if psearch == 0: read from file (donot grid_search)
        elif psearch > 0: do grid_search if not in file
        elif psearch > 1: do grid_search in any case
        elif psearch > 2: do grid_search using RandomizedSearchCV with n_iter = int(psearch)
    """
    #test_size=1./n_folds
    test_size=0.2
    to_shuffle = n_iter > 0
   
    # problems with Accelerate in MacOs
    if n_jobs is None:
        n_jobs = 1 if os.uname()[0]=='Darwin' else -1
  
    params = {}

    saved_params,best_scores = read_saved_params(params_fname)
    params.update(saved_params.get(model.get_name(), {}))
    best_score = best_scores.get(model.get_name(),None)

    if psearch>1 or (psearch and model.get_name() not in saved_params):
        # initialize model with last best params
        logger.debug('initialize model with last best params: %s',params)
        model.set_params(**params)
        param_grid = model.get_param_grid(randomized=(psearch > 2))
        if len(param_grid) == 0:
            logger.warning('empty param_grid for model %s, skiping grid_search..', model.get_name())
            return params
        if to_shuffle:
            logger.info("Shuffled %s run X:%s y:%s", 'RandomizedSearchCV' if psearch > 2 else 'GridSearchCV',
                X.shape,y.shape)
            logger.info("n_iter: %d test_size:%.1f%% n_jobs=%d",n_iter,test_size*100,n_jobs)
        else:
            logger.info("% run X:%s y:%s", 'RandomizedSearchCV' if psearch > 2 else 'GridSearchCV',
                X.shape,y.shape)
            logger.info("n_folds: %d n_jobs=%d",n_folds,n_jobs)
        #cv1 = cross_validation.KFold(len(y), n_folds=n_cv, random_state=random_state)
       
        if to_shuffle:
            cv1 = cross_validation.StratifiedShuffleSplit(y, n_iter=n_iter, test_size=test_size, 
                    random_state=random_state)
            prefix = "%d Shuffled Iter(test=%.1f%%)" % (n_iter,test_size*100.)
        else:
            cv1 = cross_validation.StratifiedKFold(y, n_folds=n_folds)
            prefix = "%d Fold" % n_folds
        
        if psearch > 2:
            clf = grid_search.RandomizedSearchCV(model, param_distributions=param_grid, n_iter=psearch, 
                cv=cv1, n_jobs=n_jobs, scoring = scoring)
        else:
            clf = grid_search.GridSearchCV(model, param_grid, 
                cv=cv1, n_jobs=n_jobs, scoring = scoring)
        clf.fit(X,y)
        logger.info("Grid Scores:\n%s", pformat(clf.grid_scores_))
        logger.info("found params (%s > %.5f): %s",
            model.get_name(),clf.best_score_, clf.best_params_)
        if best_score:
            if clf.best_score_ > best_score:
                logger.info("increased best score from %.5f to %.5f",best_score,clf.best_score_)
                best_score = clf.best_score_
                params.update(clf.best_params_)
            else:
                logger.warning("best score unchanged: %.5f > %.5f",best_score,clf.best_score_)
        else:
            best_score = clf.best_score_
            params.update(clf.best_params_)
        
        # reread saved params
        saved_params,best_scores = read_saved_params(params_fname)
        best_score_cur = best_scores.get(model.get_name(),None)
        if best_score_cur is None or best_score > best_score_cur:
            best_scores[model.get_name()] = best_score
            saved_params[model.get_name()] = params
            saved_params['_best_scores'] = best_scores
            with open(params_fname, 'w') as f:
                json.dump(saved_params, f, indent=4, separators=(',', ': '),
                                      ensure_ascii=True, sort_keys=True)
        else:
            logger.warning('While processing best_score changed: %.5f >= %.5f',best_score_cur,best_score)
    else:
        params.update(saved_params.get(model.get_name(), {}))
        if model.get_name() not in saved_params:
            logger.warning('%s not in saved_params',model.get_name())
        else:
            logger.info("using params %s: %s", model.get_name(), params)

    return params

def split_cv_grid(X,y,cv,n_samples=0.1):
    N = X.shape[0]
    if n_samples < 1:
        n_samples = int(N*n_samples)+1
    if n_samples==1 or n_samples >= N:
        raise ValueError("bad n_samples=%d" % n_samples)
    
    small = np.random.choice(N,n_samples, replace=False)
    small.sort()
    d1 = dict(zip(small,range(len(small))))
    big = np.array([x for x in range(N) if x not in d1])
    # big already sorted
    d2 = dict(zip(big,range(len(big))))
    X1, y1 = X[small,:],y[small] 
    assert len(d1) == X1.shape[0]
    X2, y2 = X[big,:],y[big] 
    def grid(cv1,d):
        cv_grid = []
        for (train,test) in cv1:
            x1 = [d[i] for i in train if i in d]
            x2 = [d[i] for i in test if i in d]
            cv_grid.append( (np.asarray(x1),np.asarray(x2)) )
        return cv_grid
    assert X1.shape[0]+X2.shape[0] == N
    return grid(cv,d1),X1,y1,grid(cv,d2),X2,y2

def make_cv_grid(X,y,cv=None,n_samples=0.1, verbose=0):
    # select small sample for grid_search 
    if isinstance(cv,int): 
        if verbose>0: print "cv:int"
        cv1 = cross_validation.KFold(X.shape[0],cv)
    elif isinstance(cv,float):
        if verbose>0: print "cv:float"
        cv1 = cross_validation.KFold(X.shape[0])
    elif not cv:
        if verbose>0: print "cv:not"
        cv1 = cross_validation.KFold(X.shape[0])
    else:
        if verbose>0: print "cv:ok"
        cv=list(cv) # to stop generator
        #if verbose: print "cv:",len(cv),"cv[0]:",cv[0]
        cv1 = cv
    N = X.shape[0]
    if n_samples == 0:
        return cv1,X,y
    elif n_samples < 1:
        n_samples = int(N*n_samples)
        if n_samples < 50:
            if verbose>0: print "Warning: too low n_samples:", n_samples
    if n_samples<=2 or n_samples >= N:
        return cv1,X,y
    
    small = np.random.choice(N,n_samples, replace=False)
    small.sort()
    d = dict(zip(small,range(len(small))))
    X_grid, y_grid = X[small,:],y[small] 
    assert len(d) == X_grid.shape[0]
    cv_grid = []
    for (train,test) in cv1:
        x1 = [d[i] for i in train if i in small]
        x2 = [d[i] for i in test if i in small]
        cv_grid.append( (np.asarray(x1),np.asarray(x2)) )
    return cv_grid,X_grid,y_grid

def _cross_val_predict(estimator, X, y, train, test, verbose,
                     fit_params, proba=False):
    """Inner loop for cross validation"""
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    
    estimator.fit(X_train, y_train, **fit_params)
    if proba:
        y_pred = estimator.predict_proba(X_test)[:,1]
    else:
        y_pred = estimator.predict(X_test)
    return (test,y_pred)


def cross_val_predict(estimator, X, y, loss=None, cv=None, n_jobs=1, 
        verbose=0, fit_params=None, proba=False,
        pre_dispatch='2*n_jobs'):
    fit_params = fit_params if fit_params is not None else {}
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    results = parallel(
        delayed(_cross_val_predict)(clone(estimator), X, y, train, test,
                                  verbose, fit_params, proba)
        for train, test in cv)
    y_pred = np.zeros(len(y))
    scores = []
    for (mask,y_p) in results:
        y_pred[mask] = y_p
        if loss:
            y_test = y[mask]
            scores.append(-loss(y_test, y_p))
    if loss:
        scores = np.asarray(scores)

    return np.asarray(y_pred),scores

from sklearn.metrics import roc_curve, auc
def compute_auc(y, y_pred):
    fpr, tpr, _ = roc_curve(y, y_pred)
    return auc(fpr, tpr)

def cross_val_predict_proba(estimator, X, y, scoring='roc_auc', cv=None, n_jobs=1, 
        verbose=0, fit_params=None,
        pre_dispatch='2*n_jobs'):
    fit_params = fit_params if fit_params is not None else {}
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    results = parallel(
        delayed(_cross_val_predict)(clone(estimator), X, y, train, test,
                                  verbose, fit_params, proba=True)
        for train, test in cv)
    y_pred = np.zeros(len(y))
    scores = []
    for (mask,y_p) in results:
        y_pred[mask] = y_p
        if scoring == 'roc_auc':
            y_test = y[mask]
            scores.append(compute_auc(y_test, y_p))
    return np.asarray(y_pred),np.asarray(scores)

### Regression specific
from regressor import MaeRegressor

def make_grid_search(clf, X, y, cv=4, n_samples=0.1, 
    n_estimators = 10, kernel='rbf', 
    n_iter=0,
    verbose=0): 
    alphas = {'alpha':[10*(0.1**i) for i in range(10)]}
    svm1 = {'C':[0.001,0.01,0.1],'gamma':[0.1,0.01,0.001,0.0001]}        
    #svm1 = {'C':[0.0001, 0.001, 0.01, 0.1, 1, 10],'gamma':[0.1,0.01,0.001]}        
    rf1 = {'max_depth':[6,12,24],'min_samples_leaf':[1,3,6,9,12,15,18,21,24,27,30]}
    sgd1 = {'alpha':alphas['alpha'],'loss':['huber'],'epsilon':[0.0001,0.001,0.01,0.1]}

    param_dist_rf = {"max_depth": sp_randint(3,30),#[3, None],
              #"max_features": sp_randint(1, X.shape[1]),
              #"min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1,30),
              #"bootstrap": [True, False],
              #"criterion": ["gini", "entropy"]
              }

    param_dist_gb = {"n_estimators" : [n_estimators],
              #"max_depth": [6,12,24], 
              "max_depth": sp_randint(1,30),
              #"max_features": sp_randint(1, X.shape[1]),
              #"min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": [1], #"min_samples_leaf": sp_randint(1,30),
              "subsample" : [0.5,1.0],
              #"bootstrap": [True, False],
              #"criterion": ["gini", "entropy"]
              }
    if clf=='gb' and n_iter == 0: n_iter = 20

    def f_sel(clf,kernel='rbf',n_estimators=10):
        "select estimator and params by estimator name"
        if clf in ('rf','ef'):
            parameters = param_dist_rf if n_iter > 0 else rf1
        if clf=='svm':
            parameters = svm1
            est = MaeRegressor(svm.SVR(kernel=kernel,verbose=verbose-1),'svm')
        elif clf=='gb':
            parameters = param_dist_gb
            est =MaeRegressor(GradientBoostingRegressor(n_estimators=n_estimators,verbose=verbose-1),'gb')
        elif clf=='rf':
            est =MaeRegressor(RandomForestRegressor(n_estimators=n_estimators,verbose=verbose-1),'rf')
        elif clf=='ef':
            est =MaeRegressor(ExtraTreesRegressor(n_estimators=n_estimators,verbose=verbose-1),'ef')
        elif clf=='lm':
            parameters = alphas        
            est = MaeRegressor(linear_model.Ridge(),'lm')
        else:
            parameters = sgd1       
            est = MaeRegressor(linear_model.SGDRegressor(),'sgd')
        return parameters,est

    parameters, est = f_sel(clf,kernel,n_estimators)
    
    cv_grid,X_grid,y_grid = make_cv_grid(X,y,cv,n_samples)
    
    if verbose:
        print "Search estimator parameters..",
    if n_iter>0 and clf in ('rf','ef','gb'):
        gs = grid_search.RandomizedSearchCV(est, param_distributions=parameters, n_iter=n_iter, 
            cv=cv_grid, n_jobs=-1, verbose=verbose-1).fit(X_grid,y_grid)
    else:
        gs = grid_search.GridSearchCV(est, parameters, 
            cv=cv_grid, n_jobs=-1, verbose=verbose-1).fit(X_grid,y_grid)
    if verbose:
        print gs.best_params_
        print "Pre Score:", gs.best_estimator_.score(X,y)
    #return clone(gs.best_estimator_)
    return gs

