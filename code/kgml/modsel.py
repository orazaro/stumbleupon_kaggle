#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
import os
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, mean_absolute_error, make_scorer
from sklearn.metrics import roc_curve, auc


def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss


def calc_roc_auc(y_test,y_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def estimate_scores(scores, scoring, sampling=True, n_sample=None, verbose=1):
    n_cv = len(scores)
    scores_mean = np.mean(scores)
    if scoring == 'roc_auc': #flip score < 0.5
        scores_mean = scores_mean if scores_mean >= 0.5 else 1 - scores_mean
    me = 1.96 * np.std(scores) 
    if sampling:
        if isinstance(scoring,basestring) and scoring=='accuracy':
            phat = scores_mean
            me = 1.96*np.sqrt(phat*(1-phat)/n_sample)
        else:
            me = me / np.sqrt(len(scores))

    if verbose > 0:
        print "%d Fold CV Score(%s): %.6f +- %.4f" % (n_cv, scoring, scores_mean, me,)
    return scores_mean, me

def bootstrap_632(n, n_iter, random_state=None):
    while n_iter > 0:
        train = np.random.randint(0,n,size=n)
        s_test = set(range(n)) - set(train)
        l_test = sorted(s_test)
        if len(l_test) > 0:
            if False:
                test_n = np.random.randint(0,len(l_test),size=n)
                test = np.asarray([l_test[i] for i in test_n])
            else:
                test = np.asarray(l_test)
            #print train,test
            yield (train,test)
            n_iter -= 1

def cv_select(y, random_state, n_cv, cv, test_size=0.1):
    if isinstance(cv,basestring):
        if cv == 'shuffle':
            return cross_validation.StratifiedShuffleSplit(y, n_cv, test_size=test_size, random_state=random_state)
        elif cv == 'loo':
            return cross_validation.LeaveOneOut(n_cv)
        elif cv == 'kfold':
            return cross_validation.StratifiedKFold(y, n_folds=n_cv)
        elif cv == 'boot':
            return cross_validation.Bootstrap(len(y), n_iter=n_cv, train_size=(1-test_size), random_state=random_state)
        elif cv == 'boot632':
            return bootstrap_632(len(y), n_iter=n_cv, random_state=random_state)
        # for regression
        elif cv == '_shuffle':
            return cross_validation.ShuffleSplit(len(y), n_iter=n_cv, test_size=test_size, random_state=random_state)
        elif cv == '_kfold':
            return cross_validation.KFold(len(y), n_folds=n_cv)
        else:
            raise ValueError("bad cv:%s"%cv)
    else:
        return cv

def cv_run(rd, X, y, random_state, n_cv=16, n_iter=0, n_jobs=-1, scoring='accuracy', cv='shuffle', test_size=0.1, sampling=True):
    """ possible scorong: accuracy,roc_auc,precision,average_precision,f1
    """
    n_jobs = 1 if os.uname()[0]=='Darwin' else n_jobs
    if n_cv == 0:
      n_cv = len(y) 
    if n_iter > 0:
        p =[]
        for i in range(n_iter):
            cv1 = cv_select(y, random_state+i, n_cv, cv, test_size)
            scores = cross_validation.cross_val_score(rd, X, y, cv=cv1, 
                scoring=scoring, n_jobs=n_jobs, verbose=0)
            scores_mean,_ = estimate_scores(scores,scoring,
                sampling=True,n_sample=len(y),verbose=1)
            p.append(scores_mean)
        scores_mean,me = estimate_scores(p,scoring,
                sampling=False,n_sample=len(y),verbose=1)
        if scoring == 'accuracy':
            phat = scores_mean
            print "\tme_binom_est =",1.96*np.sqrt(phat*(1-phat)/len(y))
    else:
        cv1 = cv_select(y, random_state, n_cv, cv, test_size)
        if isinstance(scoring,basestring) and scoring=='rmse':
            scores = cross_validation.cross_val_score(rd, X, y, cv=cv1, 
                scoring='mean_squared_error',
                n_jobs=n_jobs, verbose=0)
            scores = [np.sqrt(np.abs(e)) for e in scores]
        elif isinstance(scoring,basestring) and scoring=='nrmse':
            """normalized rmse"""
            scores = cross_validation.cross_val_score(rd, X, y, cv=cv1, 
                scoring='mean_squared_error',
                n_jobs=n_jobs, verbose=0)
            y_std = np.std(y)
            scores = [np.sqrt(np.abs(e))/y_std for e in scores]
        elif isinstance(scoring,basestring) and scoring=='mae':
            mae = make_scorer(mean_absolute_error, greater_is_better=False)
            scores = cross_validation.cross_val_score(rd, X, y, cv=cv1, 
                scoring=mae,
                n_jobs=n_jobs, verbose=0)
            scores = np.abs(scores)
        else:
            scores = cross_validation.cross_val_score(rd, X, y, cv=cv1, scoring=scoring,
                n_jobs=n_jobs, verbose=0)
        scores_mean,me = estimate_scores(scores,scoring,sampling,n_sample=len(y))
    return scores_mean,me

def cv_run_ids(rd, X, y, ids, random_state, n_cv = 16, n_jobs=-1, scoring='accuracy', cv='shuffle', test_size=0.1, sampling=True):
    """ possible scorong: accuracy,roc_auc,precision,average_precision,f1
    """
    n_jobs = 1 if os.uname()[0]=='Darwin' else n_jobs
    n_ids = len(ids)
    y_ids = [y[ids[i][0]] for i in range(n_ids)]
    #print "y_ids:",sum(y_ids),len(y_ids)
    if n_cv == 0:
      n_cv = len(y_ids) 
    cv_ids = cv_select(y_ids, random_state, n_cv, cv, test_size)
    cv1 = []
    for (a,b) in cv_ids:
      a1 = []
      for i in a:
          a1 = a1 + ids[i]
      b1 = []
      for i in b:
          b1 = b1 + ids[i]
      cv1.append( (np.array(a1),np.array(b1)) )
    scores = cross_validation.cross_val_score(rd, X, y, cv=cv1, scoring=scoring,
    n_jobs=n_jobs, verbose=0)
    #print scores
    scores_mean,me = estimate_scores(scores,scoring,sampling,n_sample=len(y))
    return scores_mean,me


def test():
    bootstrap_632(10, 5)
    print "tests ok"

if __name__ == '__main__':
    import random,sys
    random.seed(1)
    import argparse
    parser = argparse.ArgumentParser(description='ModSel.')
    parser.add_argument('cmd', nargs='?', default='test')
    args = parser.parse_args()
    print >>sys.stderr,args 
  
    if args.cmd == 'test':
        test()
    else:
        raise ValueError("bad cmd")
