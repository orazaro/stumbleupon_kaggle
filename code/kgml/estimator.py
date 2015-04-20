#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    estimators
"""

import sys, random
import numpy as np
from collections import defaultdict

from sklearn.base import BaseEstimator
from sklearn.base import clone
from base import check_n_jobs

class SplitEstimator(BaseEstimator):
  """   Estimator of set of estimators: perform split into set of estimators 
        and merge them back
        expects splitter_fn(X) which returns dictionary of tuples to select samples 
            ndarrays of indexes on axis=0 for this unit of split
        estimator -  estimator for unit of split
  """
  def __init__(self, estimator, features_dict=None, random_state=None, verbose=0):
    self.estimator = estimator
    self.features_dict = features_dict
    self.random_state = random_state
    self.verbose = verbose
  def fit(self, X, y):
    if isinstance(self.estimator, dict):
        self.estimator_dict = True
        self.estimators_ = self.estimator
    else:
        self.estimator_dict = False
        self.estimators_ = dict()
    X = np.asarray(X, dtype=np.float)
    y = np.asarray(y, dtype=np.float)
    n_samples, n_features = X.shape
    if n_samples != len(y):
        raise ValueError("Number of samples in X and y does not correspond:"
                            " %d != %d" % (n_samples, len(y)))
    
    classes = defaultdict(int)
    for (i,sel) in self.splitter_fn(X).iteritems():
        if self.verbose>0:
            print "SplitEstimator.fit on selector:",i
        if self.estimator_dict:
            if i not in self.estimators_:
                print >>sys.stderr,"Warning: class not in estimators dict"
            else:
                self.estimators_[i].fit(X[sel,:],y[sel])
        else:
            est = clone(self.estimator)
            est.fit(X[sel,:],y[sel])
            self.estimators_[i] = est
        classes[i] += len(sel)
    if self.verbose > 0:
        print "Classes:", classes.items()[:20]
    return self
  def predict(self, X):
    X = np.asarray(X, dtype=np.float)
    y_pred= np.zeros(X.shape[0])
    for (i,sel) in self.splitter_fn(X).iteritems():
        if i not in self.estimators_:
            print >>sys.stderr,"Warning: class not in training set, predict 0.0"
            y_pred[sel] = np.zeros(len(sel))
        else:
            y_pred[sel] = self.estimators_[i].predict(X[sel,:])
    return y_pred
  def splitter_fn(self,X):
    raise NotImplementedError("this function is virtual")


###### Tests

class TestSplitEstimator(SplitEstimator):
  def splitter_fn(self,X):
    sels = defaultdict(list)
    for i in range(X.shape[0]):
        sels[i%3].append(i)
    #print sels
    return sels

def test_SplitEstimator():
    from sklearn import linear_model
    from base import t_round_equal
    random.seed(1)
    splitter1 = TestSplitEstimator(linear_model.Ridge(),verbose=1)
    def test0(splitter):
        X = [[1,2,3],[4,6,8],[2,4,6]]
        X1 = [[1.3,2,3],[4.1,5.8,8],[2.1,3.9,6.4]]
        X = X + X1
        X2 = [[3,10,3],[41,5.8,8],[21,3.9,6.4]]
        y = np.array([1.1,4,20])
        y = np.hstack((y,y*1.01))
        
        splitter.fit(X,y)
        y_pred = splitter.predict(X1)
        dy = np.sum(np.abs(y[:3]-y_pred))
        print y,y_pred,dy
        assert t_round_equal(dy,0.134481527717)

        y_pred = splitter.predict(X2)
        dy = np.sum(np.abs(y[:3]-y_pred))
        print y,y_pred,dy
        assert t_round_equal(dy,0.382560233656)
    test0(splitter1)
    splitter2 = TestSplitEstimator(
        {i:linear_model.Ridge() for i in range(3)},
        verbose=1)
    test0(splitter2)



if __name__ == '__main__':
    random.seed(1)
    import argparse
    parser = argparse.ArgumentParser(description='Estimator.')
    parser.add_argument('cmd', nargs='?', default='test')
    args = parser.parse_args()
    print >>sys.stderr,args 
  
    if args.cmd == 'test':
        test_SplitEstimator()
        print "tests ok"
    else:
        raise ValueError("bad cmd")

