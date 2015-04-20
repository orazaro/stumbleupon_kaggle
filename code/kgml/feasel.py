#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    features selection
"""
import numpy as np
import csv, random
from itertools import combinations

random_state = 1

### Transformer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)

def whiten(X,fudge=1E-18):
   """ http://stackoverflow.com/questions/6574782/how-to-whiten-matrix-in-pca """
   from numpy import dot, sqrt, diag
   from numpy.linalg import eigh

   # the matrix X should be observations-by-components

   # get the covariance matrix
   Xcov = dot(X.T,X)

   # eigenvalue decomposition of the covariance matrix
   d,V = eigh(Xcov)

   # a fudge factor can be used so that eigenvectors associated with
   # small eigenvalues do not get overamplified.
   D = diag(1./sqrt(d+fudge))

   # whitening matrix
   W = dot(dot(V,D),V.T)

   # multiply by the whitening matrix
   X = dot(X,W)

   return X,W

class Whitener(BaseEstimator, TransformerMixin):
  """
    Whiten matrix of data
  """
  def __init__(self, fudge=1E-18):
    self.fudge = fudge
  
  @staticmethod
  def f_whitening(X, fudge=1E-18):
    """ http://stackoverflow.com/questions/6574782/how-to-whiten-matrix-in-pca
        the matrix X should be observations-by-components
    """
    from numpy import dot, sqrt, diag
    from numpy.linalg import eigh

    # get the covariance matrix
    Xcov = dot(X.T,X)

    # eigenvalue decomposition of the covariance matrix
    d,V = eigh(Xcov)

    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    D = diag(1./sqrt(d+fudge))

    # whitening matrix
    W = dot(dot(V,D),V.T)

    # multiply by the whitening matrix
    #X = dot(X,W)

    #return X,W
    return W
  
  def fit(self, X, y=None):
    self.W = self.f_whitening(X,fudge=self.fudge)
    return self

  def transform(self, X_df):
    return np.dot(X_df,self.W)

class Cutter(BaseEstimator, TransformerMixin):
  """
    Transformer to select features from n1 to n2
  """
  def __init__(self, n1 = 0, n2 = 1000):
    self.n1 = n1
    self.n2 = n2
  
  def fit(self, X, y = None):
    return self

  def transform(self, X_df):
    #print "transforming data X_df:", X_df.shape
    return X_df[:,self.n1:self.n2]


class VarSel(BaseEstimator, TransformerMixin):
  """
    Transformer to select features with max variance
  """
  def __init__(self, k = 300, std_ceil = 0):
    self.k = k
    self.std_ceil = std_ceil
  
  def fit(self, X, y = None):
    feature_importances = np.var(X,axis=0)
    fi = sorted(zip(feature_importances,range(len(feature_importances))),reverse=True)
    if self.std_ceil > 0:
        fi_mean = np.mean(feature_importances)
        fi_std_max = fi_mean + np.std(feature_importances) * self.std_ceil
        print "fi filter: from",len(fi),
        fi = [e for e in fi if e[0] < fi_std_max]
        print "to",len(fi)
    if self.k > 0:
        self.features_selected = [e[1] for e in fi[:self.k]]
    else:
        self.features_selected = [e[1] for e in fi]
    #print "sorted features:",fi[:20],"selected:",self.features_selected
    return self

  def transform(self, X_df):
    return X_df[:,self.features_selected]

class Quadratic(BaseEstimator, TransformerMixin):
  """
    Transformer to add quadratic combinations of k first features
  """
  def __init__(self, k = 6, inter = False):
    self.k = k
    self.inter = inter
  
  def fit(self, X, y = None):
    return self

  def transform(self, X_df):
    X1 = X_df[:,:self.k]
    X1 = X1 * X1
    X1 = np.hstack([X_df,X1])
    if self.inter:
        X2 = []
        for (i,j) in combinations(range(self.k),2):
            X2.append(X_df[:,i]*X_df[:,j])
        X2 = np.vstack(X2).T
        X1 = np.hstack([X1,X2])
    #print "X1:",X1.shape
    return X1

class RFSel(BaseEstimator, TransformerMixin):
  """
    Transformer to select best features
    using RF
  """
  def __init__(self, clf = None, k = 10, n_estimators=100, n_jobs=1):
    self.clf = clf
    self.k = k
    self.n_estimators = n_estimators
    self.n_jobs = n_jobs
  
  def fit(self, X, y):
    clf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.k,
                n_jobs=self.n_jobs, random_state=random_state, verbose=0)
    #clf = DecisionTreeClassifier(criterion='entropy', max_depth=self.k, max_features=1.0, min_density=None, min_samples_leaf=1, min_samples_split=2, random_state=random_state, splitter='best')
    self.clf = clf.fit(X,np.ravel(y))
    feature_importances = clf.feature_importances_
    fi = sorted(zip(feature_importances,range(len(feature_importances))),reverse=True)
    self.features_selected = [e[1] for e in fi[:self.k]]
    #print "sorted features:",fi[:10],"selected:",self.features_selected
    return self

  def transform(self, X_df):
    #print "transforming data X_df:", X_df.shape
    return X_df[:,self.features_selected]

class SparseToDense(BaseEstimator, TransformerMixin):
  """
    Transformer to convers sparse X to dense X 
  """
  def __init__(self):
    pass
  
  def fit(self, X, y = None):
    return self

  def transform(self, X_df):
    return X_df.toarray()

