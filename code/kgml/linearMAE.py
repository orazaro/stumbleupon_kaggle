#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
   Linear model using MAE scoring (adapted to sklearn by O.Razgulyaev)
__author__ = 'Miroslaw Horbal'
__email__ = 'miroslaw@gmail.com'
__date__ = '2013-03-09'
"""
import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

OPTIMIZATION_FUNCTIONS = { 'CG':   optimize.fmin_cg,
                           'BFGS': optimize.fmin_bfgs,
                           'L-BFGS-B': None,
                           'TNC': None,
                           'Newton-CG': None,
                          }
    
class LinearMAE(BaseEstimator, RegressorMixin):
    """Linear model with Mean Absolute Error"""
    def __init__(self, l1=0.0, l2=0.0, opt='BFGS', maxiter=1000, 
                 tol=1e-6, verbose=False, use_minimize=True):
        """
        Parameters:
          l1 - magnitude of l1 penalty (default 0.0)
          l2 - magnitude of l2 penalty (default 0.0)
          opt - optimization algorithm to use for gardient decent 
                options are 'CG', 'bfgs' (default 'bfgs')
          maxiter - maximum number of iterations (default 1000)
          tol - terminate optimization if gradient l2 is smaller than tol (default 1e-4)
          verbose - display convergence information at each iteration (default False)
        """
        self.opt = opt
        self.maxiter = maxiter
        self.tol = tol
        self.l1 = l1
        self.l2 = l2
        self.verbose = verbose
        self.use_minimize = use_minimize
    
    @property
    def opt(self):
        """Optimization algorithm to use for gradient decent"""
        return self._opt
    
    @opt.setter
    def opt(self, o):
        """
        Set the optimization algorithm for gradient decent
        
        Parameters:
          o - 'CG' for conjugate gradient decent
              'bfgs' for BFGS algorithm
        """
        if o not in OPTIMIZATION_FUNCTIONS:
            raise ValueError('Unknown optimization routine %s' % o)
        self._opt = o
        self._optimize = OPTIMIZATION_FUNCTIONS[o]
   
    def score(self, X, y):
        """
        Compute the MAE of the linear model prediction on X against y
        
        Must only be run after calling fit
        
        Parameters:
          X - data array for the linear model. Has shape (m,n)
          y - output target array for the linear model. Has shape (m,o)
        """
        y = _2d(y)
        pred = self.predict(X)
        return np.mean(np.abs(pred - y))
    
    def predict(self, X):
        """
        Compute the linear model prediction on X
        
        Must only be run after calling fit
        
        Parameters:
          X - data array for the linear model. Has shape (m,n)
        """
        return X.dot(self.coef_[1:]) + self.coef_[0]
    
    def fit(self, X, y, coef=None):
        """
        Fit the linear model using gradient decent methods
        
        Parameters:
          X - data array for the linear model. Has shape (m,n)
          y - output target array for the linear model. Has shape (m,o)
          coef - None or array of size (n+1) * o
        
        Sets attributes:
          coef_ - the weights of the linear model
        """
        y = _2d(y)
        m, n = X.shape
        m, o = y.shape
        if coef is None:
            coef = np.zeros((n+1, o))
        elif coef.shape != (n+1, o):
            raise ValueError('coef must be None or be shape %s' % (str((n+1, o))))
        self._coef_shape = coef.shape
        if self.use_minimize:
            res = optimize.minimize(fun=cost_grad, 
                              x0=coef.flatten(), 
                              args=(X, y, self.l1, self.l2),
                              method = self.opt,
                              jac = True,
                              tol=self.tol,
                              options = dict(gtol=self.tol,maxiter=self.maxiter, disp=0),
                              callback=self._callback(X,y))
            self.coef_ = np.reshape(res.x, self._coef_shape)
        else:
            coef = self._optimize(f=cost, 
                              x0=coef.flatten(), 
                              fprime=grad, 
                              args=(X, y, self.l1, self.l2),
                              gtol=self.tol,
                              maxiter=self.maxiter,
                              disp=0,
                              callback=self._callback(X,y))
            self.coef_ = np.reshape(coef, self._coef_shape)
        return self

    def _callback(self, X, y):
        """
        Helper method that generates a callback function for the optimization
        algorithm opt if verbose is set to True
        """
        def callback(coef):
            self.i += 1
            self.coef_ = np.reshape(coef, self._coef_shape)
            score = self.score(X, y)
            print 'iter %i | Score: %f\r' % (self.i, score)
        self.i = 0
        return callback if self.verbose else None

def cost(coef, X, y, l1=0, l2=0):
    """
    Compute the cost of a linear model with mean absolute error:
    
      cost = X.dot(coef) + l1*mean(abs(coef[1:])) + 0.5*l2*mean(coef[1:] ** 2) 
    
    Parameters:
      coef - the weights of the linear model must have size (n + 1)*o
      X - data array for the linear model. Has shape (m,n)
      y - output target array for the linear model. Has shape (m,o)
      l1 - magnitude of the l1 penalty
      l2 - magnitude of the l2 penalty
    """
    y = _2d(y)
    m, n = X.shape
    m, o = y.shape
    Xb = np.hstack((np.ones((m,1)), X))
    coef = np.reshape(coef, (n+1, o))
    pred = Xb.dot(coef)
    c = np.mean(np.abs(pred - y))
    c_l1 = np.mean(np.abs(coef[1:])) if l1 > 0 else 0
    c_l2 = np.mean(np.square(coef[1:])) if l2 > 0 else 0
    return c + l1 * c_l1 + 0.5 * l2 * c_l2
    
def grad(coef, X, y, l1=0, l2=0):
    """
    Compute the gradient of a linear model with mean absolute error:
    
    Parameters:
      coef - the weights of the linear model must have size (n + 1)*o
      X - data array for the linear model. Has shape (m,n)
      y - output target array for the linear model. Has shape (m,o)
      l1 - magnitude of the l1 penalty
      l2 - magnitude of the l2 penalty
    """
    #~ y = np.atleast_2d(y).T if len(y.shape) == 1 else y
    y = _2d(y)
    m, n = X.shape
    m, o = y.shape
    Xb = np.hstack((np.ones((m,1)), X))
    coef = np.reshape(coef, (n+1, o))
    pred = Xb.dot(coef)
    err = pred - y
    derr = Xb.T.dot(np.sign(err)) / m
    dl1 = np.vstack((np.zeros(o), np.sign(coef[1:]))) if l1 > 0 else 0 
    dl2 = np.vstack((np.zeros(o), np.copy(coef[1:]))) if l2 > 0 else 0
    return (derr + l1 * dl1 + l2 * dl2).flatten()

def cost_grad(coef, X, y, l1=0, l2=0):
    """
    Compute the cost of a linear model with mean absolute error:
    
      cost = X.dot(coef) + l1*mean(abs(coef[1:])) + 0.5*l2*mean(coef[1:] ** 2) 
    
    Compute the gradient of a linear model with mean absolute error
    
    Parameters:
      coef - the weights of the linear model must have size (n + 1)*o
      X - data array for the linear model. Has shape (m,n)
      y - output target array for the linear model. Has shape (m,o)
      l1 - magnitude of the l1 penalty
      l2 - magnitude of the l2 penalty
    """
    y = _2d(y)
    m, n = X.shape
    m, o = y.shape
    Xb = np.hstack((np.ones((m,1)), X))
    coef = np.reshape(coef, (n+1, o))
    pred = Xb.dot(coef)
    err = pred - y
   
    # compute cost
    c = np.mean(np.abs(err))
    c_l1 = np.mean(np.abs(coef[1:])) if l1 > 0 else 0
    c_l2 = np.mean(np.square(coef[1:])) if l2 > 0 else 0
    cost =  c + l1 * c_l1 + 0.5 * l2 * c_l2
    
    # compute gradient
    derr = Xb.T.dot(np.sign(err)) / m
    dl1 = np.vstack((np.zeros(o), np.sign(coef[1:]))) if l1 > 0 else 0 
    dl2 = np.vstack((np.zeros(o), np.copy(coef[1:]))) if l2 > 0 else 0
    grad = (derr + l1 * dl1 + l2 * dl2).flatten()

    return cost,grad

def _2d(a):
    """Returns a 2d array of a if rank a <= 2"""
    if len(a.shape) == 1:
        a.shape = (len(a), 1)
    return a

def test():
    from numpy import array

    X = array([[0,1,2,4], [2,1,0,5]])
    y = array([[0,1], [2,3]])

    lin = LinearMAE(l1=0.1, l2=0.1, verbose=True, opt='CG', maxiter=10)

    lin.fit(X,y)
    print
    print 'Prediction' 
    print lin.predict(X)
    print 'Target'
    print y

import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.datasets.samples_generator import (make_classification,
                                            make_regression)

from sklearn.datasets import make_friedman1
from sklearn.datasets import make_friedman2
from sklearn.datasets import make_friedman3
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import linear_model


def test_vs_linear_model(N=10,M=10,informative=5):
    
    #random.seed(1)
   
    def run(X,y):
        def fit_predict(name):
            lin.fit(X_train,y_train.ravel())
            y_pred = lin.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            alpha = lin.alpha_ if 'alpha_' in dir(lin) else np.nan
            print "%s: mae=%f mse=%f alpha=%f" % (name,mae,mse,alpha)

        X_train = X[:N,:]
        y_train = y[:N]
        X_test = X[N:,:]
        y_test = y[N:]
        alphas = [10*(0.1**i) for i in range(10)]
        print "X_train:",X_train.shape,"X_test:",X_test.shape

        lin = linear_model.RidgeCV(alphas=alphas)
        fit_predict("ridge")
        
        lin = linear_model.LassoCV(alphas=alphas)
        fit_predict("lasso")
        
        lin_r = linear_model.RidgeCV(alphas=alphas).fit(X_train,y_train)
        lin_l = linear_model.LassoCV(alphas=alphas).fit(X_train,y_train)
        lin = LinearMAE(l1=lin_l.alpha_, l2=lin_r.alpha_, verbose=0, opt='CG', maxiter=300)
        fit_predict("LinearMAE")

        lin = RandomForestRegressor(n_estimators=100, 
            max_depth = 12,
            n_jobs = -1,
            verbose = 0, 
            random_state=3465343)
        fit_predict("RFRegressor")
        
        lin = GradientBoostingRegressor(n_estimators=100, 
            loss = 'lad',
            verbose = 0, 
            max_depth = 12,
            learning_rate = 0.1,
            subsample = 1.0,
            random_state=3465343)
        fit_predict("GBRegressor")
    
    #for noise in [0.01,0.1]:
    for noise in [0.1]:
        print "\nLinear Problem: noise=%.2f%%\n========" % (noise*100,)
        a = np.random.sample(M)
        N2 = N*2
        X = np.reshape(np.random.sample(N2*M),(N2,M))
        y = np.dot(X,a) + np.random.sample(N2)*noise
        run(X,y)

        print "\nRegression Problem: noise=%.2f%%\n========" % (noise*100,)
        X,y = make_regression(n_samples=N*2, n_features=M, n_informative=informative,
            n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, 
            noise=noise, shuffle=True, coef=False, random_state=None)
        run(X,y)

        print "\nRegression Problem, effective_rank=5 noise=%.2f%%\n========" % (noise*100,)
        X,y = make_regression(n_samples=N*2, n_features=M, n_informative=informative,
            n_targets=1, bias=0.0, effective_rank=5, tail_strength=0.5, 
            noise=noise, shuffle=True, coef=False, random_state=None)
        run(X,y)

        print "\nFriedman1 Problem noise=%.2f%%\n========" % (noise*100,)
        X,y = make_friedman1(n_samples=N*2, n_features=M, noise=noise, random_state=None)
        run(X,y)

from timeit import Timer
import time

def test_speed(N=10,M=10,informative=5):
    
    #random.seed(1)
   
    def run(X,y):
        def fit_predict(name):
            def fit_predict0(name):
                lin.fit(X_train,y_train.ravel())
                y_pred = lin.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                alpha = lin.alpha_ if 'alpha_' in dir(lin) else np.nan
                print "%s: mae=%f mse=%f alpha=%f" % (name,mae,mse,alpha)
            start = time.time()
            fit_predict0(name)
            end = time.time()
            print end - start

        X_train = X[:N,:]
        y_train = y[:N]
        X_test = X[N:,:]
        y_test = y[N:]
        alphas = [10*(0.1**i) for i in range(10)]
        print "X_train:",X_train.shape,"X_test:",X_test.shape

        lin = linear_model.RidgeCV(alphas=alphas)
        fit_predict("ridge")
        
        lin = linear_model.LassoCV(alphas=alphas)
        fit_predict("lasso")
        
        lin_r = linear_model.RidgeCV(alphas=alphas).fit(X_train,y_train)
        lin_l = linear_model.LassoCV(alphas=alphas).fit(X_train,y_train)
        maxiter = 1000
        lin = LinearMAE(l1=lin_l.alpha_, l2=lin_r.alpha_, verbose=0, opt='CG', 
                            maxiter=maxiter, use_minimize=True)
        fit_predict("LinearMAE_CG")
        lin = LinearMAE(l1=lin_l.alpha_, l2=lin_r.alpha_, verbose=0, opt='BFGS', 
                            maxiter=maxiter, use_minimize=True)
        fit_predict("LinearMAE_BFGS")
        lin = LinearMAE(l1=lin_l.alpha_, l2=lin_r.alpha_, verbose=0, opt='L-BFGS-B', 
                            maxiter=maxiter*100, use_minimize=True)
        fit_predict("LinearMAE_L-BFGS-B")
        """
        lin = LinearMAE(l1=lin_l.alpha_, l2=lin_r.alpha_, verbose=0, opt='TNC', 
                            maxiter=maxiter*20, use_minimize=True)
        fit_predict("LinearMAE_TNC")
        """
        lin = LinearMAE(l1=lin_l.alpha_, l2=lin_r.alpha_, verbose=0, opt='CG', 
                            maxiter=maxiter, use_minimize=False)
        fit_predict("LinearMAE old")
        from linearMAE2 import LinearMAE2
        lin = LinearMAE2(l1=lin_l.alpha_, l2=lin_r.alpha_, verbose=0, opt='bfgs', 
                            maxfun=maxiter)
        fit_predict("LinearMAE2")

    for noise in [0.10]:
        print "\nRegression Problem, effective_rank=5 noise=%.2f%%\n========" % (noise*100,)
        X,y = make_regression(n_samples=N*2, n_features=M, n_informative=informative,
            n_targets=1, bias=0.0, effective_rank=5, tail_strength=0.5, 
            noise=noise, shuffle=True, coef=False, random_state=None)
        run(X,y)

if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    #test_vs_linear_model()
    #test_vs_linear_model(N=1000, M=50, informative=10)
    test_speed(N=100, M=50, informative=10)

