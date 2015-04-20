#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    Feature selectors 
"""

import sys, random, os, logging

import numpy as np

from sklearn.feature_selection import f_regression
from sklearn.datasets.samples_generator import (make_classification,
                                                make_regression)
from sklearn.feature_selection import SelectPercentile, f_classif, chi2

logger = logging.getLogger(__name__)

def f_regression_select(X, y, maxf = 300, pvals = True, names = None, verbose = 0, old_idx_sel=None):
    "Select features using f_regression"
    if names == None:
        names = ["f_%d"%(i+1) for i in range(X.shape[1])]
    if not old_idx_sel:
        old_idx_sel = range(X.shape[1])
    f=f_regression(X,y,center=False)
    # (F-value, p-value, col, name)
    a = [(f[0][i], f[1][i], old_idx_sel[i], names[i]) 
            for i in range(X.shape[1])]
    if pvals:
        a = [e for e in a if e[1]<0.05]
    a = sorted(a, reverse=True)
    idx_sel = [ e[2] for e in a[:maxf] ]
    if verbose > 0:
        b = a[:maxf]
        def out():
            if min(maxf,len(b)) > 100:
                print >>sys.stderr,"F_select(%d):"%len(b),b[:90],"...",b[-10:]
            else:
                print >>sys.stderr,"F_select(%d):"%len(b),b[:maxf]
        def out2():
            print >>sys.stderr,"F_select(%d):" % len(b)
            def pr(m1,m2):
                for i in range(m1,m2):
                    row = b[i]
                    print >>sys.stderr,"%10s %10.2f %15g %10d" % (row[3],row[0],row[1],row[2])
            n = min(len(b),maxf)
            m = 90 if n > 100 else n
            pr(0,m)
            if n > 100:
                print >>sys.stderr,"..."
                pr(len(b)-10,len(b))
        if verbose > 1:
            out2()
        else:
            out()
    return np.asarray(idx_sel, dtype=int)

def test_f_regression_select():
    print "==> a lot of features"
    X, y = make_regression(n_samples=20000, n_features=200, n_informative=150,
                             shuffle=False, random_state=0)
    idx_sel = f_regression_select(X, y, verbose=2)
    print "==> few ones"
    X, y = make_regression(n_samples=200, n_features=20, n_informative=5, noise=0.5,
                             shuffle=False, random_state=0)
    idx_sel = f_regression_select(X, y, verbose=1)
    print "tests ok"

if __name__ == '__main__':
    random.seed(1)
    import argparse
    parser = argparse.ArgumentParser(description='Feature selectors.')
    parser.add_argument('cmd', nargs='?', default='test')
    args = parser.parse_args()
    print >>sys.stderr,args 
  
    if args.cmd == 'test':
        test_f_regression_select()
    else:
        raise ValueError("bad cmd")
