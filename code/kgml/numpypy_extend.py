#!/usr/bin/env pypy
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
   Extend numpypy module of pypy 
"""


def _diff(v,n=1):
    if n <= 0: return np.array(v)
    p = []
    for i in range(len(v)-1):
        p.append(v[i+1]-v[i])
    return _diff(p,n=n-1)

def _sort(v, reverse=False):
    return np.array(sorted(v, reverse=reverse))

def _corrcoef(x,y):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    mx = x.mean()
    my = y.mean()
    xm, ym = x-mx, y-my
    r_num = n*(np.add.reduce(xm*ym))
    r_den = n*np.sqrt(np.sum(xm*xm)*np.sum(ym*ym))
    r = (r_num / r_den) if r_den != 0 else 0
    return r

def _argsort(seq,reverse=False):
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)

from platform import python_implementation;
if python_implementation() == 'PyPy':
    import numpypy as np 
    if 'diff' not in dir(np):
        np.diff = _diff
    a = np.array([1,3,2,4])
    b = np.array([1,4,0,4])
    try: np.sort(a)
    except: np.sort = _sort
    try: 
        np.cor(a,b)
        raise RuntimeError("numpypy.cor already exists!")
    except: np.cor = _corrcoef
    try: 
        np.argsort(a,reverse=True)
        raise RuntimeError("numpypy.argsort already exists!")
    except: np.argsort = _argsort
else:
    import numpy as np
    np.cor = _corrcoef

def pearsonr(x, y):
    # Assume len(x) == len(y)
    from itertools import imap
    n = len(x)
    sum_x = float(sum(x))
    sum_y = float(sum(y))
    sum_x_sq = sum(map(lambda x: pow(x, 2), x))
    sum_y_sq = sum(map(lambda x: pow(x, 2), y))
    psum = sum(imap(lambda x, y: x * y, x, y))
    num = psum - (sum_x * sum_y/n)
    den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
    if den == 0: return 0
    return num / den

def t_round_equal(x,y,n = 5):
    return round(abs(x-y),n) == 0.0

def test():
    def check(x,y):
        a = pearsonr(x,y)
        b = np.cor(x,y)
        print "pearsonr:", a
        print "corcoef:", b
        assert t_round_equal(a,b)
    x = np.array([1,2,3,5,9,12])
    y = x*2
    y[3] *= 1.5
    y[0] *= 0.6
    check(x,y) 

    import random 
    x = np.array([random.random() for _ in range(10)])
    y = np.array([random.random() for _ in range(10)])
    check(x,y)

if __name__ == '__main__':
    test()
