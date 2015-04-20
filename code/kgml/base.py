#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    base obects and functions
"""

import platform;
if platform.python_implementation() == 'PyPy':
    import numpypy as np
    import numpypy_extend
else:
    import numpy as np

class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def t_round_equal(x,y,n = 5):
    return round(abs(x-y),n) == 0.0

def test():
    b = Bunch(x=2,y=3)
    print b
    assert str(b) == "{'y': 3, 'x': 2}"
    assert mean_absolute_error(np.array([1.,2.3]),np.array([-2.1,3.3])) == 2.05

def check_n_jobs(n_jobs):
    import multiprocessing
    if multiprocessing.current_process()._daemonic:
        return 1
    else:
        return n_jobs

