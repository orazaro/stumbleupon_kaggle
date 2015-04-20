#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    Imbalanced datasets    
"""
import numpy as np
import random

from collections import defaultdict
from sklearn import metrics

def find_best_cutoff(y1,ypp,verbose=0):
    from scipy import optimize
    def f(x,*params):
        y_true,ypp = params
        y_pred = np.array(map(int,ypp>x))
        res = metrics.f1_score(y_true, y_pred)
        #print "x:",x,"res:",res
        return -res
    rranges = (slice(0,1,0.01),)
    resbrute = optimize.brute(f, rranges, args=(y1,ypp), full_output=False,
                                  finish=optimize.fmin)
    if verbose: print "resbrute:",resbrute
    return resbrute[0]

def ids_invert(X,ids):
    """ invert ids
        ids: dictionary of relation id -> rows in X
    """
    ids_inv = [None]*X.shape[0]
    for (k,v) in ids.iteritems():
        for i in v:
            ids_inv[i] = k
    return ids_inv

def round_down(Xall,y1,ids=None):
    p_zeros = [i for i,e in enumerate(y1) if e == 0]
    p_ones = [i for i,e in enumerate(y1) if e > 0]
    delta = len(p_zeros) - len(p_ones)
    if delta > 0:
        sel = random.sample(p_zeros,len(p_ones))
        sel = sel + p_ones
    elif delta < 0:
        sel = random.sample(p_ones,len(p_zeros))
        sel = sel + p_zeros
    else:
        return Xall,y1,ids
    #print "round down:",len(p_zeros),len(p_ones),len(sel)
    if ids is not None:
        ids_inv = ids_invert(Xall,ids)
        ids2 = defaultdict(list)
        for (j,i) in enumerate(sel):
            ids2[ids_inv[i]].append(j)
        #print len(sel),sum([len(ids[k]) for k in ids])
        # renumerate ids2 as range(0,len(ids2))
        ids = defaultdict(list)
        for i,id2 in enumerate(ids2):
            ids[i] = ids2[id2]
        return Xall[sel,:],y1[sel],ids
    else:
        return Xall[sel,:],y1[sel],ids

def round_up(Xall,y1,ids=None):
    if ids is not None: 
        ids_inv = ids_invert(Xall,ids)
    p_zeros = [i for i,e in enumerate(y1) if e == 0]
    p_ones = [i for i,e in enumerate(y1) if e > 0]
    delta = len(p_zeros) - len(p_ones)
    if delta > 0:
        sel = [random.choice(p_ones) for _ in range(delta)]
    elif delta < 0:
        delta = -delta
        sel = [random.choice(p_zeros) for _ in range(delta)]
    else:
        return Xall,y1,ids
    X1 = [Xall]
    z1 = list(y1)
    j = Xall.shape[0]
    for i in sel:
        X1.append(Xall[i,:])
        z1.append(y1[i])
        if ids is not None:
            ids[ids_inv[i]].append(j)
        j += 1
    X1 = np.vstack(X1)
    z1 = np.array(z1).ravel()
    #print "round_up: 0:",len(p_zeros),"1:",len(p_ones),"X1:",X1.shape,"y1:",z1.shape,"j_last:",j
    return X1,z1,ids

from smote import SMOTE

def round_smote(Xall,y1,k=5,h=1.0):
    p_zeros = [i for i,e in enumerate(y1) if e == 0]
    p_ones = [i for i,e in enumerate(y1) if e > 0]
    delta = len(p_zeros) - len(p_ones)
    if delta > 0:
        N = ( int(len(p_zeros)/len(p_ones))+1 ) * 100
        T = Xall[p_ones,:]
        S = SMOTE(T, N, k, h)
        sel = random.sample(range(S.shape[0]),delta)
        X1 = np.vstack([Xall,S[sel,:]])
        z1 = np.hstack([y1,np.ones(delta)])
    elif delta < 0:
        delta = -delta
        N = ( int(len(p_ones)/len(p_zeros))+1 ) * 100
        T = Xall[p_zeros,:]
        S = SMOTE(T, N, k, h)
        sel = random.sample(range(S.shape[0]),delta)
        X1 = np.vstack([Xall,S[sel,:]])
        z1 = np.hstack([y1,np.zeros(delta)])
    else:
        return Xall,y1
    #print "round smote:","X1:",X1.shape,"z1:",z1.shape
    return X1,z1

