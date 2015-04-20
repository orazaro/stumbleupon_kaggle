#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    outliers detection
"""
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn import preprocessing, decomposition
from sklearn.pipeline import Pipeline

def search_outliers_EllipticEnvelope(X):
    clf = EllipticEnvelope(contamination=0.2)
    clf.fit(X)
    is_outliers = clf.predict(X)
    return is_outliers

def search_outliers_array(data, m = 6.):
    return abs(data - np.mean(data)) > m * np.std(data)

def search_outliers_array2(data, m = 6.):
    """ http://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0
    return s>m

def search_outliers(X, m = 6., mode = 1, verbose=1):
    """ Search outliers in X matrix with mode:
        1. Select outliers in every column, than select rows-outliers 
            with too much columns-outlier  
        2. Select rows-outliers of sum of its all columns
        3. Select rows-outliers of max value of its all columns
        4. make PCA of the matrix X, than select rows-outliers of its 
            first four principal components
        parameter m - Z-score in std to select outliers
    """
    nrows,ncols = X.shape
    mode_search_outliers_array = int(mode/10)
    if mode_search_outliers_array == 0:
        s_o_a = search_outliers_array
    else:
        s_o_a = search_outliers_array2
    mode_mode = mode%10
    if mode_mode == 1:
        outliers = np.array([0.0] * nrows)
        for j in range(ncols):
            isout = s_o_a(X[:,j],m)
            if np.any(isout):
                bad = np.where(isout)[0]
                outliers[bad] += 1.0
                if verbose>1:
                    print("outliers col:%d row_vals:%r"%(j,zip(bad,X[bad,j]))),
                    print "data: ",np.mean(X[:,j]),"+-",np.std(X[:,j])
        sel_outliers = s_o_a(outliers,m=m)
    elif mode_mode == 2:
        outliers = np.sum(X,axis=1)
        sel_outliers = s_o_a(outliers,m=m)
    elif mode_mode == 3:
        outliers = np.max(X,axis=1)
        sel_outliers = s_o_a(outliers,m=m)
    elif mode_mode == 4:
        from feasel import VarSel
        pline = [
            ("varsel", VarSel(k=4000)),
            #("scaler", preprocessing.StandardScaler(with_mean=True)),
            ("pca", decomposition.RandomizedPCA(n_components=20, whiten=True,random_state=1))
            ]  ;
        X1 = Pipeline(pline).fit_transform(X)
        #print "X1:",X1.shape,X1[:,:4]
        sel_outliers = np.array([False] * nrows)
        for j in range(4):
            outliers = X1[:,j]
            sel_outliers = sel_outliers | s_o_a(outliers,m=m)
            if np.any(sel_outliers): break
    else:
        raise ValueError("bad search_outliers mode: %r"%mode)
    if verbose>0:
        #print "sel_outliers:",sel_outliers
        if type(sel_outliers)!=bool:
            print "outliers:",outliers[sel_outliers]
    return np.where(sel_outliers)[0] 


def test():
    data = np.arange(100)/100.
    data[50] = 3.
    print np.any(search_outliers_array(data))
    X = np.array([range(50),range(50),range(50)])/50.
    X[1,10] = 3.
    X[1,14] = 15.
    X[0,1] = 33.
    #print search_outliers(X)
    bad = search_outliers(X.T,m=3,mode=4)
    print "bad rows:",sum(bad)
    print X.T[bad,:]

if __name__ == "__main__":
    test()
