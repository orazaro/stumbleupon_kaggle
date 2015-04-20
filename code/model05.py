#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model05.py (copy from model02)
TfidfVectorizer: norm='l2'
("svd", TruncatedSVD(n_components=100, random_state=1 )),
"""

import sys, random, pickle
import numpy as np
import datetime, csv, gzip
import pandas as pd

from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

from kgml.rfecv import RFECVp

from prepare import Prepare_1

def cv_run(rd, X, y):
    n_cv = 16
    cv1 = cross_validation.KFold(len(y), n_folds=n_cv, random_state=1)
    scores = cross_validation.cross_val_score(rd, X, y, cv=cv1, 
        scoring='roc_auc', 
        #scoring=make_scorer(roc_auc_score), 
        n_jobs=1, verbose=1)
    print "scores:",scores
    print "%d Fold CV Score: %.6f +- %.4f" % (n_cv, np.mean(scores), 2*np.std(scores),)


def main():
    (X_all,y,lentrain) = Prepare_1().fit(update=True)

    X = X_all[:lentrain]

    clf1 = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=123)
    clf2 = RandomForestClassifier(n_estimators=200, max_depth=24,
            n_jobs=-1, random_state=1, verbose=0)

    clf3 = GradientBoostingClassifier(n_estimators=42, max_depth=24,
            random_state=1, verbose=2, subsample=0.9)

    clf4 = svm.SVC(probability=True)

    clf5 = KNeighborsClassifier(n_neighbors=5)

    clf6 = SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
           fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
           loss='hinge', n_iter=50, n_jobs=1, penalty='elasticnet', power_t=0.5,
           random_state=None, rho=None, shuffle=False, verbose=0,
           warm_start=False)

    clf = clf1

    """
    selector = RFECVp(clf2,clf2, step=50, cv=4, scoring="roc_auc", verbose=2)
    selector = selector.fit(X, y)
    clf = selector
    """

    rd = Pipeline([
        #("selector", SelectPercentile(chi2, percentile=90)),
        #("selector", SelectPercentile(f_classif, percentile=50)),
        #("selector", lm.RandomizedLogisticRegression(C=1, random_state=1, verbose=1)),
        #("pca", PCA(n_components='mle')),
        #("pca", PCA(n_components=500)),
        #("svd", TruncatedSVD(n_components=200, random_state=1 )),
        #("lasso",svm.LinearSVC(C=0.5, penalty="l1", dual=False)),
        ("est", clf)
        ])

    if True:
        cv_run(rd, X, y)
        return
    else:
        print "Prepare submission.."

    print "training on full data"
    rd.fit(X,y)
    X_test = X_all[lentrain:]
    pred = rd.predict_proba(X_test)[:,1]
    testfile = pd.read_csv('../data/test.tsv', sep="\t", na_values=['?'], index_col=1)
    pred_df = pd.DataFrame(pred, index=testfile.index, columns=['label'])
    submname = 'submission_%s' % (datetime.datetime.today().strftime("%Y%m%d_%H%M%S"),)
    #print submname
    pred_df.to_csv('../data/%s.csv' % submname)
    print "%s file created.." % submname

if __name__=="__main__":
    random.seed(1)
    np.random.seed(1)
    main()
