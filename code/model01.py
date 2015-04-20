"""
model01.py
modify beat_banch.py: fit TfidfVectorizer only on test data
X_all(post): (10566, 84460)
20 Fold CV Score:  0.877200246763
=> res_model01.csv
LEADERBOARD: 0.87981
"""

# -*- coding: utf-8 -*-
import sys, random
import numpy as np
import datetime, csv, gzip
import pandas as pd

from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectPercentile, f_classif, chi2

from kgml.preprocessing import preprocess_pipeline

loadData = lambda f: np.genfromtxt(open(f,'r'), delimiter=' ')

def load_Boilerplate():
    print "loading data.."
    traindata_raw = list(np.array(pd.read_table('../data/train.tsv'))[:,2])
    testdata_raw = list(np.array(pd.read_table('../data/test.tsv'))[:,2])
    y = np.array(pd.read_table('../data/train.tsv'))[:,-1]

    if False:
        print "pre-processing data"
        traindata = []
        testdata = []
        for observation in traindata_raw:
            traindata.append(preprocess_pipeline(observation, "english", 
                "WordNetLemmatizer", True, False, False))
        for observation in testdata_raw:
            testdata.append(preprocess_pipeline(observation, "english", 
                "WordNetLemmatizer", True, False, False))
    else:
        traindata, testdata = traindata_raw, testdata_raw

    X_all = traindata + testdata
    lentrain = len(traindata)
    return X_all,y,lentrain

def transform_Tfidf(X_all, lentrain):
  tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)

  print "fitting pipeline"
  tfv.fit(X_all[lentrain:])
  print "transforming data"
  X_all = tfv.transform(X_all)
  print "X_all(post):",X_all.shape
  return X_all, tfv

def cv_run(rd, X, y):
  n_cv = 20
  scores = cross_validation.cross_val_score(rd, X, y, cv=n_cv, scoring='roc_auc', 
    n_jobs=-1, verbose=1)
  print "%d Fold CV Score: %.6f +- %.4f" % (n_cv, np.mean(scores), 2*np.std(scores),)

def select_features(X,y):
    selector = SelectPercentile(f_classif, percentile=10)
    print "fit selector"
    selector.fit(X, y)
    print "transform features"
    X = selector.transform(X)
    return X,selector

def main():
  X_all, y, lentrain = load_Boilerplate()

  X_all, tfv = transform_Tfidf(X_all, lentrain)

  X = X_all[:lentrain]

  clf = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=123)

  rd = Pipeline([
    #("selector", SelectPercentile(chi2, percentile=90)),
    #("pca", PCA(n_components='mle')),
    #("pca", PCA(n_components=500)),
    #("svd", TruncatedSVD(n_components=500)),
    ("est", clf)
    ])

  cv_run(rd, X, y)

  #X,selector = select_features(X,y)
  #X_test = selector.transform(X_test)

  X_test = X_all[lentrain:]
  print "training on full data"
  rd.fit(X,y)
  pred = rd.predict_proba(X_test)[:,1]
  testfile = pd.read_csv('../data/test.tsv', sep="\t", na_values=['?'], index_col=1)
  pred_df = pd.DataFrame(pred, index=testfile.index, columns=['label'])
  submname = 'submission_%s' % (datetime.datetime.today().strftime("%Y%m%d_%H%M%S"),)
  #print submname
  pred_df.to_csv('../data/%s.csv' % submname)
  print "%s file created.." % submname

if __name__=="__main__":
    random.seed(123)
    np.random.seed(123)
    main()
