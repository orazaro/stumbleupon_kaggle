#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Prepare data and cache result
"""
import sys, random, logging
import cPickle as pickle
import joblib
import numpy as np
import datetime, csv, gzip, json
import pandas as pd
from collections import defaultdict

from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
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
from sklearn.feature_extraction import DictVectorizer

from kgml.nltk_preprocessing import preprocess_pipeline
from kgml.rfecv import RFECVp

import project
logger = logging.getLogger(__name__)

loadData = lambda f: np.genfromtxt(open(f,'r'), delimiter=' ')

class Prepare(object):
  """
  Prepare abstract class
  """
  def __init__(self, model=1):
    self.model = model
    self.datapath = project.Project().datapath

  def fit(self, update=False):
    raise NotImplementedError("this function is virtual")
  
  def get_fname(self, update, suff='prep'):
    fname = "%s/data_%s_%02d.pkl" % (self.datapath, suff, self.model)
    if not update:
        try:
            with open(fname,"rb") as f:
              pass
        except IOError:
            update = True
    return fname, update
  
  def _cache_hkey(self, jparams):
    return abs(hash(jparams))
  
  def _cache_load_index(self):
    fname = '%s/cache/cache_index.jbl' % self.datapath
    try:
        cache_index  = joblib.load(fname)
    except IOError:
        cache_index = defaultdict(list)
    return cache_index
  def _cache_dump_index(self,cache_index):
    fname = '%s/cache/cache_index.jbl' % self.datapath
    joblib.dump(cache_index, fname)
 
  def cache_ls(self):
    cache_index = self._cache_load_index()
    for k in cache_index:
        for i,jparams in enumerate(cache_index[k]):
            print '%s_%d: %s' % (k,i,jparams)

  def cache_load(self,params,prefix='prep'):
    cache_index = self._cache_load_index()
    jparams = json.dumps((prefix,params))
    hkey = self._cache_hkey(jparams)
    if hkey not in cache_index or jparams not in cache_index[hkey]:
            return None
    offset = cache_index[hkey].index(jparams)
    fname = '%s/cache/%s_%s_%d.jbl' % (self.datapath,prefix,hkey,offset)
    data,jparams = joblib.load(fname)
    return data

  def cache_dump(self,data,params,prefix='prep'):
    jparams = json.dumps((prefix,params))
    cache_index = self._cache_load_index()
    hkey = self._cache_hkey(jparams)
    if hkey in cache_index:
        if jparams in cache_index[hkey]:
            offset = cache_index[hkey].index(jparams)
        else:
            offset = len(cache_index[hkey])
            cache_index[hkey].append(jparams)
    else:
        cache_index[hkey].append(jparams)
        offset = 0
    fname = '%s/cache/%s_%s_%d.jbl' % (self.datapath,prefix,hkey,offset)
    joblib.dump((data,jparams),fname)
    self._cache_dump_index(cache_index)
  
  def cache_params(self, X, y):
    "generate params to identify X and y"
    x1,y1 = np.asarray(X),np.asarray(y)
    params = dict(
        x_mean=np.mean(x1),
        x_std=np.std(x1),
        x_ss = np.sum(x1*x1),
        y_mean=np.mean(y1),
        y_std=np.std(y1),
        y_ss = np.sum(y1*y1),
        )
    return params

  def _cache_register(self,fname_short):
    "register file: to copy files from remote cache"
    prefix = fname_short.split('_')[0]
    fname1 = '%s/cache/%s' % (self.datapath,fname_short)
    data,jparams = joblib.load(fname1)
    cache_index = self._cache_load_index()
    hkey = self._cache_hkey(jparams)
    if hkey in cache_index:
        if jparams in cache_index[hkey]:
            offset = cache_index[hkey].index(jparams)
        else:
            offset = len(cache_index[hkey])
            cache_index[hkey].append(jparams)
    else:
        cache_index[hkey].append(jparams)
        offset = 0
    fname = '%s/cache/%s_%s_%d.jbl' % (self.datapath,prefix,hkey,offset)
    joblib.dump((data,jparams),fname)
    self._cache_dump_index(cache_index)

##### BP

class BPobj(object):
  def __init__(self):
    pass
  def fit(self, X_all_df, y, n_components = 128, min_df=3,
            use_svd=True, tfidf=2, fit_area='test', extra='{}'):
    # data for unsupervized learning
    BP = X_all_df['boilerplate'].values
    lentrain = len(y)
    self.tfv, self.svd = _fit_BP(BP, lentrain, n_components = n_components, min_df=min_df,
            use_svd=use_svd, tfidf=tfidf, fit_area=fit_area, extra=extra)
  def transform(self,X_df):
    print "transforming data X_df:", X_df.shape
    BP = X_df.iloc[:,2]
    BP = self.tfv.transform(BP)
    if self.svd != None:
        print "transforming svd BP:", BP.shape,
        BP = self.svd.transform(BP)
        print "=>", BP.shape
    return BP

def _fit_BP(BP=None, lentrain=None, n_components = 128, min_df=3,
        use_svd=True, tfidf=2, fit_area='test', extra='{}'):
    """
    aka Latent semantic analysis 
    but ignoring X_df on fit
    """
    ex = json.loads(extra)
    ngram_range=(1, ex.get('ngram_max',2))
    max_features = ex.get('max_features',None)
    max_df = ex.get('max_df',1.0)
    binary = ex.get('binary',False)
    use_idf = ex.get('use_idf',1)
    smooth_idf = ex.get('smooth_idf',1)
    sublinear_tf = ex.get('sublinear_tf',1)
    norm = ex.get('norm','l2')
    token_min = ex.get('token_min',1)
    token_pattern = r'\w{%d,}'%token_min

    logger.debug('ngram_range=%s max_features=%s max_df=%s binary=%s',
        ngram_range,max_features,max_df,binary)

    if tfidf > 0:
        logging.info("fitting TfidfVectorizer")
        tfv = TfidfVectorizer(
            min_df=min_df, max_df=max_df, max_features=max_features, strip_accents='unicode', 
            analyzer='word',token_pattern=token_pattern, ngram_range=ngram_range, binary=binary,
            use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf, norm=norm)
    else:
        logging.info("fitting CountVectorizer")
        tfv = CountVectorizer(
            min_df=min_df, max_df=max_df, max_features=max_features, strip_accents='unicode', 
            analyzer='word',token_pattern=token_pattern, ngram_range=ngram_range, binary=binary,
            )
    if fit_area == 'test':
        tfv.fit(BP[lentrain:])
    elif fit_area == 'train':
        tfv.fit(BP[:lentrain])
    elif fit_area == 'all':
        tfv.fit(BP)
    else:
        raise ValueError("Bad fit_area: %s" % fit_area)

    BP = tfv.transform(BP)
    logging.debug("BP(post tfv):%s",BP.shape)
    if use_svd:
        logging.info("fit svd")
        svd = TruncatedSVD(n_components=n_components, random_state=1)#, algorithm='arpack')
        svd.fit(BP)
    else:
        svd = None
    return tfv,svd

gBP = BPobj()
def BP_tfv_transform(X):
    return gBP.tfv.transform(X)
def BP_svd_transform(X):
    return gBP.svd.transform(X)


#### Prepare_0

supported_stemmers = ["PorterStemmer","SnowballStemmer","LancasterStemmer","WordNetLemmatizer"]
class Prepare_0(Prepare):
  """
  Prepare abstract 
  """
  def __init__(self, model=2, n_components=128, min_df=1, preproc=-1, 
        use_svd=True, tfidf=2, stemmer=3, fit_area='test', extra='{}'):
    super(Prepare_0, self).__init__(model = model)
    self.stemmer = stemmer
    self.n_components = n_components
    self.min_df = min_df
    self.preproc = preproc
    self.use_svd = use_svd
    self.tfidf = tfidf
    self.fit_area = fit_area
    self.extra = extra


  def load_transform(self, update=False):
    """ Load data, fit BP, and transform it and return back
    """
    params = (self.n_components,self.min_df,self.preproc,self.use_svd,self.tfidf,
        self.stemmer,self.fit_area,self.extra)
    if update:
        ex = json.loads(self.extra)
        do_remove_stopwords = ex.get('do_remove_stopwords',True)
        logging.info("updating data%s",params)
        train_df = pd.read_table('%s/train.tsv'%self.datapath, na_values='?')
        test_df = pd.read_table('%s/test.tsv'%self.datapath, na_values='?')
        assert train_df.shape[1] == test_df.shape[1] + 1
        names = train_df.columns
        #logging.debug("%s",list(names))
        y = np.array(train_df['label'].values, dtype=np.int64)
        X_all_df = pd.concat((train_df.ix[:,:-1],test_df))
        logging.debug("train:%s test:%s X_all_df:%s",train_df.shape,test_df.shape,X_all_df.shape)
        assert X_all_df.shape == (train_df.shape[0]+test_df.shape[0],test_df.shape[1])
        
        if self.preproc > 0:
            logging.info("pre-processing data and update Xal_df:%s",X_all_df.shape)
            N = X_all_df.shape[0]
            for i in range(N):
                observation = X_all_df.iloc[i,2]
                d = json.loads(observation)
                for k in ['title','body']:
                    if k in d and d[k]:
                        d[k] = preprocess_pipeline(d[k], 
                            lang="english", stemmer_type=supported_stemmers[self.stemmer], return_as_str=True,
                            do_remove_stopwords=do_remove_stopwords, do_clean_html=False)
                X_all_df.iloc[i,2] = json.dumps(d)
        
        BPobj1 = BPobj()
        BPobj1.fit(X_all_df,y,n_components=self.n_components,min_df=self.min_df, 
            use_svd=self.use_svd, tfidf=self.tfidf, fit_area=self.fit_area,
            extra=self.extra)

        BP = BPobj1.transform(X_all_df)
        dat =  (X_all_df,y,BP,params)
        logging.info("save data X_all_df:%s y:%s BP:%s",X_all_df.shape,y.shape,BP.shape)
        self.cache_dump(dat,params)
    else:
        logging.debug("load data%s",params)
        data = self.cache_load(params)
        if data is None:
            logging.info("Data with params not cached: updating..")
            return self.load_transform(update=True)
        else:
            (X_all_df,y,BP,params) = data
    return (X_all_df,y,BP,params)

  def load(self, preproc = -1, update=False):
    fname, update = self.get_fname(update, suff='bp')
    global gBP
    if update:
        print "loading data.."
        train_df = pd.read_table('%s/train.tsv'%self.datapath, na_values='?')
        test_df = pd.read_table('%s/test.tsv'%self.datapath, na_values='?')
        assert train_df.shape[1] == test_df.shape[1] + 1
        names = train_df.columns
        print(list(names))
        y = np.array(train_df['label'].values, dtype=np.int64)
        X_all_df = pd.concat((train_df.ix[:,:-1],test_df))
        print("train:",train_df.shape,"test:",test_df.shape,"X_all_df:",X_all_df.shape)
        assert X_all_df.shape == (train_df.shape[0]+test_df.shape[0],test_df.shape[1])
        
        if preproc > 0:
            print "pre-processing data and update Xal_df:",X_all_df.shape
            N = X_all_df.shape[0]
            for i in range(N):
                observation = X_all_df.iloc[i,2]
                d = json.loads(observation)
                for k in ['title','body']:
                    if k in d and d[k]:
                        d[k] = preprocess_pipeline(d[k], 
                            lang="english", stemmer_type=supported_stemmers[self.stemmer], return_as_str=True,
                            do_remove_stopwords=True, do_clean_html=False)
                X_all_df.iloc[i,2] = json.dumps(d)
        else:
            preproc = 0
        
        gBP.fit(X_all_df,y,n_components=self.n_components,min_df=self.min_df)
        #gBP.fit(X_all_df.iloc[:400,:],y[:200],preproc=preproc)
        
        dat =  (X_all_df,y,preproc,gBP)
        print "save data X_all_df",X_all_df.shape,"y",y.shape
        joblib.dump(dat,fname)
    else:
        print "load data..",
        with open(fname,"rb") as f:
            dat = joblib.load(fname)
            (X_all_df, y, preproc1, gBP) = dat 
        print "=> X_all_df:",dat[0].shape,"y:",dat[1].shape,"preproc:",preproc1
        if preproc >= 0 and preproc1 != preproc:
            return self.load(preproc=preproc, update=True)
    return X_all_df, y

  def load_y_colnames(self):
    logger.debug("loading y and train column names..")
    train_df = pd.read_table('%s/train.tsv'%self.datapath, na_values='?')
    colnames = train_df.columns
    y = np.array(train_df['label'].values, dtype=np.int64)
    n_train = len(y)
    n_test = 3171
    n_all = n_train + n_test
    return y, colnames, n_train, n_test, n_all
 
  def dump_ypred_residuals(self,y,y_pred):
    logger.debug("dump y_pred and residuals")
    y_df = pd.DataFrame({'ypred':y_pred,'resid':y-y_pred})
    y_df.to_csv('%s/ypred.csv'%self.datapath,index=False)

  def _load_data(self):
    print "loading data.."
    train_df = pd.read_table('%s/train.tsv'%self.datapath, na_values='?')
    test_df = pd.read_table('%s/test.tsv'%self.datapath, na_values='?')
    names = train_df.columns
    print(list(names))
    BP_train = list(train_df['boilerplate'].values)
    BP_test = list(test_df['boilerplate'].values)
    BP = BP_train + BP_test
    y = np.array(train_df['label'].values, dtype=np.int64)
    return train_df,test_df,BP,y,names

  def fit(self, update=False):
    fname, update = self.get_fname(update)
    if update:
        dat = self._get_data() 
        print "save data"
        with open(fname,"wb") as f:
            pickle.dump(dat,f)
    else:
        print "load data"
        with open(fname,"rb") as f:
            dat = pickle.load(f)
    return dat # (X_all, y, feature_names)

  def _lsa(self, BP, lentrain, n_components=16, preproc=True, 
    fit_area='test', min_df=3):
    return lsa(BP, lentrain, n_components, preproc, fit_area, min_df)

#### Prepare_1

class Prepare_1(Prepare):
  """
  Prepare model_1
  """
  def __init__(self):
    super(Prepare_1, self).__init__(model = 1)

  def fit(self, update=False):
    fname, update = self.get_fname(update)
    if update:
        X_all, y, lentrain = load_Boilerplate()
        X_all, tfv = transform_Tfidf(X_all, lentrain)
        if 1:
            # svd here
            print "use svd"
            svd = TruncatedSVD(n_components=500, random_state=1)
            X_all = svd.fit_transform(X_all)
            print "X_all(post svd):",X_all.shape
        dat = (X_all,y,lentrain)
        print "save data"
        with open(fname,"wb") as f:
            pickle.dump(dat,f)
    else:
        print "load data"
        with open(fname,"rb") as f:
            (X_all,y,lentrain) = pickle.load(f)
    return (X_all,y,lentrain)


def load_Boilerplate():
    print "loading data.."
    traindata_raw = list(np.array(pd.read_table('../data/train.tsv'))[:,2])
    testdata_raw = list(np.array(pd.read_table('../data/test.tsv'))[:,2])

    if 0:    
        traindata_raw = tr_json(traindata_raw)
        testdata_raw = tr_json(testdata_raw)

    y = np.array(pd.read_table('../data/train.tsv'))[:,-1]

    if False:
        print "pre-processing data"
        traindata = []
        testdata = []
        for observation in traindata_raw:
            traindata.append(preprocess_pipeline(observation, "english", 
                "WordNetLemmatizer", True, True, False))
        for observation in testdata_raw:
            testdata.append(preprocess_pipeline(observation, "english", 
                "WordNetLemmatizer", True, True, False))
    else:
        traindata, testdata = traindata_raw, testdata_raw

    X_all = traindata + testdata
    lentrain = len(traindata)
    return X_all,y,lentrain

def transform_Tfidf(X_all, lentrain):
    tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,
        smooth_idf=1, sublinear_tf=1, norm='l2')

    print "fitting pipeline"
    tfv.fit(X_all[lentrain:])
    print "transforming data"
    X_all = tfv.transform(X_all)
    print "X_all(post):",X_all.shape
    return X_all, tfv

def select_features(X,y):
    selector = SelectPercentile(f_classif, percentile=10)
    print "fit selector"
    selector.fit(X, y)
    print "transform features"
    X = selector.transform(X)
    return X,selector

def lsa(BP, lentrain, n_components=16, preproc=True, 
    fit_area='test', min_df=3):
    """
    aka Latent semantic analysis
    """
    if preproc:
        print "pre-processing data"
        traindata = []
        for observation in BP:
            traindata.append(preprocess_pipeline(observation, "english", 
                "WordNetLemmatizer", True, True, False))
        BP = traindata

    print "fitting TfidfVectorizer"
    tfv = TfidfVectorizer(min_df=min_df,  max_features=None, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,
        smooth_idf=1, sublinear_tf=1, norm='l2')
    if fit_area == 'test':
        tfv.fit(BP[lentrain:])
    elif fit_area == 'train':
        tfv.fit(BP[:lentrain])
    else:
        tfv.fit(BP)
    print "transforming data"
    BP = tfv.transform(BP)
    print "BP(post):",BP.shape

    if 1:
        # svd here
        print "use svd"
        svd = TruncatedSVD(n_components=n_components, random_state=1)
        BP = svd.fit_transform(BP)
    
    return BP


#### Prepare_2

class Prepare_2(Prepare_0):
  """
  Prepare model_2
  """
  def __init__(self, model=2):
    super(Prepare_2, self).__init__(model = model)

  def _get_data(self):
    train_df, test_df, BP, y, names = self._load_data()
    lentrain = len(y)
    #print BP[:10],y[:10]
    n_components = 128
    BP = self._lsa(BP, lentrain, n_components=n_components, preproc=False)
    
    X_dicts = []
    def add_dics(df, start = 0):
        for i in range(df.shape[0]):
            row = df.iloc[i,:]
            d = dict()
            if not pd.isnull(row['alchemy_category']):
                f = 'ac_%s' % row['alchemy_category']
                d[f] = float(row['alchemy_category_score'])
            for j,name in enumerate(names[:-1]):
                if j < 5:
                    pass
                elif j in [17,20]:
                    if pd.isnull(row[name]):
                        d[name] = 0
                    elif row[name]:
                        d[name] = 1
                    else:
                        d[name] = -1
                elif not pd.isnull(row[name]):
                    f = name
                    d[f] = row[name]
                else:
                    raise ValueError("bad row:",row)
            for j in range(n_components):
                f = "bp_%03d" % j
                d[f] = BP[start+i,j]
            X_dicts.append(d)
    add_dics(train_df,start=0)
    add_dics(test_df,start=lentrain)
    print len(X_dicts[0])
    #for l in X_dicts: print l

    self.vectorizer = DictVectorizer()
    X_all = self.vectorizer.fit_transform(X_dicts)
    #print "X_all:",X_all.shape,X_all
    print self.vectorizer.get_feature_names()

    return X_all,y,self.vectorizer.get_feature_names()

#### Prepare_3

def tr_json(data):
    dicts = dict()
    dicts['title'] = []
    dicts['body'] = [] 
    dicts['url'] = []
    for line in data:
        d = json.loads(line)
        for k in ['title','body','url']:
            if k in d and d[k]:
                r = d[k]
            else:
                r = 'NA'
            dicts[k].append(r)
    return dicts

class Prepare_3(Prepare_2):
  """
  Prepare model_3
  """
  def __init__(self, model=3):
    super(Prepare_3, self).__init__(model = model)

  def _get_data(self):
    train_df, test_df, BP, y, names = self._load_data()
    lentrain = len(y)
    #print BP[:10],y[:10]
    dicts = tr_json(BP) 
    params = dict()
    params['title'] = (64,False,'test')
    params['body'] = (128,False,'test')
    params['url'] = (64,False,'test')
    for k in ['title','body','url']:
        p = params[k] 
        dicts[k] = self._lsa(dicts[k], lentrain, n_components=p[0], 
            preproc=p[1], fit_area=p[2])
    
    X_dicts = []
    def add_dics(df, start = 0):
        for i in range(df.shape[0]):
            row = df.iloc[i,:]
            d = dict()
            if not pd.isnull(row['alchemy_category']):
                f = 'ac_%s' % row['alchemy_category']
                d[f] = float(row['alchemy_category_score'])
            for j,name in enumerate(names[:-1]):
                if j < 5:
                    pass
                elif j in [17,20]:
                    if pd.isnull(row[name]):
                        d[name] = 0
                    elif row[name]:
                        d[name] = 1
                    else:
                        d[name] = -1
                elif not pd.isnull(row[name]):
                    f = name
                    d[f] = row[name]
                else:
                    raise ValueError("bad row:",row)
            for k in ['title','body','url']:
                for j in range(params[k][0]):
                    f = "bp_%s_%03d" % (k,j)
                    d[f] = dicts[k][start+i,j]
            X_dicts.append(d)
    add_dics(train_df,start=0)
    add_dics(test_df,start=lentrain)
    print len(X_dicts[0])
    #for l in X_dicts: print l

    self.vectorizer = DictVectorizer()
    X_all = self.vectorizer.fit_transform(X_dicts)
    #print "X_all:",X_all.shape,X_all
    print self.vectorizer.get_feature_names()

    return X_all,y,self.vectorizer.get_feature_names()


def test():
    print "tests ok"

if __name__ == '__main__':
    random.seed(1)
    import argparse
    parser = argparse.ArgumentParser(description='Prepare.')
    parser.add_argument('cmd', nargs='?', default='test')
    parser.add_argument('-update', default='0')    
    parser.add_argument('-preproc', default=0)    
    parser.add_argument('-fn', default=None)    
    args = parser.parse_args()
    #print >>sys.stderr,args 
  
    if args.cmd == 'test':
        test()
    elif args.cmd == 'update_bp':
        Prepare_0().load(preproc=int(args.preproc),update=True)
    elif args.cmd == 'update_10':
        Prepare_0(model=10).load_transform(update=1)
    elif args.cmd == 'cache_ls':
        Prepare().cache_ls()
    elif args.cmd == 'cache_register':
        Prepare()._cache_register(args.fn)
    elif args.cmd == 'run':
        update=int(args.update)
        if update == 1:
            Prepare_1().fit(update=int(args.update))
        elif update == 2:
            Prepare_2().fit(update=int(args.update))
        elif update == 3:
            Prepare_3().fit(update=int(args.update))
        else:
            print "norun"
    else:
        raise ValueError("bad cmd")

