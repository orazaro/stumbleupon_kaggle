#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    Feature extraction 
"""

import sys, random, os, logging
from collections import defaultdict
import re, string
import collections

import numpy as np

from sklearn.feature_selection import f_regression
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif, chi2

logger = logging.getLogger(__name__)

# Extract best words functools

def paste_words(text, vocab = None):
    s = text.translate(string.maketrans("&","_"),'"/ ,').lower() 
    if vocab and s not in vocab:
        return {}
    return { s:1 }

def extract_words(text, vocab = None, use_bigrams=0, minlen_word=2):
    reWords = re.compile('\w+')
    words = re.findall(reWords, text.lower())
    gwords = [w for w in words if w not in ENGLISH_STOP_WORDS] 
    bigrams = ['_'.join(p) for p in  zip(gwords,gwords[1:])]
    if use_bigrams:
        words += bigrams
    c = collections.Counter(words)
    if vocab:
        d = { e:c[e] for e in c if e in vocab }
    else:
        d = { e:c[e] for e in c if len(e) >= minlen_word }
    return d

def select_bestwords(D, y, nmax = 100, is_classif=True):
    """ Select nmax best correleted words in D (list of dicts) 
        with goal = y
    """
    y = np.asarray(y)
    v = DictVectorizer(sparse=True)
    try:
        X = v.fit_transform(D)
    except ValueError:
        logger.warning("===Except*** in select_bestwords D:%d y:%d",len(D),len(y))
        return (set([]))
    if is_classif:
        f=f_classif(X,y)
    else:
        f=f_regression(X,y)
    names = v.get_feature_names()
    # (F-value, p-value, word)
    a = [(f[0][i], f[1][i], names[i]) 
            for i in range(len(names))]
    a = sorted([e for e in a if e[1]<0.05], reverse=True)
    logger.debug("select_bestwords:%s",a[:16])
    top = set([ e[2] for e in a[:nmax] ])
    return top


def select_topwords(dlist, y, use_bigrams=1, mincount=10, nmax=100):
    """
    select top features (words and bigrams) from list of texts (dlist)
    best correlated with goals (y)
    """
    N = len(dlist)
    if nmax <= 0:   # return empty lists
        return set([]),[{} for _ in range(N)]
    wordlist = []
    voc = defaultdict(int)
    for i in range(N):
        words = extract_words(dlist[i], vocab=None,
            use_bigrams=use_bigrams)
        for w in words:
            voc[w] += words[w]
        wordlist.append(words)
    
    # select big words in voc
    voc = {w:i for (w,i) in voc.iteritems() 
        if w not in ENGLISH_STOP_WORDS and i >= mincount}
    
    # select words from voc only and set count=1
    wordlist1 = []
    for i in range(N):
        wordlist1.append({w:1 for w in wordlist[i] if w in voc})
    
    top1 = select_bestwords(wordlist1, y, nmax)

    # filter word list from top
    for i in range(N):
        wordlist[i] = {w:i for (w,i) in wordlist[i].iteritems() if w in top1}
    
    return top1, wordlist

# extract best words END

