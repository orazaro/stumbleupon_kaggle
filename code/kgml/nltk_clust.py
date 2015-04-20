#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# used sources from:
#   https://gist.github.com/xim/1279283
# License:  BSD 3 clause

import numpy as np
import nltk
import nltk.corpus
from nltk import decorators
import nltk.stem

supported_stemmers = ["PorterStemmer","SnowballStemmer","LancasterStemmer","WordNetLemmatizer"]

#stemmer_func = nltk.stem.EnglishStemmer().stem
#stemmer_func = nltk.stem.PorterStemmer().stem
stemmer_func = nltk.stem.snowball.EnglishStemmer().stem
stopwords = set(nltk.corpus.stopwords.words('english'))

@decorators.memoize
def normalize_word(word):
    return stemmer_func(word.lower())

def get_all_words(sentences):
    words = set()
    for sent in sentences:
        for word in sent.split():
            words.add(normalize_word(word))
    return list(words)

#@decorators.memoize
def vectorspaced(words, sentence):
    sentence_components = [normalize_word(word) for word in sentence.split()]
    return np.array([
        word in sentence_components and not word in stopwords
        for word in words], np.short)

