#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Submit functions
"""
import sys, random, pickle, copy
import numpy as np
import datetime, csv, gzip
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from project import Project

def do_submit(pred):
    submname = 'submission_%s' % (datetime.datetime.today().strftime("%Y%m%d_%H%M%S"),)
    write_submit(pred,submname) 
    do_cmp(submname)

def write_submit(pred,submname):
    testfile = pd.read_csv('../data/test.tsv', sep="\t", na_values=['?'], index_col=1)
    pred_df = pd.DataFrame(pred, index=testfile.index, columns=['label'])
    fn = '%s/%s.csv.gz' % (Project().datapath,submname)
    with gzip.open(fn,'wb') as fp:
        pred_df.to_csv(fp)
        print "%s file created.." % submname

def mix_load(fn):
    fn = '%s/%s.csv.gz' % (Project().datapath, fn)
    with gzip.open(fn,'rb') as fp:
        reader = csv.reader(fp) 
        header = next(reader)
        return np.array(list(reader),dtype=float)[:,1:]
def mix_short(sn):
    for pref in ['submission_','mix_']:
        l = len(pref)
        if sn[:l] == pref:
            return sn[l:]
    return sn
def mix(sn1, sn2):
    s1 = mix_load(sn1)
    s2 = mix_load(sn2)
    pred = (s1 + s2) / 2.0
    submname = 'mix_%s_and_%s' % (mix_short(sn1),mix_short(sn2))
    write_submit(pred,submname) 
    do_cmp(submname)
def mix3(sn1, sn2, sn3):
    s1 = mix_load(sn1)
    s2 = mix_load(sn2)
    s3 = mix_load(sn3)
    pred = (s1 + s2 + s3) / 3.0
    submname = 'mix_%s_and_%s_and_%s' % (mix_short(sn1),mix_short(sn2),mix_short(sn3))
    write_submit(pred,submname) 
    do_cmp(submname)

def do_cmp(sn1, sn2="submission_model02"):
    def load(fn):
        print fn,
        fn = '%s/%s.csv.gz' % (Project().datapath, fn)
        with gzip.open(fn,'rb') as fp:
            reader = csv.reader(fp) 
            header = next(reader)
            return np.array(list(reader),dtype=float)[:,1:]
    s1 = load(sn1)
    print
    s2 = load(sn2)
    print "mae: %.4f" % mean_absolute_error(s1,s2)
    s2 = load('submission_mix')
    print "mae2: %.4f" % mean_absolute_error(s1,s2)

def test():
    print "tests ok"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Mix.')
    parser.add_argument('cmd', nargs='?', default='test')
    parser.add_argument('-rs', default=None)
    parser.add_argument('-mix', default='')
    args = parser.parse_args()
    print >>sys.stderr,args 
  
    if args.cmd == 'test':
        test()
    elif args.cmd == 'cmp':
        names = args.mix.split('+')
        do_cmp(*names)
    elif args.cmd == 'mix':
        names = args.mix.split('+')
        mix(*names)
    elif args.cmd == 'mix3':
        names = args.mix.split('+')
        mix3(*names)
    else:
        raise ValueError("bad cmd")

