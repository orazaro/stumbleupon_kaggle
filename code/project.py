#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Common functions to read data for project
    Set globals: codepath, projpath, datapath, gP
"""

import sys, csv, os, random, gzip
from collections import defaultdict
import numpy as np
import json

#from sklearn.datasets.base import Bunch
from kgml.base import Bunch

# set globals 
gP = None

class Project(object):
  def __init__(self, prpath = None):
    self.kglpath = os.getenv('KGL_PATH')
    if prpath:
        self.projpath = prpath
    elif self.kglpath:    
        self.projpath = self.kglpath+'/stumbleupon'
    else:
        codepath = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.projpath = '/'.join(codepath.split('/')[:-1])
    self.codepath = self.projpath+'/code'
    self.datapath = self.projpath+'/data'

  def dataset_save(self, data, dname, tozip=True):
    """Write data bunch as one csv file:
        1st row: feature_names
        1st col: ids (target_names)
        last col: targets
    """
    print >>sys.stderr, "save %s: [%d,%d]" % (dname,len(data.target_names),len(data.feature_names))
    fname = '%s/%s.csv' % (self.datapath, dname)
    if tozip:
        fname = fname + ".gz"
        fopen = gzip.open
    else:
        fopen = open
    with fopen(fname,"wb") as fp:
        writer = csv.writer(fp)
        header = ['id'] + data.feature_names + ['target']
        writer.writerow(header)
        for (i,data_row) in enumerate(data.data):
            row = [data.target_names[i]] + list(data_row) + [data.target[i]]
            writer.writerow(row)
  
  def dataset_load(self, dname, tozip=True):
    """read data bunch as one csv file:
        1st row: feature_names
        1st col: ids (target_names)
        last col: targets
        returns data bunch
    """
    reader = self.reader(dname,tozip=tozip)
    feature_names = next(reader)[1:-1]
    target_names, target, data = [],[],[]
    for row in reader:
        target_names.append(row[0])
        target.append(row[-1])
        data.append(row[1:-1])
    return Bunch(feature_names=feature_names, 
        data=np.asarray(data, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64), 
        target_names=target_names)
  
  def dataset_range(self, dataset, start, stop):
    return Bunch(feature_names=dataset.feature_names, 
        data=dataset.data[start:stop], 
        target=dataset.target[start:stop], 
        target_names=dataset.target_names[start:stop])
  
  def reader(self, tbl, delim=',', tozip=False):
    "Reader generator for csv files"
    fname = '%s/%s.csv' % (self.datapath, tbl)
    if tozip:
        fname = fname + ".gz"
        fopen = gzip.open
    else:
        fopen = open
    for e in csv.reader(fopen(fname),delimiter=delim):
        yield e
  
  def read(self, tbl, skip_head=True, tozip=False):
    "Read csv file into list"
    reader = self.reader(tbl,tozip=tozip)
    if skip_head:
        header = next(reader)
    return list(reader)
  
  def read_dict(self, tbl):
    "Read data as dictionary with first column as key"
    reader = self.reader(tbl)
    header = next(reader)
    header[10] = 'YearMade' # MfgYear is the same as YearMade
    return ( dict((e[0],e) for e in reader), header )

def test():
    p = Project()
    data = Bunch(feature_names=["x","y"], 
        data=np.array([[1,2],[11,22],[111,222]]), 
        target=np.array([1,2,3]), 
        target_names=["s1","s2","s3"])
    p.dataset_save(data,"test_project")
    data2 = p.dataset_load("test_project")
    print data2
    assert len(data)==len(data2)
    assert (data.data==data2.data).all()
    p.dataset_save(data,"test_project",tozip=True)
    data3 = p.dataset_load("test_project",tozip=True)
    print data3
    assert len(data)==len(data3)
    assert (data.data==data3.data).all()

    print "tests ok"

if __name__ == '__main__':
    random.seed(1)
    import argparse
    parser = argparse.ArgumentParser(description='Project.')
    parser.add_argument('cmd', nargs='?', default='test')
    args = parser.parse_args()
    print >>sys.stderr,args 
  
    if args.cmd == 'test':
        test()
    else:
        raise ValueError("bad cmd")

