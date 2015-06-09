# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:24:12 2015

@author: amlopez
"""

from dtw import dtw
import numpy as np
import sklearn.preprocessing as prep


global sen1
global sen2

def my_custom_norm(x, y):
    return np.linalg.norm(sen1[:,x]-sen2[:,x])
    

def multi_dtw(a, b, normalize = True):
    sen1 = a
    sen2 = b
    
    if(normalize):
        a = prep.scale(a)
        b = prep.scale(b)
    
    indexa = np.arange(a.shape[1])    
    indexb = np.arange(b.shape[1])    
    
    return dtw(indexa, indexb)
    
