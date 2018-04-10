# -*- coding: utf-8 -*-
"""
Created on Thu Apr 05 17:35:21 2018

@author: Maxime
"""

from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    p = Pool()
    print(p.map(f,[1,12,23]))