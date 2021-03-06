# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:26:05 2018

@author: Maxime
"""

from sklearn.cluster import KMeans,MiniBatchKMeans
from .extractor import Extractor,roi2data
import numpy as np
import psutil
from sklearn.tree import DecisionTreeClassifier
from multiprocessing.pool import ThreadPool


class KMeanClustering:
    def __init__(self,X,y,sliceSize=(3,3),step=1,notALabelFlag=0):
        """
        " X: a list of n array-like or sparse matrix, shape=(width,heigth, n_features) Training instances to labeled clusters.
        " y: a list of n array-like of shape =(width,heigth) The target values (class labels).
        " notALabelFlag, the value in y which does not correspond to a label (ie unlabeled coordinate)
        """


        X,y = Extractor().rois2data(zip(X,y),sliceSize,step,notALabelFlag)
        clusters_labels = np.unique(y)
        self.n_clusters = len(clusters_labels)
        self.X = X
        self.sliceSize = sliceSize
        self.step = step
        self.notALabelFlag = notALabelFlag
        self.cluster_centers = np.empty((self.n_clusters,X.shape[1]))
        for i,label in enumerate(clusters_labels):
            self.cluster_centers[i] = np.mean(X[y==label],axis=0)


    def fit(self,X,**kargs):
        """
        " X: array-like or sparse matrix, shape=(width,heigth, n_features) Training instances to unlabeled clusters.
        """
        kargs["n_clusters"] = self.n_clusters

        kargs["init"] = self.cluster_centers.copy()

        Xdata,coord = roi2data(X,self.sliceSize,self.step,splitted=True)
        Xdata = np.array(Xdata)

        if Xdata.shape[0] >= 10000 and False:
            self.kmeans = MiniBatchKMeans(**kargs)
        else:
            self.kmeans = KMeans(**kargs)

        return self.kmeans.fit(np.concatenate((Xdata,self.X),axis=0))


    def predict(self,X):
        """
        " X: array-like or sparse matrix, shape=(width,heigth, n_features) Training instances to unlabeled clusters.
        """
        Xdata,coord = roi2data(X,self.sliceSize,self.step,splitted=True)
        Xdata = np.array(Xdata)
        labels = self.kmeans.predict(X)

        y = np.empty(X.shape[:-1])
        y[:,:] = self.notALabelFlag

        for i,label in enumerate(labels):
            y[coord[i][0],coord[i][1]] = label

        return y

class SpectralModel:
    def __init__(self,
                 base_estimator=DecisionTreeClassifier,
                 base_estimator_param=None,
                 choice_function=np.argmax,
                 n_estimators=10,
                 slice_size=(3,3),
                 n_jobs=-1,
                 notALabelFlag=0):
        """
        " notALabelFlag, the value in y which does not correspond to a label (ie unlabeled coordinate)
        """
        self.n_jobs = n_jobs if n_jobs > 0 else max(psutil.cpu_count() + n_jobs + 1,1) if n_jobs < 0 else 0
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.base_estimator_param = base_estimator_param
        self.choice_function = choice_function
        self.sliceSize = slice_size
        self.notALabelFlag = notALabelFlag
    def set_N_Jobs(self,n_jobs):
        if n_jobs:
            self.n_jobs = n_jobs if n_jobs > 0 else max(psutil.cpu_count() + n_jobs + 1,1)

    def fit(self,X,y=None,use=0.8):
        """
        " X: a list of n array-like or sparse matrix, shape = (width,height, n_features) Training instances to labeled clusters.
        " y: a list of n array-like of shape = (width,height) The target values (class labels).
        " Note: if y is None, then X must be a list of tuple (x,y) with
        "                     - x array-like or sparse matrix, shape = (width,height, n_features)
        "                     - y array-like of shape = (width,height)
        """
        X,y = Extractor().rois2data(zip(X,y) if not y is None else X,self.sliceSize,1,self.notALabelFlag)
        self.labels = np.unique(y)
        self.estimators = []
        ln = len(y)

        if self.n_jobs > 1:
            indexes = [list(range(ln)) for i in range(self.n_jobs)]
            pool = ThreadPool(self.n_jobs)
            st = 0
            try:
                for i in range(self.n_estimators + 1):
                    np.random.shuffle(indexes[int(i%self.n_jobs)])
                    if i and i % self.n_jobs == 0 or i == self.n_estimators:
                        self.estimators.extend(pool.map(fit,[(self.base_estimator,
                                                              self.base_estimator_param,
                                                              [X[indexes[j][:int(use*ln)],:],y[indexes[j][:int(use*ln)]]]) for j in range(i-st)]))
                        st = i
            finally:
                pool.close()
        else:
            indexes = list(range(ln))
            for i in range(self.n_estimators):
                np.random.shuffle(indexes)
                self.estimators.append(fit([self.base_estimator,self.base_estimator_param,[X[indexes[:int(use*ln)],:],y[indexes[:int(use*ln)]]]]))


        return self

    def predict(self,X):
        """
        " X: array-like or sparse matrix, shape=(width,height, n_features)
        "
        """

        Xdata,coord = roi2data(X,self.sliceSize,1,splitted=True)
        Xdata = np.array(Xdata)

        ylabels = []
        if self.n_jobs > 1:
            pool = ThreadPool(self.n_jobs)
            st = 0
            try:
                for i in range(self.n_estimators):
                    if i and i % self.n_jobs == 0 or i == self.n_estimators-1:
                        ylabels.extend(pool.map(predict,[(self.estimators[j],Xdata) for j in range(st,i+1)]))
                        st = i
            finally:
                pool.close()

        else:
            for est in self.estimators:
                ylabels.append(est.predict(Xdata))

        ylabels = zip(*ylabels)

        y = np.empty(X.shape[:-1])
        y[:,:] = self.notALabelFlag

        for i,labels in enumerate(ylabels):
            label,count = np.unique(labels,True)
            label = label[self.choice_function(count)]
            y[coord[i][0],coord[i][1]] = label

        return y



def fit(arg):
    estimator,arg,fit_arg = arg
    if arg is None:
      estimator = estimator()
    else:
      estimator = estimator(**arg)
    estimator.fit(*fit_arg)
    return estimator

def predict(arg):
    estimator,y = arg
    return estimator.predict(y)