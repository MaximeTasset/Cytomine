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
    def __init__(self,base_estimator=DecisionTreeClassifier,n_estimators=10,step=1,slice_size=(3,3),n_job=-1,notALabelFlag=0):
        """
        " notALabelFlag, the value in y which does not correspond to a label (ie unlabeled coordinate)
        """
        self.n_job = n_job if n_job > 0 else max(psutil.cpu_count() + n_job + 1,1)
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.step = min(step,1)
        self.sliceSize = slice_size
        self.notALabelFlag = notALabelFlag

    def fit(self,X,y,use=0.8):
        """
        " Xs: a list of n array-like or sparse matrix, shape=(width,heigth, n_features) Training instances to labeled clusters.
        " y: a list of n array-like of shape =(width,heigth) The target values (class labels).
        """
        X,y = Extractor().rois2data(zip(X,y),self.sliceSize,self.step,self.notALabelFlag)

        self.estimators = []
        ln = len(y)
        indexes = [list(range(ln)) for i in range(self.n_job)]
        pool = ThreadPool(self.n_job)
        st = 0
        try:
            for i in range(self.n_estimators + 1):
                np.random.shuffle(indexes[int(i%self.n_job)])
                if i and i % self.n_job == 0 or i == self.n_estimators:
                    print(i-st)
                    self.estimators.extend(pool.map(fit,[(self.base_estimator,[X[indexes[j][:int(use*ln)],:],y[indexes[j][:int(use*ln)]]]) for j in range(i-st)]))
                    st = i
        finally:
            pool.close()

        return self

    def predict(self,X):
        """
        " X: array-like or sparse matrix, shape=(width,heigth, n_features) Training instances to unlabeled clusters.
        """

        Xdata,coord = roi2data(X,self.sliceSize,self.step,splitted=True)
        Xdata = np.array(Xdata)

        ylabels = []
        pool = ThreadPool(self.n_job)
        st = 0
        try:
            for i in range(self.n_estimators):
                if i and i % self.n_job == 0 or i == self.n_estimators-1:
                    ylabels.extend(pool.map(predict,[(self.estimators[j],Xdata) for j in range(st,i+1)]))
                    st = i
        finally:
            pool.close()

        ylabels = zip(*ylabels)

        y = np.empty(X.shape[:-1])
        y[:,:] = self.notALabelFlag

        for i,labels in enumerate(ylabels):
            label,count = np.unique(labels,True)
            label = label[np.argmax(count)]
            y[coord[i][0],coord[i][1]] = label

        return y

def fit(arg):
    estimator,arg = arg
    estimator = estimator()
    estimator.fit(*arg)
    return estimator

def predict(arg):
    estimator,y = arg
    return estimator.predict(y)