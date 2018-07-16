# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:26:05 2018

@author: Maxime
"""

from sklearn.cluster import KMeans,MiniBatchKMeans
from .extractor import Extractor,roi2data
import numpy as np
import psutil

class KMeanClustering:
    def __init__(self,X,y,sliceSize=(3,3),step=1,notALabelFlag=0):
        """
        " Xs: a list of n array-like or sparse matrix, shape=(width,heigth, n_features) Training instances to labeled clusters.
        " y: a list of n array-like of shape =(width,heigth) The target values (class labels).
        " notALabelFlag, the value in y which does not correspond to a label (ie unlabeled coordinate)
        """


        X,y = Extractor().rois2data(zip(X,y),sliceSize,step,notALabelFlag)
        clusters_labels = np.unique(y)
        self.n_clusters = len(clusters_labels)
        self.X = X
        self.sliceSize = sliceSize
        self.step = step

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


class SpectralModel:
    def __init__(self,base_estimator,nb_estimator,slice_size=(3,3),nb_job=-1):
        self.nb_job = nb_job if nb_job > 0 else max(psutil.cpu_count() + nb_job,1)
        self.sliceSize = slice_size

    def fit(X,y):
        pass


