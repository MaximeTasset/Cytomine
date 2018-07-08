# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:26:05 2018

@author: Maxime
"""

from sklearn.cluster import KMeans,MiniBatchKMeans
import numpy as np

class KMeanClustering:
    def __init__(self,X,y):
        """
        " X: array-like or sparse matrix, shape=(n_samples, n_features) Training instances to labeled clusters.
        " y: array-like of shape = [n_samples] The target values (class labels).
        """
        clusters_labels = np.unique(y)
        self.n_clusters = len(clusters_labels)
        self.X = X.copy()

        self.cluster_centers = np.empty((self.n_clusters,X.shape[1]))
        for i,label in enumerate(clusters_labels):
            self.cluster_centers[i] = np.mean(X[y==label],axis=0)


    def fit(self,X,**kargs):
        """
        " X: array-like or sparse matrix, shape=(n_samples, n_features) Training instances to unlabeled clusters.
        """
        kargs["n_clusters"] = self.n_clusters

        kargs["init"] = self.cluster_centers.copy()

        if X.shape[0] >= 10000 and False:
          self.kmeans = MiniBatchKMeans(**kargs)
        else:
          self.kmeans = KMeans(**kargs)

        return self.kmeans.fit(np.concatenate((X,self.X),axis=0))
