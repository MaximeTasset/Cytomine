# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 16:02:17 2018

@author: Maxime
"""


import logging
from cytomine import *
from cytomine.models import *
from cytomine.spectral import *
from sklearn.ensemble import ExtraTreesClassifier
from cytomine.utilities.reader import CytomineSpectralReader,Bounds

import pickle
import gzip
import numpy as np

import sys
filename = "flutistev4.save"
n_jobs = 90
id_project = 56924820
slice_size = 10

print(filename)


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
class kerastimator:
    def __init__(self,input_size,n_layer):
        self.input_size = input_size
        self.n_layer = n_layer

    def __build_model(self):
        model = Sequential()
        model.add(Dense(50, input_dim=self.input_size, activation='relu'))
        for i in range(1):
            model.add(Dense(50, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self,X,y):
        self.estimator = KerasClassifier(build_fn=self.__build_model, epochs=150, batch_size=int(y.shape[0]/100), verbose=0)
        self.estimator.fit(X,y)
        return self
    def predict(self,X):
        return self.estimator.predict(X).reshape(X.shape[0])

def test1():
#    ig = ImageGroupCollection({"project":56924820}).fetch()[0]
    ext = Extractor(filename,True,n_jobs)
    ext.readFile()
    def estimator():
        return ExtraTreesClassifier(n_estimators=1000,n_jobs=n_jobs)
    for slice_size in range(1,11):
        sys.stdout.writelines("slice size = {}\n".format(slice_size))
        sys.stdout.flush()

        sm = SpectralModel(base_estimator=estimator,n_estimators=1,slice_size=(slice_size,slice_size),n_jobs=1)
#        sm = SpectralModel(base_estimator=kerastimator,base_estimator_param={"input_size":(slice_size**2)*ext.numFeature},n_estimators=1,slice_size=(slice_size,slice_size),n_jobs=0)
        sm.fit(ext.rois,use=1)

        sys.stdout.writelines("End fit\n")
        sys.stdout.flush()

#        reader = CytomineSpectralReader(ig.id,Bounds(5531,2353,512,512),tile_size=Bounds(0,0,512,512),num_thread=8)
#        reader.read()
#        muli_image = reader.getResult(True,False)
#        with gzip.open("flutiste_data.im",'wb') as fp:
#            pickle.dump(muli_image,fp,protocol=pickle.HIGHEST_PROTOCOL)
        with gzip.open("flutiste_data.im","rb") as fp:
            multi_image,coord = pickle.load(fp)
        full_mask = np.zeros(multi_image.shape[:-1])
        for i in range(0,full_mask.shape[0],30-slice_size):
            for j in range(0,full_mask.shape[1],30-slice_size):
                    prediction = sm.predict(multi_image[i:i+30,j:j+30])
                    for k,l in np.ndindex((30,30)):
                        if i + k >= full_mask.shape[0] or j + l >= full_mask.shape[1]:
                            break
                        if prediction[k,l]:
                            full_mask[i+k,j+l] = prediction[k,l]
                    if j + 30 >= full_mask.shape[1]:
                        break
            if i + 30 >= full_mask.shape[0]:
                break

        with open("flutiste_data_best_{}.mask".format(slice_size),'wb') as fp:
            pickle.dump(full_mask,fp,protocol=pickle.HIGHEST_PROTOCOL)

def test2():
    ext = Extractor(filename,True,n_jobs)
    ext.readFile()

    slice_size = 10
    step = 50
    for slice_size in [1,10]:
        for name,max_features in [('randomized',1),("log2","log2"),("sqrt","sqrt"),("middle",int(slice_size*slice_size*ext.numFeature/2)),("all",None)]:
            sys.stdout.writelines("slice size = {}\n".format(slice_size))
            sys.stdout.flush()
            def estimator():
                return ExtraTreesClassifier(n_estimators=1000,n_jobs=n_jobs,max_features=max_features)
            sm = SpectralModel(base_estimator=estimator,n_estimators=1,slice_size=(slice_size,slice_size),n_jobs=1)
    #        sm = SpectralModel(base_estimator=kerastimator,base_estimator_param={"input_size":(slice_size**2)*ext.numFeature},n_estimators=1,slice_size=(slice_size,slice_size),n_jobs=0)
            sm.fit(ext.rois,use=1)

            sys.stdout.writelines("End fit\n")
            sys.stdout.flush()

    #        reader = CytomineSpectralReader(ig.id,Bounds(5531,2353,512,512),tile_size=Bounds(0,0,512,512),num_thread=8)
    #        reader.read()
    #        muli_image = reader.getResult(True,False)
    #        with gzip.open("flutiste_data.im",'wb') as fp:
    #            pickle.dump(muli_image,fp,protocol=pickle.HIGHEST_PROTOCOL)
            with gzip.open("flutiste_data.im","rb") as fp:
                multi_image,coord = pickle.load(fp)
            full_mask = np.zeros(multi_image.shape[:-1])
            for i in range(0,full_mask.shape[0],step-slice_size):
                for j in range(0,full_mask.shape[1],step-slice_size):
                        prediction = sm.predict(multi_image[i:i+step,j:j+step])
                        for k,l in np.ndindex((step,step)):
                            if i + k >= full_mask.shape[0] or j + l >= full_mask.shape[1]:
                                break
                            if prediction[k,l]:
                                full_mask[i+k,j+l] = prediction[k,l]
                        if j + step >= full_mask.shape[1]:
                            break
                if i + step >= full_mask.shape[0]:
                    break

            with open("flutiste_data_best_{}_{}.mask".format(slice_size,name),'wb') as fp:
                pickle.dump(full_mask,fp,protocol=pickle.HIGHEST_PROTOCOL)

with Cytomine(host="demo.cytomine.be", public_key="f1f8cacc-b71a-4bc2-a6cd-e6bb40fd19b5", private_key="9e94aa70-4e7c-4152-8067-0feeb58d42eb",
                  verbose=logging.WARNING) as cytomine:

     test2()