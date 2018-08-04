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
filename = "flutistev3.save"
n_jobs = 48
id_project = 56924820
slice_size = 10

print(filename)

def estimator():
    return ExtraTreesClassifier(n_estimators=1000,n_jobs=n_jobs)

with Cytomine(host="demo.cytomine.be", public_key="f1f8cacc-b71a-4bc2-a6cd-e6bb40fd19b5", private_key="9e94aa70-4e7c-4152-8067-0feeb58d42eb",
                  verbose=logging.WARNING) as cytomine:
#    ig = ImageGroupCollection({"project":56924820}).fetch()[0]
    ext = Extractor(filename,True,n_jobs)
    ext.readFile()
    for slice_size in range(1,11):
        sys.stdout.writelines("slice size = {}\n".format(slice_size))
        sys.stdout.flush()
        sm = SpectralModel(base_estimator=estimator,n_estimators=1,step=1,slice_size=(slice_size,slice_size),n_jobs=1)
        sm.fit(ext.rois,use=1)
        sys.stdout.writelines("End fit\n")
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

        with open("flutiste_data_{}.mask".format(slice_size),'wb') as fp:
            pickle.dump(full_mask,fp,protocol=pickle.HIGHEST_PROTOCOL)
