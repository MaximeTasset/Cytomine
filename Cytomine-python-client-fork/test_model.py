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

filename = "FlutisteData_n_u.save"
n_jobs = 24
id_project = 56924820

def estimator():
    return ExtraTreesClassifier(n_estimators=1000,n_jobs=n_jobs)

with Cytomine(host="demo.cytomine.be", public_key="XXX", private_key="XXX",
                  verbose=logging.WARNING) as cytomine:
    ig = ImageGroupCollection({"project":56924820}).fetch()[0]
    ext = Extractor(filename,True,n_jobs)
    ext.readFile()

    sm = SpectralModel(base_estimator=estimator,n_estimators=1,step=1,slice_size=(5,5),n_jobs=1)
    sm.fit(ext.rois,use=1)

#    reader = CytomineSpectralReader(ig.id,Bounds(5531,2353,512,512),tile_size=Bounds(0,0,512,512),num_thread=8)
#    reader.read()
#    muli_image = reader.getResult(True,False)
#    with open("flutiste_data.im",'wb') as fp:
#        pickle.dump(muli_image,fp,protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open("flutiste_data.im","rb") as fp:
        multi_image,coord = pickle.load(fp)
    full_mask = np.zeros(multi_image.shape[:-1])
    for i in range(0,full_mask.shape[0],15):
        for j in range(0,full_mask.shape[1],15):
                full_mask[i:i+20,j:j+20] = sm.predict(multi_image[i:i+20,j:j+20])
                if j + 20 >= full_mask.shape[1]:
                    break
        if i + 20 >= full_mask.shape[0]:
            break

    with open("flutiste_data.mask",'wb') as fp:
        pickle.dump(full_mask,fp,protocol=pickle.HIGHEST_PROTOCOL)
