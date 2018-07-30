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

with Cytomine(host="demo.cytomine.be", public_key="f1f8cacc-b71a-4bc2-a6cd-e6bb40fd19b5", private_key="9e94aa70-4e7c-4152-8067-0feeb58d42eb",
                  verbose=logging.WARNING) as cytomine:
    ig = ImageGroupCollection({"project":56924820}).fetch()[0]
    ext = Extractor(filename)
    ext.readFile()

    sm = SpectralModel(base_estimator=estimator,n_estimators=1,step=1,slice_size=(5,5),n_jobs=1)
    sm.fit(ext.rois,use=1)

    reader = CytomineSpectralReader(ig,Bounds(5531,2353,512,512),tile_size=Bounds(0,0,512,512),n_jobs=8)
    reader.read()
    muli_image = reader.getResult(True,False)
    with open("flutiste_data.im",'wb') as fp:
        pickle.dump(muli_image,fp,protocol=pickle.HIGHEST_PROTOCOL)
    mask = sm.predict(muli_image)
    with open("flutiste_data.mask",'wb') as fp:
        pickle.dump(mask,fp,protocol=pickle.HIGHEST_PROTOCOL)