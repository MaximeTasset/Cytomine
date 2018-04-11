# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2015. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


__author__          = "Marée Raphaël <raphael.maree@ulg.ac.be>"
__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"


from client.cytomine import Cytomine
from client.cytomine.models import *
from client.cytomine.spectral.sampler import *
from shapely.geometry import Polygon,Point
from shapely.wkt import loads
from sklearn.ensemble import ExtraTreesClassifier as ETC
import numpy as np

#id_mtasset = 25637310
#c = Cytomine('demo.cytomine.be','f1f8cacc-b71a-4bc2-a6cd-e6bb40fd19b5','9e94aa70-4e7c-4152-8067-0feeb58d42eb', verbose= True)

#Cytomine connection parameters
cytomine_host="demo.cytomine.be"
cytomine_public_key="f1f8cacc-b71a-4bc2-a6cd-e6bb40fd19b5"
cytomine_private_key="9e94aa70-4e7c-4152-8067-0feeb58d42eb"
id_project=28146931
id_project=31054043
#id_users=[25637310]

#Connection to Cytomine Core
conn = Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path = '/api/', working_path = '/tmp/', verbose= False,timeout=1200)

image_groups_id = [im.id for im in conn.get_image_groups(id_project)]
##predict_terms_list = [term.id for term in conn.get_project_terms(id_project) if str(term.name) != 'BloodVessels']
predict_terms_list = [term.id for term in conn.get_project_terms(id_project)]
##sampler = Sampler()
##sampler.loadDataFromCytomine(conn=conn,imagegroupls=image_groups_id,id_project = id_project,id_users=None,predict_terms_list=predict_terms_list)
##sampler.saveFeatureSelectionInCSV("extraction-Urothelium.csv",n_estimators=1000,max_features='auto')
##print(sampler.getinfo())
##data,label = sampler.rois2data()
##et = ETC(n_estimators=100,n_jobs=-1)
##ind = list(range(len(data)))
##np.random.shuffle(ind)
##et.fit(data[ind[:int(.5*len(data))]],label[ind[:int(.5*len(data))]])
##print(et.score(data[ind[int(.5*len(data)):]],label[ind[int(.5*len(data)):]]))
##
##predict_terms_list = [term.id for term in conn.get_project_terms(id_project) if str(term.name) != 'Urothelium']
#samplerr = Sampler()
#samplerr.loadDataFromCytomine(conn=conn,imagegroup_id_list=image_groups_id,id_project = id_project,id_users=None,predict_terms_list=predict_terms_list)
##samplerr.saveFeatureSelectionInCSV("extraction-BloodVessels.csv",n_estimators=100,max_features=100000)
#print(samplerr.getinfo())
#
#data,label = samplerr.rois2data()
#et = ETC(n_estimators=100,n_jobs=-1)
#ind = list(range(len(data)))
#np.random.shuffle(ind)
#et.fit(data[ind[:int(.5*len(data))]],label[ind[:int(.5*len(data))]])
#print(et.score(data[ind[int(.5*len(data)):]],label[ind[int(.5*len(data)):]]))

#predict_terms_list = [term.id for term in conn.get_project_terms(id_projects)]
#samplerrr = Sampler()
#samplerrr.loadDataFromCytomine(conn=conn,imagegroupls=image_groups_id,id_project = id_project,id_users=None,predict_terms_list=predict_terms_list)
#samplerrr.saveFeatureSelectionInCSV("extraction-all.csv",n_estimators=10,max_features='auto',usedata=.2)
#print(samplerrr.getinfo())
#
#data,label = samplerrr.rois2data(rois=None,sliceSize=(3,3),step=1)
#et = ETC(n_estimators=100,n_jobs=-1)
#ind = list(range(len(data)))
#np.random.shuffle(ind)
#et.fit(data[ind[:int(.5*len(data))]],label[ind[:int(.5*len(data))]])
#print(et.score(data[ind[int(.5*len(data)):]],label[ind[int(.5*len(data)):]]))

