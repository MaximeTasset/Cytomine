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


from cytomine import Cytomine
from cytomine.models import *
from cytomine.spectral.extractor import *
from shapely.geometry import Polygon,Point
from shapely.wkt import loads
from sklearn.ensemble import ExtraTreesClassifier as ETC
import numpy as np

import logging

#id_mtasset = 25637310
#c = Cytomine('demo.cytomine.be','f1f8cacc-b71a-4bc2-a6cd-e6bb40fd19b5','9e94aa70-4e7c-4152-8067-0feeb58d42eb', verbose= True)

#Cytomine connection parameters
cytomine_host="demo.cytomine.be"
cytomine_public_key="XXX"
cytomine_private_key="XXX"
id_project=28146931
#id_project=31054043
#id_users=[25637310]

#Connection to Cytomine Core
with Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path = '/api/', working_path = '/tmp/', verbose= logging.WARNING,timeout=1200) as cytomine:

  from cytomine.utilities.reader import *
  reader = CytomineSpectralReader(28407375,bounds = Bounds(0,0, 100000, 100000),tile_size = Bounds(0,0,30,30),overlap=0,num_thread=10)
  reader.read()
  reader.getResult()
#    image_groups_id = [im.id for im in imagegroup.ImageGroupCollection(filters={'project':id_project}).fetch()]
#    print(image_groups_id)
#    ##predict_terms_list = [term.id for term in conn.get_project_terms(id_project) if str(term.name) != 'BloodVessels']
#    predict_terms_list = [term.id for term in ontology.TermCollection(filters={'project':id_project}).fetch()]
#
#    extra = Extractor(nb_job=-1)
#    extra.loadDataFromCytomine(imagegroup_id_list=image_groups_id,id_project = id_project,id_users=None,predict_terms_list=predict_terms_list)
##    extra.saveFeatureSelectionInCSV("extraction-28146931.csv",n_estimators=50,max_features=100000,usedata=(1 if id_project==31054043 else 0.2))
#
#    print(extra.getinfo())


