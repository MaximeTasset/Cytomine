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


__author__          = "Maxime Tasset <maxime.tasset@student.ulg.ac.be>"
__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"


import cytomine
import sys

#Connect to cytomine, edit connection values
cytomine_host="demo.cytomine.be"
cytomine_public_key="XXX"
cytomine_private_key="XXX"
id_project=28146931

#Connection to Cytomine Core
conn = cytomine.Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path = '/api/', working_path = '/tmp/', verbose= False)

#
#define software parameter template

software = conn.add_software("Feature_Selection", "pyxitSuggestedTermJobService","ValidateAnnotation")

conn.add_software_parameter("cytomine_predict_term",software.id,"ListDomain",None,False,10,False,"/api/project/$currentProject$/term.json","name","name")
conn.add_software_parameter("cytomine_positive_predict_term",software.id,"ListDomain",None,True,10,False,"/api/project/$currentProject$/term.json","name","name")
conn.add_software_parameter("cytomine_users_annotation",software.id,"ListDomain",None,False,20,False,"/api/project/$currentProject$/user.json","username","username")
conn.add_software_parameter("cytomine_imagegroup",software.id,"ListDomain",None,False,30,False,"/api/project/$currentProject$/imagegroup.json","id","id")

conn.add_software_parameter("forest_n_estimators", software.id, "Number", 10, True, 1100, False)
conn.add_software_parameter("forest_min_samples_split", software.id, "Number", 2, True, 1300, False)
conn.add_software_parameter("forest_max_features", software.id, "String", 'auto', True, 1400, False)

conn.add_software_parameter("pyxit_save_to", software.id, "String", "/tmp", False, 1500, False)

#add software to a given project
addSoftwareProject = conn.add_software_project(id_project,software.id)
