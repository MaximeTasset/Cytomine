# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2018. Authors: see NOTICE file.
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

__copyright__       = "Copyright 2010-2018 University of Liège, Belgium, http://www.cytomine.be/"

from cytomine import Cytomine
from cytomine.models import Software, SoftwareParameter,SoftwareProject

#cytomine_host="demo.cytomine.be"
#cytomine_public_key="XXX"
#cytomine_private_key="XXX"
#id_project= XXX

cytomine_host="research.cytomine.be"
cytomine_public_key="f1f8cacc-b71a-4bc2-a6cd-e6bb40fd19b5"
cytomine_private_key="9e94aa70-4e7c-4152-8067-0feeb58d42eb"
id_project=28146931

save_path = "./tmp"

startx = 0
starty = 0
endx = 100
endy = 100


def main(argv):
    with Cytomine.connect_from_cli(argv):
        software = Software(name="Feature_Selection",
                            service_name="pyxitSuggestedTermJobService",
                            result_name="ValidateAnnotation").save()

        # filtering annotations
        SoftwareParameter("cytomine_imagegroup", type="Domain", id_software=software.id, index=800, default_value='',
                          uri="/api/project/$currentProject$/imagegroup.json",uri_sort_attribut="id",uri_print_attribut="id").save()
        SoftwareParameter("cytomine_tile_size", type="Number", id_software=software.id, index=900, default_value=30).save()

        SoftwareParameter("startx", type="Number", id_software=software.id, index=1000, default_value=0).save()
        SoftwareParameter("starty", type="Number", id_software=software.id, index=1100, default_value=1000).save()
        SoftwareParameter("endx", type="Number", id_software=software.id, index=1200, default_value=0).save()
        SoftwareParameter("endy", type="Number", id_software=software.id, index=1300, default_value=1000).save()

        SoftwareParameter("cytomine_predict_term", type="ListDomain", id_software=software.id, index=1400, default_value='',
                          uri="/api/project/$currentProject$/term.json",uri_sort_attribut="name",uri_print_attribut="name").save()

        # running parameters
        SoftwareParameter("n_jobs", type="Number", id_software=software.id, default_value=1, index=1500).save()
        SoftwareParameter("n_jobs_reader", type="Number", id_software=software.id, default_value=1, index=1600).save()

        SoftwareParameter("model_path", type="String", id_software=software.id, default_value="./tmp", index=1700).save()
        SoftwareParameter("model_name", type="String", id_software=software.id, default_value="model.pkl", index=1800).save()
        SoftwareParameter("model_nb_jobs", type="Number", id_software=software.id, default_value=1, index=1900).save()

        SoftwareParameter("cytomine_overlap", type="Number", id_software=software.id, default_value=10, index=2000).save()



        print(software.id)
        SoftwareProject(software.id,id_project).save()
        return software


if __name__ == "__main__":
    import sys
    import logging
    if len(sys.argv[1:]):
      software = main(sys.argv[1:])
    else:
      argv = ['--cytomine_host',cytomine_host,
              "--cytomine_public_key",cytomine_public_key,
              "--cytomine_private_key",cytomine_private_key]
      software = main(argv)
      from feature_selection import main as main_feature
      argv.extend(["--cytomine_id_project",str(id_project),
                   "--cytomine_id_software",str(software.id),
                   "--cytomine_predict_term",'', #list of the terms names used for the feature selection format 'name1,name2,name3' (note: if '', all terms will be used)
                   "--cytomine_positive_predict_term",'', #list of the terms names that will be merge format 'name1,name2,name3' (note: if '', no merge)
                   "--cytomine_users_annotation",'', #the annotations which have as user, one in the list will be used, not the others (note: if '' all annotation will be used)
                   "--cytomine_imagegroup",'', #list of the project imagegroup's id that will be used (note: if '' all imagegroup will be used)
                   "--startx",str(0),
                   "--startx",str(0),
                   "--n_jobs",str(4),
                   "--model_path",save_path,
                   "--forest_max_features","auto",
                   "--forest_n_estimators",str(10),
                   "--forest_min_samples_split",str(2),
                   "--save_path",save_path]) # complete path where the 'results.csv' will be saved
      try:
        job = main_feature(argv)
      finally:

        with Cytomine(host="demo.cytomine.be", public_key=cytomine_public_key, private_key=cytomine_private_key,
                  verbose=logging.INFO) as cytomine:
                      software.delete()




#import cytomine
#import sys
#
##Connect to cytomine, edit connection values
#cytomine_host="demo.cytomine.be"
#cytomine_public_key="f1f8cacc-b71a-4bc2-a6cd-e6bb40fd19b5"
#cytomine_private_key="9e94aa70-4e7c-4152-8067-0feeb58d42eb"
#id_project=28146931
#
##Connection to Cytomine Core
#conn = cytomine.Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path = '/api/', working_path = '/tmp/', verbose= False)
#
##define software parameter template
#
#software = conn.add_software("Feature_Selection", "pyxitSuggestedTermJobService","ValidateAnnotation")
#
#conn.add_software_parameter("cytomine_predict_term",software.id,"ListDomain",None,False,10,False,"/api/project/$currentProject$/term.json","name","name")
#conn.add_software_parameter("cytomine_positive_predict_term",software.id,"ListDomain",None,True,10,False,"/api/project/$currentProject$/term.json","name","name")
#conn.add_software_parameter("cytomine_users_annotation",software.id,"ListDomain",None,False,20,False,"/api/project/$currentProject$/user.json","username","username")
#conn.add_software_parameter("cytomine_imagegroup",software.id,"ListDomain",None,False,30,False,"/api/project/$currentProject$/imagegroup.json","id","id")
#
#conn.add_software_parameter("forest_n_estimators", software.id, "Number", 10, True, 1100, False)
#conn.add_software_parameter("forest_min_samples_split", software.id, "Number", 2, True, 1300, False)
#conn.add_software_parameter("forest_max_features", software.id, "String", 'auto', True, 1400, False)
#
#conn.add_software_parameter("pyxit_save_to", software.id, "String", "/tmp", False, 1500, False)
#
##add software to a given project
#addSoftwareProject = conn.add_software_project(id_project,software.id)