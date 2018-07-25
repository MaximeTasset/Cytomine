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

save_path = "./tmp"

cytomine_host="demo.cytomine.be"
cytomine_public_key="XXX"
cytomine_private_key="XXX"
id_project= XXX
id_project=28146931

def main(argv):
    with Cytomine.connect_from_cli(argv):
        software = Software(name="Feature_Selection_2",
                            service_name="pyxitSuggestedTermJobService",
                            result_name="ValidateAnnotation").save()

#        SoftwareParameter("cytomine_id_software", type="Number", id_software=software.id,
#                          index=100, set_by_server=True, required=True).save()
#        SoftwareParameter("cytomine_id_project", type="Number", id_software=software.id,
#                          index=100, set_by_server=True, required=True).save()

        # filtering annotations
        SoftwareParameter("cytomine_predict_term", type="ListDomain", id_software=software.id, index=500, default_value='',
                          uri="/api/project/$currentProject$/term.json",uri_sort_attribut="name",uri_print_attribut="name").save()
        SoftwareParameter("cytomine_positive_predict_term", type="ListDomain", id_software=software.id, index=600, default_value='',
                          uri="/api/project/$currentProject$/term.json",uri_sort_attribut="name",uri_print_attribut="name").save()


        SoftwareParameter("cytomine_users_annotation", type="ListDomain", id_software=software.id, index=700, default_value='',
                          uri="/api/project/$currentProject$/user.json",uri_sort_attribut="username",uri_print_attribut="username").save()
        SoftwareParameter("cytomine_imagegroup", type="ListDomain", id_software=software.id, index=800, default_value='',
                          uri="/api/project/$currentProject$/imagegroup.json",uri_sort_attribut="id",uri_print_attribut="id").save()

        # running parameters
        SoftwareParameter("n_jobs", type="Number", id_software=software.id, default_value=1, index=1000).save()

        SoftwareParameter("forest_max_features", type="String", id_software=software.id, default_value="auto", index=1200).save()
        SoftwareParameter("forest_n_estimators", type="Number", id_software=software.id, default_value=10, index=1300).save()
        SoftwareParameter("forest_min_samples_split", type="Number", id_software=software.id, default_value=2, index=1400).save()
        SoftwareParameter("save_path", type="String", id_software=software.id, default_value="/tmp", index=1600).save()
        SoftwareParameter("save_as_xlsx", type="Number", id_software=software.id, default_value=0, index=1800).save()

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
                   "--n_jobs",str(4),
                   "--forest_max_features","auto",
                   "--forest_n_estimators",str(10),
                   "--forest_min_samples_split",str(2),
                   "--save_path",save_path,# complete path where the 'results.csv' will be saved
                   "--save_as_xlsx",str(1)])
      try:
        job = main_feature(argv)
      finally:

        with Cytomine(host="demo.cytomine.be", public_key=cytomine_public_key, private_key=cytomine_private_key,
                  verbose=logging.INFO) as cytomine:
                      software.delete()
