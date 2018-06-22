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

import os

from os.path import join
from cytomine.spectral import Extractor
from cytomine.models import TermCollection,UserCollection,ImageGroupCollection
from cytomine import CytomineJob

import logging

def main(argv):
  with CytomineJob.from_cli(argv,verbose=logging.WARNING) as cj:
      id_project = cj.project.id

      save_path = cj.parameters.save_path
      try:
          max_features = int(cj.parameters.forest_n_estimators)
      except ValueError:
          max_features = cj.parameters.forest_n_estimators


      n_estimators = cj.parameters.forest_n_estimators
      min_samples_split = cj.parameters.forest_min_samples_split

      termCollection = TermCollection(filters={'project':id_project}).fetch()
      if cj.parameters.cytomine_predict_term != '':
          terms_name = cj.parameters.cytomine_predict_term.split(',')
          predict_terms_list = [term.id for term in termCollection if str(term.name) in terms_name]
      else:
          predict_terms_list = [term.id for term in termCollection]

      if cj.parameters.cytomine_positive_predict_term != '':
          terms_name = cj.parameters.cytomine_positive_predict_term.split(',')
          positive_predict_terms_list = [term.id for term in termCollection if str(term.name) in terms_name and term.id in predict_terms_list]
          if not len(positive_predict_terms_list):
              positive_predict_terms_list = None
      else:
          positive_predict_terms_list = None

      if cj.parameters.cytomine_users_annotation != '':
          users_annotation = cj.parameters.cytomine_users_annotation.split(',')
          users_annotation = [user.id for user in UserCollection(filters={"project": id_project}).fetch() if user.username in  users_annotation]
      else:
          users_annotation = None

      if not cj.parameters.cytomine_imagegroup is None and cj.parameters.cytomine_imagegroup != '':
          imagegroup_ids = [int(image_group) for image_group in cj.parameters.cytomine_imagegroup.split(',')]
      else:
          imagegroup_ids =  [im.id for im in ImageGroupCollection(filters={'project':id_project}).fetch()]


      print("Fetching data...")

      cj.job.update(statusComment = "Run...")

      ext = Extractor()
      cj.job.update(statusComment = "Fetching data...", progress = 5)


      ext.loadDataFromCytomine(imagegroup_id_list=imagegroup_ids,id_project = id_project,
                               id_users=users_annotation,predict_terms_list=predict_terms_list)


      os.makedirs(save_path,exist_ok=True)

      if positive_predict_terms_list is not None:
          cj.job.update(statusComment = "Regrouping Positive terms...", progress = 50)
          positive_id = max(ext.data["Y"]) + 1
          for pos_id in positive_predict_terms_list:
              ext.data["Y"][ext.data["Y"] == pos_id] = positive_id
      print("Feature Selection...")
      cj.job.update(statusComment = "Feature Selection...", progress = 55)

      ext.saveFeatureSelectionInCSV(join(save_path,"results.csv"),n_estimators= n_estimators,
                                    max_features=max_features,min_samples_split=min_samples_split)

      cj.job.update(statusComment = "Finished.", progress = 100)
      return cj.job

def update_job_status(job, status, status_comment, progress):
    job.status = status if status else job.status
    job.statusComment = status_comment if status_comment else job.statusComment
    job.progress = progress if progress else job.progress
    job.update()

if __name__ == "__main__":
    import sys
    if len(sys.argv[1:]):
      software = main(sys.argv[1:])
    else:
      software = main(['--cytomine_host',"demo.cytomine.be",
                       "--cytomine_public_key","f1f8cacc-b71a-4bc2-a6cd-e6bb40fd19b5",
                       "--cytomine_private_key","9e94aa70-4e7c-4152-8067-0feeb58d42eb",
                       ])

    main(sys.argv[1:])