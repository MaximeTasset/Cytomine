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
from cytomine.spectral import Extractor,SpectralModel
from cytomine.models import TermCollection,UserCollection,ImageGroupCollection
from cytomine import CytomineJob
import pickle

import logging
from sklearn.ensemble import ExtraTreesClassifier


def main(argv):
  with CytomineJob.from_cli(argv,verbose=logging.WARNING) as cj:
      id_project = cj.project.id

      save_path = cj.parameters.save_path
      try:
          max_features = int(cj.parameters.forest_max_features)
      except ValueError:
          max_features = cj.parameters.forest_max_features

      n_estimators = cj.parameters.forest_n_estimators
      min_samples_split = cj.parameters.forest_min_samples_split
      step = cj.parameters.step
      slice_size = (cj.parameters.slice_size,cj.parameters.slice_size)
      n_jobs = cj.parameters.n_jobs
      use = cj.parameters.data_by_estimator

      termCollection = TermCollection(filters={'project':id_project}).fetch()
      if cj.parameters.cytomine_predict_term != '':
          terms_name = cj.parameters.cytomine_predict_term.split(',')
          predict_terms_list = [term.id for term in termCollection if str(term.name) in terms_name]
      else:
          predict_terms_list = [term.id for term in termCollection]

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

      print("Fitting Model...")
      cj.job.update(statusComment = "Fitting Model...", progress = 55)

      X = [x for x,y in ext.rois]
      y = [y for x,y in ext.rois]

      # Part that can be modify for the model
      def estimator():
        return ExtraTreesClassifier(min_samples_split=min_samples_split,max_features=max_features,n_estimators=n_estimators,n_jobs=n_jobs)

      # As we use the ExtraTreesClassifier, we fixe the 'n_estimators', 'n_jobs' and 'use' parameters to 1
      sm = SpectralModel(base_estimator=estimator,n_estimators=1,step=step,slice_size=slice_size,n_jobs=1)
      use = 1
      sm.fit(X,y,use=use)


      print("Saving The Model...")
      cj.job.update(statusComment = "Saving The Model...", progress = 95)
      with open(join(save_path,"model.pickle"),"wb") as m:
          pickle.dump(sm,m,pickle.HIGHEST_PROTOCOL)

      cj.job.update(statusComment = "Finished.", progress = 100)
      return cj.job

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