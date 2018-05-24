# -*- coding: utf-8 -*-
"""
Created on Wed Mar 07 00:25:54 2018

@author: Maxime
"""

import os
from cytomine import Cytomine

from os.path import join
from cytomine.spectral import Extractor
from cytomine.models import TermCollection,UserCollection
from cytomine import CytomineJob

parameters = {
'cytomine_host' : None,
'cytomine_public_key' : None,
'cytomine_private_key' : None,
'cytomine_base_path' : '/api/',
'cytomine_working_path' : '/tmp/',
'cytomine_id_software' : 0,
'cytomine_id_project' : 0,
'cytomine_predict_term' : [],
'cytomine_positive_predict_term' : [],
'cytomine_users_annotation': [],
'cytomine_imagegroup': [],
'forest_n_estimators': 10,
'forest_min_samples_split': 1,
'pyxit_save_to' : None,
}

def main(argv):
  with CytomineJob.from_cli(argv) as cj:

      id_project = cj.parameters.cytomine_id_project

      save_path = cj.parameters.save_path
      try:
          max_features = int(cj.parameters.forest_n_estimators)
      except ValueError:
          max_features = cj.parameters.forest_n_estimators


      n_estimators = cj.parameters.forest_n_estimators
      min_samples_split = cj.parameters.forest_min_samples_split

      if cj.parameters.cytomine_predict_term is not None and  cj.parameters.cytomine_predict_term != '':
          terms_name = cj.parameters.cytomine_predict_term.split(',')
          predict_terms_list = [term.id for term in cj.get_project_terms(id_project) if str(term.name) in terms_name]
      else:
          predict_terms_list = [term.id for term in TermCollection(filters={'project':id_project}).fetch()]

      if cj.parameters.cytomine_positive_predict_term is not None and cj.parameters.cytomine_positive_predict_term != '':
          terms_name = cj.parameters.cytomine_positive_predict_term.split(',')
          positive_predict_terms_list = [term.id for term in cj.get_project_terms(id_project) if str(term.name) in terms_name and term.id in predict_terms_list]
          if not len(positive_predict_terms_list):
              positive_predict_terms_list = None
      else:
          positive_predict_terms_list = None

      if cj.parameters.cytomine_users_annotation is not None:
          users_annotation = cj.parameters.cytomine_users_annotation.split(',')
          users_annotation = [user.id for user in UserCollection(filters={"project": id_project}).fetch() if user.username in  users_annotation]
      else:
          users_annotation = None

      imagegroup_ids = [int(image_group) for image_group in cj.parameters.cytomine_imagegroup.split(',')]

      print("Fetching data...")

      cj.job.update(statusComment = "Run...")

      ext = Extractor()
      cj.job.update(statusComment = "Fetching data...", progress = 5)


      ext.loadDataFromCytomine(imagegroupls=imagegroup_ids,id_project = id_project,
                               id_users=users_annotation,predict_terms_list=predict_terms_list)

      d = os.path.dirname(save_path)
      if not os.path.exists(d):
          os.makedirs(d)

      if positive_predict_terms_list is not None:
          cj.job.update(statusComment = "Regrouping Positive terms...", progress = 50)
          positive_id = max(ext.data["Y"]) + 1
          for pos_id in positive_predict_terms_list:
              ext.data["Y"][ext.data["Y"] == pos_id] = positive_id

      cj.job.update(statusComment = "Feature Selection...", progress = 55)

      ext.saveFeatureSelectionInCSV(join(d,"results.csv"),n_estimators= n_estimators,
                                    max_features=max_features,min_samples_split=min_samples_split)

      cj.job.update(statusComment = "Finished.", progress = 100)

def update_job_status(job, status, status_comment, progress):
    job.status = status if status else job.status
    job.statusComment = status_comment if status_comment else job.statusComment
    job.progress = progress if progress else job.progress
    job.update()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])