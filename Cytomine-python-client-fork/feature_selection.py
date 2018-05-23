# -*- coding: utf-8 -*-
"""
Created on Wed Mar 07 00:25:54 2018

@author: Maxime
"""

import os, optparse
from cytomine import Cytomine

from os.path import join
from cytomine.spectral import Extractor
from cytomine.models import *
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
  # Define command line options
  print("Main function")

  with CytomineJob.from_cli(argv) as cj:

      id_software = cj.parameters.cytomine_id_software
      id_project = cj.parameters.cytomine_id_project

      save_path = cj.parameters.save_path
      try:
          max_features = int(cj.parameters.forest_n_estimators)
      except ValueError:
          max_features = cj.parameters.forest_n_estimators


      n_estimators = cj.parameters.forest_n_estimators
      min_samples_split = cj.parameters.forest_min_samples_split

      if cj.parameters.cytomine_predict_term is not None:
          terms_name = cj.parameters.cytomine_predict_term.split(',')
          predict_terms_list = [term.id for term in cj.get_project_terms(id_project) if str(term.name) in terms_name]
      else:
          predict_terms_list = [term.id for term in TermCollection(filters={'project':id_project}).fetch()]

      if cj.parameters.cytomine_positive_predict_term is not None:
          positive_predict_terms_list = cj.parameters.cytomine_predict_term.split(',')
      else:
          positive_predict_terms_list = None

      if cj.parameters.cytomine_users_annotation is not None:
          parameters['cytomine_users_annotation'] = cj.parameters.cytomine_users_annotation.split(',')
          parameters['cytomine_users_annotation'] = [user.id for user in UserCollection(filters={"project": id_project}).fetch() if user.username in  parameters['cytomine_users_annotation']]
      else:
          parameters['cytomine_users_annotation'] = None

      parameters['cytomine_imagegroup'] = options.cytomine_imagegroup.split(',')

      #Create a new userjob if cjected as human user

      current_user = CurrentUser().fetch()
      run_by_user_job = False
      if current_user.algo==False:
          print("adduserJob...")
          from cytomine.models.user import User
          from cytomine.models.software import Job
          job = Job(id_project, id_software).save()
          user_job = User().fetch(job.userJob)
          print("set_credentials...")
          cj.set_credentials(str(user_job.publicKey), str(user_job.privateKey))
          print("done")
      else:
          user_job = current_user
          print("Already running as userjob")
          run_by_user_job = True


      job = cj.get_job(user_job.job)

      print("Fetching data...")

      update_job_status(job, status = job.RUNNING, status_comment = "Run...", progress = 0)

      ext = Extractor()
      update_job_status(job, status = job.RUNNING, status_comment = "Fetching data...", progress = 25)




      ext.loadDataFromCytomine(imagegroupls=parameters['cytomine_imagegroup'],id_project = id_project,
                               id_users=parameters['cytomine_users_annotation'],predict_terms_list=predict_terms_list)

      d = os.path.dirname(save_path)
      if not os.path.exists(d):
          os.makedirs(d)

      update_job_status(job, status = job.RUNNING, status_comment = "Feature Selection...", progress = 50)

      ext.saveFeatureSelectionInCSV(join(d,"results.csv"),n_estimators= n_estimators,
                                    max_features=max_features,min_samples_split=min_samples_split)

      update_job_status(job, status = job.TERMINATED, status_comment = "Finish", progress = 100)

def update_job_status(job, status, status_comment, progress):
    job.status = status if status else job.status
    job.statusComment = status_comment if status_comment else job.statusComment
    job.progress = progress if progress else job.progress
    job.update()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])