# -*- coding: utf-8 -*-
"""
Created on Wed Mar 07 00:25:54 2018

@author: Maxime
"""

import os, optparse
from cytomine import Cytomine
from os.path import join
from cytomine.spectral.sampler import *

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
  p = optparse.OptionParser(description='Cytomine Detect Sample',
                              prog='Cytomine Detect Sample on Slide',
                              version='0.1')

  p.add_option('--cytomine_host', type="string", default = 'beta.cytomine.be', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
  p.add_option('--cytomine_public_key', type="string", default = 'XXX', dest="cytomine_public_key", help="Cytomine public key")
  p.add_option('--cytomine_private_key',type="string", default = 'YYY', dest="cytomine_private_key", help="Cytomine private key")
  p.add_option('--cytomine_base_path', type="string", default = '/api/', dest="cytomine_base_path", help="Cytomine base path")
  p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")
  p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")
  p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")

  p.add_option('--cytomine_predict_term', type='string',default=None, dest='cytomine_predict_term', help="list of term names take into account")
  p.add_option('--cytomine_positive_predict_term', type='string',default=None, dest='cytomine_positive_predict_term', help="list of positive term names take into account")
  p.add_option('--cytomine_imagegroup', type='string',default=None, dest='cytomine_imagegroup', help="list of imagegroup used for feature selection")

  p.add_option('--forest_min_samples_split', type="int",default=2, dest="forest_min_samples_split", help="The parameter \'min_samples_split\' of the forest")
  p.add_option('--forest_n_estimators', type="int",default=10, dest="forest_n_estimators", help="The parameter \'n_estimators\' of the forest")
  p.add_option('--forest_max_features', type="string",default='auto', dest="forest_max_features", help="The parameter \'max_features\' of the forest")

  p.add_option('--pyxit_save_to', type='string', dest='pyxit_save_to', help="the directory in which the csv will be saved")


  options, arguments = p.parse_args( args = argv)

  parameters['cytomine_host'] = options.cytomine_host
  parameters['cytomine_public_key'] = options.cytomine_public_key
  parameters['cytomine_private_key'] = options.cytomine_private_key
  parameters['cytomine_base_path'] = options.cytomine_base_path
  parameters['cytomine_working_path'] = options.cytomine_working_path
  parameters['cytomine_base_path'] = options.cytomine_base_path
  parameters['cytomine_id_software'] = options.cytomine_id_software
  parameters['cytomine_id_project'] = options.cytomine_id_project

  parameters['pyxit_save_to'] = options.pyxit_save_to
  try:
      parameters['forest_max_features'] = int(options.forest_n_estimators)
  except ValueError:
      parameters['forest_max_features'] = options.forest_n_estimators


  parameters['forest_n_estimators'] = options.forest_n_estimators
  parameters['forest_min_samples_split'] = options.forest_min_samples_split

  conn = Cytomine(parameters['cytomine_host'], parameters['cytomine_public_key'], parameters['cytomine_private_key'], base_path = parameters['cytomine_base_path'], working_path = parameters['cytomine_working_path'], verbose= False)

  if options.cytomine_predict_term is not None:
      terms_name = options.cytomine_predict_term.split(',')
      parameters['cytomine_predict_term'] = [term.id for term in conn.get_project_terms(parameters['cytomine_id_project']) if str(term.name) in terms_name]
  else:
      parameters['cytomine_predict_term'] = [term.id for term in conn.get_project_terms(parameters['cytomine_id_project'])]

  if options.cytomine_positive_predict_term is not None:
      parameters['cytomine_positive_predict_term'] = options.cytomine_predict_term.split(',')
  else:
      parameters['cytomine_positive_predict_term'] = None

  if options.cytomine_positive_predict_term is not None:
      parameters['cytomine_positive_predict_term'] = options.cytomine_predict_term.split(',')
  else:
      parameters['cytomine_positive_predict_term'] = None

  if options.cytomine_users_annotation is not None:
      parameters['cytomine_users_annotation'] = options.cytomine_users_annotation.split(',')
      parameters['cytomine_users_annotation'] = [user.id for user in conn.get_project_users(parameters['cytomine_id_project']) if user.username in  parameters['cytomine_users_annotation']]
  else:
      parameters['cytomine_users_annotation'] = None

  parameters['cytomine_imagegroup'] = options.cytomine_imagegroup.split(',')

  #Create a new userjob if connected as human user
  current_user = conn.get_current_user()
  run_by_user_job = False
  if current_user.algo==False:
      print("adduserJob...")
      user_job = conn.add_user_job(parameters['cytomine_id_software'], parameters['cytomine_id_project'])
      print("set_credentials...")
      conn.set_credentials(str(user_job.publicKey), str(user_job.privateKey))
      print("done")
  else:
      user_job = current_user
      print("Already running as userjob")
      run_by_user_job = True
  job = conn.get_job(user_job.job)

  print("Fetching data...")
  job = conn.update_job_status(job, status = job.RUNNING, status_comment = "Run...", progress = 0)

  samplerr = Sampler()
  job = conn.update_job_status(job, status = job.RUNNING, status_comment = "Fetching data...", progress = 75)
  samplerr.loadDataFromCytomine(conn=conn,imagegroupls=parameters['cytomine_imagegroup'],id_project = parameters['cytomine_id_project'],id_users=parameters['cytomine_users_annotation'],predict_terms_list=parameters['cytomine_predict_term'])

  d = os.path.dirname(parameters['pyxit_save_to'])
  if not os.path.exists(d):
      os.makedirs(d)
  samplerr.saveFeatureSelectionInCSV(join(d,"results.csv"),n_estimators=1000,max_features=100000)
  job = conn.update_job_status(job, status = job.TERMINATED, status_comment = "Finish", progress = 100)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])