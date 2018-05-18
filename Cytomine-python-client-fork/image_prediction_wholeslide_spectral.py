# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2016. Authors: see NOTICE file.
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


__author__          = "Marée Raphael <raphael.maree@ulg.ac.be>"
__contributors__    = ["Gilles Louppe <g.louppe@gmail.com>", "Stévens Benjamin <b.stevens@ulg.ac.be>", "Olivier Caubo"]
__copyright__       = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"



#exemple usage, see test-predict.sh
#This is a whole workflow (code to be redesigned in a future release): It can work within ROIs (e.g. tissue section),
#apply a segmentation model (pixel classifier) tile per tile, detect connected components, perform union of detected geometries
#in the whole image, apply post-processing based on min/max are, apply a final classifier on geometries,
#and finally output statistics (counts/area).

try:
    import Image, ImageStat
except:
    from PIL import Image, ImageStat

import sys

import pickle

from progressbar import *
import os, optparse
from time import localtime, strftime


from shapely.geometry import Polygon,MultiPolygon
from shapely.ops import cas
import numpy as np

import cytomine
from cytomine.utilities.reader import Bounds, CytomineSpectralReader
from cytomine.spectral.extractor import coordonatesToPolygons

from cytomine.models.user import User,CurrentUser
from cytomine.models.software import Job

#Parameter values are now set through command-line
parameters = {
'cytomine_host' : None,
'cytomine_public_key' : None,
'cytomine_private_key' : None,
'cytomine_base_path' : None,
'cytomine_working_path' : '/home/maree/tmp/cytomine/',
'cytomine_id_software' : 0,
'cytomine_id_project' : 0,
'cytomine_id_image' : None,
'cytomine_zoom_level' : 0,
'cytomine_tile_size' : 512,
'cytomine_tile_min_stddev' : 0,
'cytomine_tile_max_mean' : 255,
'cytomine_predict_term' : 0,
'cytomine_union' : False,
'cytomine_postproc' : False,
'cytomine_count' : False,
'cytomine_min_size' : 0,
'cytomine_max_size' : 1000000000,
'cytomine_roi_term': None,
'cytomine_reviewed_roi': None,
'pyxit_target_width' : 24,
'pyxit_target_height' : 24,
'pyxit_predict_step' : 1,
'pyxit_save_to' : '/home/maree/tmp/cytomine/models/test.pkl',
'pyxit_colorspace' : 2,
'pyxit_nb_jobs' : 10,
'nb_jobs' : 20,
'publish_annotations': True ,

}



#-----------------------------------------------------------------------------------------------------------
#Functions


def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

#-----------------------------------------------------------------------------------------------------------



def main(argv):
    current_path = os.getcwd() +'/'+ os.path.dirname(__file__)
    # Define command line options
    print("Main function")
    p = optparse.OptionParser(description='Cytomine Segmentation prediction',
                              prog='Cytomine segmentation prediction',
                              version='0.1')

    p.add_option('--cytomine_host', type="string", default = 'beta.cytomine.be', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
    p.add_option('--cytomine_public_key', type="string", default = '', dest="cytomine_public_key", help="Cytomine public key")
    p.add_option('--cytomine_private_key',type="string", default = '', dest="cytomine_private_key", help="Cytomine private key")
    p.add_option('--cytomine_base_path', type="string", default = '/api/', dest="cytomine_base_path", help="Cytomine base path")
    p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")
    p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")
    p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")
    p.add_option('--cytomine_union', type="string", default="0", dest="cytomine_union", help="Turn on union of geometries")
    p.add_option('--cytomine_postproc', type="string", default="0", dest="cytomine_postproc", help="Turn on postprocessing")

    p.add_option('--cytomine_min_size', type="int", default=0, dest="cytomine_min_size", help="minimum size (area) of annotations")
    p.add_option('--cytomine_max_size', type="int", default=10000000000, dest="cytomine_max_size", help="maximum size (area) of annotations")

    p.add_option('-i', '--cytomine_id_imagegroup', type='int', dest='cytomine_id_imagegroup', help="imagegroup id from cytomine")
    p.add_option('-t', '--cytomine_tile_size', type='int', dest='cytomine_tile_size', help="sliding tile size")

    p.add_option('--cytomine_union_min_length', type='int', default=5, dest='cytomine_union_min_length', help="union")

    p.add_option('--cytomine_union_area', type='int', default=5, dest='cytomine_union_area', help="union")

    p.add_option('-j', '--nb_jobs', type='int', dest='nb_jobs', help="number of parallel jobs")
    p.add_option('--startx', type='int', default=0, dest='cytomine_startx', help="start x position")
    p.add_option('--starty', type='int', default=0, dest='cytomine_starty', help="start y position")
    p.add_option('--endx', type='int', dest='cytomine_endx', help="end x position")
    p.add_option('--endy', type='int', dest='cytomine_endy', help="end y position")
    p.add_option('--cytomine_predict_term', type='int', dest='cytomine_predict_term', help="term id of predicted term (binary mode)")
    p.add_option('--cytomine_roi_term', type='string', dest='cytomine_roi_term', help="term id of region of interest where to count)")

   p.add_option('--cytomine_reviewed_roi', type='string', default="0", dest="cytomine_reviewed_roi", help="Use reviewed roi only")

    p.add_option('--pyxit_target_width', type='int', dest='pyxit_target_width', help="pyxit subwindows width")
    p.add_option('--pyxit_target_height', type='int', dest='pyxit_target_height', help="pyxit subwindows height")
    p.add_option('--cytomine_predict_step', type='int', dest='cytomine_predict_step', help="pyxit step between successive subwindows")
    p.add_option('--pyxit_save_to', type='string', dest='pyxit_save_to', help="pyxit segmentation model file") #future: get it from server db

    p.add_option('--pyxit_post_classification', type="string", default="0", dest="pyxit_post_classification", help="pyxit post classification of candidate annotations")

    p.add_option('--pyxit_nb_jobs', type='int', dest='pyxit_nb_jobs', help="pyxit number of jobs for trees") #future: get it from server db

    p.add_option('--verbose', type='string', default="0", dest="verbose", help="Turn on (1) or off (0) verbose mode")


    options, arguments = p.parse_args( args = argv)

    parameters['cytomine_host'] = options.cytomine_host
    parameters['cytomine_public_key'] = options.cytomine_public_key
    parameters['cytomine_private_key'] = options.cytomine_private_key
    parameters['cytomine_base_path'] = options.cytomine_base_path
    parameters['cytomine_working_path'] = options.cytomine_working_path
    parameters['cytomine_base_path'] = options.cytomine_base_path
    parameters['cytomine_id_project'] = options.cytomine_id_project
    parameters['cytomine_id_software'] = options.cytomine_id_software
    parameters['cytomine_predict_term'] = options.cytomine_predict_term
    parameters['model_id_job'] = 0
    if options.cytomine_roi_term:
        parameters['cytomine_roi_term'] = map(int,options.cytomine_roi_term.split(','))
    parameters['cytomine_reviewed_roi'] = str2bool(options.cytomine_reviewed_roi)
    parameters['cytomine_union'] = str2bool(options.cytomine_union)
    parameters['cytomine_postproc'] = str2bool(options.cytomine_postproc)
    parameters['cytomine_mask_internal_holes'] = str2bool(options.cytomine_mask_internal_holes)
    parameters['cytomine_count'] = str2bool(options.cytomine_count)
    if options.cytomine_min_size:
        parameters['cytomine_min_size'] = options.cytomine_min_size
    if options.cytomine_max_size:
        parameters['cytomine_max_size'] = options.cytomine_max_size
    parameters['cytomine_predict_step'] = options.cytomine_predict_step
    parameters['pyxit_save_to'] = options.pyxit_save_to


    parameters['pyxit_nb_jobs'] = options.pyxit_nb_jobs
    parameters['cytomine_nb_jobs'] = options.pyxit_nb_jobs

    parameters['cytomine_id_imagegroup'] = options.cytomine_id_imagegroup

    parameters['cytomine_tile_size'] = options.cytomine_tile_size

    parameters['cytomine_startx'] = options.cytomine_startx
    parameters['cytomine_starty'] = options.cytomine_starty
    parameters['cytomine_endx'] = options.cytomine_endx
    parameters['cytomine_endy'] = options.cytomine_endy
    parameters['nb_jobs'] = options.nb_jobs


    print(parameters)


    # Check for errors in the options
    if options.verbose:
        print("[pyxit.main] Options = ", options)

    #Initialization
    print("TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime()))
    print("Loading prediction model (local)")
    fp = open(parameters["pyxit_save_to"], "r")
    print(fp)

    pyxit = pickle.load(fp)
    pyxit.n_jobs = parameters['nb_jobs'] #multithread subwindows extraction in pyxit
    pyxit.base_estimator.n_jobs= parameters['pyxit_nb_jobs']  #multithread tree propagation
    #Reading parameters
    id_imagegroup= parameters['cytomine_id_imagegroup'] #int(sys.argv[1])


    #Create local directory to dump tiles
    local_dir = "%s/slides/project-%d/tiles/" % (parameters['cytomine_working_path'], parameters["cytomine_id_project"])
    if not os.path.exists(local_dir):
        print("Creating tile directory: %s" %local_dir)
        os.makedirs(local_dir)



    print("TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime()))
    print("Parameters: %s" %parameters)


    #Cytomine connection
    print("Connection to Cytomine server")
    conn = cytomine.Cytomine(parameters["cytomine_host"],
                             parameters["cytomine_public_key"],
                             parameters["cytomine_private_key"],
                             verbose= True)


    print("Create Job and UserJob...")
    id_software = parameters['cytomine_id_software']
    #Create a new userjob if connected as human user
    current_user = CurrentUser().fetch()
    if current_user.algo==False:
        print("adduserJob...")

        job = Job( parameters['cytomine_id_project'], parameters['cytomine_id_software']).save()
        user_job = User().fetch(job.userJob)
        print("set_credentials...")
        conn.set_credentials(str(user_job.publicKey), str(user_job.privateKey))
        print("done")
    else:
        user_job = current_user
        print("Already running as userjob")

    job = Job().fetch(user_job.job)

    job.statusComment = "Publish software parameters values"
    job.update()


#    job = conn.update_job_status(job, status = job.RUNNING, progress = 0, status_comment = "Loading data...")
    job.statusComment = "Initialization of the reader"
    job.status = job.RUNNING
    job.progress = 0
    job.update()



    reader = CytomineSpectralReader(id_imagegroup,
                                    bounds = None,
                                    tile_size = Bounds(0,0,parameters['cytomine_tile_size'],parameters['cytomine_tile_size']),
                                    overlap=0,num_thread=parameters['nb_jobs'])

    x,y = reader.reverseHeight((parameters['cytomine_startx'],parameters['cytomine_endy']))

    width = abs(parameters['cytomine_startx']-parameters['cytomine_endx'])
    height = abs(parameters['cytomine_starty']-parameters['cytomine_endy'])
    reader.setBounds(Bounds(x,y,width,height))

    job.statusComment = "Start fetching data"
    job.status = job.RUNNING
    job.progress = 5
    job.update()

    results = MultiPolygon()

    job.statusComment = "Initial read"
    job.status = job.RUNNING
    job.progress = 6
    job.update()
    for i in range(5):
      reader.read(async=True)
      if not reader.next():
        break

    job.statusComment = "Start fetching data"
    job.status = job.RUNNING
    job.progress = 7
    job.update()

    stop = False
    while not stop:

      for i in range(5):
        reader.read(async=True)
        if not reader.next():
          stop = True
          break
      fetch = []
      coords = []
      if not stop:
        it = 5
      else:
        it = 10
      for i in range(it):
        result = reader.getResult(all_coord=True,in_list=True)
        if result is None: #that means no results left to fetch
          break
        else:
          spectras,coord = result
          fetch.extend(spectras)
          coords.extend(coord.flatten)
      fetch = np.asarray(fetch)
      predictions = pyxit.predict(fetch)

      #get the coordonate of the pxl that correspond to the request
      coord = [reader.reverseHeight(coord[index[0]]) for index in np.argwhere(predictions==parameters['cytomine_predict_term'])]
      results = results.union(coordonatesToPolygons(coord,nb_job=parameters['nb_jobs']))




    job.statusComment = "Finish Job.."
    job.status = job.TERMINATED
    job.progress = 100
    job.update()


    sys.exit()


if __name__ == "__main__":
    import sys
    exit(0)
    main(sys.argv[1:])