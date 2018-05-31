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

import sys

import pickle

import os
from time import localtime, strftime


from shapely.geometry import Polygon,MultiPolygon
import numpy as np

import cytomine
from cytomine.utilities import Bounds, CytomineSpectralReader
from cytomine.spectral.extractor import coordonatesToPolygons,polygonToAnnotation

from cytomine.models import User,CurrentUser,Job,Annotation

from multiprocessing import Pool

from shapely import wkt

from cytomine import CytomineJob



#-----------------------------------------------------------------------------------------------------------
#Functions


def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

#-----------------------------------------------------------------------------------------------------------



def main(argv):
    current_path = os.getcwd() +'/'+ os.path.dirname(__file__)
    # Define command line options


    print("Main function")
    with CytomineJob.from_cli(argv,verbose=logging.WARNING) as cj:
      id_project = cj.project.id

    p = optparse.OptionParser(description='Cytomine Segmentation prediction',
                              prog='Cytomine segmentation prediction',
                              version='0.1')

    p.add_option('--cytomine_host', type="string", default = 'beta.cytomine.be', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
    p.add_option('--cytomine_public_key', type="string", default = '', dest="cytomine_public_key", help="Cytomine public key")
    p.add_option('--cytomine_private_key',type="string", default = '', dest="cytomine_private_key", help="Cytomine private key")

    p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")
    p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")


    p.add_option('-i', '--cytomine_id_imagegroup', type='int', dest='cytomine_id_imagegroup', help="imagegroup id from cytomine")
    p.add_option('-t', '--cytomine_tile_size', type='int', dest='cytomine_tile_size', help="sliding tile size")

    p.add_option('-j', '--nb_jobs', type='int', dest='nb_jobs', help="number of parallel jobs")
    p.add_option('--startx', type='int', default=0, dest='cytomine_startx', help="start x position")
    p.add_option('--starty', type='int', default=0, dest='cytomine_starty', help="start y position")
    p.add_option('--endx', type='int', dest='cytomine_endx', help="end x position")
    p.add_option('--endy', type='int', dest='cytomine_endy', help="end y position")
    p.add_option('--cytomine_predict_term', type='int', dest='cytomine_predict_term', help="term id of predicted term (binary mode)")

    p.add_option('--pyxit_save_to', type='string', dest='pyxit_save_to', help="pyxit segmentation model file") #future: get it from server db

    p.add_option('--cytomine_overlap', type='int', dest='cytomine_overlap', help="reader overlap")

    p.add_option('--model_nb_jobs', type='int', dest='pyxit_nb_jobs', help="pyxit number of jobs for trees") #future: get it from server db


    options, arguments = p.parse_args( args = argv)
    if False:
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
        SoftwareParameter("n_jobs", type="Number", id_software=software.id, default_value=1, index=1000).save()
        SoftwareParameter("n_jobs_reader", type="Number", id_software=software.id, default_value=10, index=1000).save()

        SoftwareParameter("model_path", type="String", id_software=software.id, default_value="/tmp", index=1600).save()
        SoftwareParameter("model_name", type="String", id_software=software.id, default_value="model.pickle", index=1600).save()
        SoftwareParameter("model_nb_jobs", type="Number", id_software=software.id, default_value=4, index=1600).save()

        SoftwareParameter("cytomine_overlap", type="Number", id_software=software.id, default_value=10, index=1600).save()



    #Initialization
    print("TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime()))
    print("Loading prediction model (local)")
    with open(os.path.join(cj.parameters.model_path,cj.parameters.model_name), "rb") as fb:
        classifier = pickle.load(fp)
        if hasattr(classifier,"set_N_Jobs"):
            classifier.set_N_Jobs(cj.parameters.model_nb_jobs)
        elif hasattr(classifier,"n_jobs"):
            classifier.n_jobs = cj.parameters.model_nb_jobs

    n_jobs = cj.parameters.n_jobs
    n_jobs_reader = cj.parameters.n_jobs_reader
    predict_term = cj.parameters.cytomine_predict_term

    #Reading parameters

    id_imagegroup= cj.parameters.cytomine_imagegroup #int(sys.argv[1])


    tile_size = cj.parameters.cytomine_tile_size
    overlap = cj.parameters.cytomine_overlap

    cj.job.update(statusComment = "Initialization finished...")



    cj.job.update(statusComment = "Initialization of the reader", progress = 1)


    reader = CytomineSpectralReader(id_imagegroup,
                                    bounds = None,
                                    tile_size = Bounds(0,0,tile_size,tile_size),
                                    overlap=overlap,num_thread=n_jobs_reader)

    startx = cj.parameters.startx
    starty = cj.parameters.starty
    endx = cj.parameters.endx
    endy = cj.parameters.endy

    x,y = reader.reverseHeight((startx,endy))

    width = abs(startx-endx)
    height = abs(starty-endy)

    reader.setBounds(Bounds(x,y,width,height))

    cj.job.update(statusComment = "Start fetching data", progress = 5)

    results = MultiPolygon()

    cj.job.update(statusComment = "Initial read", progress = 6)

    for i in range(5):
      reader.read(async=True)
      if not reader.next():
        break


    cj.job.update(statusComment = "Start fetching data", progress = 7)

    stop = False

    iterate = 5
    pool = Pool(n_jobs)

    while not stop:
      for i in range(iterate):
        reader.read(async=True)
        if not reader.next():
          stop = True
          break
      fetch = []
      coords = []
      if not stop:
        it = iterate
      else:
        it = iterate*2
      for i in range(it):
        result = reader.getResult(all_coord=True,in_list=True)
        if result is None: #that means no results left to fetch
          break
        else:
          spectras,coord = result
          fetch.extend(spectras)
          coords.extend(coord.flatten)
      fetch = np.asarray(fetch)

      #make prediction here
      predictions = pyxit.predict(fetch)

      #get the coordonate of the pxl that correspond to the request
      coord = [reader.reverseHeight(coord[index[0]]) for index in np.argwhere(predictions==predict_term)]
      if len(coord):
        results = results.union(coordonatesToPolygons(coord,nb_job=n_jobs,pool,False))

    reader.first_id

    Annotation()
    results = results.buffer(-0.5).simplify(1,False)

    cj.job.update(statusComment = "Converting ROI to polygon", progress = 70)

    amp = pool.map(polygonToAnnotation,[p for p in resuls])

    cj.job.update(statusComment = "Uploading of the annotations on the first image of the imagegroup", progress = 95)

    for p in amp:
      Annotation(p,reader.first_id,
                 id_terms=predict_term,
                 id_project=id_project).save()


    cj.job.update(statusComment = "Finish Job..", progress = 100)

    pool.close()


if __name__ == "__main__":
    import sys
    exit(0)
    main(sys.argv[1:])