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
      id_project = cj.project.i


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
        predictions = classifier.predict(fetch)

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