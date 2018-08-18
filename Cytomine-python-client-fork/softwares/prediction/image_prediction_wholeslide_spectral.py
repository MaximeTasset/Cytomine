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

import pickle

import os
from time import localtime, strftime

from shapely.geometry import MultiPolygon
import numpy as np

from cytomine.utilities.reader import Bounds, CytomineSpectralReader
from cytomine.spectral import coordonatesToPolygons,polygonToAnnotation

from cytomine.models import Annotation,TermCollection,ImageGroupCollection

from multiprocessing import Pool

from cytomine import CytomineJob
import logging
import json

def main(argv):
#    current_path = os.getcwd() +'/'+ os.path.dirname(__file__)

    print("Main function")
    with CytomineJob.from_cli(argv,verbose=logging.WARNING) as cj:
      id_project = cj.project.id


      #Initialization
      print("TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime()))
      print("Loading prediction model (local)")
      with open(os.path.join(cj.parameters.model_path,cj.parameters.model_name), "rb") as fp:
          classifier = pickle.load(fp)
      if hasattr(classifier,"set_N_Jobs"):
          classifier.set_N_Jobs(cj.parameters.model_nb_jobs)
      elif hasattr(classifier,"n_jobs"):
          classifier.n_jobs = cj.parameters.model_nb_jobs

      n_jobs = cj.parameters.n_jobs

      n_jobs_reader = cj.parameters.n_jobs_reader
      termCollection = TermCollection(filters={'project':id_project}).fetch()
      if cj.parameters.cytomine_predict_term != '':
          terms_name = json.loads(cj.parameters.cytomine_predict_term)
          if type(terms_name) == dict:
            terms_name = terms_name["collection"]

          if type(terms_name) == list:
            if not len(terms_name):
              predict_term_list = [term for term in classifier.labels]
            elif type(terms_name[0]) == str:
               predict_term_list = [term.id for term in termCollection if str(term.name) in terms_name and term.id in classifier.labels]
               if not len(predict_term_list):
                 predict_term_list = [term for term in classifier.labels]

      else:
          predict_term_list = [term for term in classifier.labels]

      #Reading parameters
      id_imagegroup= cj.parameters.cytomine_imagegroup
      if id_imagegroup != '':
          if not type(id_imagegroup) is int:
              id_imagegroup = json.loads(id_imagegroup)
              if type(terms_name) == dict:
                  id_imagegroup = id_imagegroup["id"]
              else:
                  id_imagegroup = int(id_imagegroup)
      else:
        id_imagegroup = ImageGroupCollection({"project":id_project}).fetch()[0].id




      if hasattr(classifier,"sliceSize"):
          classif_slize = max(classifier.sliceSize)
      else:
          classif_slize = 0
      tile_size = max(cj.parameters.cytomine_tile_size,classif_slize)

      overlap = max(cj.parameters.cytomine_overlap,classif_slize-1)

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

      results = {predict_term:MultiPolygon() for predict_term in predict_term_list}

      cj.job.update(statusComment = "Initial read", progress = 6)

      for i in range(5):
        reader.read(async=True)
        if not reader.next():
          break


      cj.job.update(statusComment = "Start fetching data", progress = 7)

      stop = False

      iterate = 5
      pool = Pool(n_jobs)
      try:
          while not stop:
              for i in range(iterate):
                  reader.read(async=True)
                  if not reader.next():
                      stop = True
                      break
              fetched_data = []
              coords = []
              if not stop:
                  it = iterate
              else:
                  it = iterate*2
              for i in range(it):
                  result = reader.getResult(all_coord=True,in_list=False)
                  if result is None: #that means no result left to fetch
                      break
                  else:
                      spectras,coord = result
                      fetched_data.append(spectras)
                      coords.append(coord)

              #make prediction here

              predictions = [classifier.predict(fetch) for fetch in fetched_data]

              #get the coordonate of the pxl that correspond to the request
              wanted_coords = {predict_term:[reader.reverseHeight(coords[i][index[0],index[1]]) for i,prediction in enumerate(predictions)
                                                                           for index in np.argwhere(prediction==predict_term)]
                                                                           for predict_term in predict_term_list}
              for predict_term in wanted_coords:
                  coords = wanted_coords[predict_term]
                  if len(coords):
                      results[predict_term] = results[predict_term].union(coordonatesToPolygons(coords,nb_job=n_jobs,pool=pool,trim=False))


          results = {predict_term:results[predict_term].buffer(-0.5).simplify(1,False)}

          cj.job.update(statusComment = "Converting ROI to polygon", progress = 70)

          amp = {predict_term:pool.map(polygonToAnnotation,[p for p in results[predict_term]]) for predict_term in results}

          cj.job.update(statusComment = "Uploading of the annotations on the first image of the imagegroup", progress = 95)

          for predict_term in amp:
              for p in amp[predict_term]:
                  Annotation(p,reader.first_id,
                             id_terms=predict_term,
                             id_project=id_project).save()

          cj.job.update(statusComment = "Finish Job..", progress = 100)
      finally:
          pool.close()
