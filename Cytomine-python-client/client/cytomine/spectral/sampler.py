# -*- coding: utf-8 -*-
"""
@author: Maxime
"""

import numpy as np
from .. import cytomine
from multiprocessing import RLock
from multiprocessing.pool import ThreadPool
from shapely.geometry import Polygon,Point
from shapely.wkt import loads
import pickle
from heapq import nlargest
from sklearn.feature_selection import chi2,f_classif
from sklearn.ensemble import ExtraTreesClassifier as ETC
import csv
import sys
import six
from copy import deepcopy

def f(i):
    return i*i
def ff(i):
    return i*i
class Sampler:
    def __init__(self, filename=None, cytomine_host=None, cytomine_public_key=None, cytomine_private_key=None, base_path = '/api/', working_path = '/tmp/',file_type=None,verbose=True):
        """
        Parameters
        ----------
        filename : the file name in which the coordonates will be read by default.
        type : the type of file that is read can be either 'binary' or 'text'
        square_size : the square side size (in pixel)
        """
        self.filename = filename
        self.cytomine_host = cytomine_host
        self.cytomine_public_key = cytomine_public_key
        self.cytomine_private_key = cytomine_private_key
        self.base_path = base_path
        self.working_path = working_path
        self.verbose = verbose
        self.pool = ThreadPool()

        self.data = None
        if not file_type:
            return
        elif file_type == "text":
            self.readFromTextFile()
        elif file_type == "binary":
            self.readFromBinaryFile()
        else:
            raise ValueError("error: expected \"binary\" or \"text\"")

    def readFile(self,filename=None):

        if not filename and self.filename:
             filename = self.filename
        elif not filename:
            raise ValueError("No filename given")

        f = open(filename, "rb")
        try:
            self.data = pickle.load(f)
        finally:
            f.close()

    def writeFile(self,filename=None,data=None):
        if not filename and self.filename:
             filename = self.filename
        elif not filename:
            raise ValueError("No filename")
        if not data and self.data:
             data = self.data
        elif not data:
            raise ValueError("No data")

        f = open(filename, "wb")
        try:
            pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
        finally:
            f.close()

    def chi2(self, sort=False, N=0,usedata=0.2):
        n_sample = len(self.data[1])
        ind = list(range(n_sample))
        np.random.shuffle(ind)
        ch,_ = chi2(self.data[1][ind[:int(usedata*n_sample)]],self.data[2][ind[:int(usedata*n_sample)]])
        if not N:
            N = len(ch)

        if sort:
          return nlargest(N,[(ch[i],i) for i in range(len(ch))])
        else:
          return [(ch[i],i) for i in range(len(ch))]

    def f_classif(self, sort=False, N=0,usedata=0.2):
        n_sample = len(self.data[1])
        ind = list(range(n_sample))
        np.random.shuffle(ind)
        f,_ = f_classif(self.data[1][ind[:int(usedata*n_sample)]],self.data[2][ind[:int(usedata*n_sample)]])
        if not N:
            N = len(f)

        if sort:
          return nlargest(N,[(f[i],i) for i in range(len(f))])
        else:
          return [(f[i],i) for i in range(len(f))]

    def features_ETC(self, sort=False, N=0,n_estimators=1000,max_features='auto',min_samples_split=2,usedata=0.2):
        n_sample = len(self.data[1])
        ind = list(range(n_sample))
        np.random.shuffle(ind)
        if not isinstance(max_features, six.string_types) and max_features is not None:
            max_features = max(1,min(max_features,int(self.data[1].shape[1])))
        etc = ETC(n_estimators=n_estimators,max_features=max_features,min_samples_split=min_samples_split).fit(self.data[1][ind[:int(usedata*n_sample)]],self.data[2][ind[:int(usedata*n_sample)]])
        f = etc.feature_importances_
        if not N:
            N = len(f)

        if sort:
          return nlargest(N,[(f[i],i) for i in range(len(f))])
        else:
          return [(f[i],i) for i in range(len(f))]

    def saveFeatureSelectionInCSV(self,filename,n_estimators=1000,max_features=None,min_samples_split=2,usedata=0.2):
      filename = str(filename)
      if not filename.endswith('.csv'):
        filename += ".csv"
      print("chi2")
      chi2 = self.chi2(usedata=usedata)
      print("f_classif")
      fclassif = self.f_classif(usedata=usedata)
      print("features_ETC")
      etc = self.features_ETC(n_estimators=n_estimators,max_features=max_features,min_samples_split=min_samples_split,usedata=usedata)

      with open(filename, 'w') as csvfile:
        fieldnames = ['layer','chi2', 'f_classif','ExtraTree']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,dialect='excel')

        writer.writeheader()
        for i in range(len(chi2)):
          writer.writerow({'layer':i,'chi2':chi2[i][0], 'f_classif':fclassif[i][0],'ExtraTree':etc[i][0]})

    def loadDataFromCytomine(self,conn=None,imagegroupls=[28417287],id_project = 28146931,id_users=None,predict_terms_list=None):
        if conn is None:
            conn = cytomine.Cytomine(self.cytomine_host, self.cytomine_public_key,
                                       self.cytomine_private_key, base_path = self.base_path,
                                       working_path = self.working_path, verbose= False)

        if predict_terms_list is None:
          terms = conn.get_project_terms(id_project)
          predict_terms_list = {term.id for term in terms}

        #load thread

        predict_terms_list = set(predict_terms_list)
        polys = []
        spect = []
        annot = []
        rois = []

        n = 0
        for imagegroup in imagegroupls:
            #Get project imagegroupHDF5 and images from imagegroup
            imagegroupHDF5 = conn.get_imageGroupHDF5(imagegroup).id
            images = conn.get_project_image_instances(id_project)

            for i in images:
                if self.verbose:
                    sys.stdout.write("\r                                                                   {}      ".format(n))
                    sys.stdout.flush()
                n += 1
                if i.numberOfAnnotations:
                  image = conn.get_image_instance(i.id)

                  #Get annotations in this image
                  if id_users is None:
                      annotationsList = [conn.get_annotations(
                                              id_project = id_project,
                                              id_user = None,
                                              id_image = image.id,
                                              showWKT=True,
                                              reviewed_only = False)]
                  else:
                      def ann(id_user) : return conn.get_annotations(id_project = id_project,id_user = id_user,id_image = image.id,showWKT=True,reviewed_only = False)
                      annotationsList = self.pool.map(ann,id_users)


                  height = image.height

                  annott,polyss,roiss,rect = extract_roi(annotationsList,predict_terms_list,image.width,image.height)
                  rl = RLock()
                  nb = len(rect)
                  self.done = 0
                  def getRect(rectangle):

                      (w,h,sizew,sizeh) = rectangle
                      co = deepcopy(conn)
                      sp = co.get_rectangle_spectre(imagegroupHDF5,w,h,sizew,sizeh)
                      if self.verbose:
                          with rl:
                              self.done += 1
                              sys.stdout.write("\r{}/{}      ".format(self.done,nb))
                              sys.stdout.flush()
                      return sp
                  spectras = self.pool.map(getRect,rect)
#                  f = lambda (w,h,sizew,sizeh):conn.get_rectangle_spectre(imagegroupHDF5,w,h,sizew,sizeh)
#                  spectras = [f(r) for r in rect]
                  for j,s in enumerate(spectras):
                     if s is not None and len(s):
                         annot.append(annott[j])
                         polys.append(polyss[j])
                         rois.append(roiss[j])
                         spect.append(s)
                         nimage=len(s[0].spectra)


#                  for annotations in annotationsList:
#                      for a in annotations.data():
#                          a.term = list(set(a.term) & predict_terms_list)
#
#                          #if the annotation has no asked term, do not take it
#                          if not len(a.term):
#                            continue
#
#                          annot.append(a)
#                          pol = Polygon(loads(a.location))
#                          polys.append(pol)
#
#                          minx, miny, maxx, maxy = pol.bounds
#
#                          sizew = int(abs(maxx - minx))
#                          sizeh = int(abs(maxy - miny))
#
#                          #conversion to API up-left (0,0) coordonate
#                          w = max(min(int(round(minx)),image.width),0)
#                          h = max(min(int(round(image.height - maxy)),image.height),0)
#
#                          if self.verbose:
#                              sys.stdout.write("\r{}      ".format((w,h,sizew,sizeh,image.height,n)))
#                              sys.stdout.flush()
#
#                          s =
#
#                          spect.append(s)
#                          nimage=len(s[0].spectra)
#                          rois.append((round(maxx),round(maxy),sizew,sizeh))

        if self.verbose:
            sys.stdout.write("\n")
            sys.stdout.flush()
        dataCoord = []
        dataX = []
        dataY = []
        unknownCoord = []
        unknownX = []

        self.rois = []

        for i in range(len(spect)):
            roi = np.zeros((rois[i][2],rois[i][3],nimage))
            roil = np.zeros((rois[i][2],rois[i][3]))
            for pixel in spect[i]:
                #conversion to Cytomine down-left (0,0) coordonate
                lp = [pixel.pxl[0],height-pixel.pxl[1]]

                rois[i][0]
                roi[int(abs(lp[0]-rois[i][0])-1),int(abs(lp[1]-rois[i][1])-1)] = pixel.spectra
                p = Point(lp)
                if polys[i].contains(p):
                    if len(annot[i].term):
                        roil[int(abs(lp[0]-rois[i][0])-1),int(abs(lp[1]-rois[i][1])-1)] = annot[i].term[0]

                    for term in annot[i].term:
                      dataCoord.append(lp)
                      dataX.append(pixel.spectra)
                      dataY.append(term)
                else:
                    unknown = True
                    for a,pol in enumerate(polys):
                        if pol.contains(p):
                            roil[int(abs(lp[0]-rois[i][0])-1),int(abs(lp[1]-rois[i][1])-1)] = annot[a].term[0]
                            unknown = False
                            break
                    if unknown:
                        unknownCoord.append(lp)
                        unknownX.append(pixel.spectra)
            self.rois.append((roi,roil))
#        self.polys = polys
#        self.spect = spect
#        self.annot = annot
        self.numData = int(len(dataCoord))
        self.numUnknown = int(len(unknownCoord))

        self.data = (np.asarray(dataCoord),
                     np.asarray(dataX),
                     np.asarray(dataY),
                     np.asarray(unknownCoord),
                     np.asarray(unknownX))
        self.numFeature = int(self.data[1].shape[1])

    def rois2data(self,rois=None,sliceSize=(3,3),step=1,flatten=True):
        """
        rois a list of tuple (np.array((width,height,features)),np.array((width,height)))
        """

        if rois is None:
            if hasattr(self,"rois"):
                rois = self.rois
            else:
                return
        x = []
        y = []
        for roi,labels in rois:
            for i in range(0,roi.shape[0]-sliceSize[0],step):
                for j in range(0,roi.shape[1]-sliceSize[1],step):
                    ic = int(abs(2*i+sliceSize[0])/2)
                    jc = int(abs(2*j+sliceSize[1])/2)
                    if labels[ic,jc]:
                        if flatten:
                            x.append(roi[i:i+sliceSize[0],j:j+sliceSize[1]].flatten())
                        else:
                            x.append(roi[i:i+sliceSize[0],j:j+sliceSize[1]])
                        y.append(labels[ic,jc])

        return np.array(x),np.array(y)


    def getinfo(self):
        if hasattr(self, "numData") and hasattr(self, "numUnknown") and hasattr(self, "numFeature"):
            return {"numData":self.numData,"numUnknown":self.numUnknown,"numFeature":self.numFeature}

def extract_roi(annotationsList,predict_terms_list,image_width,image_height):
    annot = []
    polys = []
    rois = []
    rect = []
    for annotations in annotationsList:
        for a in annotations.data():
            a.term = list(set(a.term) & predict_terms_list)

            #if the annotation has no asked term, do not take it
            if not len(a.term):
                continue

            annot.append(a)
            pol = Polygon(loads(a.location))
            polys.append(pol)

            minx, miny, maxx, maxy = pol.bounds

            sizew = int(abs(maxx - minx))
            sizeh = int(abs(maxy - miny))

            #conversion to API up-left (0,0) coordonate
            w = max(min(int(round(minx)),image_width),0)
            h = max(min(int(round(image_height - maxy)),image_height),0)

            rect.append((w,h,sizew,sizeh))
            rois.append((round(maxx),round(maxy),sizew,sizeh))
    return annot,polys,rois,rect
