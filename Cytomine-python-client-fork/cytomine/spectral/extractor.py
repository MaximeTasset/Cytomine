# -*- coding: utf-8 -*-
"""
@author: Maxime
"""

import numpy as np
from ..cytomine import *
from ..models.imagegroup import *
from ..models.ontology import TermCollection
from ..models.image import ImageInstance
from ..models.annotation import AnnotationCollection


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
import socket,time
import psutil

class Extractor:
    def __init__(self, filename=None,file_type=None,verbose=True,nb_job=1):
        """
        Parameters
        ----------
        filename : the file name in which the coordonates will be read by default.
        type : the type of file that is read can be either 'binary' or 'text'
        square_size : the square side size (in pixel)
        """
        self.filename = filename
        self.nb_job = nb_job if nb_job > 0 else max(psutil.cpu_count() + nb_job,1)
        self.verbose = verbose
        self.pool = ThreadPool(self.nb_job)

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

    def chi2(self, sort=False, N=0,usedata=1):
        n_sample = len(self.data["X"])
        ind = list(range(n_sample))
        np.random.shuffle(ind)
        ch,_ = chi2(self.data["X"][ind[:int(usedata*n_sample)]],self.data["Y"][ind[:int(usedata*n_sample)]])
        if not N:
            N = len(ch)

        if sort:
          return nlargest(N,[(ch[i],i) for i in range(len(ch))])
        else:
          return [(ch[i],i) for i in range(len(ch))]

    def f_classif(self, sort=False, N=0,usedata=1):
        n_sample = self.numData
        ind = list(range(n_sample))
        np.random.shuffle(ind)
        f,_ = f_classif(self.data["X"][ind[:int(usedata*n_sample)]],self.data["Y"][ind[:int(usedata*n_sample)]])
        if not N:
            N = len(f)

        if sort:
          return nlargest(N,[(f[i],i) for i in range(len(f))])
        else:
          return [(f[i],i) for i in range(len(f))]

    def features_ETC(self, sort=False, N=0,n_estimators=1000,max_features='auto',min_samples_split=2,usedata=1):
        n_sample = self.numData
        ind = list(range(n_sample))
        np.random.shuffle(ind)
        if not isinstance(max_features, six.string_types) and max_features is not None:
            max_features = max(1,min(max_features,int(self.data["X"].shape[1])))
        etc = ETC(n_estimators=n_estimators,max_features=max_features,min_samples_split=min_samples_split,n_jobs=-1).fit(self.data["X"][ind[:int(usedata*n_sample)]],self.data["Y"][ind[:int(usedata*n_sample)]])
        f = etc.feature_importances_
        if not N:
            N = len(f)

        if sort:
          return nlargest(N,[(f[i],i) for i in range(len(f))])
        else:
          return [(f[i],i) for i in range(len(f))]

    def saveFeatureSelectionInCSV(self,filename,n_estimators=1000,max_features=None,min_samples_split=2,usedata=1):
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


    def loadDataFromCytomine(self,imagegroup_id_list=[28417287],id_project = 28146931,id_users=None,predict_terms_list=None,max_fetch_size=(10,10)):
        """
        " read the annotations of a list of imagegroup from a project
        """

        if predict_terms_list is None:
          terms = TermCollection(filters={'project':id_project})
          predict_terms_list = {term.id for term in terms}

        predict_terms_list = set(predict_terms_list)
        polys = []
        spect = []
        annot = []
        rois = []

        n = 0

        for imagegroup_id in imagegroup_id_list:
            #Get project imagegroupHDF5 and images from imagegroup_id
            imagegroupHDF5 = ImageGroup(id=imagegroup_id).image_groupHDF5()
            if not imagegroupHDF5:
                continue
            else:
              imagegroupHDF5 = imagegroupHDF5.id

            #allow to get only the images used in the HDF5 imageGroup
            images = ImageSequenceCollection(filters={"imagegroup":imagegroup_id}).fetch()
            if not images:
              continue

            for im in images:

                if self.verbose:
                    sys.stdout.write("\r                                                                   {}      ".format(n))
                    sys.stdout.flush()
                n += 1
                image = ImageInstance(id=im.image).fetch()
                if image.numberOfAnnotations:
                  #Get annotations in this image
                  if id_users is None:
                      annotations_list = [AnnotationCollection(project=id_project, user=None, image=image.id, term=None,
                                                               showMeta=None, bbox=None, bboxAnnotation=None, reviewed=False,
                                                               showTerm=True).fetch()]
                  else:
                      def ann(id_user) :
                        return AnnotationCollection(project=id_project, user=id_user, image=image.id, term=None,
                                                               showMeta=None, bbox=None, bboxAnnotation=None, reviewed=False,
                                                               showTerm=True).fetch()

                      annotations_list = self.pool.map(ann,id_users)

                  annott,polyss,roiss,rect = extract_roi(annotations_list,predict_terms_list,image.width,image.height)

                  if self.verbose:
                      rl = RLock()
                      nb = len(rect)
                      self.done = 0
                  #function made on fly to fetch rectangles
                  def getRect(rectangle):
                      sp = None
                      im =  ImageGroupHDF5(id=imagegroupHDF5)
                      requests = splitRect(rectangle,max_fetch_size[0],max_fetch_size[1])
                      while len(requests):
                          (w,h,sizew,sizeh) = requests.pop()
                          try:
                            if sp is None:
                                sp = im.rectangle_all(w,h,sizew,sizeh)
                            else:
                                sp += im.rectangle_all(w,h,sizew,sizeh)
                          except socket.error :
                            print(socket.error)
                            time.sleep(5)
                            if sizew > 1 and sizeh > 1:
                              requests.extend(splitRect((w,h,sizew,sizeh),sizew/2,sizeh/2))
                            else:
                              print("error, cannot retreive data")
                            continue
                          except socket.timeout :
                            print(socket.timeout)
                            time.sleep(5)
                            if sizew > 1 and sizeh > 1:
                              requests.extend(splitRect((w,h,sizew,sizeh),sizew/2,sizeh/2))
                            else:
                              print("error, cannot retreive data")

                            continue

                      if self.verbose:
                          with rl:
                              self.done += 1
                              sys.stdout.write("\r{}/{}      ".format(self.done,nb))
                              sys.stdout.flush()
                      sp.sort(key=lambda data: data["pxl"])
                      return sp

                  spectras = self.pool.map(getRect,rect)
                  for j,s in enumerate(spectras):
                     if s is not None and len(s):
                         annot.append(annott[j])
                         polys.append(polyss[j])
                         rois.append(roiss[j])
                         spect.append(s)
                         nimage=len(s[0]['spectra'])
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

            roi = [np.zeros((rois[i][2],rois[i][3],nimage),dtype=np.uint8) for _ in range(len(annot[i].term))]
            roil = [np.zeros((rois[i][2],rois[i][3]),dtype=int) for _ in range(len(annot[i].term))]
            for pixel in spect[i]:
                #conversion to Cytomine down-left (0,0) coordonate
                lp = [pixel['pxl'][0],rois[i][4]-pixel['pxl'][1]]
                for t in range(len(annot[i].term)):
                    roi[t][int(abs(lp[0]-rois[i][0])-1),int(abs(lp[1]-rois[i][1])-1)] = pixel['spectra']
                p = Point(lp)
                if polys[i].contains(p):
                    if len(annot[i].term):
                        for t in range(len(annot[i].term)):
                            roil[t][int(abs(lp[0]-rois[i][0])-1),int(abs(lp[1]-rois[i][1])-1)] = annot[i].term[0]

                    for term in annot[i].term:
                      dataCoord.append(lp)
                      dataX.append(pixel['spectra'])
                      dataY.append(term)
                else:
                    unknown = True
                    for a,pol in enumerate(polys):
                        if pol.contains(p):
                            for t in range(len(annot[i].term)):
                                roil[t][int(abs(lp[0]-rois[i][0])-1),int(abs(lp[1]-rois[i][1])-1)] = annot[a].term[0]
                            unknown = False
                            break
                    if unknown:
                        unknownCoord.append(lp)
                        unknownX.append(pixel['spectra'])
            self.rois.extend([(roi[t],roil[t]) for t in range(len(annot[i].term))])

#        self.polys = polys
#        self.spect = spect
#        self.annot = annot
        self.numData = int(len(dataCoord))
        self.numUnknown = int(len(unknownCoord))

        self.data = {"data_coord":np.asarray(dataCoord),
                     "X":np.asarray(dataX),
                     "Y":np.asarray(dataY),
                     "unknown_coord":np.asarray(unknownCoord),
                     "unknown_X":np.asarray(unknownX)}
        self.numFeature = int(self.data["X"].shape[1]) if len(spect) else None

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

def extract_roi(annotations_list,predict_terms_list,image_width,image_height):
    annot = []
    polys = []
    rois = []
    rect = []
    for annotations in annotations_list:
        for a in annotations.data():
            a.term = list(set(a.term) & predict_terms_list)

            #if the annotation has no asked term, do not take it
            if not len(a.term):
                continue
            a = a.fetch()
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
            rois.append((round(maxx),round(maxy),sizew,sizeh,image_height))
    return annot,polys,rois,rect

def coordonatesToPolygons(coordonates,nb_job=1):
    """
    coordonates: a list of tuple (x,y) with x,y integers
    return a MultiPolygon that contains all points that are in a valid polygon
    (ie non-zeros area (or more than one pixel width))
    """
    from shapely.geometry import Polygon,MultiPolygon
    from shapely.ops import cascaded_union
    nb_job = max(1,nb_job)

    #converts points to polygon

    polys = [Polygon([(x-.5,y-.5),
                      (x-.5,y+.5),
                      (x+.5,y+.5),
                      (x+.5,y-.5)])for x,y in coordonates]
    if nb_job == 1:
        return cascaded_union(polys)
    else:
        from multiprocessing import Pool
        pool = Pool(nb_job)

        lim = len(coordonates)/nb_job
        polyss = [polys[int(i*lim):int((i+1)*lim)] for i in range(nb_job)]
        print("ok")
        results = pool.map(cascaded_union,polyss)
        print("almost done")
        multipolys = MultiPolygon()
        for mp in results:
            multipolys = multipolys.union(mp)
        return multipolys.buffer(-.5)


def splitRect(rect,maxw,maxh):

    rects = []

    (w,h,sizew,sizeh) = rect
    limitw = int(w + sizew)
    limith = int(h + sizeh)
    currw = w

    while currw < limitw:
        tmpw = min(maxw,abs(limitw-currw))
        currh = h
        while currh < limith:
            tmph = min(maxh,abs(limith-currh))
            rects.append((int(currw),int(currh),int(tmpw),int(tmph)))
            currh += tmph
        currw += tmpw
    return rects
