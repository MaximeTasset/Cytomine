# -*- coding: utf-8 -*-

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

__author__          = "Maxime Tasset <maxime.tasset@student.ulg.ac.be>"
__contributors__ = []
__copyright__       = "Copyright 2010-2018 University of Liège, Belgium, http://www.cytomine.be/"


import numpy as np
from ..models import Term,TermCollection,ImageInstance,AnnotationCollection
from ..models import ImageGroupHDF5,ImageSequenceCollection,ImageGroup,ImageGroupCollection


from multiprocessing import RLock
from multiprocessing.pool import ThreadPool
from shapely.geometry import Polygon,Point
from shapely.wkt import loads
import pickle
from heapq import nlargest
from sklearn.feature_selection import chi2,f_classif
from sklearn.ensemble import ExtraTreesClassifier as ETC

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

    def _populate(self):
        if hasattr(self,'data'):
            for name in self.data:
                setattr(self,name,self.data[name])

    def readFile(self,filename=None):

        if not filename and self.filename:
             filename = self.filename
        elif not filename:
            raise ValueError("No filename given")

        f = open(filename, "rb")
        try:
            self.data = pickle.load(f)
            self._populate()
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
        n_sample = len(self.X)
        ind = list(range(n_sample))
        np.random.shuffle(ind)
        ch,_ = chi2(self.X[ind[:int(usedata*n_sample)]],self.Y[ind[:int(usedata*n_sample)]])
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
        f,_ = f_classif(self.X[ind[:int(usedata*n_sample)]],self.Y[ind[:int(usedata*n_sample)]])
        if not N:
            N = len(f)

        if sort:
          return nlargest(N,[(f[i],i) for i in range(len(f))])
        else:
          return [(f[i],i) for i in range(len(f))]

    def features_ETC(self, sort=False, N=0,n_estimators=10,max_features='auto',min_samples_split=2,usedata=1):
        n_sample = self.numData
        ind = list(range(n_sample))
        np.random.shuffle(ind)
        if not isinstance(max_features, six.string_types) and max_features is not None:
            max_features = max(1,min(max_features,int(self.X.shape[1])))
        etc = ETC(n_estimators=n_estimators,max_features=max_features,
                  min_samples_split=min_samples_split,n_jobs=-1)
        etc.fit(self.X[ind[:int(usedata*n_sample)]],self.Y[ind[:int(usedata*n_sample)]])
        f = etc.feature_importances_
        if not N:
            N = len(f)

        if sort:
          return nlargest(N,[(f[i],i) for i in range(len(f))])
        else:
          return [(f[i],i) for i in range(len(f))]

    def saveFeatureSelectionInCSV(self,filename,n_estimators=10,max_features=None,min_samples_split=2,usedata=1):
      print("chi2")
      chi2 = self.chi2(usedata=usedata)
      print("f_classif")
      fclassif = self.f_classif(usedata=usedata)
      print("features_ETC")
      etc = self.features_ETC(n_estimators=n_estimators,max_features=max_features,min_samples_split=min_samples_split,usedata=usedata)

      filename = str(filename)
      print("Saving")
      if filename.endswith('.xlsx'):
        import xlsxwriter
        with xlsxwriter.Workbook(filename) as workbook:
          worksheet = workbook.add_worksheet()
          fields = ['layer','chi2', 'f_classif','ExtraTree']
          worksheet.write_row(0, 0, fields)
          for row, val in enumerate(range(len(chi2))):
              worksheet.write_row(row+1,0,[row,(chi2[row][0] if not (np.isnan(chi2[row][0]) or np.isinf(chi2[row][0])) else 0),
                                           (fclassif[row][0] if not (np.isnan(fclassif[row][0]) or np.isinf(fclassif[row][0])) else 0),
                                           (etc[row][0] if not (np.isnan(etc[row][0]) or np.isinf(etc[row][0])) else 0)])
          row += 2
          worksheet.write_row(row,0,["nb_annotation",self.numAnnotation])
          row += 1
          worksheet.write_row(row,0,["nb_pixel",len(self.data["X"])])
          row += 1
          fields = ['term_name','nb_annotation', 'nb_pixel']
          worksheet.write_row(row, 0, fields)
          row += 1
          for i in self.numAnnotationTerm:
              if i in self.mapIdTerm:
                  worksheet.write_row(row,0,[self.mapIdTerm[i],self.numAnnotationTerm[i], self.numPixelTerm[i]])
                  row += 1
      else:
        if not filename.endswith('.csv'):
          filename += ".csv"
        import csv
        with open(filename, 'w') as csvfile:
          fieldnames = ['layer','chi2', 'f_classif','ExtraTree']
          writer = csv.DictWriter(csvfile, fieldnames=fieldnames,dialect='excel')

          writer.writeheader()
          for i in range(len(chi2)):
              writer.writerow({'layer':i,'chi2':chi2[i][0], 'f_classif':fclassif[i][0],'ExtraTree':etc[i][0]})
          writer = csv.writer(csvfile,dialect='excel')
          writer.writerow(["nb_annotation",self.numAnnotation])
          writer.writerow(["nb_pixel",len(self.data["X"])])
          fieldnames = ['term_name','nb_annotation', 'nb_pixel']
          writer = csv.DictWriter(csvfile, fieldnames=fieldnames,dialect='excel')
          writer.writeheader()
          for i in self.numAnnotationTerm:
              if i in self.mapIdTerm:
                  writer.writerow({'term_name':self.mapIdTerm[i],'nb_annotation':self.numAnnotationTerm[i], 'nb_pixel':self.numPixelTerm[i]})


    def loadDataFromCytomine(self,imagegroup_id_list=None,id_project = 28146931,id_users=None,predict_terms_list=None,max_fetch_size=(10,10)):
        """
        " read the annotations of a list of imagegroup from a project
        """

        if predict_terms_list is None:
          terms = TermCollection(filters={'project':id_project}).fetch()
          predict_terms_list = {term.id for term in terms}
          nb_annotation_term = {term.id:0 for term in terms}
          map_id_name_terms = {term.id:term.name for term in terms}
        else:
          predict_terms_list = set(predict_terms_list)
          map_id_name_terms = {}
          nb_annotation_term = {term:0 for term in predict_terms_list}
          for term in predict_terms_list:
              try:
                map_id_name_terms[term] = Term(id=term).fetch().name
              except AttributeError:
                pass
        if imagegroup_id_list is None:
          imagegroup_id_list = ImageGroupCollection({"project":id_project}).fetch()
          imagegroup_id_list = [im.id for im in imagegroup_id_list]

        nb_annotation = 0


        polys = []
        spect = []
        annot = []
        rois = []

        nb_fetched_image = 1

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
            if self.verbose:
                nb_image = len(images)
            for im in images:

                if self.verbose:
                    sys.stdout.write("\r                 {}/{}      ".format(nb_fetched_image,nb_image))
                    sys.stdout.flush()
                nb_fetched_image += 1
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


                  annott,polyss,roiss,rect,dic,nb_ann = extract_roi(annotations_list,predict_terms_list,image.width,image.height)
                  for id in dic:
                    nb_annotation_term[id] += dic[id]
                  nb_annotation += nb_ann


                  if self.verbose:
                      rl = RLock()
                      nb = len(rect)
                      self.done = 0
                  #function made on fly to fetch rectangles
                  def getRect(rectangle):
                      sp = None
                      im =  ImageGroupHDF5(id=imagegroupHDF5)
                      requests = splitRect(rectangle,max_fetch_size[0],max_fetch_size[1])
                      if self.verbose:
                          with rl:
                              sys.stdout.write("\r{}/{}      ".format(self.done,nb))
                              sys.stdout.flush()
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



        self.data = {"data_coord":np.asarray(dataCoord),
                     "X":np.asarray(dataX),
                     "Y":np.asarray(dataY),
                     "unknown_coord":np.asarray(unknownCoord),
                     "unknown_X":np.asarray(unknownX)}
        self.data["rois"] = self.rois
        self.data["numData"] = int(len(dataCoord))
        self.data["numUnknown"] = int(len(unknownCoord))
        self.data["numFeature"] = int(self.data["X"].shape[1]) if len(spect) else None
        self.data["numAnnotation"] = nb_annotation
        self.data["numAnnotationTerm"] = nb_annotation_term
        self.data["mapIdTerm"] = map_id_name_terms
        self.data["numPixelTerm"] = {i:len(self.data["Y"][self.data["Y"] == i]) for i in nb_annotation_term}

        self._populate()


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
            for rect,coord in roi2data(roi,sliceSize,step,flatten):
                ic,jc = coord
                if labels[ic,jc]:
                    x.append(rect)
                    y.append(labels[ic,jc])

        return np.array(x),np.array(y)


    def getinfo(self):
        if hasattr(self, "numData") and hasattr(self, "numUnknown") and hasattr(self, "numFeature"):
            return {"numData":self.numData,"numUnknown":self.numUnknown,"numFeature":self.numFeature}

def roi2data(roi,sliceSize=(3,3),step=1,flatten=True):
        """
        roi a np.array((width,height,features))

        """
        x_coord = []
        for i in range(0,roi.shape[0]-sliceSize[0],step):
            for j in range(0,roi.shape[1]-sliceSize[1],step):
                ic = int(abs(2*i+sliceSize[0])/2)
                jc = int(abs(2*j+sliceSize[1])/2)
                if flatten:
                    x_coord.append((roi[i:i+sliceSize[0],j:j+sliceSize[1]].flatten(),(ic,jc)))
                else:
                    x_coord.append((roi[i:i+sliceSize[0],j:j+sliceSize[1]],(ic,jc)))

        return x_coord

def extract_roi(annotations_list,predict_terms_list,image_width,image_height):
    annot = []
    polys = []
    rois = []
    rect = []
    dic = {}
    nb_annotation = 0
    for annotations in annotations_list:
        for a in annotations.data():
            a.term = list(set(a.term) & predict_terms_list)

            #if the annotation has no asked term, do not take it
            if not len(a.term):
                continue
            nb_annotation += 1
            for term in a.term:
                dic[term] = dic.setdefault(term,0) + 1
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
    return annot,polys,rois,rect,dic,nb_annotation

def coordonatesToPolygons(coordonates,nb_job=1,pool=None,trim=True):
    """
    coordonates: a list of tuple (x,y) with x,y integers
    return a MultiPolygon that contains all points that are in a valid polygon (if trim == True)
    (ie non-zeros area (or more than one pixel width))
    """
    from shapely.geometry import Polygon
    from shapely.ops import cascaded_union

    #converts points to polygon

    polys = [Polygon([(x-.5,y-.5),
                      (x-.5,y+.5),
                      (x+.5,y+.5),
                      (x+.5,y-.5)])for x,y in coordonates]
    nb_job = max(1,nb_job)

    if nb_job == 1:
        multipolys = cascaded_union(polys)
    else:

        if pool is None:
            from multiprocessing import Pool
            pool = Pool(nb_job)
            toclose = True
        else:
            toclose = False

        lim = len(coordonates)/nb_job
        polyss = [polys[int(i*lim):int((i+1)*lim)] for i in range(nb_job)]

        results = pool.map(cascaded_union,polyss)
        if toclose:
          pool.close()
        multipolys = cascaded_union(results)
    if trim:
        return multipolys.buffer(-.5)
    else:
        return multipolys

def polygonToAnnotation(polygon):
    tests = [np.ceil,np.floor]
    t = polygon.buffer(0.01)
    ext = []
    for point in polygon.exterior.coords:
      ok = []
      for t1 in tests:
        for t2 in tests:
          x,y = int(t1(point[0])),int(t2(point[1]))
          if t.contains(Point((x,y))):
            ok.append((x,y))
      if len(ok):
        ext.append(min(ok,key=lambda x: np.sqrt((x[0]+point[0])**2+(x[1]+point[1])**2)))

    #actual annotation's polygon
    return Polygon(ext).buffer(0)

def split(array3D):
    """
    array3D array-like of shape (n,m,k)
    return a list of k elements where the ith element corresponds to the array3D[:,:,i]
    """
    return np.split(array3D,array3D.shape[2],2)

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
