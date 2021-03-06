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
from shapely.geometry.geo import box

from shapely.wkt import loads
import pickle
from heapq import nlargest
from sklearn.feature_selection import chi2,f_classif
from sklearn.ensemble import ExtraTreesClassifier as ETC

import sys
import six
import socket,time
import psutil
import gzip

class Extractor:
    def __init__(self, filename=None,verbose=True,nb_job=1,dtype=np.int8):
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
        self.dtype=dtype

        self.data = None

    def _populate(self):
        if hasattr(self,'data'):
            for name in self.data:
                setattr(self,name,self.data[name])


    def read(filename=None):

        try:
            with gzip.open(filename,'rb') as f:
                data = pickle.load(f)
        except OSError:
            with open(filename, "rb") as f:
                data = pickle.load(f)
        return data
    classmethod(read)

    def readFile(self,filename=None):

        if not filename and self.filename:
             filename = self.filename
        elif not filename:
            raise ValueError("No filename given")
        self.data = Extractor.read(filename)

        self._populate()
        return self

    def write(filename,data,compressed=True,compresslevel=4):
        f = open(filename, "wb") if not compressed else gzip.open(filename, "wb",compresslevel=compresslevel)
        try:
            pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
        finally:
            f.close()
    classmethod(write)

    def writeFile(self,filename=None,compressed=True,compresslevel=4):
        if not filename and self.filename:
             filename = self.filename
        elif not filename:
            raise ValueError("No filename")

        Extractor.write(filename,self.data,compressed,compresslevel)

    def chi2(self, sort=False, N=0,usedata=1,data=None):
        if data is None:
            n_sample = self.numData
            data = self.X,self.Y
        else:
            n_sample = len(data[1])

        ind = list(range(n_sample))
        np.random.shuffle(ind)
        ch,_ = chi2(data[0][ind[:int(usedata*n_sample)]],data[1][ind[:int(usedata*n_sample)]])
        if not N:
            N = len(ch)

        if sort:
          return nlargest(N,[(ch[i],i) for i in range(len(ch))])
        else:
          return [(ch[i],i) for i in range(len(ch))]

    def f_classif(self, sort=False, N=0,usedata=1,data=None):
        if data is None:
            n_sample = self.numData
            data = self.X,self.Y
        else:
            n_sample = len(data[1])

        ind = list(range(n_sample))
        np.random.shuffle(ind)
        f,_ = f_classif(data[0][ind[:int(usedata*n_sample)]],data[1][ind[:int(usedata*n_sample)]])
        if not N:
            N = len(f)

        if sort:
          return nlargest(N,[(f[i],i) for i in range(len(f))])
        else:
          return [(f[i],i) for i in range(len(f))]

    def features_ETC(self, sort=False, N=0,n_estimators=10,max_features='auto',min_samples_split=2,usedata=1,data=None):
        if data is None:
            n_sample = self.numData
            data = self.X,self.Y
        else:
            n_sample = len(data[1])

        ind = list(range(n_sample))
        np.random.shuffle(ind)
        if not isinstance(max_features, six.string_types) and max_features is not None:
            max_features = max(1,min(max_features,int(self.X.shape[1])))
        etc = ETC(n_estimators=n_estimators,max_features=max_features,
                  min_samples_split=min_samples_split,n_jobs=-1)
        etc.fit(data[0][ind[:int(usedata*n_sample)]],data[1][ind[:int(usedata*n_sample)]])
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
          worksheet.write_row(row,0,["nb_pixel",len(self.X)])
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
          writer.writerow(["nb_pixel",len(self.X)])
          fieldnames = ['term_name','nb_annotation', 'nb_pixel']
          writer = csv.DictWriter(csvfile, fieldnames=fieldnames,dialect='excel')
          writer.writeheader()
          for i in self.numAnnotationTerm:
              if i in self.mapIdTerm:
                  writer.writerow({'term_name':self.mapIdTerm[i],'nb_annotation':self.numAnnotationTerm[i], 'nb_pixel':self.numPixelTerm[i]})


    def loadDataFromCytomine(self,imagegroup_id_list=None,id_project = None,id_users=None,predict_terms_list=None,max_fetch_size=(10,10),pixel_border=0,trim=False):
        """
        " read the annotations of a list of imagegroup from a project
        """

        if predict_terms_list is None and id_project is not None:
            terms = TermCollection(filters={'project':id_project}).fetch()
            predict_terms_list = {term.id for term in terms}
            nb_annotation_term = {term.id:0 for term in terms}
            map_id_name_terms = {term.id:term.name for term in terms}
        elif predict_terms_list is not None:
            predict_terms_list = set(predict_terms_list)
            map_id_name_terms = {}
            nb_annotation_term = {term:0 for term in predict_terms_list}
            for term in predict_terms_list:
                try:
                  map_id_name_terms[term] = Term(id=term).fetch().name
                except AttributeError:
                  pass
        else:
            raise ValueError("Not enough information to continue")

        if imagegroup_id_list is None:
            if id_project is not None:
                imagegroup_id_list = ImageGroupCollection({"project":id_project}).fetch()
                imagegroup_id_list = [im.id for im in imagegroup_id_list]
            else:
                raise ValueError("Not enough information to continue")

        nb_annotation = 0
        pixel_border = min(0,pixel_border)

        polys = []
        spect = []
        annot = []
        rois = []

        pool = ThreadPool(self.nb_job)
        try:
            nb_fetched_image = 1

            for imagegroup_id in imagegroup_id_list:
                #Get project imagegroupHDF5 and images from imagegroup_id
                imagegroupHDF5 = ImageGroup(id=imagegroup_id).image_groupHDF5()
                if not imagegroupHDF5:
                    continue
                else:
                  imagegroupHDF5 = imagegroupHDF5.id
                  id_project = ImageGroup(id=imagegroup_id).fetch().project

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
                            global image
                            return AnnotationCollection(project=id_project, user=id_user, image=image.id, term=None,
                                                                   showMeta=None, bbox=None, bboxAnnotation=None, reviewed=False,
                                                                   showTerm=True).fetch()

                          annotations_list = pool.map(ann,id_users)

                      width = image.width
                      height = image.height
                      annott,polyss,roiss,rect,dic,nb_ann = extract_roi(annotations_list,predict_terms_list,image.width,image.height,pixel_border)

                      for id in dic:
                          nb_annotation_term[id] += dic[id]
                      nb_annotation += nb_ann

                      if self.verbose:
                          rl = RLock()
                          nb = len(rect)
                          self.done = 0
                      #function made on fly to fetch rectangles
                      def getRect(rect_poly):
                          rectangle,poly = rect_poly
                          sp = None
                          im =  ImageGroupHDF5(id=imagegroupHDF5)
                          if not trim:
                              requests = splitRect(rectangle,max_fetch_size[0],max_fetch_size[1])
                          else:
                              rec = (rectangle[0],height-(rectangle[1]+rectangle[3]),rectangle[2],rectangle[3])
                              requests = removeUnWantedRect(rec,poly.buffer(pixel_border),max_fetch_size)
                              requests = [(r[0],height-(r[1]+r[3]),r[2],r[3]) for r in requests]


                          if self.verbose:
                              with rl:
                                  sys.stdout.write("\r{}/{}      ".format(self.done,nb))
                                  sys.stdout.flush()
                          while len(requests):
                              (w,h,sizew,sizeh) = requests.pop()
                              try:
                                if sp is None:
                                    sp = [{"pxl":p["pxl"],"spectra":np.array(p["spectra"],dtype=self.dtype)} for p in im.rectangle_all(w,h,sizew,sizeh)]
                                else:
                                    sp += [{"pxl":p["pxl"],"spectra":np.array(p["spectra"],dtype=self.dtype)} for p in im.rectangle_all(w,h,sizew,sizeh)]
                              except socket.error :
                                print(socket.error)
                                time.sleep(5)
                                if sizew > 1 and sizeh > 1:
                                  requests.extend(splitRect((w,h,sizew,sizeh),sizew/2,sizeh/2))
                                else:
                                  print("error, cannot retrieve data")
                                continue
                              except socket.timeout :
                                print(socket.timeout)
                                time.sleep(5)
                                if sizew > 1 and sizeh > 1:
                                  requests.extend(splitRect((w,h,sizew,sizeh),sizew/2,sizeh/2))
                                else:
                                  print("error, cannot retrieve data")

                                continue

                          if self.verbose:
                              with rl:
                                  self.done += 1
                                  sys.stdout.write("\r{}/{}      ".format(self.done,nb))
                                  sys.stdout.flush()
                          dsp = {tuple(data['pxl']):data for data in sp}
                          sti = rectangle[0]
                          stj = rectangle[1]
                          si = rectangle[2]
                          sj = rectangle[3]
                          sp = [dsp[sti+i,stj+j] if (sti+i,stj+j) in dsp else {'pxl':(sti+i,stj+j),'spectra':np.zeros(nb_image,dtype=self.dtype),"fetched":False} for i,j in np.ndindex((si,sj))]

                          return sp

                      spectras = []
                      requests = list(zip(rect,polyss))
                      while self.nb_job < len(requests):
                          tmp_req = []
                          for _ in range(self.nb_job):
                              tmp_req.append(requests.pop(0))
                          spectras.extend(pool.map(getRect,tmp_req))

                      spectras.extend(pool.map(getRect,requests))

                      for j,s in enumerate(spectras):
                         if s is not None and len(s):
                             annot.append(annott[j])
                             polys.append(polyss[j])
                             rois.append(roiss[j])
                             spect.append(s)
                             nimage=len(s[0]['spectra'])
        finally:
            pool.close()
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

            spectrum = [pixel for pixel in spect[i]]
            spectrum.sort(key=lambda spectra: tuple(spectra['pxl']))
            image = np.array([spectra['spectra'] for spectra in spectrum],dtype=self.dtype)
            image_coord = np.array([spectra['pxl'] for spectra in spectrum])
            if trim:
                lookat = np.array([0 if 'fetched' in spectra else 1 for spectra in spectrum])
            del spectrum

            image = np.expand_dims(image,axis=1)
            image = image.reshape((rois[i][2],rois[i][3], nimage))
            if len(annot[i].term) == 1:
                roi = [image]
            else:
                roi = [image.copy() for _ in range(len(annot[i].term))]
            del image
            roil = [np.zeros((rois[i][2],rois[i][3]),dtype=int) for _ in range(len(annot[i].term))]

            image_coord = np.expand_dims(image_coord,axis=1)
            image_coord = image_coord.reshape((rois[i][2],rois[i][3], 2))
            if trim:
                lookat = np.expand_dims(lookat,axis=1)
                lookat = lookat.reshape((rois[i][2],rois[i][3]))


            for position in np.ndindex(image_coord.shape[:2]):
                if trim and not lookat[position]:
                    continue
                lp = [image_coord[position][0],rois[i][4]-image_coord[position][1]]
                p = Point(lp)
                if polys[i].contains(p):
                    for t in range(len(annot[i].term)):
                        roil[t][position] = annot[i].term[t]
                    for term in annot[i].term:
                        dataCoord.append(lp)
                        dataX.append(roi[0][position])
                        dataY.append(term)
                else:
                    unknown = True
                    for a,pol in enumerate(polys):
                        if pol.contains(p):
                            for t in range(len(annot[i].term)):
                                roil[t][position] = annot[a].term[0]
                            unknown = False
                            break
                    if unknown:
                        unknownCoord.append(lp)
                        unknownX.append(roi[0][position])

            self.rois.extend([(roi[t],roil[t]) for t in range(len(annot[i].term))])



        self.data = {"data_coord":np.asarray(dataCoord),
                     "X":np.asarray(dataX),
                     "Y":np.asarray(dataY),
                     "unknown_coord":np.asarray(unknownCoord),
                     "unknown_X":np.asarray(unknownX)}

        self.data["width"] = width
        self.data["height"] = height
        self.data["rois"] = self.rois
        self.data["numData"] = int(len(dataCoord))
        self.data["numUnknown"] = int(len(unknownCoord))
        self.data["numFeature"] = int(self.data["X"].shape[1]) if len(spect) else None
        self.data["numAnnotation"] = nb_annotation
        self.data["numAnnotationTerm"] = nb_annotation_term
        self.data["mapIdTerm"] = map_id_name_terms
        self.data["numPixelTerm"] = {i:len(self.data["Y"][self.data["Y"] == i]) for i in nb_annotation_term}

        self._populate()
        return self


    def rois2data(self,rois=None,sliceSize=(3,3),step=1,notALabelFlag=0,flatten=True,bands=None,dtype=None):
        """
        rois a list of tuple (np.array((width,height,features)),np.array((width,height)))
        """
        if dtype is None:
          dtype = self.dtype

        if rois is None:
            if hasattr(self,"rois"):
                rois = self.rois
                dtype = rois[0][0].dtype
                notALabelFlag = 0
            else:
                return
        x = []
        y = []
        for roi,labels in rois:
            for rect,coord in roi2data(roi,sliceSize,step,flatten,bands=bands,dtype=dtype):
                ic,jc = coord
                if labels[ic,jc] != notALabelFlag:
                    x.append(rect)
                    y.append(labels[ic,jc])

        return np.array(x),np.array(y)


    def getinfo(self):
        if hasattr(self, "numData") and hasattr(self, "numUnknown") and hasattr(self, "numFeature"):
            return {"numData":self.numData,"numUnknown":self.numUnknown,"numFeature":self.numFeature}

def roi2data(roi,sliceSize=(3,3),step=1,flatten=True,splitted=False,bands=None,dtype=None):
        """
        roi a np.array((width,height,features))

        """
        if not dtype is None and not roi.dtype is np.dtype(dtype):
          roi = roi.astype(dtype)
        if not splitted:
          x_coord = []
        else:
          x = []
          coord = []
        if bands is None:
            bands = list(range(roi.shape[2]))

        for i in range(0,roi.shape[0]-sliceSize[0],step):
            for j in range(0,roi.shape[1]-sliceSize[1],step):
                ic = int(abs(2*i+sliceSize[0])/2)
                jc = int(abs(2*j+sliceSize[1])/2)
                if not splitted:
                    if flatten:
                        x_coord.append((roi[i:i+sliceSize[0],j:j+sliceSize[1],bands].flatten(),(ic,jc)))
                    else:
                        x_coord.append((roi[i:i+sliceSize[0],j:j+sliceSize[1],bands],(ic,jc)))
                else:
                    if flatten:
                        x.append(roi[i:i+sliceSize[0],j:j+sliceSize[1],bands].flatten())
                        coord.append((ic,jc))
                    else:
                        x.append(roi[i:i+sliceSize[0],j:j+sliceSize[1],bands])
                        coord.append((ic,jc))

        if splitted:
          return x,coord
        else:
          return x_coord

def extract_roi(annotations_list,predict_terms_list,image_width,image_height,pixel_border):
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

            minx = max(minx - pixel_border,0)
            miny = max(miny - pixel_border,0)
            maxx = min(maxx + pixel_border,image_width)
            maxy = min(maxy + pixel_border,image_height)

            sizew = int(abs(maxx - minx))
            sizeh = int(abs(maxy - miny))

            #conversion to API up-left (0,0) coordonate
            w = max(min(int(round(minx)),image_width - 1),0)
            h = max(min(int(round(image_height - maxy)),image_height - 1),0)

            rect.append((round(w),round(h),sizew,sizeh))
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


#def removeUnWantedRect(rectangle,polygon,max_size=(10,10)):
#    """
#    " rectangle a tuple (minx,miny,sizex,sizey)
#    " polygon a shepley.geometry.Polygon
#    " max_size a tuple (max_size_x,max_size_y) the maximum size of the returned rectangles
#    "
#    " Return a list of tuple (minx,miny,sizex,sizey) which corresponds to rectangles
#    " include in 'rectangle' and that intersect 'polygon'. sizex <= ,sizey <=
#    "
#    """
#    if (np.array(rectangle[2:]) - np.array(max_size) <= 0).all():
#        return [rectangle]
#    else:
#
#        newwidth =  (int(np.floor(rectangle[2]/2.0)),int(np.ceil(rectangle[2]/2.0))) if rectangle[2] > max_size[0] else (rectangle[2],rectangle[2])
#        newheight = (int(np.floor(rectangle[3]/2.0)),int(np.ceil(rectangle[3]/2.0))) if rectangle[3] > max_size[1] else (rectangle[3],rectangle[3])
#
#        coord = [(rectangle[0],rectangle[1],newwidth[0],newheight[0])]
#        coord.append((rectangle[0],rectangle[1]+newheight[0],newwidth[0],newheight[1]))
#        coord.append((rectangle[0]+newwidth[0],rectangle[1],newwidth[1],newheight[0]))
#        coord.append((rectangle[0]+newwidth[0],rectangle[1]+newheight[0],newwidth[1],newheight[1]))
#
#        new_rectangles = []
#        for rect in coord:
#            minx,miny,maxx,maxy = rect[0],rect[1],rect[0]+rect[2],rect[1]+rect[3]
#            pol = box(minx,miny,maxx,maxy)
#            #only add rectangle that intersect 'polygon'
#            if pol.intersects(polygon):
#                new_rectangles.extend(removeUnWantedRect(rect,polygon,max_size))
#        return new_rectangles

def removeUnWantedRect(rectangle,polygon,max_size=(10,10)):
    """
    " rectangle a tuple (minx,miny,sizex,sizey)
    " polygon a shepley.geometry.Polygon
    " max_size a tuple (max_size_x,max_size_y) the maximum size of the returned rectangles
    "
    " Return a list of tuple (minx,miny,sizex,sizey) which corresponds to rectangles
    " include in 'rectangle' and that intersect 'polygon'. sizex <= ,sizey <=
    "
    """
    rects = splitRect(rectangle,max_size[0],max_size[1])
    return [rect for rect in rects if box(rect[0],rect[1],rect[0]+rect[2],rect[1]+rect[3]).intersects(polygon)]



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
