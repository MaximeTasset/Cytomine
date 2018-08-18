# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2015. Authors: see NOTICE file.
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


from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.manifold import TSNE

from numpy.random import shuffle
from sklearn.decomposition import PCA
from cytomine.spectral.extractor import Extractor
from cytomine.models import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
plt.switch_backend("agg")
import numpy as np
import sys,PIL
import pickle

cytomine_host="research.cytomine.be"
cytomine_public_key="XXX"
cytomine_private_key="XXX"
id_project=0

n_jobs = 16
n_estimators = 1000
test = .2
validation = .1
train = test + validation

ext = Extractor("extractedData.save")
try:
    print("load data from file")
    ext.readFile()
except FileNotFoundError:
    print("File not found... Trying to fetch it from Cytomine")
    from cytomine import Cytomine
    import logging
    with Cytomine(cytomine_host,cytomine_public_key,cytomine_private_key,verbose=logging.WARNING):
      ext.loadDataFromCytomine(id_project=id_project)
      print("Saving data to file for later uses")
      ext.writeFile()

print("fit_transform pca")
pca = PCA().fit(ext.X)
print("transform")
X = pca.transform(ext.X)

indexes = list(range(X.shape[0]))
shuffle(indexes)

print("ExtraTreesClassifier with {} estimators".format(n_estimators))
etc = ETC(n_jobs=n_jobs,n_estimators=n_estimators)

def test_DimensionReduction():
  print("================================================")
  print("Test: explained variance ratio in PCA")
  print("================================================")
  count = 0
  a = 0.99
  val = len(pca.explained_variance_ratio_)
  for i,c in enumerate(pca.explained_variance_ratio_):
      count += c
      if count >= a:
        print("ratio explained >= {:.2f} with {} components".format(a,i+1))
        val = i + 1
        break

  tsne_score = {}
  etc.fit(ext.X[indexes[:int(test*len(indexes))]],ext.Y[indexes[:int(test*len(indexes))]])

  print("Score with PCA: {}".format(etc.score(ext.X[indexes[int(train*len(indexes)):]],ext.Y[indexes[int(train*len(indexes)):]])))

  for name,n in [("tsne2",2),("tsne3",3)]:
      print("fit_transform TSNE {}".format(n))
      sys.stdout.flush()
      tsne = TSNE(n,n_iter=1000).fit_transform(X[:,:val])
      with open("tsne_{}.pickle".format(n),'wb') as f:
          pickle.dump(tsne,f)
      test_SampleTSNE_X,test_SampleTSNE_Y = tsne[indexes[:int(test*len(indexes))]],ext.Y[indexes[:int(test*len(indexes))]]
      train_SampleTSNE_X,train_SampleTSNE_Y = tsne[indexes[int(train*len(indexes)):]],ext.Y[indexes[int(train*len(indexes)):]]
      etc.fit(train_SampleTSNE_X,train_SampleTSNE_Y)
      score = etc.score(test_SampleTSNE_X,test_SampleTSNE_Y)
      tsne_score[name] = score
      print("Score {}: {}".format(name,score))

  return tsne_score

if __name__ == '__main__':
    counts = test_DimensionReduction()

    with Cytomine(host=cytomine_host, public_key=cytomine_public_key, private_key=cytomine_private_key,
                  verbose=logging.WARNING) as cytomine:

        igh = ImageGroupCollection({"project":31054043}).fetch()[0].fetch().image_groupHDF5()
        im = igh.rectangle(0,0,260,134)
        pixels = []
        for i,j in np.ndindex(im.shape[:-1]):
            pixels.append(im[i,j])
        newP = TSNE(3).fit_transform(pixels)
        imm = np.zeros((260, 134,3))
        it = iter(newP)
        for i,j in np.ndindex(im.shape[:-1]):
            imm[i,j] = it.__next__()
        immr = imm - imm.min()
        immr = immr/immr.max()
        immr *= 255
        immr = immr.astype(np.uint8)
        immr = np.swapaxes(immr,0,1)
        PIL.Image.fromarray(immr,mode='RGB').save("MaldiDemo_tsne/MALDI-DEMO_tsne.png")
        PIL.Image.fromarray(np.swapaxes(im[:,:,[1,1,1]].astype(np.uint8),0,1),mode='RGB').save("MaldiDemo_tsne/MALDI-DEMO.png")

        filename = "MaldiDemo/MaldiDemoData.save"
        nfilename = "MaldiDemo_tsne/MaldiDemoData_tsne.save"
        ext = Extractor(filename,nb_job=n_jobs)
        try:
            print("load data from file {}".format(filename))
            ext.readFile()
        except FileNotFoundError:
            print("File not found... Trying to fetch it from Cytomine")
            from cytomine import Cytomine
            import logging
            ext.loadDataFromCytomine(id_project=id_project)
            print("Saving data to file for later uses")
            ext.writeFile()
        lab = np.zeros((260,134),dtype=int)
        newX = []
        for i,y in enumerate(ext.Y):
            lab[ext.data_coord[i][0],134-ext.data_coord[i][1]] = y
            newX.append(imm[ext.data_coord[i][0],134-ext.data_coord[i][1]])
        ext.data["rois"] = [(imm,lab)]
        ext.data["X"] = np.asarray(newX)
        ext.data["numFeature"] = 3
        print("save data to file {}".format(nfilename))
        Extractor.write(nfilename,ext.data)

