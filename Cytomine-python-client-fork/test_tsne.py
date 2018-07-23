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
import matplotlib.pyplot as plt
plt.switch_backend("agg")
import numpy as np
import sys
import pickle

cytomine_host="demo.cytomine.be"
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
