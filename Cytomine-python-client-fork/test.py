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
import numpy as np

test = .2
validation = .1
train = test + validation

print("load data from file")
ext = Extractor("extractedData.save")
ext.readFile()
print("fit_transform pca")
pca = PCA().fit(ext.X)
print("transform")
X = pca.transform(ext.X)

indexes = list(range(X.shape[0]))
shuffle(indexes)

etc = ETC(n_jobs=3,n_estimators=100)

def test_comparaisonFeatureImportance():

    print("getting feature importances from ETC")
    imp_val = ext.features_ETC(n_estimators=100)
    imp = [i for imp,i in sorted(imp_val)]
    imp.reverse()
    print("getting feature importances from f_classif")
    f_c_val = ext.f_classif()
    f_c = [i for imp,i in sorted(f_c_val)]
    f_c.reverse()
    print("getting feature importances from chi2")
    chi2_val = ext.chi2()
    chi2 = [i for imp,i in sorted(chi2_val)]
    chi2.reverse()

    print("getting sum of ETC, chi2 and f_classif")
    imp_val = np.asarray([val for val,i in imp_val])
    imp_val[np.isnan(imp_val)] = np.min([i for i in imp_val if not np.isnan(i)])
    tmax = np.max(imp_val)
    imp_val /= tmax if tmax else 1

    f_c_val = np.asarray([val for val,i in f_c_val])
    f_c_val[np.isnan(f_c_val)] = np.min([i for i in f_c_val if not np.isnan(i)])
    tmax = np.max(f_c_val)
    f_c_val /= tmax if tmax else 1

    chi2_val = np.asarray([val for val,i in chi2_val])
    chi2_val[np.isnan(chi2_val)] = np.min([i for i in chi2_val if not np.isnan(i)])
    tmax = np.max(chi2_val)
    chi2_val /= tmax if tmax else 1

    sum_val = chi2_val + f_c_val + imp_val
    sumv = [i for val,i in sorted([(val,i) for i,val in enumerate(sum_val)])]
    del imp_val,f_c_val,chi2_val

    #Tes, train and validation sets
    test_SamplePCA_X,test_SamplePCA_Y = X[indexes[:int(test*len(indexes))]],ext.Y[indexes[:int(test*len(indexes))]]
    test_SampleX,test_SampleY = ext.X[indexes[:int(test*len(indexes))]],ext.Y[indexes[:int(test*len(indexes))]]

    train_SamplePCA_X,train_SamplePCA_Y = X[indexes[int(train*len(indexes)):]],ext.Y[indexes[int(train*len(indexes)):]]
    train_SampleX,train_SampleY = ext.X[indexes[int(train*len(indexes)):]],ext.Y[indexes[int(train*len(indexes)):]]

    val_SamplePCA_X,val_SamplePCA_Y = X[indexes[int(test*len(indexes)):int(train*len(indexes))]],ext.Y[indexes[int(test*len(indexes)):int(train*len(indexes))]]
    val_SampleX,val_SampleY = ext.X[indexes[int(test*len(indexes)):int(train*len(indexes))]],ext.Y[indexes[int(test*len(indexes)):int(train*len(indexes))]]

    print("train set size: {}".format(len(train_SampleY)))
    print("test set size: {}".format(len(test_SampleY)))
    print("validation set size: {}".format(len(val_SampleY)))

    pca_score = []
    imp_score = []
    f_c_score = []
    chi2_score = []
    sum_score = []
    n_feature = []

    best = {}

    for i in sorted(list(set(list(range(0,1650,100))+list(range(1600,1650,10))))):
        print("\nScores with best {} features/components:\n".format(1650-i))
        n_feature.append(1650-i)

        print("Scores before PCA:")
        etc.fit(train_SampleX[:,imp[:int(1650-i)]],train_SampleY)
        score = etc.score(test_SampleX[:,imp[:int(1650-i)]],test_SampleY)
        if best.get("imp",(0,0))[1] < score:
          best["imp"] = (int(1650-i),score,imp)
        imp_score.append(score)
        print("\t -imp :\t{}".format(score))
        etc.fit(train_SampleX[:,f_c[:int(1650-i)]],train_SampleY)
        score = etc.score(test_SampleX[:,f_c[:int(1650-i)]],test_SampleY)
        if best.get("f_c",(0,0))[1] < score:
          best["f_c"] = (int(1650-i),score,f_c)
        f_c_score.append(score)
        print("\t -f_classif :\t{}".format(score))
        etc.fit(train_SampleX[:,chi2[:int(1650-i)]],train_SampleY)
        score = etc.score(test_SampleX[:,chi2[:int(1650-i)]],test_SampleY)
        if best.get("chi2",(0,0))[1] < score:
          best["chi2"] = (int(1650-i),score,chi2)
        chi2_score.append(score)
        print("\t -chi2 :\t{}".format(score))
        etc.fit(train_SampleX[:,sumv[:int(1650-i)]],train_SampleY)
        score = etc.score(test_SampleX[:,sumv[:int(1650-i)]],test_SampleY)
        if best.get("sum",(0,0))[1] < score:
          best["sum"] = (int(1650-i),score,sumv)
        sum_score.append(score)
        print("\t -Sum :\t{}".format(score))

        etc.fit(train_SamplePCA_X[:,:int(1650-i)],train_SamplePCA_Y)
        score = etc.score(test_SamplePCA_X[:,:int(1650-i)],test_SamplePCA_Y)
        if best.get("pca",(0,0))[1] < score:
          best["pca"] = (int(1650-i),score,None)
        pca_score.append(score)
        print("Score after PCA: {}".format(score))

    print("Bests tested on validation set:")
    for f in best:
        score_test,i,f_imp = best[f]
        if f == "pca":
            etc.fit(train_SamplePCA_X[:,:int(1650-i)],train_SamplePCA_Y)
            score = etc.score(val_SamplePCA_X[:,:int(1650-i)],val_SamplePCA_Y)
            print("Score PCA (with {} cpts):\t{} (test:\t{})".format(int(1650-i),score,score_test))
        else:
            etc.fit(train_SampleX[:,f_imp[:int(1650-i)]],train_SampleY)
            score = etc.score(val_SampleX[:,f_imp[:int(1650-i)]],val_SampleY)
            print("Score {} (with {} features):\t{} (test:\t{})".format(f,int(1650-i),score,score_test))

    plt.plot(n_feature,pca_score,label="pca")
    plt.plot(n_feature,chi2_score,label="chi2")
    plt.plot(n_feature,imp_score,label="imp")
    plt.plot(n_feature,f_c_score,label="f_classif")
    plt.plot(n_feature,sum_score,label="sum")
    plt.legend()
    plt.show()

    return pca_score, imp_score, f_c_score, chi2_score, sum_score, n_feature

def test_DimensionReduction():
  count = 0
  for i,c in enumerate(pca.explained_variance_ratio_):
      count += c
      if count >= 0.9:
        break

  tsne_score = {}
  for name,n in [("tsne2",2),("tsne3",3)]:
      print("fit_transform TSNE {}".format(n))
      tsne = TSNE(n,n_iter=1000).fit_transform(X[:,:i+1])
      test_SampleTSNE_X,test_SampleTSNE_Y = tsne[indexes[:int(test*len(indexes))]],ext.Y[indexes[:int(test*len(indexes))]]
      train_SampleTSNE_X,train_SampleTSNE_Y = tsne[indexes[int(train*len(indexes)):]],ext.Y[indexes[int(train*len(indexes)):]]
      etc.fit(train_SampleTSNE_X,train_SampleTSNE_Y)
      score = etc.score(test_SampleTSNE_X,test_SampleTSNE_Y)
      tsne_score[name] = score
      print("Score {}: {}".format(name,score))

  return tsne_score

def test_Spaciality():
  best = (0,0,0)
  for i in range(1,6):
    X,y = ext.rois2data(None,(i,i))
    indexes = list(range(X.shape[0]))
    shuffle(indexes)
    train_SampleX,train_SampleY = X[indexes[int(train*len(indexes)):]],y[indexes[int(train*len(indexes)):]]
    test_SampleX,test_SampleY = X[indexes[:int(test*len(indexes))]],y[indexes[:int(test*len(indexes))]]
    val_SampleX,val_SampleY = X[indexes[int(test*len(indexes)):int(train*len(indexes))]],y[indexes[int(test*len(indexes)):int(train*len(indexes))]]

    print("train set size: {}".format(len(train_SampleY)))
    print("test set size: {}".format(len(test_SampleY)))
    print("validation set size: {}".format(len(val_SampleY)))

    etc.fit(train_SampleX,train_SampleY)
    score = etc.score(test_SampleX,test_SampleY)
    print("Score with a slice size of {}:\t{}".format(i,score))

    if score >= best[1]:
        best = (i,score,etc.score(val_SampleX,val_SampleY))

  print("Best score with a slice size of {} (test set {}):\t{} on the validation set".format(best[0],best[1],best[2]))

if __name__ == '__main__':
  test_comparaisonFeatureImportance()
  test_DimensionReduction()
  test_Spaciality()