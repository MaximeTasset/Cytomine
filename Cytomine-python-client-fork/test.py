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
#from sklearn.manifold import TSNE

from numpy.random import shuffle
from sklearn.decomposition import PCA
from cytomine.spectral.extractor import Extractor
import matplotlib.pyplot as plt
plt.switch_backend("agg")
import numpy as np
import os
#import sys
#import pickle

cytomine_host="demo.cytomine.be"
cytomine_public_key="XXX"
cytomine_private_key="XXX"
id_project=0

n_jobs = 24
n_estimators = 100
test = .2
validation = .1
train = test + validation

filename = "extractedData.save"
#filename = "MaldiDemoData.save"
#filename = "FlutisteData.save"
save_path = "./colors"
#save_path = "./MaldiDemo"
#save_path = "./Flutiste"

os.makedirs(save_path,exist_ok=True)
ext = Extractor(filename)
try:
    print("load data from file {}".format(filename))
    ext.readFile()
except FileNotFoundError:
    print("File not found... Trying to fetch it from Cytomine")
    from cytomine import Cytomine
    import logging
    with Cytomine(cytomine_host,cytomine_public_key,cytomine_private_key,verbose=logging.WARNING):
      ext.loadDataFromCytomine(id_project=id_project)
      print("Saving data to file for later uses")
      ext.writeFile()
nb_feature = ext.numFeature
print("fit_transform pca")
pca = PCA().fit(ext.X)
print("transform")
X = pca.transform(ext.X)

print("fit_transform pca with whiten")
pca_w = PCA(whiten=True).fit(ext.X)
print("transform")
X_w = pca_w.transform(ext.X)

indexes = list(range(X.shape[0]))
shuffle(indexes)

etc = None


def test_comparaisonFeatureImportance():
    print("================================================")
    print("Test: Comparaison Of Feature Importance Measure")
    print("================================================")
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

    #Test, train and validation sets
    test_SamplePCA_X = X[indexes[:int(test*len(indexes))]]
    test_SamplePCA_w_X = X_w[indexes[:int(test*len(indexes))]]
    test_SampleX = ext.X[indexes[:int(test*len(indexes))]]

    test_SampleY = ext.Y[indexes[:int(test*len(indexes))]]

    train_SamplePCA_X = X[indexes[int(train*len(indexes)):]]
    train_SamplePCA_w_X = X_w[indexes[int(train*len(indexes)):]]
    train_SampleX = ext.X[indexes[int(train*len(indexes)):]]

    train_SampleY = ext.Y[indexes[int(train*len(indexes)):]]

    val_SamplePCA_X = X[indexes[int(test*len(indexes)):int(train*len(indexes))]]
    val_SamplePCA_w_X = X_w[indexes[int(test*len(indexes)):int(train*len(indexes))]]
    val_SampleX = ext.X[indexes[int(test*len(indexes)):int(train*len(indexes))]]

    val_SampleY = ext.Y[indexes[int(test*len(indexes)):int(train*len(indexes))]]

    print("train set size: {}".format(len(train_SampleY)))
    print("test set size: {}".format(len(test_SampleY)))
    print("validation set size: {}".format(len(val_SampleY)))

    pca_score = []
    pca_w_score = []
    imp_score = []
    f_c_score = []
    chi2_score = []
    sum_score = []
    n_feature = []

    best = {}

    for i in sorted(list(set(list(range(0,int(nb_feature-50),100))+list(range(int(nb_feature-50),nb_feature,1))))):
        print("\nScores with best {} features/components:\n".format(nb_feature-i))
        n_feature.append(nb_feature-i)

        print("Scores before PCA:")
        etc.fit(train_SampleX[:,imp[:int(nb_feature-i)]],train_SampleY)
        score = etc.score(test_SampleX[:,imp[:int(nb_feature-i)]],test_SampleY)
        if best.get("imp",(0,0))[1] < score:
          best["imp"] = (int(nb_feature-i),score,imp)
        imp_score.append(score)
        print("\t -imp :\t{}".format(score))
        etc.fit(train_SampleX[:,f_c[:int(nb_feature-i)]],train_SampleY)
        score = etc.score(test_SampleX[:,f_c[:int(nb_feature-i)]],test_SampleY)
        if best.get("f_c",(0,0))[1] < score:
          best["f_c"] = (int(nb_feature-i),score,f_c)
        f_c_score.append(score)
        print("\t -f_classif :\t{}".format(score))
        etc.fit(train_SampleX[:,chi2[:int(nb_feature-i)]],train_SampleY)
        score = etc.score(test_SampleX[:,chi2[:int(nb_feature-i)]],test_SampleY)
        if best.get("chi2",(0,0))[1] < score:
          best["chi2"] = (int(nb_feature-i),score,chi2)
        chi2_score.append(score)
        print("\t -chi2 :\t{}".format(score))
        etc.fit(train_SampleX[:,sumv[:int(nb_feature-i)]],train_SampleY)
        score = etc.score(test_SampleX[:,sumv[:int(nb_feature-i)]],test_SampleY)
        if best.get("sum",(0,0))[1] < score:
          best["sum"] = (int(nb_feature-i),score,sumv)
        sum_score.append(score)
        print("\t -Sum :\t{}".format(score))

        etc.fit(train_SamplePCA_X[:,:int(nb_feature-i)],train_SampleY)
        score = etc.score(test_SamplePCA_X[:,:int(nb_feature-i)],test_SampleY)
        if best.get("pca",(0,0))[1] < score:
          best["pca"] = (int(nb_feature-i),score,None)
        pca_score.append(score)
        print("Score after PCA: {}".format(score))

        etc.fit(train_SamplePCA_w_X[:,:int(nb_feature-i)],train_SampleY)
        score = etc.score(test_SamplePCA_w_X[:,:int(nb_feature-i)],test_SampleY)
        if best.get("pca_w",(0,0))[1] < score:
          best["pca_w"] = (int(nb_feature-i),score,None)
        pca_w_score.append(score)
        print("Score after PCA with whiten: {}".format(score))

    print("Bests tested on validation set:")
    for f in best:
        n_feature_k,score_test,f_imp = best[f]
        if f == "pca":
            etc.fit(train_SamplePCA_X[:,:int(n_feature_k)],train_SampleY)
            score = etc.score(val_SamplePCA_X[:,:int(n_feature_k)],val_SampleY)
            print("Score PCA (with {} cpts):\t{} (test:\t{})".format(n_feature_k,score,score_test))
        elif f == "pca_w":
            etc.fit(train_SamplePCA_w_X[:,:int(n_feature_k)],train_SampleY)
            score = etc.score(val_SamplePCA_w_X[:,:int(n_feature_k)],val_SampleY)
            print("Score PCA with Whiten (with {} cpts):\t{} (test:\t{})".format(n_feature_k,score,score_test))
        else:
            etc.fit(train_SampleX[:,f_imp[:int(n_feature_k)]],train_SampleY)
            score = etc.score(val_SampleX[:,f_imp[:int(n_feature_k)]],val_SampleY)
            print("Score {} (with {} features):\t{} (test:\t{})".format(f,n_feature_k,score,score_test))

    plt.plot(n_feature,pca_score,label="pca")
    plt.plot(n_feature,pca_w_score,label="pca whithen")

    plt.plot(n_feature,chi2_score,label="chi2")
    plt.plot(n_feature,imp_score,label="imp")
    plt.plot(n_feature,f_c_score,label="f_classif")
    plt.plot(n_feature,sum_score,label="sum")
    plt.ylabel("Score")
    plt.xlabel("Number Of Feature")
    plt.legend()
    plt.savefig(os.path.join(save_path,"comparaison_feature_imp_{}.png").format(n_estimators))
    plt.close()

    return pca_score,pca_w_score, imp_score, f_c_score, chi2_score, sum_score, n_feature

def test_DimensionReduction():
  print("================================================")
  print("Test: explained variance ratio in PCA")
  print("================================================")
  count = 0
  counts = [0]
  n_component = [0]
  a = 0.9
  for i,c in enumerate(pca.explained_variance_ratio_):
      count += c
      if count >= a:
        print("ratio explained >= {:.2f} with {} components".format(a,i+1))
        a += 0.01
      counts.append(count)
      n_component.append(i+1)
  plt.plot(n_component,counts)
  plt.ylabel("Explained Variance Ratio")
  plt.xlabel("Number Of Principal Components")
  plt.savefig(os.path.join(save_path,"explained_variance_ratio_pca.png"))
  plt.close()
  return counts

def test_depth():
    print("================================================")
    print("Test: Depth Importance (etc max_depth)")
    print("================================================")
    max_depth = etc.max_depth

    #Test, train and validation sets
    test_SamplePCA_X = X[indexes[:int(test*len(indexes))]]
    test_SamplePCA_w_X = X_w[indexes[:int(test*len(indexes))]]
    test_SampleX = ext.X[indexes[:int(test*len(indexes))]]

    test_SampleY = ext.Y[indexes[:int(test*len(indexes))]]

    train_SamplePCA_X = X[indexes[int(train*len(indexes)):]]
    train_SamplePCA_w_X = X_w[indexes[int(train*len(indexes)):]]
    train_SampleX = ext.X[indexes[int(train*len(indexes)):]]

    train_SampleY = ext.Y[indexes[int(train*len(indexes)):]]

    val_SamplePCA_X = X[indexes[int(test*len(indexes)):int(train*len(indexes))]]
    val_SamplePCA_w_X = X_w[indexes[int(test*len(indexes)):int(train*len(indexes))]]
    val_SampleX = ext.X[indexes[int(test*len(indexes)):int(train*len(indexes))]]

    val_SampleY = ext.Y[indexes[int(test*len(indexes)):int(train*len(indexes))]]

    depth = 100

    scores = []
    scores_pca = []
    scores_pca_w = []

    best = (0,0,0)
    best_pca = (0,0,0)
    best_pca_w = (0,0,0)

    for i in range(depth):
        etc.max_depth = i + 1

        etc.fit(train_SampleX,train_SampleY)
        score = etc.score(test_SampleX,test_SampleY)
        scores.append(score)
        if score >= best[1]:
          best = (i+1,score,etc.score(val_SampleX,val_SampleY))

        etc.fit(train_SamplePCA_X,train_SampleY)
        score = etc.score(test_SamplePCA_X,test_SampleY)
        scores_pca.append(score)
        if score >= best_pca[1]:
          best_pca = (i+1,score,etc.score(val_SamplePCA_X,val_SampleY))

        etc.fit(train_SamplePCA_w_X,train_SampleY)
        score = etc.score(test_SamplePCA_w_X,test_SampleY)
        scores_pca_w.append(score)
        if score >= best_pca_w[1]:
          best_pca_w = (i+1,score,etc.score(val_SamplePCA_w_X,val_SampleY))

    plt.plot(range(1,depth+1),scores,label="raw data")
    plt.plot(range(1,depth+1),scores_pca,label="PCA")
    plt.plot(range(1,depth+1),scores_pca_w,label="PCA whiten")
    plt.ylabel("Score")
    plt.xlabel("Max Depth")
    plt.legend()
    plt.savefig(os.path.join(save_path,"score_max_depth_{}.png".format(n_estimators)))
    plt.close()
    print("Raw: Best score with a max depth of {} (test set {}):\t{} on the validation set".format(*best))
    print("PCA: Best score with a max depth of {} (test set {}):\t{} on the validation set".format(*best_pca))
    print("PCA with whiten: Best score with a max depth of {} (test set {}):\t{} on the validation set".format(*best_pca_w))

    etc.max_depth = max_depth


def test_Spaciality():
    print("================================================")
    print("Test: Spaciality Importance (tile size)")
    print("================================================")
    best = (0,0,0)
    for i in range(1,11):
        X,y = ext.rois2data(None,(i,i))
        indexes = list(range(X.shape[0]))
        shuffle(indexes)
        train_SampleX,train_SampleY = X[indexes[int(train*len(indexes)):]],y[indexes[int(train*len(indexes)):]]
        test_SampleX,test_SampleY = X[indexes[:int(test*len(indexes))]],y[indexes[:int(test*len(indexes))]]
        val_SampleX,val_SampleY = X[indexes[int(test*len(indexes)):int(train*len(indexes))]],y[indexes[int(test*len(indexes)):int(train*len(indexes))]]
        if not (len(train_SampleY) and len(test_SampleY) and len(val_SampleY)):
            break
        del X,y

        print("train set size: {}".format(len(train_SampleY)))
        print("test set size: {}".format(len(test_SampleY)))
        print("validation set size: {}".format(len(val_SampleY)))

        etc.fit(train_SampleX,train_SampleY)
        del train_SampleX,train_SampleY

        score = etc.score(test_SampleX,test_SampleY)
        del test_SampleX,test_SampleY
        print("Score with a slice size of {}:\t{}".format(i,score))

        if score >= best[1]:
            best = (i,score,etc.score(val_SampleX,val_SampleY))
        del val_SampleX,val_SampleY

    print("Best score with a slice size of {} (test set {}):\t{} on the validation set".format(best[0],best[1],best[2]))


if __name__ == '__main__':

    for n_estimators in [100,1000]:
        print("ExtraTreesClassifier with {} estimators".format(n_estimators))
        etc = ETC(n_jobs=n_jobs,n_estimators=n_estimators)
        test_comparaisonFeatureImportance()
        test_depth()
        test_Spaciality()

    counts = test_DimensionReduction()