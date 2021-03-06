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
import pickle,gzip

cytomine_host="demo.cytomine.be"
cytomine_public_key="XXX"
cytomine_private_key="XXX"
id_project=0

n_jobs = 48
n_estimators = 100
test = .2
validation = .1
train = test + validation
fileid = 10

if fileid == 1:
    filename = "extractedData.save"
    save_path = "./colors"
elif fileid == 10:
    filename = "MaldiDemoData_tsne.save"
    save_path = "./MaldiDemo_tsne"
elif fileid == 100:
    filename = "MaldiDemoData.save"
    save_path = "./MaldiDemo"
else:
    filename = "flutiste.save"
    save_path = "./Flutiste"

os.makedirs(save_path,exist_ok=True)
ext = Extractor(os.path.join(save_path,filename),nb_job=n_jobs)
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
del ext.data,ext.unknown_X,ext.unknown_coord,ext.data_coord

indexes = list(range(ext.X.shape[0]))
shuffle(indexes)


nb_feature = ext.numFeature

#Test, train and validation sets
train_SampleX = ext.X[indexes[int(train*len(indexes)):]].astype(np.uint8)

print("fit_transform pca")
pca = PCA().fit(train_SampleX)
print("fit_transform pca with whiten")
pca_w = PCA(whiten=True).fit(train_SampleX)

train_SamplePCA_X = pca.transform(train_SampleX)
train_SamplePCA_w_X = pca_w.transform(train_SampleX)

train_SampleY = ext.Y[indexes[int(train*len(indexes)):]]

test_SampleX = ext.X[indexes[:int(test*len(indexes))]].astype(np.uint8)
test_SamplePCA_X = pca.transform(test_SampleX)
test_SamplePCA_w_X = pca_w.transform(test_SampleX)

test_SampleY = ext.Y[indexes[:int(test*len(indexes))]]

val_SampleX = ext.X[indexes[int(test*len(indexes)):int(train*len(indexes))]].astype(np.uint8)
val_SamplePCA_X = pca.transform(val_SampleX)
val_SamplePCA_w_X = pca_w.transform(val_SampleX)

val_SampleY = ext.Y[indexes[int(test*len(indexes)):int(train*len(indexes))]]

#ext.saveFeatureSelectionInCSV(os.path.join(save_path,"result.xlsx"),1000)
#with gzip.open(os.path.join(save_path,"result_feature.pkl"),"wb",compresslevel=4) as fb:
#    pickle.dump((ext.features_ETC(n_estimators=1000,sort=True),ext.chi2(sort=True),ext.f_classif(sort=True)),fb)

del ext.X,ext.Y
etc = None

best_FI = {}
best_sp = {}
best_dp = {}

def test_comparaisonFeatureImportance():
    global best_FI
    print("================================================")
    print("Test: Comparaison Of Feature Importance Measure")
    print("================================================")
    print("getting feature importances from ETC")
    imp_val = ext.features_ETC(n_estimators=n_estimators,data=(train_SampleX,train_SampleY))
    imp = [i for imp,i in sorted(imp_val)]
    imp.reverse()
    print("getting feature importances from f_classif")
    f_c_val = ext.f_classif(data=(train_SampleX,train_SampleY))
    f_c = [i for imp,i in sorted(f_c_val)]
    f_c.reverse()
    print("getting feature importances from chi2")
    chi2_val = ext.chi2(data=(train_SampleX,train_SampleY))
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

    best_FI = {}

    for i in sorted(list(set(list(range(0,max(0,int(nb_feature-50)),100))+list(range(max(0,int(nb_feature-50)),nb_feature,1))))):
        print("\nScores with best {} features/components:\n".format(nb_feature-i))
        n_feature.append(nb_feature-i)

        print("Scores before PCA:")
        etc.fit(train_SampleX[:,imp[:int(nb_feature-i)]],train_SampleY)
        score = etc.score(test_SampleX[:,imp[:int(nb_feature-i)]],test_SampleY)
        if best_FI.get("imp",(0,0))[1] < score:
          best_FI["imp"] = (int(nb_feature-i),score,imp)
        imp_score.append(score)
        print("\t -imp :\t{}".format(score))
        etc.fit(train_SampleX[:,f_c[:int(nb_feature-i)]],train_SampleY)
        score = etc.score(test_SampleX[:,f_c[:int(nb_feature-i)]],test_SampleY)
        if best_FI.get("f_c",(0,0))[1] < score:
          best_FI["f_c"] = (int(nb_feature-i),score,f_c)
        f_c_score.append(score)
        print("\t -f_classif :\t{}".format(score))
        etc.fit(train_SampleX[:,chi2[:int(nb_feature-i)]],train_SampleY)
        score = etc.score(test_SampleX[:,chi2[:int(nb_feature-i)]],test_SampleY)
        if best_FI.get("chi2",(0,0))[1] < score:
          best_FI["chi2"] = (int(nb_feature-i),score,chi2)
        chi2_score.append(score)
        print("\t -chi2 :\t{}".format(score))
        etc.fit(train_SampleX[:,sumv[:int(nb_feature-i)]],train_SampleY)
        score = etc.score(test_SampleX[:,sumv[:int(nb_feature-i)]],test_SampleY)
        if best_FI.get("sum",(0,0))[1] < score:
          best_FI["sum"] = (int(nb_feature-i),score,sumv)
        sum_score.append(score)
        print("\t -Sum :\t{}".format(score))

        etc.fit(train_SamplePCA_X[:,:int(nb_feature-i)],train_SampleY)
        score = etc.score(test_SamplePCA_X[:,:int(nb_feature-i)],test_SampleY)
        if best_FI.get("pca",(0,0))[1] < score:
          best_FI["pca"] = (int(nb_feature-i),score,None)
        pca_score.append(score)
        print("Score after PCA: {}".format(score))

        etc.fit(train_SamplePCA_w_X[:,:int(nb_feature-i)],train_SampleY)
        score = etc.score(test_SamplePCA_w_X[:,:int(nb_feature-i)],test_SampleY)
        if best_FI.get("pca_w",(0,0))[1] < score:
          best_FI["pca_w"] = (int(nb_feature-i),score,None)
        pca_w_score.append(score)
        print("Score after PCA with whiten: {}".format(score))

    print("Bests tested on validation set:")
    for f in best_FI:
        n_feature_k,score_test,f_imp = best_FI[f]
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
    global best_dp
    print("================================================")
    print("Test: Depth Importance (etc max_depth)")
    print("================================================")
    max_depth = etc.max_depth

    depth = 100

    scores = []
    scores_pca = []
    scores_pca_w = []

    best = (0,0,0)
    best_pca = (0,0,0)
    best_pca_w = (0,0,0)
#   pour avoir la vrai profondeur => max([et.tree_.max_depth for et in etc.estimators_])

    real_depth = []
    real_depth_pca = []
    real_depth_pca_w = []

    for i in range(depth):
        etc.max_depth = i + 1

        etc.fit(train_SampleX,train_SampleY)
        score = etc.score(test_SampleX,test_SampleY)
        real_depth.append(max([et.tree_.max_depth for et in etc.estimators_]))
        scores.append(score)
        if score >= best[1]:
          best = (i+1,score,etc.score(val_SampleX,val_SampleY))

        etc.fit(train_SamplePCA_X,train_SampleY)
        score = etc.score(test_SamplePCA_X,test_SampleY)
        real_depth_pca.append(max([et.tree_.max_depth for et in etc.estimators_]))
        scores_pca.append(score)
        if score >= best_pca[1]:
          best_pca = (i+1,score,etc.score(val_SamplePCA_X,val_SampleY))

        etc.fit(train_SamplePCA_w_X,train_SampleY)
        score = etc.score(test_SamplePCA_w_X,test_SampleY)
        real_depth_pca_w.append(max([et.tree_.max_depth for et in etc.estimators_]))
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
    plt.plot(range(1,depth+1),real_depth,label="raw data")
    plt.plot(range(1,depth+1),real_depth_pca,label="PCA")
    plt.plot(range(1,depth+1),real_depth_pca_w,label="PCA whiten")
    plt.ylabel("Real Max Depth")
    plt.xlabel("Max Depth")
    plt.legend()
    plt.savefig(os.path.join(save_path,"real_max_depth_{}.png".format(n_estimators)))
    plt.close()
    print("Raw: Best score with a max depth of {} (test set {}):\t{} on the validation set".format(*best))
    print("PCA: Best score with a max depth of {} (test set {}):\t{} on the validation set".format(*best_pca))
    print("PCA with whiten: Best score with a max depth of {} (test set {}):\t{} on the validation set".format(*best_pca_w))

    best_dp = {"raw":best,"pca":best_pca,"pca_w":best_pca_w}

    etc.max_depth = max_depth

def spaciality(X,y,i,indexes,old_score,name):
    X = np.asarray(X)
    X[np.nonzero(np.isfinite(X) == False)] = 0
    y = np.asarray(y)
    train_SampleX,train_SampleY = X[indexes[int(train*len(indexes)):]],y[indexes[int(train*len(indexes)):]]
    test_SampleX,test_SampleY = X[indexes[:int(test*len(indexes))]],y[indexes[:int(test*len(indexes))]]
    val_SampleX,val_SampleY = X[indexes[int(test*len(indexes)):int(train*len(indexes))]],y[indexes[int(test*len(indexes)):int(train*len(indexes))]]
    if not (len(train_SampleY) and len(test_SampleY) and len(val_SampleY)):
        return True,old_score
    del X,y

    print("train set size: {}".format(len(train_SampleY)))
    print("test set size: {}".format(len(test_SampleY)))
    print("validation set size: {}".format(len(val_SampleY)))

    etc.fit(train_SampleX,train_SampleY)

    del train_SampleX,train_SampleY
    score = etc.score(test_SampleX,test_SampleY)
    del test_SampleX,test_SampleY
    print("{}: Score with a slice size of {}:\t{}".format(name,i,score))

    return False,((i,score,etc.score(val_SampleX,val_SampleY)) if score >= old_score[1] else old_score)

def roi_pca(X,indexes,whiten):

    a = 0.97
    X = np.asarray(X,dtype=np.uint8)
    train_SampleX = X[indexes[int(train*len(indexes)):]]

    totransform = []
    for roi in train_SampleX:
        for i,j in np.ndindex(roi.shape[:2]):
            totransform.append(roi[i,j])

    pca = PCA(whiten=whiten).fit(totransform)
    count = 0
    for i,c in enumerate(pca.explained_variance_ratio_):
        count += c
        if count >= a:
            print("{}: ratio explained >= {:.2f} with {} components".format("pca_w" if whiten else "pca",a,i+1))
            pca_kept = i+1
            break
    pca = PCA(n_components=pca_kept,whiten=whiten).fit(totransform)
    already = {}
    totransform = []
    for r,roi in enumerate(X):
        for i,j in np.ndindex(roi.shape[:2]):
            key = tuple(roi[i,j])
            try:
                index,coords = already[key]
                coords.append((r,i,j))
                already[key] = index,coords
            except KeyError:
                already[key] = len(totransform),[(r,i,j)]
                totransform.append(roi[i,j])

    inin = {}
    for key,coords in already.values():
        for coord in coords:
            inin[coord] = key
    del already
    it = pca.transform(totransform).astype(np.float32)

    rois_pca = []
    for r,roi in enumerate(X):
        roi = np.empty((roi.shape[0],roi.shape[1],pca_kept),dtype=np.float32)
        for i,j in np.ndindex(roi.shape[:2]):
            roi[i,j] = it[inin[r,i,j]]
        rois_pca.append(roi.flatten())
        del roi

    return np.array(rois_pca,dtype=np.uint8)

def test_Spaciality(reduce):
    print("================================================")
    print("Test: Spaciality Importance (tile size): reduce {}".format(reduce))
    print("================================================")
    best = (0,0,0)


    for i in range(1,11):
        if reduce:
            X,y = ext.rois2data(None,(i,i),bands=best_FI["imp"][2][:best_FI["imp"][0]])
        else:
            X,y = ext.rois2data(None,(i,i),bands=None)
        indexes = list(range(X.shape[0]))
        shuffle(indexes)
        stop, best = spaciality(X,y,i,indexes,best,"raw")
        if stop:
          break
        del X,y


    print("Best score with a slice size of {} (test set {}):\t{} on the validation set".format(best[0],best[1],best[2]))

def test_max_feature():
    print("================================================")
    print("Test: feature selection (etc max_feature)")
    print("================================================")
    last = etc.max_features

    best = (0,0,0)
    best_pca = (0,0,0)
    best_pca_w = (0,0,0)
    for name,max_features in [('totally randomized trees',1),("sqrt",int(np.sqrt(ext.numFeature))),("middle",int(ext.numFeature/2)),("all",ext.numFeature),("log2",int(np.log2(ext.numFeature)))]:
        etc.max_features = max_features

        etc.fit(train_SampleX,train_SampleY)
        score = etc.score(test_SampleX,test_SampleY)
        print("Raw: score with max features {} on test set {}".format(name,score))
        if score >= best[1]:
          best = (name,score,etc.score(val_SampleX,val_SampleY))

        etc.fit(train_SamplePCA_X,train_SampleY)
        score = etc.score(test_SamplePCA_X,test_SampleY)
        print("PCA: score with max features {} on test set {}".format(name,score))
        if score >= best_pca[1]:
          best_pca = (name,score,etc.score(val_SamplePCA_X,val_SampleY))

        etc.fit(train_SamplePCA_w_X,train_SampleY)
        score = etc.score(test_SamplePCA_w_X,test_SampleY)
        print("PCA with whiten: score with max features {} on test set {}".format(name,score))
        if score >= best_pca_w[1]:
          best_pca_w = (name,score,etc.score(val_SamplePCA_w_X,val_SampleY))

    print("Raw: Best score with max features {} (test set {}):\t{} on the validation set".format(*best))
    print("PCA: Best score with max features {} (test set {}):\t{} on the validation set".format(*best_pca))
    print("PCA with whiten: Best score with max features {} (test set {}):\t{} on the validation set".format(*best_pca_w))

    etc.max_features = last
def feature():
    with gzip.open("Flutiste/result_feature.pkl",'rb') as fb:
        imp,chi,fc = pickle.load(fb)
    imp = [i for _,i in imp]
    chi = [i for _,i in chi]
    fc = [i for _,i in fc]
    ft = [('imp',imp),('chi2',chi),('f_classif',fc)]
    perm = [(0,1),(0,2),(1,2),(0,1,2)]
    for p in perm:
        common = []
        for i in range(1,len(chi)+1):
            for j,ind in enumerate(p):
                if not j:
                    X = set(ft[ind][1][:i])
                else:
                    X &= set(ft[ind][1][:i])
            common.append(len(X)/i)
        name = ""
        for j,ind in enumerate(p):
            if j:
                name += "&"
            name +=ft[ind][0]
        plt.plot(range(1,len(chi)+1),common,label=name)
    plt.ylabel("Number of Common Features")
    plt.xlabel("Number of Features")
    plt.legend()
    plt.savefig(os.path.join("Flutiste","comparaison_feature_imp_p.png"))
    plt.close()
if __name__ == '__main__':
    for n_estimators in [100,1000]:
        best_FI = {}
        print("ExtraTreesClassifier with {} estimators".format(n_estimators))
        etc = ETC(n_jobs=n_jobs,n_estimators=n_estimators)
        test_comparaisonFeatureImportance()
        test_depth()
        test_Spaciality(True)
        test_Spaciality(False)
        test_max_feature()

    counts = test_DimensionReduction()
