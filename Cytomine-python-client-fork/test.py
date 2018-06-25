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

print("load data from file")
ext = Extractor("extractedData.save")
ext.readFile()
print("fit_transform pca")
X = PCA().fit_transform(ext.data["X"])

val = list(range(X.shape[0]))
shuffle(val)
etc = ETC(n_jobs=4,n_estimators=100)

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

#Test and train sets
test_SamplePCA_X,test_SamplePCA_Y = X[val[:int(0.2*len(val))]],ext.data["Y"][val[:int(0.2*len(val))]]
test_SampleX,test_SampleY = ext.data["X"][val[:int(0.2*len(val))]],ext.data["Y"][val[:int(0.2*len(val))]]

train_SamplePCA_X,train_SamplePCA_Y = X[val[int(0.2*len(val)):]],ext.data["Y"][val[int(0.2*len(val)):]]
train_SampleX,train_SampleY = ext.data["X"][val[int(0.2*len(val)):]],ext.data["Y"][val[int(0.2*len(val)):]]


pca_score = []
imp_score = []
f_c_score = []
chi2_score = []
sum_score = []
n_feature = []


for i in list(set(list(range(0,1650,100))+list(range(1600,1650,10)))):
    print("Scores with best {} features/components:\n".format(1650-i))
    n_feature.append(1650-i)

    print("Scores before PCA:")
    etc.fit(train_SampleX[:,imp[:int(1650-i)]],train_SampleY)
    score = etc.score(test_SampleX[:,imp[:int(1650-i)]],test_SampleY)
    imp_score.append(score)
    print("\t -imp :\t{}".format(score))
    etc.fit(train_SampleX[:,f_c[:int(1650-i)]],train_SampleY)
    score = etc.score(test_SampleX[:,f_c[:int(1650-i)]],test_SampleY)
    f_c_score.append(score)
    print("\t -f_classif :\t{}".format(score))
    etc.fit(train_SampleX[:,chi2[:int(1650-i)]],train_SampleY)
    score = etc.score(test_SampleX[:,chi2[:int(1650-i)]],test_SampleY)
    chi2_score.append(score)
    print("\t -chi2 :\t{}".format(score))
    etc.fit(train_SampleX[:,sumv[:int(1650-i)]],train_SampleY)
    score = etc.score(test_SampleX[:,sumv[:int(1650-i)]],test_SampleY)
    sum_score.append(score)
    print("\t -Sum :\t{}".format(score))

    etc.fit(train_SamplePCA_X[:,:int(1650-i)],train_SamplePCA_Y)
    score = etc.score(test_SamplePCA_X[:,:int(1650-i)],test_SamplePCA_Y)
    pca_score.append(score)
    print("Score after PCA: {}".format(score))


plt.plot(n_feature,pca_score,label="pca")
plt.plot(n_feature,chi2_score,label="chi2")
plt.plot(n_feature,imp_score,label="imp")
plt.plot(n_feature,f_c_score,label="f_classif")
plt.plot(n_feature,sum_score,label="sum")
plt.legend()
plt.show()

print("fit_transform TSNE 2")
tsne2 = TSNE(2,n_iter=1000).fit_transform(X[:,:50])
print("fit_transform TSNE 3")
tsne3 = TSNE(3,n_iter=1000).fit_transform(X[:,:50])
print("fit_transform TSNE 4")
tsne4 = TSNE(4,n_iter=1000).fit_transform(X[:,:50])

tsne_score = {}
for name,tsne in [("tsne2",tsne2),("tsne3",tsne3),("tsne4",tsne4)]:
    test_SampleTSNE_X,test_SampleTSNE_Y = tsne[val[:int(0.2*len(val))]],ext.data["Y"][val[:int(0.2*len(val))]]
    train_SampleTSNE_X,train_SampleTSNE_Y = tsne[val[int(0.2*len(val)):]],ext.data["Y"][val[int(0.2*len(val)):]]
    etc.fit(train_SampleTSNE_X,train_SampleTSNE_Y)
    score = etc.score(test_SampleTSNE_X,test_SampleTSNE_Y)
    tsne_score[name] = score
    print("Score {}: {}".format(name,score))
