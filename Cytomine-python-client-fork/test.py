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
from numpy.random import shuffle
from sklearn.decomposition import PCA
from cytomine.spectral.extractor import Extractor
import matplotlib.pyplot as plt

print("load data from file")
ext = Extractor("extractedData.save")
ext.readFile()
print("fit_transform pca")
X = PCA().fit_transform(ext.data["X"])

val = list(range(X.shape[0]))
shuffle(val)
etc = ETC(n_jobs=4,n_estimators=100)

print("get feature importances from ETC")
imp = [i for imp,i in sorted(ext.features_ETC(n_estimators=100))]
imp.reverse()
print("get feature importances from f_classif")
f_c = [i for imp,i in sorted(ext.f_classif())]
f_c.reverse()
print("get feature importances from chi2")
chi2 = [i for imp,i in sorted(ext.chi2())]
chi2.reverse()

test_SamplePCA_X,test_SamplePCA_Y = X[val[:int(0.2*len(val))]],ext.data["Y"][val[:int(0.2*len(val))]]
test_SampleX,test_SampleY = ext.data["X"][val[:int(0.2*len(val))]],ext.data["Y"][val[:int(0.2*len(val))]]
train_SamplePCA_X,train_SamplePCA_Y = X[val[int(0.2*len(val)):]],ext.data["Y"][val[int(0.2*len(val)):]]
train_SampleX,train_SampleY = ext.data["X"][val[int(0.2*len(val)):]],ext.data["Y"][val[int(0.2*len(val)):]]

pca_score = []
imp_score = []
f_c_score = []
chi2_score = []
n_feature = []


for i in range(0,1650,100):
    print("with best {} features/components".format(1650-i))
    n_feature.append(1650-i)

    print("score before PCA:")
    etc.fit(train_SampleX[:,imp[:int(1650-i)]],train_SampleY)
    score = etc.score(test_SampleX[:,imp[:int(1650-i)]],test_SampleY)
    imp_score.append(score)
    print("\t imp {}".format(score))
    etc.fit(train_SampleX[:,f_c[:int(1650-i)]],train_SampleY)
    score = etc.score(test_SampleX[:,f_c[:int(1650-i)]],test_SampleY)
    f_c_score.append(score)
    print("\t f_classif {}".format(score))
    etc.fit(train_SampleX[:,chi2[:int(1650-i)]],train_SampleY)
    score = etc.score(test_SampleX[:,chi2[:int(1650-i)]],test_SampleY)
    chi2_score.append(score)
    print("\t chi2 {}".format(score))

    etc.fit(train_SamplePCA_X[:,:int(1650-i)],train_SamplePCA_Y)
    score = etc.score(test_SamplePCA_X[:,:int(1650-i)],test_SamplePCA_Y)
    pca_score.append(score)
    print("score after PCA: {}".format(score))

for i in range(1600,1650,10):
    print("with best {} features/components".format(1650-i))
    n_feature.append(1650-i)

    print("score before PCA:")
    etc.fit(train_SampleX[:,imp[:int(1650-i)]],train_SampleY)
    score = etc.score(test_SampleX[:,imp[:int(1650-i)]],test_SampleY)
    imp_score.append(score)
    print("\t imp {}".format(score))
    etc.fit(train_SampleX[:,f_c[:int(1650-i)]],train_SampleY)
    score = etc.score(test_SampleX[:,f_c[:int(1650-i)]],test_SampleY)
    f_c_score.append(score)
    print("\t f_classif {}".format(score))
    etc.fit(train_SampleX[:,chi2[:int(1650-i)]],train_SampleY)
    score = etc.score(test_SampleX[:,chi2[:int(1650-i)]],test_SampleY)
    chi2_score.append(score)
    print("\t chi2 {}".format(score))

    etc.fit(train_SamplePCA_X[:,:int(1650-i)],train_SamplePCA_Y)
    score = etc.score(test_SamplePCA_X[:,:int(1650-i)],test_SamplePCA_Y)
    pca_score.append(score)
    print("score after PCA: {}".format(score))

plt.plot(n_feature,pca_score,label="pca")
plt.plot(n_feature,chi2_score,label="chi2")
plt.plot(n_feature,imp_score,label="imp")
plt.plot(n_feature,f_c_score,label="f_classif")
plt.legend()
plt.show()