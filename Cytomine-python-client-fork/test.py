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

print("load data from file")
ext = Extractor("extractedData.save")
ext.readFile()
print("fit_transform pca")
X = PCA().fit_transform(ext.data["X"])

val = list(range(X.shape[0]))
shuffle(val)
etc = ETC(n_jobs=4,n_estimators=100)

print("get feature importances from ETC")
etc.fit(ext.data["X"],ext.data["Y"])
imp = [i for imp,i in sorted([(imp,i) for i,imp in enumerate(etc.feature_importances_)])]
imp.reverse()

test_SamplePCA_X,test_SamplePCA_Y = X[val[:int(0.2*len(val))]],ext.data["Y"][val[:int(0.2*len(val))]]
test_SampleX,test_SampleY = ext.data["X"][val[:int(0.2*len(val))]],ext.data["Y"][val[:int(0.2*len(val))]]
train_SamplePCA_X,train_SamplePCA_Y = X[val[int(0.2*len(val)):]],ext.data["Y"][val[int(0.2*len(val)):]]
train_SampleX,train_SampleY = ext.data["X"][val[int(0.2*len(val)):]],ext.data["Y"][val[int(0.2*len(val)):]]


for i in range(0,1650,100):
    print("with best {} features/components".format(1650-i))
    etc.fit(train_SampleX[:,imp[:int(1650-i)]],train_SampleY)
    print("score before PCA: {}".format(etc.score(test_SampleX[:,imp[:int(1650-i)]],test_SampleY)))

    etc.fit(train_SamplePCA_X[:,:int(1650-i)],train_SamplePCA_Y)
    print("score after PCA: {}".format(etc.score(test_SamplePCA_X[:,:int(1650-i)],test_SamplePCA_Y)))

