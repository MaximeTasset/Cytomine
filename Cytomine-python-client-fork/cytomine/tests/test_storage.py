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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from cytomine.models.storage import *
from cytomine.tests.conftest import random_string

__author__ = "Rubens Ulysse <urubens@uliege.be>"


class TestStorage:
    def test_storage(self, connect, dataset):
        name = random_string()
        storage = Storage(name, random_string(), dataset["user"].id).save()
        assert(isinstance(storage, Storage))
        assert(storage.name == name)

        storage = Storage().fetch(storage.id)
        assert(isinstance(storage, Storage))
        assert(storage.name == name)

        name = random_string()
        storage.name = name
        storage.update()
        assert(isinstance(storage, Storage))
        assert(storage.name == name)

        # TODO: Storage delete does not work on Cytomine-Core
        # storage.delete()
        # assert(not Storage().fetch(storage.id))

    def test_storages(self, connect, dataset):
        storages = StorageCollection().fetch()
        assert(isinstance(storages, StorageCollection))


class TestUploadedFile:
    def test_uploaded_file(self, connect, dataset):
        storages = StorageCollection().fetch()
        path = "path"
        uf = UploadedFile("original", "filename", path, 1, "ext", "contentType", None, storages[0].id,
                          connect.current_user.id, UploadedFile.UPLOADED, None).save()
        assert(isinstance(uf, UploadedFile))
        assert(uf.path == path)

        path = path + "bis"
        uf.path = path
        uf.update()
        assert(isinstance(uf, UploadedFile))
        assert(uf.path == path)

        uf.delete()
        uf = UploadedFile().fetch(uf.id)
        assert(uf.deleted is not None)
        # assert(not UploadedFile().fetch(uf.id))

    def test_uploaded_files(self, connect, dataset):
        uploaded_files = UploadedFileCollection().fetch()
        assert (isinstance(uploaded_files, UploadedFileCollection))
