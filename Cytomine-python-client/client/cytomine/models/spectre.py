# -*- coding: utf-8 -*-


#
# * Copyright (c) 2009-2017. Authors: see NOTICE file.
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

__author__          = "Tasset Maxime <maxime.tasset@student.ulg.ac.be>"
__copyright__       = "Copyright 2010-2017 University of Liège, Belgium, http://www.cytomine.be/"


from model import Model
from collection import Collection


class Spectre(Model):
    def __init__(self, params = None):
        super(Spectre, self).__init__(params)
        self._callback_identifier = "spectre"

    def to_url(self):

        if hasattr(self, "id") and hasattr(self, "width") and hasattr(self, "height"):
            return "imagegroupHDF5/%d/%d/%d/pixel.json" % (self.id,self.width,self.height)
        else:
            raise ValueError('Uninitialized or used Spectre')


class SpectreCollection(Collection):

    def __init__(self, params = None):
        super(SpectreCollection, self).__init__(Spectre, params)

    def to_url(self):
        if hasattr(self, "id") and hasattr(self, "width") and hasattr(self, "height") and hasattr(self, "rwidth") and hasattr(self, "rheight"):
            return "imagegroupHDF5/%d/%d/%d/%d/%d/rectangle.json" % (self.id,self.width,self.height,self.rwidth,self.rheight)
        else:
            raise ValueError('Uninitialized or used Spectre')

class ImageGroupHDF5(Model):
    def __init__(self, params = None):
        super(ImageGroupHDF5, self).__init__(params)
        self._callback_identifier = "imageGroupHDF5"

    def to_url(self):
        return "imagegroup/%d/imagegroupHDF5.json" % (self.group)

class ImageGroup(Model):
    def __init__(self, params = None):
        super(ImageGroup, self).__init__(params)
        self._callback_identifier = "imageGroup"

    def to_url(self):
        return "imagegroup/%d.json" % (self.id)


class ImageGroupCollection(Collection):
    def __init__(self, params = None):
        super(ImageGroupCollection, self).__init__(ImageGroup,params)

    def to_url(self):
        return "project/%d/imagegroup.json" % (self.project)
