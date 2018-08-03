# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 20:43:50 2018

@author: Maxime
"""

import gzip,pickle,PIL
import numpy as np
with gzip.open('flutiste_data.im','rb') as fb:
    im,coord = pickle.load(fb)
im = im.astype(np.uint8)
PIL.Image.fromarray(im[:,:,:3].swapaxes(0,1)).save('original.png')
num_image = 10
for i in range(1,num_image+1):
    with gzip.open("flutiste_data_{}.mask".format(i),'rb') as fb:
        mask = pickle.load(fb)
    t_im = im[:,:,:3].copy()
    x,y = np.nonzero(mask == 42276340)
    t_im[x,y] = (255,0,0)
    t_im = t_im.swapaxes(0,1)
    image = PIL.Image.fromarray(t_im,'RGB')
    image.save("detection_{}.png".format(i))

pc =ProjectCollection({"user":cytomine.current_user.id}).fetch()
for p in pc:
    print(p.name,p.id)
igc = ImageGroupCollection({"project":56924820}).fetch()
ig = igc[0]
isc = ImageSequenceCollection({'imagegroup':ig.id}).fetch()
for im in isc:
    ii = ImageInstance(id=im.image).fetch()
    if ii.numberOfAnnotations:
        annot = AnnotationCollection(project=56924820, user=None, image=ii.id, term=None,
                                                               showMeta=None, bbox=None, bboxAnnotation=None, reviewed=False,
                                                               showTerm=True).fetch()
        break
from shapely.geometry import Polygon,MultiPolygon
from shapely.wkt import loads
polys = []
for ann in annot:
    ann = ann.fetch()
    polys.append(Polygon(loads(ann.location)))
pol = polys[7]
import numpy as np
from shapely.geometry.geo import box
minx,miny,maxx,maxy = pol.bounds
rect = (int(minx),int(miny),np.ceil(maxx - minx),np.ceil(maxy - miny))
rects = removeUnWantedRect(rect,pol,(5,5))
boxes = []
for rect in rects:
    boxes.append(box(rect[0],rect[1],rect[0]+rect[2],rect[1]+rect[3]))
boxed = MultiPolygon(boxes)