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
num_image = 8
for i in range(1,num_image+1):
    with gzip.open("flutiste_data_{}.mask".format(i),'rb') as fb:
        mask = pickle.load(fb)
    t_im = im[:,:,:3].copy()
    x,y = np.nonzero(mask == 42276340)
    t_im[x,y] = (255,0,0)
    t_im = t_im.swapaxes(0,1)
    image = PIL.Image.fromarray(t_im,'RGB')
    image.save("detection_{}.png".format(i))
