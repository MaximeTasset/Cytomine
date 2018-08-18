# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 20:43:50 2018

@author: Maxime
"""

n_jobs=96
memory = {}
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.morphology import erosion,dilation,square
import numpy as np
from multiprocessing.pool import ThreadPool
import sys,gzip,pickle,PIL
from sklearn.metrics import accuracy_score
from multiprocessing import RLock
r = RLock()


def super_filter(im,th,dil=5,memoryindex=None):

    dil = max(1,dil)

    if not memoryindex is None and memoryindex in memory:
        masks = memory[memoryindex]
    else:
        masks = []
        if len(im.shape) == 3:
            gim =  0.2989 * im[:,:,0] + 0.5870 * im[:,:,1] + 0.1140 * im[:,:,2]
        elif len(im.shape) == 2:
            gim = im
        for filt in [sobel,roberts,scharr,prewitt]:
            mask = filt(gim)
            mask -= np.min(mask)
            mask /= np.max(mask)
            mask *= 255
            masks.append(mask.astype(np.uint8))
        if not memoryindex is None:
            memory[memoryindex] = masks
    mask = np.zeros_like(masks[0],dtype=np.float64)

    for m in masks:
        for v,l,u in [(0,0,th),(255,th,256)]:
            x,y = np.nonzero(m >= l)
            i = np.nonzero(m[x,y] < u)[0]
            mask[x[i],y[i]] += v/len(masks)

    umask = mask.astype(np.uint8)

    return np.asarray(erosion(dilation(umask,square(dil)),square(max(dil-1,1))),dtype=np.uint8)

def apply_super_filter(spectral_image,th_vote,th_filter,n_jobs=8, useMem=False):
  if n_jobs <= 1:
    supermask = [super_filter(spectral_image[:,:,i],th_filter,i if useMem else None) for i in range(spectral_image.shape[2])]
  else:
    pool = ThreadPool(n_jobs)
    try:
      nb = spectral_image.shape[2]
      tmp = np.zeros(1,dtype=np.int64)
      def filt(args):
          flts = super_filter(*args[:-1])
          with r:
            args[-1][0] += 1
            sys.stdout.write("\r{}\{}".format(args[-1][0],nb))
          return flts
      supermask = pool.map(filt,[(spectral_image[:,:,i],th_filter,5,i if useMem else None,tmp) for i in range(spectral_image.shape[2])])
    finally:
      pool.close()
  supermask = np.asarray(supermask)
  superfilter = np.sum(supermask>=128,axis=0)
  if not th_vote:
    return superfilter
  else:
    superfilter = superfilter >= th_vote
    return superfilter

def apply_super_filter2(spectral_image,th_vote,true_filter,n_jobs=8, useMem=False):
  nb = spectral_image.shape[2]
  tmp = np.zeros(1,dtype=np.int64)
  def filt(args):
    best = (0,0,None)
    for i in range(1,256):
      args[0]["th"] = i
      flts = super_filter(**args[0])
      score = score_filter(flts[:,:int(flts.shape[1]/2)],truefilter[:,:int(truefilter.shape[1]/2)])
      if score >= best[0]:
        best = (score,i,flts)
    if "memoryindex" in args[0] and args[0]["memoryindex"] in memory:
        del memory[args[0]["memoryindex"]]
        memory["best_{}".format(args[0]["memoryindex"])] = best[:-1]
    with r:
      args[-1][0] += 1
      sys.stdout.write("\r{}\{}".format(args[-1][0],nb))
    return best[-1]
  if n_jobs <= 1:
    supermask = [filt(({"im":spectral_image[:,:,i],"dil":5,
                        "memoryindex":i if useMem else None},tmp)) for i in range(spectral_image.shape[2])]
  else:
    pool = ThreadPool(n_jobs)
    try:
      supermask = pool.map(filt,[({"im":spectral_image[:,:,i],"dil":5,
                                    "memoryindex":i if useMem else None},tmp) for i in range(spectral_image.shape[2])])
    finally:
      pool.close()
  supermask = np.asarray(supermask)
  superfilter = np.sum(supermask>=128,axis=0)
  if not th_vote:
    return superfilter
  else:
    superfilter = superfilter >= th_vote
    return superfilter

def apply_super_filter_RGB(image,th_filter):
  supermask = super_filter(image,th_filter)
  supermask = np.array(supermask)
  superfilter = supermask >= 128
  return superfilter

def score_filter(filt,true_filter):
    return accuracy_score(true_filter.flatten(),filt.flatten())

with gzip.open("Flutiste/flutiste_data.im","rb") as fb:
    s_im,_ = pickle.load(fb)
s_im = np.swapaxes(s_im,0,1).astype(np.uint8)
truefilter = np.array(PIL.Image.open("Flutiste/labelled.png"))
xy = set([tuple(xy) for xy in np.argwhere(truefilter[:,:,0] == 255).tolist()])
xy = xy & set([tuple(xy) for xy in np.argwhere(truefilter[:,:,1] == 0).tolist()])
xy = list(xy & set([tuple(xy) for xy in np.argwhere(truefilter[:,:,2] == 0).tolist()]))
x = [x for x,y in xy]
y = [y for x,y in xy]
truefilter = np.zeros(truefilter.shape[:-1],dtype=np.bool)
truefilter[x,y] = True
im = np.array(PIL.Image.open("Flutiste/image.png"))
memory = {}
best = (0,0,None)
for i in range(1,256):#[85]:
    sys.stdout.write("\r{}  ".format(i))
    sys.stdout.flush()
    filtrgb = super_filter(im,i,4)
    supermask = filtrgb >= 128
    score = score_filter(supermask[:,:int(supermask.shape[1]/2)],truefilter[:,:int(truefilter.shape[1]/2)])
    if score >= best[1]:
        best = (i,score,supermask)
sys.stdout.write("\n")
print(best[:-1], "validation:",score_filter(best[-1][:,int(best[-1].shape[1]/2):],truefilter[:,int(truefilter.shape[1]/2):]))

memory = {}
print("multispectral, local thresold")

best_multi = (0,0,None)
filt = apply_super_filter2(s_im,0,truefilter,n_jobs=n_jobs,useMem=True)
for j in range(1,1651):
  sys.stdout.write("\r{}  ".format(j))
  sys.stdout.flush()
  superfilter = filt >= j
  score = score_filter(superfilter[:,:int(superfilter.shape[1]/2)],truefilter[:,:int(truefilter.shape[1]/2)])
  if score > best_multi[1]:
      best_multi = (j,score,filt)
      print(best_multi[:-1])
      sys.stdout.flush()
sys.stdout.write("\n")
print(best_multi[:-1], "validation:",score_filter(best_multi[-1][:,int(best_multi[-1].shape[1]/2):]>best_multi[0],truefilter[:,int(truefilter.shape[1]/2):]))
best_multi = best_multi[0],best_multi[1],best_multi[2],memory
memory = {}

print("multispectral, global thresold")

for i in range(1,255):
    filt = apply_super_filter(s_im,0,i,n_jobs=n_jobs)
    sys.stdout.write("{}  ".format(i))
    sys.stdout.flush()
    for j in range(1,1651):
        sys.stdout.flush()
        superfilter = filt >= j
        score = score_filter(superfilter[:,:int(superfilter.shape[1]/2)],truefilter[:,:int(truefilter.shape[1]/2)])
        if score > (best_multi[2] if len(best_multi) == 3 else best_multi[1]):
            best_multi = (i,j,score,filt)
            print(best_multi[:-1])
            sys.stdout.flush()
sys.stdout.write("\n")
if len(best_multi) == 3:
  print(best_multi[:-1], "validation:",score_filter(best_multi[-1][:,int(best_multi[-1].shape[1]/2):]>best_multi[-1][1],truefilter[:,int(truefilter.shape[1]/2):]))

with gzip.open("Flutiste/results.pkl","wb",compresslevel=4) as fb:
    pickle.dump((best,best_multi),fb)


#  superfilt = apply_super_filter(im,510,70)
#  x,y = np.nonzero(superfilt)
#  imm = im[:,:,:3].copy()
#  imm[x,y] = (255,0,0)
#  PIL.Image.fromarray(np.swapaxes(imm.astype(np.uint8),0,1))

#truefilter = np.array(PIL.Image.open("../../detection/labelled.png"))
#xy = set([tuple(xy) for xy in np.argwhere(truefilter[:,:,0] == 255).tolist()])
#xy = xy & set([tuple(xy) for xy in np.argwhere(truefilter[:,:,1] == 0).tolist()])
#xy = list(xy & set([tuple(xy) for xy in np.argwhere(truefilter[:,:,2] == 0).tolist()]))
#x = [x for x,y in xy]
#y = [y for x,y in xy]
#truefilter = np.zeros(truefilter.shape[:-1],dtype=np.bool)
#truefilter[x,y] = True


#import os
#
#for slice_size in [1,10]:
#  mask = {}
#  for name in ['randomized',"log2","sqrt","middle","all"]:
#      try:
#        with open("../../detection/test/flutiste_data_best_{}_{}.mask".format(slice_size,name),'rb') as fp:
#                  mask[name] = pickle.load(fp)
#      except FileNotFoundError:
#          pass
#  folder = '../../detection/test/{}'.format(slice_size)
#  os.makedirs(folder,exist_ok=True)
#  for name in mask:
#      cim = mask[name].copy()
#      cim[np.nonzero(cim != cim.max())] = 0
#      cim = cim/cim.max()
#      cim = cim.astype(np.bool)
#      cim = np.swapaxes(cim,0,1)
#      print(name,score_filter(cim[:,:int(cim.shape[1]/2)],truefilter[:,:int(truefilter.shape[1]/2)]))
#      imm = np.zeros_like(im)
#      imm[cim] = (255,0,0)
#      PIL.Image.fromarray(imm,mode="RGB").save(os.path.join(folder,'only_{}_{}.png'.format(slice_size,name)))
#      imm = s_im[:,:,:3].astype(np.uint8)
#      imm[cim] = (255,0,0)
#      PIL.Image.fromarray(imm,mode="RGB").save(os.path.join(folder,'detection_{}_{}.png'.format(slice_size,name)))

