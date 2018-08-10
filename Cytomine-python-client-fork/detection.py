# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 20:43:50 2018

@author: Maxime
"""

#import gzip,pickle,PIL
#import numpy as np
#with gzip.open('flutiste_data.im','rb') as fb:
#    im,coord = pickle.load(fb)
#im = im.astype(np.uint8)
#PIL.Image.fromarray(im[:,:,:3].swapaxes(0,1)).save('original.png')
#num_image = 10
#for i in range(1,num_image+1):
#    with gzip.open("flutiste_data_{}.mask".format(i),'rb') as fb:
#        mask = pickle.load(fb)
#    t_im = im[:,:,:3].copy()
#    x,y = np.nonzero(mask == 42276340)
#    t_im[x,y] = (255,0,0)
#    t_im = t_im.swapaxes(0,1)
#    image = PIL.Image.fromarray(t_im,'RGB')
#    image.save("detection_{}.png".format(i))
#
#pc =ProjectCollection({"user":cytomine.current_user.id}).fetch()
#for p in pc:
#    print(p.name,p.id)
#igc = ImageGroupCollection({"project":56924820}).fetch()
#ig = igc[0]
#isc = ImageSequenceCollection({'imagegroup':ig.id}).fetch()
#for im in isc:
#    ii = ImageInstance(id=im.image).fetch()
#    if ii.numberOfAnnotations:
#        annot = AnnotationCollection(project=56924820, user=None, image=ii.id, term=None,
#                                                               showMeta=None, bbox=None, bboxAnnotation=None, reviewed=False,
#                                                               showTerm=True).fetch()
#        break
#from shapely.geometry import Polygon,MultiPolygon
#from shapely.wkt import loads
#polys = []
#for ann in annot:
#    ann = ann.fetch()
#    polys.append(Polygon(loads(ann.location)))
#pol = polys[7]
#import numpy as np
#from shapely.geometry.geo import box
#minx,miny,maxx,maxy = pol.bounds
#rect = (int(minx),int(miny),np.ceil(maxx - minx),np.ceil(maxy - miny))
#rects = removeUnWantedRect(rect,pol,(5,5))
#boxes = []
#for rect in rects:
#    boxes.append(box(rect[0],rect[1],rect[0]+rect[2],rect[1]+rect[3]))
#boxed = MultiPolygon(boxes)
#
#
#totransform = []
#for roi,label in ext.rois:
#    for i,j in np.ndindex(label.shape):
#        totransform.append(roi[i,j])
#pca = PCA().fit(totransform)
#val = pca.transform(totransform)
#it = iter(val)
#rois = []
#for roi,label in ext.rois:
#    roi = np.empty_like(roi,dtype=np.float64)
#    for i,j in np.ndindex(label.shape):
#        roi[i,j] = it.__next__()
#    rois.append((roi,label))
#
#
#
#from skimage.filters import roberts, sobel, scharr, prewitt
#import numpy as np
#import PIL
#im = np.array(PIL.Image.open("../../detection/image.png"),dtype=np.uint8)
#gim =  0.2989 * im[:,:,0] + 0.5870 * im[:,:,1] + 0.1140 * im[:,:,2]
#mask = sobel(gim)
#mask -= np.min(mask)
#mask /= np.max(mask)
#mask *= 255
#smask = mask.astype(np.uint8)
#PIL.Image.fromarray(smask,'L').save("../../detection/sobel.png")
#mask = roberts(gim)
#mask -= np.min(mask)
#mask /= np.max(mask)
#mask *= 255
#rmask = mask.astype(np.uint8)
#PIL.Image.fromarray(rmask,'L').save("../../detection/roberts.png")
#mask = scharr(gim)
#mask -= np.min(mask)
#mask /= np.max(mask)
#mask *= 255
#cmask = mask.astype(np.uint8)
#PIL.Image.fromarray(cmask,'L').save("../../detection/scharr.png")
#mask = prewitt(gim)
#mask -= np.min(mask)
#mask /= np.max(mask)
#mask *= 255
#pmask = mask.astype(np.uint8)
#PIL.Image.fromarray(pmask,'L').save("../../detection/prewitt.png")
#mask = np.zeros_like(smask,dtype=np.float64)
#th = 150
#for m in [smask+rmask+cmask+pmask]:
#    for v,l,u in [(0,0,th),(255,th,256)]:
#        x,y = np.nonzero(m >= l)
#        i = np.nonzero(m[x,y] < u)
#        mask[x[i],y[i]] += v/4
#
#mask -= np.min(mask)
#mask /= np.max(mask)
#mask *= 255
#umask = mask.astype(np.uint8)
#from skimage.morphology import erosion,dilation,square
#umask = erosion(dilation(umask,square(3)),square(4))
#PIL.Image.fromarray(umask,'L').save("../../detection/union.png")
#PIL.Image.fromarray(umask,'L')
#
#mask = np.zeros_like(smask,dtype=np.float64)
#th = 70
#for m in [smask,rmask,cmask,pmask]:
#    for v,l,u in [(0,0,th),(255,th,256)]:
#        x,y = np.nonzero(m >= l)
#        i = np.nonzero(m[x,y] < u)
#        mask[x[i],y[i]] += v/4
#
#mask -= np.min(mask)
#mask /= np.max(mask)
#mask *= 255
#umask = mask.astype(np.uint8)
#from skimage.morphology import erosion,dilation,square
#umask = erosion(dilation(umask,square(4)),square(5))
#
#
#import pickle, gzip
#import PIL
#with gzip.open('flutiste_data.im','rb')as fp:
#    im,coord = pickle.load(fp)

n_jobs=24

from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.morphology import erosion,dilation,square
import numpy as np
from multiprocessing.pool import ThreadPool
import sys,gzip,pickle,PIL
from sklearn.metrics import accuracy_score

def super_filter(im,th,dil=5):
    dil = max(1,dil)
    if len(im.shape) == 3:
        gim =  0.2989 * im[:,:,0] + 0.5870 * im[:,:,1] + 0.1140 * im[:,:,2]
    elif len(im.shape) == 2:
        gim = im
    masks = []
    for filt in [sobel,roberts,scharr,prewitt]:
        mask = filt(gim)
        mask -= np.min(mask)
        mask /= np.max(mask)
        mask *= 255
        masks.append(mask.astype(np.uint8))
    mask = np.zeros_like(masks[0],dtype=np.float64)

    for m in masks:
        for v,l,u in [(0,0,th),(255,th,256)]:
            x,y = np.nonzero(m >= l)
            i = np.nonzero(m[x,y] < u)[0]
            mask[x[i],y[i]] += v/len(masks)


    mask -= np.min(mask)
    mask /= np.max(mask)
    mask *= 255
    umask = mask.astype(np.uint8)

    return np.array(erosion(dilation(umask,square(dil)),square(max(dil-1,1))))

def apply_super_filter(spectral_image,th_vote,th_filter,n_jobs=8):
  if n_jobs <= 1:
    supermask = [super_filter(spectral_image[:,:,i],th_filter) for i in range(spectral_image.shape[2])]
  else:
    pool = ThreadPool(n_jobs)
    try:
      def filt(args):
          return super_filter(*args)
      supermask = pool.map(filt,[(spectral_image[:,:,i],th_filter) for i in range(spectral_image.shape[2])])
    finally:
      pool.close()
  supermask = np.array(supermask)
  superfilter = np.sum(supermask>=128,axis=0)
  return superfilter
#  superfilter = superfilter >= th_vote
#  return superfilter


def apply_super_filter_RGB(image,th_filter):
  supermask = super_filter(image,th_filter)
  supermask = np.array(supermask)
  superfilter = supermask >= 128
  return superfilter

def score_filter(filt,true_filter):
#    n_positif =  {tuple(xy) for xy in np.argwhere(true_filter)}
#    n_positif =  len(n_positif & {tuple(xy) for xy in np.argwhere(filt)})
#    n_negatif =  {tuple(xy) for xy in np.argwhere(true_filter == False)}
#    n_negatif =  len(n_negatif & {tuple(xy) for xy in np.argwhere(filt == False)})
#
#    return (n_positif + n_negatif) / np.multiply(*true_filter.shape)
    return accuracy_score(true_filter.flatten(),filt.flatten())

with gzip.open("flutiste_data.im","rb") as fb:
    s_im,_ = pickle.load(fb)
s_im = np.swapaxes(s_im,0,1)
truefilter = np.array(PIL.Image.open("labelled7.png"))
xy = set([tuple(xy) for xy in np.argwhere(truefilter[:,:,0] == 255).tolist()])
xy = xy & set([tuple(xy) for xy in np.argwhere(truefilter[:,:,1] == 0).tolist()])
xy = list(xy & set([tuple(xy) for xy in np.argwhere(truefilter[:,:,2] == 0).tolist()]))
x = [x for x,y in xy]
y = [y for x,y in xy]
truefilter = np.zeros(truefilter.shape[:-1],dtype=np.bool)
truefilter[x,y] = True
im = np.array(PIL.Image.open("image.png"))

best = (0,0,None)
for i in range(0,256):
    sys.stdout.write("\r{}  ".format(i))
    sys.stdout.flush()
    filtrgb = super_filter(im,i,4)
    supermask = filtrgb >= 128
    score = score_filter(supermask,truefilter)
    if score >= best[1]:
        best = (i,score,supermask)
sys.stdout.write("\n")
print(best[:-1])

print("multispectral")
best_multi = (0,0,0)

for i in range(0,255):#range(max(0,best[0]-10),min(256,best[0]+11)):
    filt = apply_super_filter(s_im,0,i,n_jobs=n_jobs)
    sys.stdout.write("{}  ".format(i))
    sys.stdout.flush()
    for j in range(1651):
        sys.stdout.flush()
        superfilter = filt >= j
        score = score_filter(superfilter,truefilter)
        if score > best_multi[2]:
            best_multi = (i,j,score,filt)
            print(best_multi[:-1])
            sys.stdout.flush()

sys.stdout.write("\n")
with gzip.open("results.pkl","wb",compresslevel=4) as fb:
    pickle.dump((best,best_multi),fb)



#  superfilt = apply_super_filter(im,510,70)
#  x,y = np.nonzero(superfilt)
#  imm = im[:,:,:3].copy()
#  imm[x,y] = (255,0,0)
#  PIL.Image.fromarray(np.swapaxes(imm.astype(np.uint8),0,1))

#truefilter = np.array(PIL.Image.open("../../detection/labelled5.png"))
#xy = set([tuple(xy) for xy in np.argwhere(truefilter[:,:,0] == 255).tolist()])
#xy = xy & set([tuple(xy) for xy in np.argwhere(truefilter[:,:,1] == 0).tolist()])
#xy = list(xy & set([tuple(xy) for xy in np.argwhere(truefilter[:,:,2] == 0).tolist()]))
#x = [x for x,y in xy]
#y = [y for x,y in xy]
#truefilter = np.zeros(truefilter.shape[:-1],dtype=np.bool)
#truefilter[x,y] = True