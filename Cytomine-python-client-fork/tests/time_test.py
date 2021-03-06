"""
Created on Fri May 11 00:06:12 2018

@author: Maxime
"""

from cytomine.utilities.reader import CytomineSpectralReader,Bounds
from cytomine.cytomine import Cytomine
import sys
from timeit import default_timer
from time import sleep
import logging
from cytomine.models import imagegroup
from time import localtime,strftime


if __name__ == "__main__":
#  print('test')
#  start = default_timer()
#  sleep(10)
#  stop = default_timer()
#  elapsed_time = stop-start
#  print("elapsed time: {}s".format(elapsed_time))
#  hour = int(elapsed_time/3600)
#  minute = int((elapsed_time%3600)/60)
#  sec = int((elapsed_time%3600)%60)/60
#  print("or elapsed time: {}h{}m{}s".format(hour,minute,sec))
  print(sys.argv)
  if len(sys.argv) == 2:
    asynchrone = bool(int(sys.argv[1]))
  else:
    asynchrone = False
  print(asynchrone)
  sys.stdout.flush()

  cytomine_host="research.cytomine.be"
  cytomine_public_key="XXX"
  cytomine_private_key="XXX"
  id_project=28146931

  #Connection to Cytomine Core
  with Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, verbose= logging.WARNING,timeout=1200) as cytomine:
    stop = False
    it = 0
    print(id_project)
    while not stop:
        try:
            imagegroup_id = [im.id for im in imagegroup.ImageGroupCollection(filters={'project':int(id_project)}).fetch()][0]
            stop = True
        except TypeError:
            it += 1
            print(it)
            if it >= 10:
                sys.exit()
            stop = False
            sleep(10)

    if asynchrone:
        bounds = Bounds(0,0,3000,3000)
    else:
        bounds = Bounds(3000,3000,3000,3000)
    reader = CytomineSpectralReader(imagegroup_id,bounds = bounds,tile_size = Bounds(0,0,30,30),overlap=0,num_thread=10)
    it = 0

    start = default_timer()


    noread = False
    iteration = 10

    nb_read = 0
    while not noread:
      it += 1
      for i in range(iteration):
        reader.read(async=True)
        nb_read += 1
        if not reader.next():
          noread = True
          break
      if not noread:
        iterat = iteration
      else:
        iterat = iteration*2
      for i in range(iterat):
        result = reader.getResult(all_coord=True,in_list=True)
        if result is None: #that means no results left to fetch
          break
      #simulation of some workload
      sleep(10)

      if not it % 100:
        stop = default_timer()
        print("it {} and {} reads : position: {} after {} h {} m {} s ({} s) (net: {})".format(it,nb_read,reader.window_position,int((stop-start)/3600),
                                                                 int(((start-stop)%3600)/60),(stop-start)%60,stop-start,stop-start - it*10))
        sys.stdout.flush()

    stop = default_timer()
    elapsed_time = stop-start
    net = stop-start - it*10 #remove constant time (workload)
    print("elapsed time: {}s({}s (net))".format(elapsed_time,net))
    hour = int(elapsed_time/3600)
    minute = int((elapsed_time%3600)/60)
    sec = elapsed_time%60
    print("or elapsed time: {}h{}m{}s".format(hour,minute,sec))
    hour = int(net/3600)
    minute = int((net%3600)/60)
    sec = net%60
    print("elapsed time (net): {}h{}m{}s".format(hour,minute,sec))
