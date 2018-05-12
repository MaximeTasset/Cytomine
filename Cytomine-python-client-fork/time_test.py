# -*- coding: utf-8 -*-
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
  if len(sys.argv) == 2:
    asynchrone = bool(sys.argv[1])
  else:
    asynchrone = False
  cytomine_host="demo.cytomine.be"
  cytomine_public_key="f1f8cacc-b71a-4bc2-a6cd-e6bb40fd19b5"
  cytomine_private_key="9e94aa70-4e7c-4152-8067-0feeb58d42eb"
  id_project=28146931
#  id_project=31054043
  #id_users=[25637310]

  #Connection to Cytomine Core
  with Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path = '/api/', working_path = '/tmp/', verbose= logging.WARNING,timeout=1200) as cytomine:
    imagegroup_id = [im.id for im in imagegroup.ImageGroupCollection(filters={'project':id_project}).fetch()][0]
    reader = CytomineSpectralReader(imagegroup_id,bounds = Bounds(0,0, 100000, 100000),tile_size = Bounds(0,0,30,30),overlap=0,num_thread=10)
    it = 0

    start = default_timer()

    while True:
      reader.read(async=asynchrone)
      it += 1
      if not it % 100:
        while reader.getResult() is not None: pass
      #simulation of some workload
      sleep(10)
      if not reader.next():
        break
    while reader.getResult() is not None: pass

    stop = default_timer()
    elapsed_time = stop-start
    net = stop-start - it*10 #remove constant time (workload)
    print("elapsed time: {}s({}s (net))".format(elapsed_time,net))
    hour = int(elapsed_time/3600)
    minute = int((elapsed_time%3600)/60)
    sec = int((elapsed_time%3600)%60)/60
    print("or elapsed time: {}h{}m{}s".format(hour,minute,sec))
    hour = int(net/3600)
    minute = int((net%3600)/60)
    sec = int((net%3600)%60)/60
    print("elapsed time (net): {}h{}m{}s".format(hour,minute,sec))
