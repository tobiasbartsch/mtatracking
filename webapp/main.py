import numpy as np
import holoviews as hv
import pandas as pd
import asyncio
import param
import parambokeh
import panel as pp
import threading
import time
import colorcet as cc

import sys
sys.path.append('../../')

from bokeh.server.server import Server
from bokeh.plotting import curdoc
from holoviews.streams import Stream
from SubwayMapView import SubwayMap
from SubwayMapViewModel import initializeStationsAndLines
from SubwayMapViewModel import SubwayMapData
from mtatracking.MTAdatamine import MTAdatamine
from tornado.ioloop import IOLoop
from threading import Thread
from functools import partial

# Enter key to download RT data.
key = input("Enter your MTA realtime access key: ")

stations_str = '/home/tbartsch/data/mtadata/subway_geo/subway_geo.geojson'
lines_str = '/home/tbartsch/data/mtadata/subway_geo/subway_stations_geo.geojson'
feed_id = ['gtfs-ace', 'gtfs-bdfm', 'gtfs-g', 'gtfs-jz', 'gtfs-nqrw', 'gtfs-l', 'gtfs', 'gtfs-7', 'gtfs-si']
#load the geographic location of subway stations and lines and create the subway map
stations, lines = initializeStationsAndLines(stations_str, lines_str)

#create a subway data object to which we can bind
sdata = SubwayMapData(stations, lines) 

#create our subway map. This is the view. Initializing with the data object should take care of all data binding to property changed callbacks.
smap = SubwayMap(sdata)

doc = curdoc()

def modify_doc(doc):    
    parambokeh.Widgets(smap, continuous_update=True, callback=smap.event, on_init=True, mode='server')
    #panel = pp.Row(smap, smap.view)
    panel = pp.Row(smap.view)
    return panel.server_doc(doc=doc)

# def start_server():
    
#     print('starting server')
#     loop = IOLoop.current()
#     #loop.start()
#     server = Server({'/': modify_doc}, port=9999, io_loop=loop)
#     server.start()
#     server.show('/')
#     loop.spawn_callback(sdata.DataMineRT_async, key, loop)
#     #dmine_task = asyncio.create_task(sdata.DataMineRT_async(key, loop))
#     #loop.PeriodicCallback(sdata.DataMineRT_async, key, loop, callback_time=10000)
#     server.run_until_shutdown()
#     print('server was shutdown')

loop = IOLoop.current()
loop.spawn_callback(sdata.DataMineRT_async, key, loop)
doc = modify_doc(doc)




