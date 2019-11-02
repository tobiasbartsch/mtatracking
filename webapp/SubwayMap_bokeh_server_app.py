import numpy as np
import holoviews as hv
import pandas as pd
import asyncio
import param
import parambokeh
import panel as pp
import threading

from bokeh.server.server import Server
from holoviews.streams import Stream
from SubwayMapView import SubwayMap
from SubwayMapViewModel import initializeStationsAndLines
from SubwayMapViewModel import SubwayMapData
from tornado.ioloop import IOLoop

# Enter key to download RT data.
key = input("Enter your MTA realtime access key: ")

stations_str = '/home/tbartsch/data/mtadata/subway_geo/subway_geo.geojson'
lines_str = '/home/tbartsch/data/mtadata/subway_geo/subway_stations_geo.geojson'

#load the geographic location of subway stations and lines and create the subway map
stations, lines = initializeStationsAndLines(stations_str, lines_str)

#create a subway data object to which we can bind
sdata = SubwayMapData(stations, lines) 

#create our subway map. This is the view. Initializing with the data object should take care of all data binding to property changed callbacks.
smap = SubwayMap(sdata)


def modify_doc(doc):    
    parambokeh.Widgets(smap, continuous_update=True, callback=smap.event, on_init=True, mode='server')
    panel = pp.Row(smap, smap.view)
    return panel.server_doc(doc=doc)

def start_server():
    
    print('starting server')
    loop = IOLoop.current()
    #loop.start()
    server = Server({'/': modify_doc}, port=9999, io_loop=loop)
    server.start()
    server.show('/')
    loop.spawn_callback(sdata.DataMineRT_async, key, loop)
    #dmine_task = asyncio.create_task(sdata.DataMineRT_async(key, loop))
    #loop.PeriodicCallback(sdata.DataMineRT_async, key, loop, callback_time=10000)
    server.run_until_shutdown()
    print('server was shutdown')

#datamine_thread = threading.Thread(target = sdata.DataMineRT, args = (key, ))
# def test(**kwargs):
#     for i in np.arange(100000):
#         pass
#     print('hi, we are in the test function')

# def test2(**kwargs):
#     print('hi, we are in the test2 function')

#smap.add_subscriber(test)
#smap.add_subscriber(test2)
#smap.event()
#start_server()

start_server()