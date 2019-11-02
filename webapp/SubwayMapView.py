import numpy as np
import holoviews as hv
hv.extension("bokeh")

import geoviews as gv
gv.extension("bokeh")

import pandas as pd
import geopandas as gpd
import colorcet as cc
import param

from holoviews.streams import Stream
from bokeh.server.server import Server
from holoviews.streams import Pipe
from bokeh.models import HoverTool
from cartopy import crs



class SubwayMap(Stream):
    '''subway map (holoviews dynamic map) including pyviz.param-based control interface.
    inherited from a holoviews stream object.'''

    #class variables
    direction_list = ['North', 'South']
    lines_list = ['All', 'Q', 'N', 'R']

    #Selector class variables (parameters) for holoviews panel
    direction = param.ObjectSelector(default='North', objects=direction_list)
    line = param.ObjectSelector(default='Q', objects=lines_list)

    hover = HoverTool(tooltips=[("station", "@name")])

    @staticmethod
    def callback(data):
        return gv.Points(data, vdims=['color', 'displaysize', 'name']).opts(tools=[SubwayMap.hover], size='displaysize', color='color')

    def __init__(self, mapdata):
        '''initialize a SubwayMap object
        Args:
            mapdata (SubwayMapData): container-class for stations and lines dataframes with implemented observer pattern.
                                    This is necessary for data binding of the view to the viewmodel.
        '''
        Stream.__init__(self)

        #create an initial map
        stations, lines = mapdata.stationsdf, mapdata.linesdf
        self.pipe = Pipe(data=[])
        self.subway_map = gv.Path(lines, vdims=['color']).opts(projection=crs.LambertConformal(), height=800, width=800, color='color') * gv.DynamicMap(self.callback, streams=[self.pipe])
        self.pipe.send(stations)

        #bind changes in the stationsdf to pipe.send
        mapdata.bind_to_stationsdf(self.pipe.send)
        

    def view(self):
        return self.subway_map
