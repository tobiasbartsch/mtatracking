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
    lines_list = ['All', '1', '2', '3', '4', '5', '6', '7', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'L', 'M', 'N', 'Q', 'R', 'SI', 'W']
    display_list = ['Time since last train', 'Probability of train delay']

    #Selector class variables (parameters) for holoviews panel
    direction = param.ObjectSelector(default='North', objects=direction_list)
    line = param.ObjectSelector(default='All', objects=lines_list)
    display = param.ObjectSelector(default= 'Probability of train delay', objects = display_list)

    def callback(self, data):
        if(self.display == 'Probability of train delay'):
            layout = gv.Points(data, vdims=['color', 'displaysize', 'name','waittime_str', 'delay_prob', 'MTAdelay']).opts(tools=[SubwayMap.hover], size='displaysize', color='color')
        #elif(self.display == 'Time since last train'):
        else:
            layout = gv.Points(data, vdims=['waittimecolor', 'waittimedisplaysize', 'name', 'waittime_str', 'delay_prob','MTAdelay']).opts(tools=[SubwayMap.hover], size='waittimedisplaysize', color='waittimecolor')
        return layout

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
        self.mapdata = mapdata


    hover = HoverTool(tooltips=[("station", "@name"), ("time since last train", "@waittime_str"), ("probability that incoming train is delayed", "@delay_prob"), ("MTA reports delay?", "@MTAdelay")])


    def view(self):
        return self.subway_map
    
    #https://panel.pyviz.org/user_guide/Param.html
    @param.depends('direction', 'line', watch=True)
    def update(self):
        self.mapdata.selected_dir = self.direction
        self.mapdata.selected_line = self.line