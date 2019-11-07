import pandas as pd
import numpy as np
import asyncio
import time
import asyncio
import colorcet as cc #it's a bit unclear whether this should be here (move to view?). Not clear how to implement that there though
import geopandas as gpd

from cartopy import crs
from SubwayMapModel import CurrentTransitTimeDelays
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

#homemade modules
import sys
sys.path.append('/home/tbartsch/source/repos')

import mtatracking.MTAdatamodel as MTAdatamodel
from mtatracking.MTAgeo import addStationID_andNameToGeoPandas
from mtatracking.MTAdatamine import MTAdatamine
from mtatracking.MTAdatamodel import SubwaySystem
from mtatracking.utils import utils

_executor = ThreadPoolExecutor(1)

class SubwayMapData():
    '''class to encapsulate stations and lines dataframes. Implements observer pattern, implements a method to track trains in realtime.
    Register callbacks in self._stations_observers and self._lines_observers.
    '''

    def __init__(self, stationsdf, linesdf, historicDataPath = '/home/tbartsch/data/testsys.pkl'):
        '''initialize a new SubwayMapData object
        Args:
            stationsdf: dataframe containing stations geomertry
            linesdf: dataframe containing lines geometry
            historicDataPath (string): path to a pickeled historic subway system object. This is used to compute
                                        mean transit times between stations against which we compare the current times.
                                        Todo: find a way to recompute this with updated data every day to keep up-to-date 
                                            with changes in the system.
        '''
        self._stationsdf = stationsdf
        self._linesdf = linesdf
        self._stations_observers = []
        self._lines_observers = []
        self._delays = CurrentTransitTimeDelays(histDataPath=historicDataPath).delays
        self._RTsys = SubwaySystem()
        self._mineRTdata = True
        self._selected_dir = 'N'
        self._selected_line = 'Q'

    @property
    def selected_dir(self):
        '''the direction selected in the view'''
        return self._selected_dir

    @selected_dir.setter
    def selected_dir(self, v):
        self._selected_dir = v

    @property
    def selected_line(self):
        '''the line selected in the view'''
        return self._selected_line
    
    @selected_line.setter
    def selected_line(self, v):
        self._selected_line = v
        print('highlighting line ', v)
        self.linesdf = highlightOneLine(self.linesdf, v)

    @property
    def mineRTdata(self):
        return self._mineRTdata
    
    @mineRTdata.setter
    def mineRTdata(self, v):
        self._mineRTdata = v

    @property
    def stationsdf(self):
        return self._stationsdf
    
    @stationsdf.setter
    def stationsdf(self, v):
        self._stationsdf = v
        for callback in self._stations_observers:
            callback(self._stationsdf)

    @property
    def linesdf(self):
        return self._linesdf
    
    @linesdf.setter
    def linesdf(self, v):
        self._linesdf = v
        for callback in self._lines_observers:
            callback(self._linesdf)

    def bind_to_stationsdf(self, callback):
        print('bound')
        self._stations_observers.append(callback)
    
    def bind_to_linesdf(self, callback):
        print('bound')
        self._lines_observers.append(callback)


    async def DataMineRT_async(self, key, loop):
        '''get realtime information of the current position of trains in the subway system, 
        track their position and update the probability of train delays for traversed segments of the system.
        This in turn should then trigger callbacks in the setter of the stationsdf property.
        This method has to be called as an async task as it contains while(True) and runs forever unless cancelled.

        Args:
            key (string): your private key for the MTA realtime feeds.
            loop: IOLoop for async execution
        '''
        myRTsys = self._RTsys
        delays = self._delays
        mine = MTAdatamine(key=key)
        while(self.mineRTdata):
            print('beginning iteration')
            #time.sleep(5)
            stations = self.stationsdf.copy()
            feed_id = [1,26,16,21,2,11,31,36,51] #all the feeds. probably not great that this is hardcoded, what if they add a new one?
            print('start tracking')
            tracking_results = mine.TrackTrains(feed_id)
            print('end tracking')
            current_time = time.time()
            print('have time, attaching tracking data')
            myRTsys.attach_tracking_data(tracking_results, current_time)
            print('done attaching')
            trains = np.array(list(myRTsys.trains.values()))
            ids = np.asarray([(train.route_id, train.direction) for train in trains])
        
            #reset all stations to grey. Todo: this should probably be in the view, shouldn't it?
            stations.loc[:,'color']=cc.blues[1]
            stations.loc[:,'displaysize']=3    
            print('beginning for loop')
            for line_id, delay in delays.items():
                line = line_id[:-1]
                direction = line_id[-1:]
                these_trains = trains[np.bitwise_and(ids[:,0] == line, ids[:,1] == direction)]
                #print('updating line ' + line_id)
                await loop.run_in_executor(_executor, delay.updateDelayProbs, these_trains, current_time)

                for key, val in delay.delayProbs.items():
                    k = key.split()
                    if not np.isnan(val):
                        col = cc.CET_D4[int(np.floor(val*255))]
                        size = 5 + 3 * val
                    else:
                        col = cc.CET_D4[0]
                        size = 5
                    stations.loc[stations['stop_id']==k[2][:-1], 'color']=col
                    stations.loc[stations['stop_id']==k[2][:-1], 'displaysize']=size
            self.stationsdf = stations
            print('done with iteration')
            delays_filename = 'delays' + datetime.today().strftime('%Y-%m-%d') + '.pkl'

            utils.write(delays, delays_filename)

    


def initializeStationsAndLines(lines_geojson, stations_geojson):
    '''load the locations of stations and lines in the NYC subway system. 
    Args:
        lines_geojson (string): path to the lines_geojson file
        stations_geojson (string): path to the stations_geojson file

    Returns:
        (stations, lines) (tuple of dataframes including "geometry" columns for plotting with geoviews)
    '''
    
    lines = gpd.read_file(lines_geojson, crs = crs.LambertConformal())
    stations = gpd.read_file(stations_geojson, crs = crs.LambertConformal())
    stationsDF = MTAdatamodel.MTAstaticdata.ImportStationsDataFrame()
    addStationID_andNameToGeoPandas(stations, stationsDF)

    stations['color'] = cc.blues[1]
    stations['displaysize'] = 3
    
    lines['color'] = cc.blues[1]
    lines = colorizeAllLines(lines)

    return (stations, lines)


def colorizeAllLines(linesdf):
    ''' set all lines in the linesdf to their respective colors.
    Args:
        linesdf: the lines dataframe

    Returns:
        linesdf: lines dataframe with modified colors column
    '''

    line_ids = ['A', 'C', 'E', 'B', 'D', 'F', 'M', 'G', 'L', 'J', 'Z', 'N', 'Q', 'R', 'W', '1', '2', '3', '4', '5', '6', '7', 'T', 'S']

    for line_id in line_ids:
        linesdf.loc[linesdf['name'].str.contains(line_id), 'color'] = LineColor(line_id)    
    
    return linesdf

def highlightOneLine(linesdf, lineid):
    ''' set a single line in the linesdf to its respective color. All others are set to grey.
    Args:
        linesdf: the lines dataframe
        lineid: id of the line to colorize. This can be either with or without its direction ('Q' and 'QN' produce the same result)

    Returns:
        linesdf: lines dataframe with modified colors column
    '''
    lineid = lineid[0]
    linesdf['color'] = cc.blues[1]
    linesdf.loc[linesdf['name'].str.contains(lineid), 'color'] = LineColor(lineid) 

    return linesdf


def LineColor(lineid):
    '''return the color of line lineid
    Args:
        lineid: id of the line to colorize. This can be either with or without its direction ('Q' and 'QN' produce the same result)
    Returns:
        color
    '''

    lineid = lineid[0]

    colors = ['#2850ad', '#ff6319', '#6cbe45', '#a7a9ac', '#996633', '#fccc0a', '#ee352e', '#00933c', '#b933ad', '#00add0', '#808183']
    lines_ids = [['A', 'C', 'E'], ['B', 'D', 'F', 'M'], ['G'], ['L'], ['J', 'Z'], ['N', 'Q', 'R', 'W'], ['1', '2', '3'], ['4', '5', '6'], ['7'], ['T'], ['S']]

    c = pd.Series(colors)
    ids = pd.DataFrame(lines_ids)
    c[(ids == lineid).any(axis=1)]

    return c[(ids == lineid).any(axis=1)].to_numpy()[0]