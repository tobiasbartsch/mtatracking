import pandas as pd
import numpy as np
import asyncio
import time
import asyncio
import colorcet as cc #it's a bit unclear whether this should be here (move to view?). Not clear how to implement that there though
import geopandas as gpd
import datetime as datet

from cartopy import crs
from SubwayMapModel import CurrentTransitTimeDelays
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from tornado import gen

#homemade modules
import sys
#sys.path.append('/home/tbartsch/source/repos')
sys.path.append('../../')
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
            stationsdf: dataframe containing stations geometry
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
    def line_ids(self):
        '''list of all line ids in the system'''
        print([element[:-1] for element in list(self._delays.keys())])
        return [element[:-1] for element in list(self._delays.keys())]

    @property
    def selected_dir(self):
        '''the direction selected in the view'''
        return self._selected_dir

    @selected_dir.setter
    def selected_dir(self, v):
        self._selected_dir = v[:1] #only save first letter (North = N)

    @property
    def selected_line(self):
        '''the line selected in the view'''
        return self._selected_line
    
    @selected_line.setter
    def selected_line(self, v):
        self._selected_line = v
        print('highlighting line ', v)
        if v == 'All':
            self.linesdf = colorizeAllLines(self.linesdf)
        else:
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


    def PushStationsDF(self):
        '''await functions that update the stationsdf, then await this function.
        This triggers push of the dataframe to the view.
        '''
        print('hello from the push callback')
        for callback in self._stations_observers:
            callback(self._stationsdf)
        

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
            #feed_id = [1,26,16,21,2,11,31,36,51] #all the feeds. probably not great that this is hardcoded, what if they add a new one?
            feed_id = ['gtfs-ace', 'gtfs-bdfm', 'gtfs-g', 'gtfs-jz', 'gtfs-nqrw', 'gtfs-l', 'gtfs', 'gtfs-7', 'gtfs-si']
            print('start tracking')
            tracking_results = await self._getdata(mine, feed_id, 10)
            print('end tracking')
            current_time = time.time()
            print('have time, attaching tracking data')
            myRTsys.attach_tracking_data(tracking_results, current_time)
            print('done attaching')
            trains = np.array(list(myRTsys.trains.values()))
            #ids = np.asarray([(train.route_id, train.direction) for train in trains])
        
            #reset all stations to grey.
            stations.loc[:,'color']=cc.blues[1]
            stations.loc[:,'displaysize']=3
            stations.loc[:, 'MTAdelay']=False
            stations.loc[:, 'waittimecolor']=cc.blues[1]
            stations.loc[:, 'delay_prob'] = np.nan
            stations.loc[:, 'waittime_str']='unknown'
            stations.loc[:, 'inboundtrain']='N/A'
            stations.loc[:, 'inbound_from']='N/A'
            

            print('loop')
            await self._updateStationsDfDelayInfo(delays, trains, stations, current_time, loop)
            await self._updateStationsDfWaitTime(myRTsys, stations, current_time, self.selected_dir, self.selected_line)

            self.stationsdf = stations
            print('done with iteration')
            #delays_filename = 'delays' + datetime.today().strftime('%Y-%m-%d') + '.pkl'
            #utils.write(delays, delays_filename)

    @gen.coroutine
    def update(self, stations):
        self.stationsdf = stations
        

    async def _getdata(self, dmine, feed_id, waittime):
        tracking_results = dmine.TrackTrains(feed_id)
        await asyncio.sleep(waittime)
        return tracking_results


    async def _updateStationsDfDelayInfo(self, delays, trains, stations, current_time, loop):
        '''update 'color' and 'displaysize' columns in the data frame, reflecting the probability that a subway will reach a station with a delay
        
        Args:
            delays: dictionary of delay objects
            trains: the trains we are currently tracking
            stations: stations data frame
            current_time: current time stamp
            loop: IOLoop for async execution
        
        '''
        ids = np.asarray([(train.route_id, train.direction) for train in trains])
        for line_id, delay in delays.items():
            line = line_id[:-1]
            direction = line_id[-1:]
            these_trains = trains[np.bitwise_and(ids[:,0] == line, ids[:,1] == direction)]
            #print('updating line ' + line_id)
            await loop.run_in_executor(_executor, delay.updateDelayProbs, these_trains, current_time)
            
            for train in these_trains:
                #get the MTA delay info and populate df with that
                MTADelayMessages = train.MTADelayMessages
                if len(MTADelayMessages) > 0:
                    if(np.abs(current_time - np.max(MTADelayMessages))) < 40:
                        arr_station = train.arrival_station_id[:-1]
                        stations.loc[stations['stop_id']==arr_station, 'MTAdelay']=True

            if (line == self.selected_line or self.selected_line == 'All') and direction == self.selected_dir:
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
                    stations.loc[stations['stop_id']==k[2][:-1], 'delay_prob']=val
                    stations.loc[stations['stop_id']==k[2][:-1], 'inboundtrain']=delay.train_ids[key]
                    stations.loc[stations['stop_id']==k[2][:-1], 'inbound_from']=k[0][:-1]


    async def _updateStationsDfWaitTime(self, subwaysys, stationsdf, currenttime, selected_dir, selected_line):
        '''update "waittime", "waittimedisplaysize", and "waittimecolor" column in data frame, reflecting the time (in seconds) that has passed since the last train visited this station.
        This is trivial if we are only interested in trains of a particular line, but gets more tricky if the user selected to view "All" lines
        
        Args: 
            subwaysys: subway system object containing the most recent tracking data
            stationsdf: stations data frame 
        '''
        for station_id, station in subwaysys.stations.items():
            if station_id is not None and len(station_id) > 1:
                station_dir = station_id[-1:]
                s_id = station_id[:-1]
                wait_time = None
                if station_dir == selected_dir and selected_line is not 'All': #make sure we are performing this update according to the direction selected by the user
                    wait_time = station.timeSinceLastTrainOfLineStoppedHere(selected_line, selected_dir, currenttime)
                elif station_dir == selected_dir and selected_line == 'All':
                    wait_times = []
                    #iterate over all lines that stop here
                    lines_this_station = list(station.trains_stopped.keys()) #contains direction (i.e. QN instead of Q)
                    lines_this_station = list(set([ele[:-1] for ele in lines_this_station]))
                    for line in lines_this_station:
                        wait_times.append(station.timeSinceLastTrainOfLineStoppedHere(line, selected_dir, currenttime))
                    wait_times = np.array(wait_times)
                    wts = wait_times[wait_times != None]
                    if len(wts) > 0:
                        wait_time = np.min(wait_times[wait_times != None])
                    else:
                        wait_time = None
                if(wait_time is not None):
                    stationsdf.loc[stationsdf['stop_id']==s_id, 'waittime']=wait_time #str(datet.timedelta(seconds=wait_time))
                    stationsdf.loc[stationsdf['stop_id']==s_id, 'waittime_str'] = timedispstring(wait_time)
                    #spread colors over 30 min. We want to eventually replace this with a scaling by sdev
                    if(int(np.floor(wait_time/(30*60)*255)) < 255):
                        col = cc.fire[int(np.floor(wait_time/(30*60)*255))]
                    else:
                        col = cc.fire[255]
                    stationsdf.loc[stationsdf['stop_id']==s_id, 'waittimecolor']=col
                    stationsdf.loc[stationsdf['stop_id']==s_id, 'waittimedisplaysize']=5 #constant size in this display mode        

        
        
    


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

    stations['color'] = cc.blues[1] #this color reflects the delay of the incoming train. We should rename this.
    stations['displaysize'] = 3
    stations['delay_prob'] = np.nan
    stations['MTAdelay']=False
    stations['inboundtrain'] = 'N/A'
    stations['inbound_from'] = 'N/A'

    stations['waittime']=np.nan
    stations['waittime_str']='unknown'
    stations['waittimedisplaysize']=3
    stations['waittimecolor']=cc.blues[1] #this color reflects the time that has passed since a train visited this station. The view should decide which color to display.
    stations['waittimedisplaysize']=3 
    
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

def timedispstring(secs):
    hms = str(datet.timedelta(seconds=round(secs))).split(':')
    if hms[0] == '0':
        if hms[1].lstrip("0") == '':
            return hms[2].lstrip("0") + ' s'
        else:
            return hms[1].lstrip("0") + ' min ' + hms[2].lstrip("0") + ' s'
    else:
        return hms[0].lstrip("0") + ' hours ' + hms[1].lstrip("0") + ' min ' + hms[2].lstrip("0") + ' s'