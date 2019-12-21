import numpy as np
import pandas as pd
import os as os
from os import walk
from os.path import dirname
from multiprocessing import Pool
from functools import partial
 
#custom modules, see https://github.com/tobiasbartsch/mtatracking and https://github.com/tobiasbartsch/algorithms
import sys
sys.path.append('/home/tbartsch/source/repos')
from mtatracking.MTAdatamodel import SubwaySystem
from mtatracking.utils import utils
 
import mtatracking.gtfs_realtime_pb2 as gtfs_realtime_pb2
sys.modules['gtfs_realtime_pb2'] =gtfs_realtime_pb2
 
'''this loads a dictionary of class mtatracking.MTASubwayAnalyzer.delayOfTrainsInLine_Bayes
'''
delays = utils.read('../webapp/delays2019-10-31.pkl')
 
#functions
 
def attach(tracking_results, timestamp, subwaysys):
    current_time = timestamp #the timestamp of the current GTFS snapshot
    subwaysys.attach_tracking_data(tracking_results, current_time)
    trains = np.array(list(subwaysys.trains.values()))
    ids = np.asarray([(train.route_id, train.direction) for train in trains])
      
    numdelays= {}
    numarrivalstations = {}
    for line_id, delay in delays.items():
        numdelays[line_id] = 0
        numarrivalstations[line_id] = 0
        line = line_id[:-1]
        direction = line_id[-1:]
        these_trains = trains[np.bitwise_and(ids[:,0] == line, ids[:,1] == direction)]
       
        #get the probability for the delay of each train
        delay.updateDelayProbs(trains_in_line=these_trains, timestamp=current_time)
        for key, val in delay.delayProbs.items():
            k = key.split()
            if not np.isnan(val):
                numarrivalstations[line_id] +=1
                if(val==1):
                    numdelays[line_id]+=1
    numtrains = len(trains)
    return numdelays, numarrivalstations, numtrains
 
 
# load GTFS realtime data and create a dataframe of delay frequencies
#create a subway system object to which we can attach the data:
subwaysys = SubwaySystem()
 
#this list will hold the result dictionaries. We will later convert it to the pandas df
totaldelays = []
 
 
 
mypath = '/home/tbartsch/data/SubwayTracking/'
f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break
f_stamps = [1, 2]
for s in f:
    timestamp = s[16:]
    timestamp = timestamp[:-4]
    f_stamps.append(int(timestamp))
f_stamps.sort()
 
# attach to the subway system
c = 0
for counter, stamp in enumerate(f_stamps):
    if(stamp < 1561939200 and stamp > 1564531200): #1st of July to 31st of July
        continue
    if(counter % 5000 == 0):
        print(counter)
    if len(str(stamp)) == 10:
        n = 'tracking_results' + str(stamp) + '.pkl'
        myfile = os.path.join(mypath, n)
        try:
            data = utils.read(myfile)
            numdelays, numarrivalstations, numtrains = attach(data, stamp, subwaysys)
            for (line_id, ndel), (line_id2, nstations) in zip(numdelays.items(), numarrivalstations.items()):
                assert(line_id==line_id2)
                row = {'time' : stamp, 'numdelays' : ndel, 'numarrivalstations' : nstations, 'line_id': line_id, 'numtrains': numtrains}
                totaldelays.append(row)
        except:
            print("Error loading file" + str(stamp))
 
delaysdf = pd.DataFrame(totaldelays)
 
#average over lines
timestamps = list(set(delaysdf['time']))
arrst = []
dels = []
dirs = []
time = []
numtrains = []
for i, stamp in enumerate(timestamps):
    if(i%1000 == 0):
        print(i)
    #northbound
    s = delaysdf[(delaysdf['time']==stamp) & (delaysdf['line_id'].str[-1:] == 'N')]
    arrst.append(np.sum(s['numarrivalstations']))
    dels.append(np.sum(s['numdelays']))
    time.append(s['time'].values[0])
    dirs.append('N')
    numtrains.append(s['numtrains'].values[0])
    #southbound
    s = delaysdf[(delaysdf['time']==stamp) & (delaysdf['line_id'].str[-1:] == 'S')]
    arrst.append(np.sum(s['numarrivalstations']))
    dels.append(np.sum(s['numdelays']))
    time.append(s['time'].values[0])
    dirs.append('S')
    numtrains.append(s['numtrains'].values[0])
 
 
delaysdf = pd.DataFrame({'time': time, 'arrival_stations': arrst, 'delays': dels, 'numtrains': numtrains, 'direction': dirs})
delaysdf['delaypercent'] = delaysdf['delays']/delaysdf['arrival_stations']
 
times = np.array(list(set(delaysdf['time'])), dtype=int)
 
chunksize = int(len(times)/12)
with Pool(12) as pool:
    daytime = pool.map(utils.timestamp_to_day_time, times, chunksize=chunksize)
daytime = np.array(list(daytime))
daytime[:,1]=np.floor(daytime[:,1]/3600)
day = np.array(daytime[:,0], dtype=int)
hours = np.array(daytime[:,1], dtype=int)
 
numprocs = 15
 
delaysdf['hours']=np.nan
 
#sort
delaysdf = delaysdf.sort_values(by=['time'])
 
#for multiprocessing:
data_split = np.array_split(delaysdf, numprocs)
 
def get_data_from_lookuptable(delaysdf_chunk, lookup_table):
    #determine relevant range of the lookup table:
    start = min(delaysdf_chunk['time'])
    stop = max(delaysdf_chunk['time'])
    table = lookup_table[int(np.where(lookup_table[:,0]==start)[0]):int(np.where(lookup_table[:,0]==stop)[0])+1]
    for i, row in enumerate(table):
        delaysdf_chunk.loc[row[0]==delaysdf['time'],'day']=row[1]
        delaysdf_chunk.loc[row[0]==delaysdf['time'],'hours']=row[2]
        if(i%1000==0):
            print(i)
    return delaysdf_chunk
 
lookup_table = np.array(list(zip(day, hours)), dtype=object)
lookup_table=lookup_table[lookup_table[:,0].argsort()] #sort
 
with Pool(numprocs) as pool:
    res = pool.map(partial(get_data_from_lookuptable, lookup_table=lookup_table), data_split)
 
delaysdf_updated['weekday'] = 0
delaysdf_updated['weekday'] = delaysdf_updated['day'] < 5 #monday is 0, friday is 4
 
delaysdf_updated['north'] = delaysdf_updated['direction'] == 'N'
delaysdf_updated['category'] = delaysdf_updated['north'] * 2**0 + delaysdf_updated['weekday'] * 2**1
 
delaysdf_updated = pd.concat(res)
 
utils.write(delaysdf_updated, 'tdi_delaysdf.pkl')