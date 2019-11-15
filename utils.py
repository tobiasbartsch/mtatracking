import pickle
import sys
sys.path.append('/home/tbartsch/source/repos')
from mtatracking.MTAdatamodel import SubwaySystem
from mtatracking.MTASubwayAnalyzer import MTASubwayAnalyzer
import os as os
from os.path import dirname
from os import walk
from collections import defaultdict
from datetime import datetime
import pytz

class utils(object):
    """Write or read python objects to/from the harddisk."""

    @staticmethod
    def write(obj, filename):
        with open(filename, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)   
        return True           
    
    @staticmethod
    def read(filename):
            with open(filename, 'rb') as input:
                x = pickle.load(input)
            return x

    @staticmethod
    def attach_all_tracking(mySubwaySystem, mypath):
        f = []
        print(mypath)
        for (dirpath, dirnames, filenames) in walk(mypath):
            f.extend(filenames)
            break
        f_stamps = [1, 2]
        for s in f:
            timestamp = s[16:]
            timestamp = timestamp[:-4]
            f_stamps.append(int(timestamp))
        f_stamps.sort()
        #we now have a sorted list of all the time stamps of all the tracking data. Now we can attach to the subway system
        print("Number of time stamps", len(f_stamps))
        for counter, stamp in enumerate(f_stamps):
            if(counter % 5000 == 0):
                print(counter)
            if len(str(stamp)) == 10:
                n = 'tracking_results' + str(stamp) + '.pkl'
                myfile = os.path.join(mypath, n)
                try:
                    data = utils.read(myfile)
                    mySubwaySystem.attach_tracking_data(data, stamp)
                except:
                    print("Error loading file" + str(stamp))
                    pass
                
    @staticmethod
    def findAvgArrTimeDeltasForAllLines(mySubwaySystem, bin_width, timestamp_start, timestamp_end):
        """Compute the mean arrival time deltas between successive trains for each station along each line in the system.
        Data are returned as a function of "seconds past midnight"; the time axis is binned into bins of preset width. 
        This function writes the returned results to the working directory.
        
        Args: 
            mySybwaySystem (SubwaySystem): SubwaySystem containing the tracking data to be averaged.
            bin_width (int): Bin width of the time axis in seconds.
            timestamp_start (int): timestamp at which to start averaging. If you pass a timestamp that is earlier than when the data begins, we will start averaging with the first timestamp in the dataset.
            timestamp_end (int): timestamp at which to end averaging.

        Returns:
            (AvgArrTimeDeltasForAllLines, WeekendAvgArrTimeDeltasForAllLines) (dict, dict): keys: line_id; value: np.array, rows: time axis, cols: stations. The values are [meanTimeDelta, sdevTimeDelta, observedCounts]
        """
        
        AvgArrTimeDeltasForAllLines = defaultdict(object)
        WeekendAvgArrTimeDeltasForAllLines = defaultdict(object)
        analyzer = MTASubwayAnalyzer(mySubwaySystem)
        for line_id in list(mySubwaySystem.lines.keys()):
            (AvgArrTimeDeltasForAllLines[line_id], WeekendAvgArrTimeDeltasForAllLines[line_id]) = analyzer.AverageArrivalDelta_v2(bin_width, line_id, timestamp_start, timestamp_end)
        
        utils.write(AvgArrTimeDeltasForAllLines, 'AvgArrTimeDeltasForAllLines.pkl')
        utils.write(WeekendAvgArrTimeDeltasForAllLines, 'WeekendAvgArrTimeDeltasForAllLines.pkl')

        return (AvgArrTimeDeltasForAllLines, WeekendAvgArrTimeDeltasForAllLines)

    @staticmethod
    def timestamp_to_day_time(timestamp, tzone='US/Eastern'):
        '''Convert a time stamp to a tupel of (weekday, seconds_since_midnight).
        You can set a timezone, but by default this function assumes that we are in US/Eastern.
        
        Args:
            timestamp (int): A Unix timestamp
            tzone (string): timezone string for the pytz package
        '''
        est = pytz.timezone(tzone)
        utc = pytz.utc
        dt = datetime.utcfromtimestamp(timestamp)
        dt = utc.localize(dt)
        dt = dt.astimezone(est)

        day = dt.weekday() #0: Monday
        #Find seconds past midnight
        seconds_since_midnight = (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

        return (day, seconds_since_midnight)

    @staticmethod
    def timestamp_to_string(timestamp, tzone='US/Eastern'):
        '''Convert a time stamp to a human readable string
        You can set a timezone, but by default this function assumes that we are in US/Eastern.
        
        Args:
            timestamp (int): A Unix timestamp
            tzone (string): timezone string for the pytz package        
        
        Returns:
            timestring (string): human readable date time.
        '''
        est = pytz.timezone(tzone)
        utc = pytz.utc
        dt = datetime.utcfromtimestamp(timestamp)
        dt = utc.localize(dt)
        dt = dt.astimezone(est)

        return str(dt)