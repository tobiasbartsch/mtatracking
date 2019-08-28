import sys
sys.path.append("/home/tbartsch/source/repos")

from mtatracking.MTAdatamodel import SubwaySystem
import math
import numpy as np
from datetime import datetime
import pytz
from scipy.optimize import leastsq
from scipy.stats import norm
from scipy.integrate import quad
from scipy.special import gamma
import pymc3 as pm

from collections import defaultdict
import xarray as xr
import warnings


import algorithms
import algorithms.STaSI as st
import algorithms.HaarWavelet as hw

class delayOfTrainsInLine_Bayes(object):
    """Determine the delay of trains on a subway line."""

    def __init__(self, line_id, analyzer, n=3):
        """create a delayOfTrainsInLine_Bayes instance.
        Args:
            line_id (string): id of this subway line. For northbound Q trains, for example, pass 'QN'.
            analyzer (MTASubwayAnalyzer): analyzer object of historic subway data (ideally with data up to the present day/time)
            n (float): number of sdevs that define a threshold beyond which we treat a train as delayed (delay for t > n).
        """
        self.line_id = line_id
        self.means = dict() #mean transit time between adjacent stations. keys: pair of stations ids, e.g. 'R30N to Q01N'
        self.sdevs = dict() #sdev of transit time between adjacent stations. keys: pair of stations ids, e.g. 'R30N to Q01N'
        self.n = n
        self.analyzer = analyzer

        #initialize the delay probability along this line
        self.delayProbs = {key: 0.0 for key in self.means.keys()}

        '''def preComputeMeanAndSdev(self, segs, timestamp_start, timestamp_end):
        """Compute the means and standard deviations of transit times along the segments defined in 'segs'.
        Store the results in self.means and self.sdevs

        Args:
            segs (list of string): list of pairs of station ids, ['R30N to Q01N', ... ]. Ids must correspond to stations along the given line (self.line_id)
            timestamp_start, timestamp_end: historic time stamps for computation of the mean and sdev
        """

        for seg in segs:
            print(seg)
            ids = seg.split()
            
            self.means[seg], self.sdevs[seg]= self._getCurrentTravelTimeMeanAndSdev(ids[0], ids[2])
        '''
            

    def changeThreshold(self, n):
        """Change the definition of a 'delay': a train has to be n standard deviations late to be considered 'delayed'.
        Args:
            n (float): number of sdevs that define a threshold beyond which we treat a train as delayed (delay for t > n).
        """
        self.n = n
    
    def updateDelayProbs(self, trains_in_line, timestamp, timestamp_start = 0, timestamp_end = 999999999999):
        """Calculate the probability that a delay occurs between a pair of stations based on the wait time for a train.
        
        Args:
            trains_in_line (list of mtatracking.MTAdatamodel.SubwayTrain): the realtime train objects that are currently traversing this subway line.
            timestamp (int): the timestamp at which the current trains were observed.
            timestamp_start, timestamp_end: historic time stamps for computation of the mean and sdev
        """
        self.delayProbs = dict()
        for train in trains_in_line:
            origin_id = train.departure_station_id
            if(origin_id is None):
                continue
            destination_id = train.arrival_station_id
            wait_time = timestamp - train.departed_at_time
            key = origin_id + ' to ' + destination_id
            if(key not in self.means.keys()): #check whether the current train is travelling between two known stations.
                #get the statistics for this segment:
                self.means[key], self.sdevs[key]= self._getCurrentTravelTimeMeanAndSdev(origin_id, destination_id, timestamp_start, timestamp_end)
            
            self.delayProbs[key] = self._probability_of_delay(wait_time, self.means[key], self.sdevs[key], self.n)

    def _rho_t_given_H(self, t, n, delayed):
        '''compute the probability density of a transit time t given a train delay. A train counts as delayed if it arrives with a delay of at least n sdevs higher than the mean
        
        Args:
            t (float): time expressed in multiples of the standard deviation and with zero mean.
            n (float): number of sdevs that define a threshold beyond which we treat a train as delayed (delay for t > n).
            delayed (boolean): the hypothesis: is the train delayed? True/False
            
        Returns:
            rho (float): probability density
        '''
        
        if(t > n):
            if(delayed):
                return norm.pdf(t)/(1-norm.cdf(n))
            else:
                return 0
        else:
            if(delayed):
                return 0
            else:
                return norm.pdf(t)/norm.cdf(n)

    def _likelihood(self, t0, n, delayed):
        '''compute the likelihood of observing t>t0 given that the current train is (not) delayed.
        
        Args:
            t0 (float): current wait time for the inbound train (expressed in multiples of sdev and with zero mean).
            n (float): number of sdevs that define a threshold beyond which we treat a train as delayed (delay for t > n).
            delayed (boolean): the hypothesis: is the train delayed? True/False
            
        Returns:
            likelihood (float)
        '''
        
        like = delayed * (1/(1-norm.cdf(n)) * (1-norm.cdf(np.max([n, t0])))) + (1-delayed) * (1/norm.cdf(n) * (norm.cdf(np.max([t0,n])) - norm.cdf(t0)))
        
        return like
        
    def _p_H(self, n, delayed):
        '''compute the prior.
        
        Args:
            n (float): number of sdevs that define a threshold beyond which we treat a train as delayed (delay for t > n).
            delayed (boolean): the hypothesis: is the train delayed? True/False
        '''
        if(delayed):
            return 1-norm.cdf(n)
        else:
            return norm.cdf(n)

    def _probability_of_delay(self, t0, mu, sigma, n):
        '''compute the posterior probability: what is the predicted delay probability after we have waited a time t0 for a train.
        
        Args:
            t0 (float): time in seconds since the train left the origin station 
            mu (float): average transit time (in seconds) between the stations
            sigma (float): standard devition (in seconds) of the transit time between stations
            n (float): number of sdevs that define a threshold beyond which we treat a train as delayed (delay for (t_arrival-mu)/sdev > n).
        '''
        if(mu is None or sigma is None): #if something went wrong in computing mu or sigma we cannot compute a probability.
            return 0 #TODO handle this better.
        #normalize t0
        t0 = (t0-mu)/sigma
        
        return self._p_H(n, delayed=True) * self._likelihood(t0, n, delayed=True) / (self._p_H(n, delayed=True) * self._likelihood(t0, n, delayed=True) + self._p_H(n, delayed=False) * self._likelihood(t0, n, delayed=False))

    def _getCurrentTravelTimeMeanAndSdev(self, origin_id, destination_id, timestamp_start = 0, timestamp_end = 99999999999999):
        '''find the current travel time mean and sdev between two stations of this line using the historic data in the subway analyzer model.
        If the historic data does not contain any trains (of the given line) that tarvelled between these stations, return None.

        Args:
            line_id (string): id of the subway line, for example 'QN' for northbound Q trains
            origin_id (string): id of the origin station
            destination_id (string): id of the destination station
            timestamp_start (int): timestamp of the historic data at which to start the computation
            timestamp_end (int): timestamp of the historic data at which to stop the computation

        Returns:
            (mean, sdev)
        '''
        #get the current mean and sdev from STaSI
        data, _, result, _, _, _ = self.analyzer.AverageTransitTimeBetweenTwoStations_STaSI(line_id=self.line_id, station_id_d=destination_id, station_id_o=origin_id, timestamp_end=timestamp_end, timestamp_start=timestamp_start)
        
        if(data is None or result is None):
            return (None, None)
        
        state = int(result[-1:]['state']) #find the current state for the given pair of stations
        #key = key = origin_id + ' to ' + destination_id
        
        #get the distribution of data for the current state.
        _, means, sdevs = self.analyzer.getStatsForStates(data, result, normalized=False, statenum=state)
        
        if str(state) not in means.keys():
            return None, None
        return (means[str(state)], sdevs[str(state)])
        
        

class MTASubwayAnalyzer(object):
    """Provides methods and properties to analyze the data contained in a SubwaySystem object.
    In order to improve code reusability try to have logic happen here instead of the ViewModel layer (and certainly not in the view layer).
    """

    def __init__(self, mySubwaySystem):
        self.subwaysys = mySubwaySystem
        
    def AverageArrivalDelta_v2(self, windowsize_seconds, line_id, timestamp_start, timestamp_end):
        """Return the average delta arrival times for each station in a particular line vs time of a day. 
        Args:
            windowsize_seconds (int): Windows over which the data is blocked. Blocks are then averaged over all (appropriate) days in the dataset.
                                Windows should have an appropriate size, larger the expected delta in subway arrival time.
            line_id (string): ID of the line for which the average is computed (e.g. for northbound Q: 'QN')
            timestamp_start (int): Unix timestamp at which to start the averaging. If this timestamp is before tracking began we will start at the first time stamp in the Subway System.
            timestamp_end (int): Unix timestamp at which to end the averaging. If this timestamp is after tracking ended, we will average until the end of tracking.
        
        Returns:
            (DeltaArrTimes_weekday, DeltaArrTimes_weekend): tupel of numpy matrices, columns: stations, rows: time axis in steps of windowsize. Each entry is a list of [mean arrival time diff, sdev, number of observations]
        """
        if (timestamp_start < self.subwaysys.timestamp_startTracking):
            timestamp_start = self.subwaysys.timestamp_startTracking
        if (timestamp_end > self.subwaysys.timestamp_endTracking):
            timestamp_end = self.subwaysys.timestamp_endTracking

        #New plan: for every station and every window generate a lits of all observed delays. Then make a histogram and fit a Gauss.
        numblocks = math.floor(86400.0 /windowsize_seconds)
        numstations = len(self.subwaysys.lines[line_id].stations)
        
        DeltaArrTimes_weekday = np.empty((numblocks, numstations), dtype=object)
        DeltaArrTimes_weekend = np.empty((numblocks, numstations), dtype=object)
        #Initialize the empty np arrays with lists to which we can then append
        for index, o in np.ndenumerate(DeltaArrTimes_weekday):
            DeltaArrTimes_weekday[index] = []
        for index, o in np.ndenumerate(DeltaArrTimes_weekend):
            DeltaArrTimes_weekend[index] = []

        #iterate through all stations in the line
        for counter, station in enumerate(self.subwaysys.lines[line_id].stations):
            #get timestamps of all trains that stopped here
            stop_times_list = list(station.trains_stopped[line_id].values()) #Note: we need python 3.7 or better to guarantee dictionary order. https://mail.python.org/pipermail/python-dev/2017-December/151283.html
            #the stop times are Unix timestamps. We need to do two things:
            #1) figure out whether we are on a weekday or weekend
            #2) Stick each deltaArrivalTime into an appropriate block of a day
            #iterate through all time steps in the subway system
            for stoptime in stop_times_list: #iterating this way avoids oversampling but requires either huge amounts of tracked days or may result in windows with zero observed trains.
                if stoptime > timestamp_start and stoptime < timestamp_end:
                    est = pytz.timezone('US/Eastern')
                    utc = pytz.utc
                    #self.displaydatetime = datetime.utcfromtimestamp(value).astimezone(est).strftime('%Y-%m-%d %H:%M:%S')
                    dt = datetime.utcfromtimestamp(stoptime)
                    dt = utc.localize(dt)
                    dt = dt.astimezone(est)
                    #Return the day of the week as an integer, where Monday is 0 and Sunday is 6.               
                    day = dt.weekday()
                    #Find seconds past midnight
                    seconds_since_midnight = (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
                    block = math.floor(float(seconds_since_midnight) / windowsize_seconds)
                    self.subwaysys.display_timestamp = stoptime
                    if(day < 5): 
                        #weekday
                        DeltaArrTimes_weekday[block,counter].append(station.deltaArrivalTime[line_id])
                    else:
                        #weekend
                        DeltaArrTimes_weekend[block,counter].append(station.deltaArrivalTime[line_id])
        
        # #make histogram and gauss fit for each station and each window
        fitfunc  = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)
        errfunc  = lambda p, x, y: (y - fitfunc(p, x))
        init  = [10, 500, 10] #initialize amplitude, mean, and sdev
        binsize = 30
        for DeltaArrTimes in [DeltaArrTimes_weekday, DeltaArrTimes_weekend]:
            for index, data in np.ndenumerate(DeltaArrTimes): #make histograms with 60 s bin width. Allow for data up to 3600 seconds.
                if len(data) > 4: #we want at least 4 observations
                    (counts, deltaT) = np.histogram(data, bins=math.floor(3600/binsize), range=(0, 3600))
                    deltaT = deltaT[0:-1] + binsize/2 #convert bin edges into bin centers
                    init = [np.max(counts),np.mean(data), np.sqrt(np.var(data))]
                    out = leastsq(errfunc, init, args=(deltaT, counts))
                    c = out[0]
                    DeltaArrTimes[index]= [c[1], c[2], np.sum(counts)]
                else:
                    DeltaArrTimes[index]= [None, None, np.sum(counts)]
        
        
        #Reshape this to make it easier to address the data:
        DeltaArrTimes_weekday_dict = defaultdict(list)
        DeltaArrTimes_weekend_dict = defaultdict(list)        
        for counter, station in enumerate(self.subwaysys.lines[line_id].stations):
            DeltaArrTimes_weekday_dict[station.id] = list(DeltaArrTimes_weekday[:,counter])
            DeltaArrTimes_weekend_dict[station.id] = list(DeltaArrTimes_weekend[:,counter])
            
        return (DeltaArrTimes_weekday_dict, DeltaArrTimes_weekend_dict)

    def AverageTransitTimeBetweenTwoStations_STaSI(self, line_id, station_id_o, station_id_d, timestamp_start, timestamp_end):
        """Return the average travel times (determined by the STaSI model) between a pair of stations in a particular line vs time.
            Args:
                line_id (string): ID of the line for which the average is computed (e.g. for northbound Q: 'QN')
                station_id_o (string): ID of the origin station
                station_id_d (string): ID of the destination station
                timestamp_start (int): Unix timestamp at which to start the averaging. If this timestamp is before tracking began we will start at the first time stamp in the Subway System.
                timestamp_end (int): Unix timestamp at which to end the averaging. If this timestamp is after tracking ended, we will average until the end of tracking.
            
            Returns:
                (data, fitArray, results, fit_hist, data_hist, MDLs):
                                                                        data (xr.DataArray): the time series
                                                                        fitArray (xr.DataArray): the fit of the time series. Note that len(fitArray) may be less than len(data) as any overly extreme points are removed from data prior to the fit
                                                                        results (pd.DataFrame): contains means of segments, assigned states, and start and stop timestamps.
                                                                        fit_hist (np.histogram): histogram of the fit time series, bin size 1 second
                                                                        data_hist (np.histogram): histogram of the data, 100 bins, flexible bin size to accommodate the data
                                                                        MDLs (np.array): minimum description lengths for the various models. The fit and results reflect the model with the minimum description length.                                                          
        """

        if (timestamp_start < self.subwaysys.timestamp_startTracking):
            timestamp_start = self.subwaysys.timestamp_startTracking
        if (timestamp_end > self.subwaysys.timestamp_endTracking):
            timestamp_end = self.subwaysys.timestamp_endTracking
        
        trains_origin = self.subwaysys.stations[station_id_o].trains_stopped[line_id]
        trains_destination = self.subwaysys.stations[station_id_d].trains_stopped[line_id]

        est = pytz.timezone('US/Eastern')
        utc = pytz.utc


        #iterate through all trains that left the origin station and check how long it took them to get to the destination station
        transitTimeSeries = []
        transitTimeSeries_coords = []
        transitTimeSeries_coords_stamps = []
        for train_id in trains_origin:
            if(train_id in trains_destination): #TODO: determine whether tracking was fast enough! If we sampled in a 10 min interval, all trains will look like they spent 10 min travelling.
                departure_time = trains_origin[train_id]
                arrival_time = trains_destination[train_id]
                transit_time = arrival_time - departure_time
                if (departure_time > timestamp_start and departure_time < timestamp_end):
                    transitTimeSeries.append(transit_time)
                    dt = datetime.utcfromtimestamp(departure_time)
                    dt = utc.localize(dt)
                    dt = dt.astimezone(est)
                    transitTimeSeries_coords.append(dt)
                    transitTimeSeries_coords_stamps.append(departure_time)
            else:
                #print('Warning: train did not arrive at destination station :', train_id)
                arrival_time = np.nan
                transit_time = np.nan
                continue
        
        w1s = hw.w1(transitTimeSeries)
        sigma = hw.sdevFromW1(w1s)

        mask = np.full(len(transitTimeSeries), True)
        mean = np.mean(transitTimeSeries)
        mask[np.where(np.abs(transitTimeSeries) - mean > 10 * sigma)] = False

        tTimes_filtered = np.copy(np.asarray(transitTimeSeries)[mask])
        tTimes_coords_filtered = np.copy(np.asarray(transitTimeSeries_coords)[mask])

        if len(tTimes_filtered) == 0:
            return None, None, None, None, None, None
        
        fit, means, results, MDLs = st.fitSTaSIModel(tTimes_filtered)

        tTimes_coords_filtered = np.array(tTimes_coords_filtered, dtype=object)
        fitArray = xr.DataArray(fit,
                                dims = ['time'],
                                coords = {'time': tTimes_coords_filtered},
                                name = 'transit time fit (s)')
        data = xr.DataArray(transitTimeSeries,
                            dims = ['time'],
                            coords = {'time': transitTimeSeries_coords},
                            name = 'transit time (s)')
        
        #currently the results dataframe contains indices; make those into time stamps.
        start_stamps = np.asarray(transitTimeSeries_coords_stamps)[results['start'].values]
        stop_stamps = np.asarray(transitTimeSeries_coords_stamps)[results['stop'].values]
        results['start_timestamps'] = start_stamps
        results['stop_timestamps'] = stop_stamps
        results = results.drop('start', axis = 1)
        results = results.drop('stop', axis = 1)

        #compute a histogram of the fitted and raw data 
        hist_range = (int(mean - 4 * sigma), int(mean + 4 * sigma))
        nbins = int(8*sigma)
        print(nbins)
        if nbins < 1:
            nbins = 1
        fit_hist = np.histogram(fit, bins=nbins, range = hist_range)
        data_hist = np.histogram(transitTimeSeries, bins=100, range = hist_range)

        return data, fitArray, results, fit_hist, data_hist, MDLs

    def getStatsForStates(self, data, results, normalized, statenum=None):
        """Return the distribution of data for each state found by the STaSI algorithm.

        Args:
            data (xr.DataArray): the time series
            results (pd.DataFrame): output of AverageTransitTimeBetweenTwoStations_STaSI 
            normalized (boolean): if true: return normalized distributions
            statenum (int): if statenum is not None: process only the n-th state. Otherwise process all states.
        Returns:
            (dists, means, sdevs) 
                dists: (dict((bins, vals)): dictionary of probability distributions
                means: (dict(vals)): dictionary of means
                sdevs: (dict(vals)): dictionary of standard deviations
        """

        #get the maximum number of states
        numstates = np.max(results['state'])
        data_in_state = dict()
        utc = pytz.utc
        est = pytz.timezone('US/Eastern')
        for index, segment in results.iterrows():
            state = segment['state']
            start_timestamp = segment['start_timestamps']
            stop_timestamp = segment['stop_timestamps']
            startdate = utc.localize(datetime.utcfromtimestamp(start_timestamp)).astimezone(est)
            stopdate = utc.localize(datetime.utcfromtimestamp(stop_timestamp)).astimezone(est)
            try: 
                dataseg = data.loc[startdate:stopdate]
            except:
                continue #TODO: fix this properly! somehow under rare circumstances indexing of dataseg does not work
            if str(int(state)) in data_in_state:
                data_in_state[str(int(state))] =  np.concatenate((data_in_state[str(int(state))], dataseg))
            else:
                data_in_state[str(int(state))] = dataseg
        dists = dict()
        means = dict()
        sdevs = dict()
        for k, d in data_in_state.items():
            if(statenum is None or int(k)==statenum):
                hist = np.histogram(d, bins=2000, range=(0,60000))
                vals = hist[0]
                bins = hist[1][:-1] + hist[1][0]/2
                norm = np.trapz(x = bins, y = vals)
                vals = vals/norm
                dists[k] = (bins, vals)

                # #model this as gaussian and find mean and sdev by mcmc
                # with pm.Model() as model:
                #     mean = pm.Normal(name = 'mean', mu=np.mean(d), sd=np.std(d))
                #     sdev = pm.Normal(name = 'sdev', mu=np.std(d), sd=np.std(d))
                #     obs = pm.Normal(name = 'obs', mu=mean, sd=sdev, observed = d)
                #     trace = pm.sample(500, tune=500, nuts_kwargs=dict(target_accept=0.90), cores=16)
                
                # means[k] = float(pm.summary(trace)[0:-1]['mean'])
                # sdevs[k] = float(pm.summary(trace)[1:]['mean'])

                means[k] = np.mean(d)
                sdevs[k] = np.std(d)

        return dists, means, sdevs


    def timestamp_to_day_time(self, timestamp, tzone='US/Eastern'):
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


