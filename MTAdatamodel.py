import csv
import sys
import warnings
from bisect import bisect_left
import datetime
import time
import pandas as pd
import mtatracking.nyct_subway_pb2 as nyct_subway_pb2
from collections import defaultdict
import numpy as np

from importlib.resources import path
from . import resources

class SubwaySystem(object):
    """A subway system consists of stations, lines, and trains.
    The status of the system is viewable in real time and for past times for which the system was tracked.
    Trains do not always follow their pre-defined lines.
    """

    def __init__(self):
        """Set up the subway system. Initialize station and line objects."""

        assert sys.version_info >= (3, 7) # we want python 3.7 to guarantee that our dict entries are ordered.

        #Get a dictionary of all stations in the system
        self.stationsdict = MTAstaticdata.ImportStationsDictionary()
        self.stations = {id:SubwayStation(id, name, self) for (id, name) in self.stationsdict.items()} #makes a dictionary of station objects with ids as keys.
        self.trains = defaultdict(SubwayTrain) #current trains in the system at most recent time stamp
        self.train_history = defaultdict(defaultdict) #history of trains. outer keys: timestamps
        self.line_defs = MTAstaticdata.ImportLineDefinitions()
        self.lines = {id:SubwayLine(id, self.line_defs, self.stations) for id in list(self.line_defs)}
        self.display_timestamp = 0 #This is the timestamp for which we would like to display data. Trains, stations, and lines in the system have propeties that depend on this value. For example, station.deltaArrivalTime is the time difference between trains at time display_timestamp
        #Todo line objects
        self.last_attached_file_timestamp = np.nan # we need this to make sure we do not have unreasonably long gaps in between files during tracking
    
    def reset(self):
        """Reset stations and erase trains in the subway system, rebuild line objects, erase all tracking data."""

        self.stations = None
        self.trains = None
        self.train_history = None
        self.lines = None
        self.stations = {id:SubwayStation(id, name, self) for (id, name) in self.stationsdict.items()} #makes a dictionary of station objects with ids as keys.
        self.trains = defaultdict(SubwayTrain)
        self.train_history = defaultdict(defaultdict)
        self.lines = {id:SubwayLine(id, self.line_defs, self.stations) for id in list(self.line_defs)}

        self.last_attached_file_timestamp = None

    def stationID_to_stationName(self, id):
        return self.stations[id].name

    def ids_to_names(self, ids):
        '''change strings like 'R30N to Q01N' to 'DeKalb Av to Canal St' '''
        ids = ids.split()
        return self.stationID_to_stationName(ids[0]) + " to " + self.stationID_to_stationName(ids[2])
        

    def attach_tracking_data(self, data, currenttime):
        """Process the protocol buffer feed and populate our subway model with its data.
        
        Args: 
            data: List of protocol buffer messages containing trip_update or vehicle feed entities (presumably downloaded from the MTA realtime stream). One message per requested feed.
        """
        current_trains = list()
        for message in data:
            current_time = message.header.timestamp #try to get the time stamp from the feed rather than from the filename
            for FeedEntity in message.entity:
                if len(FeedEntity.trip_update.trip.trip_id) > 0: #entity type "trip_update"
                    self._processTripUpdate(FeedEntity, current_trains, current_time)
                if len(FeedEntity.vehicle.trip.trip_id) > 0: #entity type "vehicle"
                    self._processVehicleMessage(FeedEntity, current_trains, current_time)
 
        #delete all trains in the subway system that are not in the new feed. 
        #trains that are not in the new feed have arrived at their final station and we need to update their status accordingly
        all_trains = list(self.trains.keys())
        terminated_trains = set(all_trains) - set(current_trains)
        self.RemoveTrains(list(terminated_trains), current_time)
        #attach trains to train history
        self.train_history[current_time] = self.trains

        self.last_attached_file_timestamp = currenttime

    def _processTripUpdate(self, FeedEntity, current_trains, current_time):
        """Add data contained in the Protobuffer's Trip Update FeedEntity to the subway system.
        
        Args:
            FeedEntity: TripUpdate FeedEntity (from protobuffer).
            current_trains (list): a list of the train_ids of the trains in the current feed. This function appends additional trains found in the FeedEntity.
            current_time (timestamp): Timestamp in seconds since 1970
        """

        train_id = FeedEntity.trip_update.trip.Extensions[nyct_subway_pb2.nyct_trip_descriptor].train_id #a unique number for each train. Check whether we are already tracking this train
        current_trains.append(train_id)
        trip_id = FeedEntity.trip_update.trip.trip_id
        route_id = FeedEntity.trip_update.trip.route_id
        direction = self.direction_to_str(FeedEntity.trip_update.trip.Extensions[nyct_subway_pb2.nyct_trip_descriptor].direction)
        is_assigned = FeedEntity.trip_update.trip.Extensions[nyct_subway_pb2.nyct_trip_descriptor].is_assigned
        start_date = FeedEntity.trip_update.trip.start_date                    

        if len(FeedEntity.trip_update.stop_time_update) > 0:
            next_arrival_time = FeedEntity.trip_update.stop_time_update[0].arrival.time
            next_stop = FeedEntity.trip_update.stop_time_update[0].stop_id
            next_scheduled_track = FeedEntity.trip_update.stop_time_update[0].Extensions[nyct_subway_pb2.nyct_stop_time_update].scheduled_track
            next_actual_track = FeedEntity.trip_update.stop_time_update[0].Extensions[nyct_subway_pb2.nyct_stop_time_update].actual_track
                
            #Add to our tracker
            if train_id not in self.trains: #create a new train
                self.trains[train_id] = SubwayTrain(train_id,trip_id,route_id,direction,is_assigned,start_date,current_time, self.stations, self)
            #update tracking info for the current train
            self.trains[train_id].trip_id = trip_id
            self.trains[train_id].route_id = route_id
            self.trains[train_id].direction = self.direction_to_str(direction)
            self.trains[train_id].is_assigned = (is_assigned, current_time)
            self.trains[train_id].start_date = start_date

            self.trains[train_id].arrival_time = next_arrival_time
            self.trains[train_id].arrival_station_id = (next_stop, current_time) #replace current time with proper timestamp if available.
            self.trains[train_id].scheduled_track = next_scheduled_track
            self.trains[train_id].actual_track = next_actual_track

    def _processVehicleMessage(self, FeedEntity, current_trains, current_time):
        """Add data contained in the Protobuffer's VehicleMessage FeedEntity to the subway system.
        
        Args:
            FeedEntity: VehicleMessage FeedEntity (from protobuffer).
            current_trains (list): a list of the train_ids of the trains in the current feed. This function appends additional trains found in the FeedEntity.
            current_time (timestamp): timestamp in seconds since 1970
        """
        train_id = FeedEntity.vehicle.trip.Extensions[nyct_subway_pb2.nyct_trip_descriptor].train_id
        if train_id not in self.trains: #this is a vehicle message and there really should already be a train. However, if there is none, let's make one
            current_trains.append(train_id)
            trip_id = FeedEntity.vehicle.trip.trip_id
            route_id = FeedEntity.vehicle.trip.route_id
            direction = self.direction_to_str(FeedEntity.vehicle.trip.Extensions[nyct_subway_pb2.nyct_trip_descriptor].direction)
            is_assigned = FeedEntity.vehicle.trip.Extensions[nyct_subway_pb2.nyct_trip_descriptor].is_assigned
            start_date = FeedEntity.vehicle.trip.start_date
            self.trains[train_id] = SubwayTrain(train_id,trip_id,route_id,direction,is_assigned,start_date,current_time, self.stations, self)
                    
        #update vehicle info for the current train
        self.trains[train_id].current_stop_sequence = FeedEntity.vehicle.current_stop_sequence
        self.trains[train_id].status = (FeedEntity.vehicle.current_status, FeedEntity.vehicle.timestamp, FeedEntity.vehicle.stop_id)

    def direction_to_str(self, direction):
        """convert a direction number (1, 2, 3, 4) to a string (N, S, E, W)
        """
        if(direction == 1):
            return 'N'
        elif(direction == 2):
            return 'E'
        elif(direction == 3):
            return 'S'
        elif(direction == 4):
            return 'W'
        else:
            return direction

    @property
    def timestamp_startTracking(self):
        """return the timestamp of the first tracked tick in this Subway System dataset"""
        return list(self.train_history.keys())[0]

    @property
    def timestamp_endTracking(self):
        """return the timestamp of the last tracked tick in this Subway System dataset"""
        return list(self.train_history.keys())[-1]

    @property
    def NumberOfTrains(self):
        """The number of trains in the subway system."""
        return len(self.trains)
    
    def RemoveTrains(self, train_ids, current_time):
        """Remove trains from the subway system

        Args:
            train_ids (list): List of IDs of the trains to remove.
            current_time (timestamp): timestamp in seconds since 1970
        """
        #print("Removed trains:" )
        for train_id in train_ids:
            if train_id in self.trains:
                if self.trains[train_id].arrival_station_id not in self.stations:
                    self.stations[self.trains[train_id].arrival_station_id]= SubwayStation(self.trains[train_id].arrival_station_id, 'UNKNOWN', self)
                #print(str(self.trains[train_id].direction) + " bound " + str(self.trains[train_id].route_id) + " (" + str(self.trains[train_id].trip_id) + ") terminated at station " + self.stations[self.trains[train_id].arrival_station_id].name + " (" + str(train_id) + ")")
                self.trains[train_id].arrival_station_id = ('EndOfService', current_time) #this ensures that the arrival of the train at its last station is registered.
                del self.trains[train_id]

                
class SubwayLine(object):
    """A subway line."""

    def __init__(self, line_id, line_defs, stationdict):
        """Set up a subway line.

        Args:
            line_id (string): Id of the subway line. Include north or south-bound designation, e.g. for north bound Q : 'QN'
            line_defs (panda data frame): data frame of line definitions. Each column contains station information
            stationdict (dictionary): all subway stations in the subway system.
        """
        self.id = line_id
        self.stations = [stationdict[str(id)] for id in line_defs[line_id]]
        

class SubwayStation(object):
    """A subway station."""

    def __init__(self, station_id, station_name, this_subway_sys):
        """Set up a subway station, 
        
        Args:
            station_id (string): MTA ID of the station
            station_name (string): Human-readable name of the station
            this_subway_sys (SubwaySystem): reference to the Subway System that this station belongs to.
        """
        self.currenttime = 0
        self.id = station_id
        self.name = station_name
        self.trains_stopped = defaultdict(type(defaultdict(str))) #outer keys: line and direction, e.g. key: 'QN' is the northbound Q line. inner keys: train IDs, values: timestamps of stops
        self.subwaysys = this_subway_sys

    def registerStoppedTrain(self, train_id, route_id, direction, timestamp):
        """Register a stopped train with this station.
            
        Args:
            train_id: ID of the stopped train
            route_id: Route ID of the stopped train
            direction: direction the stopped train is travelling in
            timestamp: timestamp in seconds since 1970
        """
        line_dir = str(route_id) + str(direction) #direction is an enum: 1: N, 3: S. 2 and 4 are E and W, but those are not implemented in the NYC subway.
        self.trains_stopped[line_dir][train_id] = timestamp

    def _DeltaArrivalTime(self, route_id, direction, timestamp):
        """Time between two successive train arrivals (for a particular route and direction). Timestamp is the time in between the two arrivals (if possible this function
        will choose one train before and one arriving after timestamp). For time stamps in the future, this function returns the arrival time distance of the last two
        observed trains. For time stamps in the distant past (before observations started), the arrival time difference between the first two observed trains is returned.

        If route_id and/or direction results in fewer than two observed trains, this function raises a warning. The function returns 0 in this case.

        This function is useful to estimate delays in the SubwaySystem.

        Args:
            route_id: The route id. DO NOT INCLUDE DIRECTION, i.e. for Q trains just pass 'Q'.
            direction: direction is an enum: 1: N, 3: S. 2 and 4 are E and W, but those are not implemented in the NYC subway.
            timestamp: timestamp for which to compute DeltaArrivalTime

        Returns:
            dArrivalTime (int): arrival time difference in seconds.
        """
        line_dir = str(route_id) + str(direction)
        stop_times_list = list(self.trains_stopped[line_dir].values()) #Note: we need python 3.7 or better to guarantee dictionary order. https://mail.python.org/pipermail/python-dev/2017-December/151283.html
        position = bisect_left(stop_times_list, timestamp) #doing this with bisect_left only takes O(log(n)), see here: https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
        if(len(stop_times_list) > 1):
            if(position == 0):
                warnings.warn("timestamp refers to time before observations began")
                return stop_times_list[1]-stop_times_list[0]
            elif(position == len(stop_times_list)):
                warnings.warn("timestamp refers to time after observations ended")
                return stop_times_list[-1]-stop_times_list[-2]
            else:
                return stop_times_list[position]-stop_times_list[position-1]
        else:
            warnings.warn("line " + line_dir + ": only " + str(len(stop_times_list)) + " trains stopped here. Cannot compute delay.")
            return None        
    @property
    def deltaArrivalTime(self):
        """Computes the time in seconds between different train arrivals for each line visiting this station. The returned value is that for the time point specified by the display_timestamp of the subway system. 
            Returns: dictionary of line_id:time_difference pairs.
        """

        #find out which lines visited this station
        lines = list(self.trains_stopped.keys())
        #deltas = {id:self._DeltaArrivalTime(id[:-1], id[-1:], self.subwaysys.display_timestamp) for id in lines}
        deltas = {id:self._DeltaArrivalTime(id[:-1], id[-1:], self.subwaysys.display_timestamp) for id in lines}
        
        return deltas

class SubwayTrain(object):
    """A subway train."""

    def __init__(self, train_unique_num, trip_id, route_id, direction, is_assigned, start_date, current_time, stations, thisSubwaySystem):
        self._unique_num = train_unique_num
        self._trip_id = trip_id
        self._route_id = route_id
        self._direction = direction
        self._is_assigned = is_assigned
        self.start_date = start_date
        self._time = current_time
        self.stations = stations
        self.subwaysys = thisSubwaySystem

        self._arrival_time = None
        self._arrival_station_id = None
        self._departure_station_id = None
        self._departed_at_time = None
        self.scheduled_track = None
        self.actual_track = None

        self._status = None
        self._status_timestamp = None
        self._status_stop_id = None


    def parse_trip_id(self, trip_id):
        """Decode the trip id and find trip origin time, line number, and direction
        
        Returns:
            Tuple of (origin time, line, direction, path id)        
        """
        origin_time = int(trip_id.split('_')[0])/100 #origin time of the trip in seconds past midnight.
        trip_path = trip_id.split('_')[1]
        line = trip_path.split('.')[0]
        path_id = trip_path.split('.')[-1]
        direction = path_id[0]
        path_id = path_id[1:]
        return (origin_time, line, direction, path_id)

    @property #the unique number we get from the MTA may actually not be unique (looks to be only unique within one day). To make it entirely unique, add the time we first started tracking this train.
    def unique_num(self):
        return str(self._time) + '_' + str(self._unique_num)

    @property
    def route_id(self):
        return self._route_id

    @route_id.setter
    def route_id(self, val):
        if(val != self._route_id):
            print('route ID changed from ' + str(self._route_id) + ' to ' + str(val))
        self._route_id = val

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, val):
        if(val != self._direction):
            #print('direction changed from ' + str(self._direction) + ' to ' + str(val))
            pass
        self._direction = val

    @property
    def trip_id(self):
        return self._trip_id
    
    @trip_id.setter
    def trip_id(self, val):
        val_parsed = self.parse_trip_id(val)
        id_parsed = self.parse_trip_id(self._trip_id)
        val_parsed = val_parsed[1] + '_' + val_parsed[2] + '_' + val_parsed[3]
        id_parsed = id_parsed[1] + '_' + id_parsed[2] + '_' + id_parsed[3]
        if(id_parsed != val_parsed and self.is_assigned==True):
            #print(str(self.unique_num) + " TRIP ID CHANGED from  " + str(self._trip_id) + ' to ' + str(val) )
            pass
        self._trip_id = val
        

    @property 
    def is_assigned(self):
        return self._is_assigned

    @is_assigned.setter
    def is_assigned(self, isassigned_timestamp):
        try:
            (is_assigned, _) = isassigned_timestamp
        except ValueError:
            raise ValueError("Pass a tuple of is_assigned and timestamp")
        else:
            self._is_assigned = is_assigned

    @property
    def current_time(self):
        return self._time

    @current_time.setter
    def current_time(self, value):
        if (value < self._time or value - self._time > 60): #the next tracking data must be in the future of the current data. Future updates should be within 60s of previous updates.
            raise ValueError("tracking time stamp indicates incorrect tracking")
        self._time = value

    @property
    def arrival_time(self):
        return self._arrival_time

    @arrival_time.setter
    def arrival_time(self, value):
        self._arrival_time = value
        #TODO on property changed

    @property
    def departure_station_id(self):
        """Get the ID of the station that we last departed. If that station is unknown this property will be 'None'"""
        return self._departure_station_id

    @property
    def departed_at_time(self):
        """Get the timestamp at which we last departed a station. If that timestamp is unknown this property will be 'None'"""
        return self._departed_at_time

    @property
    def arrival_station_id(self):
        """Get the ID of the next station at which this train will arrive."""
        return self._arrival_station_id

    @arrival_station_id.setter
    def arrival_station_id(self, id_timestamp):
        """Set the ID of the next station at which this train will arrive.
            Additionally update the current origin station, i.e. the station we just left
        
        Args
            id_timestamp (tuple): (id of the next station at which this train arrives, timestamp).
        """
        try:
            (id, timestamp) = id_timestamp
        except ValueError:
            raise ValueError("Pass a tuple of ID and timestamp")
        else:
            #check whether we got the current tracking data in a timely fashion (i.e. not much later than the previous data, say within 40s.)
            if(self.route_id == 'Q'):
                print("in arrival_station_id setter, id: " + str(id))
            if(np.abs(timestamp-self.subwaysys.last_attached_file_timestamp) < 40 or self.subwaysys.last_attached_file_timestamp == np.nan):
                if(self.route_id == 'Q'):
                    print("no gap, everything ok")
                    print("id: " + str(id))
                    print("prev arrival station: " + str(self._arrival_station_id))
                    print("is assigned?: " + str(self.is_assigned))
                if(id != self._arrival_station_id and self._arrival_station_id != None and self.is_assigned==True):
                    #we just left the station _arrival_station_id, register that this train stopped at this station.
                    if(self._arrival_station_id not in self.stations):
                        self.stations[self._arrival_station_id] = SubwayStation(self._arrival_station_id, 'UNKNOWN', self.subwaysys) #if our station is not in the current dictionary, add it as an unknown station.
                    self.stations[self._arrival_station_id].registerStoppedTrain(self.unique_num, self.route_id, self.direction, timestamp) # this may mess up if a train gets rerouted: we will think that the train reached a station it did not actually stop at.
                    
                    #actually update our values:
                    self._departure_station_id = self._arrival_station_id
                    self._departed_at_time = timestamp
                    self._arrival_station_id = id
                    if(self.route_id == 'Q'):
                        print("performed normal update")
                elif(id != self._arrival_station_id and self._arrival_station_id == None and self.is_assigned==True):
                    #previous arrival station was None, so we do not know at what station we just stopped (if any). So do not register the train, just update internal properties
                    self._departure_station_id = self._arrival_station_id
                    self._departed_at_time = timestamp
                    self._arrival_station_id = id


            else:
                '''there is a gap > 40s in the tracking data -- we cannot be sure where the train previously was, all we know is where it is going. Update the arrival station id only, but leave the departure station alone''' 
                if(self.route_id == 'Q'):
                    print("40s gap")
                if(self.is_assigned==True):
                    self._arrival_station_id = id
                    self._departure_station_id = None
                    self._departed_at_time = None
                    if(self.route_id == 'Q'):
                        print("anormal update")
                else:
                    if(self.route_id == 'Q'):
                        print("failed update since not assigned.")

    @property
    def status(self):
        """The current status of the train.

        Returns:
            Tuple of (status message, timestamp of status message, station ID referred to by status message) 
        """
        return (self._status, self._status_timestamp, self._status_stop_id)
    
    @status.setter
    def status(self, value):
        (status, timestamp, stop_id) = value
        
        #The following code would be ideal to track the status of all trains, but it only works for 1,2,3,4,5,6 (and maybe L) trains.
        #Unfortunately the MTA feed does not seem to contain proper status messages for other trains.
        #Therefore, the current implementation uses the trip_update message: when arrival_station_id(self) changes we know that we just left a station and have a new arrival station.
        #While this is hardly ideal I do not currently have a better idea of implementing the tracking of all trains.

        #if(len(stop_id)==0):
        #    stop_id = self.arrival_station_id #if we do not know what the next stop is copy the value from trip_update. THIS SEEMS LIKE A HACK
        #if(len(stop_id) > 0): # it is unclear why this is necessary. Check what the meaning of empty stop ids is.
        #    #it is possible that this field only exists for the numbered lines. This is really weird. We have to figure this out.
        #    #status: 1 = STOPPED AT, 2 = IN TRANSIT TO, 3 = INCOMING AT (?)
        #    if(self._status != status and status == 2):
        #        #we are in transit to a new station, register with the station.
        #        print(str(self.unique_num) + " (Route " + str(self.route_id) + ")" + " in transit to " + str(stop_id))
        #        self.stations[stop_id].registerTrainInTransitTo(self.unique_num, self.arrival_time, timestamp)
        #    elif(self._status != status and status == 3):
        #        #we are incoming at a new station, register with the station.
        #        print(str(self.unique_num) + " (Route " + str(self.route_id) + ")" + " incoming at " + str(stop_id))
        #        self.stations[stop_id].registerIncomingTrain(self.unique_num, self.arrival_time, timestamp)
        #    elif(self._status != status and status == 1):
        #        #we are stopped at a new station, register our stop with the station.
        #        print(str(self.unique_num) + " (Route " + str(self.route_id) + ")" + " stopped at " + str(stop_id))
        #        self.stations[stop_id].registerStoppedTrain(self.unique_num, timestamp)

        self._status = status
        self._status_timestamp = timestamp
        self._status_stop_id = stop_id
    


class MTAstaticdata(object):
    """Provides access to static MTA data"""

    @staticmethod
    def ImportStationsDictionary():
        """Import a dictionary of station names. Keys are the unique IDs of the stations.
        
        Returns:
            stopsdict (dict): Dictionary of (station_ids : station names)
        """
        with path(resources, 'stops.txt') as stop_path:
            with open(stop_path, mode='r') as infile:
                reader = csv.reader(infile)
                stopsdict = {rows[0]:rows[2] for rows in reader}
        return stopsdict
    
    @staticmethod
    def ImportStationsDataFrame():
        """Import a pandas dataframe of station ids, names, and locations.
        
        Returns:
            stationsDF (DataFrame): DataFrame with columns 'ids', 'names', 'location'
        """
        
        with path(resources, 'stops.txt') as stop_path:
           stationsDF = pd.read_csv(stop_path, delimiter = ',')
        return stationsDF

    @staticmethod
    def ImportLineDefinitions():
        """Import a spreadsheet defining the stops in each subway line.

        Returns:
            linesdict (dict): Dictionary of lines (line_ids : list of station_ids)
        """
        with path(resources, 'lines_definition.csv') as line_path:
            data = pd.read_csv(line_path)
        return data

