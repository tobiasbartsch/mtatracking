from __future__ import print_function
import time
import sys
import urllib.request
import multiprocessing
import math
from datetime import datetime
from operator import itemgetter
from collections import defaultdict

import mtatracking.nyct_subway_pb2 as nyct_subway_pb2
import mtatracking.gtfs_realtime_pb2 as gtfs_realtime_pb2

class MTAdatamine(object):
    """Retrieve realtime information about the status of trains in the NYC subway system."""

    def __init__(self, key):
        """
        Args:
            key: Your access key to the MTA realtime data feeds
        """
        self.k = key

    def TrackTrains(self, feed_ids):
        """Query the locations and status of all trains of a specific set of lines.
        
        Args:
            feed_ids (list of int): IDs of the set of subway lines to track. For example, feed_ids=[16] are the NQR trains.

        Returns: List of dictionaries of tracked trains.
        """

        #messagelist = [gtfs_realtime_pb2.FeedMessage() for _ in range(len(feed_ids))]
        
        data = None
        messagelist = []
        while data is None:
            try:
                for id in feed_ids: #right now this is set to reprocess ALL feeds if one of them fails.
                    url = 'http://datamine.mta.info/mta_esi.php?key=' + str(self.k) + '&feed_id=' + str(id)
                    with urllib.request.urlopen(url) as response:
                        print(id)
                        data = response.read()
                        feed_message = gtfs_realtime_pb2.FeedMessage()
                        feed_message.ParseFromString(data)
                        messagelist.append(feed_message)                          
            except: #pretty often there are truncated messages. if there are, reset.
                data=None
                messagelist=[]
                pass
        return messagelist


    def _TrackTrainsOnce(self, feed_message, trip_dict):
      """Process the feed_message to find the status of all trains. Add the new tracking data to trip_dict
        
        Args:
            feed_message: Protocol buffer message containing train statuses (presumably downloaded from the MTA realtime stream)
            trip_dict: Dictionary of previously tracked trips to which the new information will be attached.
      """
      
      current_time = time.time()
      for FeedEntity in feed_message.entity:
        #print("ID:", FeedEntity.id) #this is a unique number that's not very useful
        #print("Trip ID:", FeedEntity.trip_update.trip.trip_id) #this contains info on which train we are tracking and which direction it is going
        if len(FeedEntity.trip_update.stop_time_update) > 0: #this only rtrieves the next stop of each train.
            stop_time = FeedEntity.trip_update.stop_time_update[0]
        
            #Add to our tracker
            trip_dict[FeedEntity.trip_update.trip.trip_id][math.floor(current_time)] = (stop_time.arrival.time, stop_time.stop_id)
            #print("Stop Code:", stop_time.stop_id) # this has funny codes for the stops. e.g.
        
            #print("Arrival delay:", stop_time.arrival.delay) this is empty
            #print("Arrival time in Unix s: ", stop_time.arrival.time)
            #print("Arrival time: ", datetime.fromtimestamp(stop_time.arrival.time).strftime('%Y-%m-%d %H:%M:%S'))
            #for every train we need a key:value pair, where the values are a 2D list of all stops and arrival times. Keys are the unique numbers of the trips
      return trip_dict
