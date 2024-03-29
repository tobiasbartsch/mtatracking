import time
import sys
#sys.path.append('/home/tbartsch/source/repos')
#from mtatracking.utils import utils
#from mtatracking.MTAdatamine import MTAdatamine
from utils import utils
from MTAdatamine import MTAdatamine

#import mtatracking.nyct_subway_pb2 as nyct_subway_pb2
#import mtatracking.gtfs_realtime_pb2 as gtfs_realtime_pb2
#from MTAdatamodel import SubwaySystem

#from MTAdatamodel import SubwaySystem

# Enter key to download RT data.
key = input("Enter your MTA realtime access key: ")

#Make a new datamine
dmine = MTAdatamine(key)

#Now let's track some trains
delay = 10.0 #delay in seconds
num = 2880 #how often do we track? 2880 = 8 hours for 10 s ticks
#feed_id = [1,26,16,21,2,11,31,36,51]
feed_id = ['gtfs-ace', 'gtfs-bdfm', 'gtfs-g', 'gtfs-jz', 'gtfs-nqrw', 'gtfs-l', 'gtfs', 'gtfs-7', 'gtfs-si']
#MySubwaySys = SubwaySystem()

starttime=time.time()

track = True
while track:
   i = 0    
   while i < num:
       i+=1
       print("tick", i)
       data = None
       messagelist = []
       tracking_results = dmine.TrackTrains(feed_id)
       #MySubwaySys.attach_tracking_data(tracking_results)
       #print("*********************** Trains in system: " + str(MySubwaySys.NumberOfTrains) + " **************************")
       fname = 'tracking_results' + str(int(time.time())) + '.pkl'
       written = False
       while written is False:
            #try:
            written = utils.write(tracking_results, fname)
            print(written)
            #except:
            #    written = False
            #    print("Writing to disk failed, retrying...")
            #    track = False
            #    break
       time.sleep(delay - ((time.time() - starttime) % delay)) #wait until we are supposed to sample the next set of data.

#   fname = 'SubwaySys' + str(int(time.time())) + '.pkl'
#   utils.write(MySubwaySys, fname)
#   MySubwaySys.reset()