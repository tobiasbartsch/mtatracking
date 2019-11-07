import time
import sys
sys.path.append('/home/tbartsch/source/repos')
from mtatracking.utils import utils
from mtatracking.MTAdatamine import MTAdatamine
# Enter key to download RT data.
key = input("Enter your MTA realtime access key: ")

#Make a new datamine
dmine = MTAdatamine(key)

#Now let's track some elevators; once every 30 min should be good enough.
delay = 10 #delay in seconds
num = 48 #how often do we track? 2880 = 8 hours for 10 s ticks
#feed_id = [1,26,16,21,2,11,31,36,51]
#MySubwaySys = SubwaySystem()

starttime=time.time()

while True:
   i = 0    
   while i < num:
       i+=1
       print("tick", i)
       data = None
       elevatorxml_results = dmine.GetElevatorData()
       #MySubwaySys.attach_tracking_data(tracking_results)
       #print("*********************** Trains in system: " + str(MySubwaySys.NumberOfTrains) + " **************************")
       fname = 'elevator' + str(int(time.time())) + '.xml'
       with open(fname, "w") as xml_file:
           elevatorxml_results.writexml(xml_file) 
       print('got data')
       time.sleep(delay - ((time.time() - starttime) % delay)) #wait until we are supposed to sample the next set of data.
