#homemade modules
import sys
sys.path.append('/home/tbartsch/source/repos')
from mtatracking.utils import utils
from mtatracking.MTAdatamodel import SubwaySystem
from mtatracking.MTASubwayAnalyzer import MTASubwayAnalyzer
from mtatracking.MTASubwayAnalyzer import delayOfTrainsInLine_Bayes
from mtatracking.utils import utils


class CurrentTransitTimeDelays():
    '''Provides methods and properties to handle the mean transit times between adjacent stations.
    Historic data are necessary to make these computations; you have to pass a historic subway system to the initializer of this class.
    The initializer does NOT automatically compute anything (such computations take a long time and are sometimes not wanted when an object is first created). 
    You can manually start computations by interacting with the self.delays property.
    '''
    
    def __init__(self, histDataPath='/home/tbartsch/data/testsys.pkl'):
        '''create a new CurrentTransitTimeDelays object.
        load historic data (ideally of the past month) so that we can compute the current mean transit times between stations and their sdevs
        
        Args:
            histDataPath (path to  a pickled subway system object) 
        '''
        self.historic_subsys = utils.read(histDataPath)
        analyzer = MTASubwayAnalyzer(mySubwaySystem=self.historic_subsys)

        #get all unique lines in the historic data
        self.unique_lines = analyzer.unique_lines

        #self.delays = {id: delayOfTrainsInLine_Bayes(line_id = id, analyzer=analyzer, n=2, timestamp_start = 1556668800, timestamp_end = 1561939200) for id in self.unique_lines}
        self.delays = utils.read('delays2019-11-06.pkl')