import numpy as np
import pandas as pd

def lookUpStationByLoc(lat, lon, stationsDF):
    '''return the station id of the station closest to stop_lat, stop_lon.
    
    Args:
        lat (float): latitude
        lon (float): longitude
        stationsDF (dataframe): dataframe of station information (from MTAdatamodel.MTAstaticdata.ImportStationsDataFrame())
    
    Returns:
        station_id, station_name (string, string): id and name of the station closest to the given lat and lon
    '''
    longs = stationsDF['stop_lon']
    lats = stationsDF['stop_lat']
    
    d = np.sqrt(np.power((longs - lon),2) + np.power((lats - lat),2))
    
    stationindices = np.where(d==np.min(np.array(d)))
    station_ids = np.array(stationsDF['stop_id'])[np.array(stationindices)]
    station_names = np.array(stationsDF['stop_name'])[np.array(stationindices)]
    parent_stations = np.array(stationsDF['parent_station'])[np.array(stationindices)]
    a = pd.isnull(np.array(parent_stations.flatten(), dtype=object))
    return station_ids.flatten()[a][0], station_names.flatten()[a][0]

def addStationID_andNameToGeoPandas(geopandasDF, stationsDF):
    '''inserts columns stop_id and stop_name from the stationsDF (from gtfs) into the geopandasDF'''
    #geopandasDF.insert(0, "stop_id", None)
    #geopandasDF.insert(1, "stop_name", None)
    
    geopandasDF['stop_id'] = geopandasDF.apply(lambda row: lookUpStationByLoc(row['geometry'].y, row['geometry'].x, stationsDF)[0], axis=1)
    geopandasDF['stop_name'] = geopandasDF.apply(lambda row: lookUpStationByLoc(row['geometry'].y, row['geometry'].x, stationsDF)[1], axis=1)