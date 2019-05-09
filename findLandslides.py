# Imports
import numpy as np
import pandas as pd
from obspy import UTCDateTime

from reviewData import reviewData

from detectLandslides import getStreamObject, findTriggers, detectLandslides
from removeTeleseisms import searchComCatforLandslides

"""
Implements the detectLandslides() function to search for landslides originating
from a certain coordinate over a given time range. 
"""

def processSignal(st, lslat, lslon, newsamprate=0.):
    """
    Ensures template and target signals are processed the same way before 
    cross-correlation.
    INPUTS
    st - obspy stream object with seismic data
    lslat (float) - latitudinal coordinate of landslide (make negative for south
        of Equator)
    lslon (float) - longitudinal coordinate of landslide (make negative for west
        of Prime Meridian)
    newsamprate (float) - optional; sampling rate to resample template to. Must
        match the sampling rate of the stream object the template is being 
        compared to.
    OUTPUT
    st - processed stream object
    """
    # Demean signal
    st.detrend('demean')
    
    # Apply bandpass filter
    fmin = 1.0 # lower frequency limit, Hz'
    fmax = 3.0 # upper frequency limit, Hz
    st.filter('bandpass', freqmin=fmin, freqmax=fmax)
    
    # Apply signal taper
    #st.taper(max_percentage=0.05, type='cosine')
    
    """
    # Apply cosine filter
    prefilt = [0.01, 0.02, 10, 20]
    st.remove_response(pre_filt=prefilt, output='VEL')
    """
    
    # Attach station coordinates to stream object
    st = reviewData.attach_distaz_IRIS(st, lslat, lslon)
    st = st.sort(keys=['rdist', 'channel'])
    
    # Resample signal
    if newsamprate > 0.:
        st.resample(newsamprate)
    
    return(st)
    
if __name__ == '__main__': 
    # Input parameters
    lslat = 60.07367 # latitude of landslide in deg
    lslon = -139.8462 # longitude of landslide in deg
    radius = 100. # search radius for nearest seismic stations in km
    
    # Define date range to look for landslides in
    starttime = UTCDateTime(2017,7,16,23,59,59)
    endtime = UTCDateTime(2017,7,18,23,59,59)
    interval = 8. * 3600.  # interval to split time period up into in seconds
    
    # Data file info
    filenamepref = 'yacutat'
    folderdat = filenamepref + '_data'
    loadfromfile = True
    savedat = True
    
    # Set thresholds for landslide search
    min_stations = 4 # number of stations that must fit model
    min_time_diff = 10.0 # number of seconds first arrival times can differ from model by
    
    # Split up big date range into smaller chunks
    starts = np.arange(starttime, endtime, interval)
    ends = starts + interval
    
    # Loop through smaller date ranges and find landslides
    trigger_times = []
    teleseisms = []
    events_df = pd.DataFrame()
    for i in range(0,len(starts)):
        print('')
        print('Assessing time range %s to %s...' % (starts[i],ends[i]))
        print('')
        st = getStreamObject(starts[i],ends[i],lslat,lslon,radius=radius,
                              loadfromfile=loadfromfile,savedat=savedat,
                              folderdat=folderdat,filenamepref=filenamepref) 
        st = processSignal(st, lslat, lslon)
        st, trig, new_trigger_times, new_teleseisms = findTriggers(lslat,lslon,st,
                                                                   trigger_times)
        for trigger_time in new_trigger_times:
            trigger_times.append(trigger_time)
        for teleseism_time in new_teleseisms:
            teleseisms.append(teleseism_time)
        new_events_df = detectLandslides(st,trig,lslat,lslon,min_stations,
                                         min_time_diff,[])
        print('%i landslides detected.' % len(new_events_df))
        events_df = events_df.append(new_events_df)
        
    # Create dataframe for signal trigger times
    triggers_df = pd.DataFrame({'Trigger times': trigger_times}) 
    
    # Reindex dataframe
    events_df = events_df.reset_index(drop=True)
    
    # Save dataframes to file
    save_dfs = False # Set to True to save event and trigger time dataframes to CSVs
    if save_dfs:
        print('Saving event predictions and all trigger times to files.')
        events_df.to_csv(filenamepref+'_predicted_events.csv')
        triggers_df.to_csv(filenamepref+'_trigger_times.csv')
    
    # Look for landslides in ComCat and return
    print('Looking for landslides in ComCat...')
    network = st[0].stats.network # Seismic network of station closest to landslide
    station = st[0].stats.station # Seismic station closest to landslide
    possible_ls = searchComCatforLandslides(starttime,endtime,lslat,lslon,
                                            network,station)  
    print('%i possible landslides found in ComCat.' % len(possible_ls))
    
    print('Done.')