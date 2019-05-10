import numpy as np
import pandas as pd
from obspy import UTCDateTime, Stream

from reviewData import reviewData

from detectLandslides import getStreamObject, findTriggers, detectAftershocks
from removeTeleseisms import searchComCatforLandslides

import warnings
warnings.filterwarnings("ignore")

"""
Takes a known landslide event and compares the seismic signals created by it at 
nearby stations to other seismic triggers to find potential 'aftershock' events.
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
    """    
    # Input parameters    
    lslat = 60.07367 # latitude of landslide in deg
    lslon = -139.8462 # longitude of landslide in deg
    radius = 100. # search radius for nearest seismic stations in km
    
    starttime = UTCDateTime(2017,7,11,0,0,0)
    endtime = UTCDateTime(2017,7,31,23,59,59)

    known_event_time = UTCDateTime(2017,7,23,21,49,46)
    
    filenamepref = 'yacutat'
    """
    lslat = 58.77918 # latitude of landslide in deg
    lslon = -136.88827 # longitude of landslide in deg
    radius = 150. # search radius for nearest seismic stations in km
    starttime = UTCDateTime(2016,6,21,0,0,0)
    endtime = UTCDateTime(2016,7,6,0,0,0)
    known_event_time = UTCDateTime(2016,6,28,16,21,3)
    filenamepref = 'lamplugh'
    
    folderdat = filenamepref + '_data'
    loadfromfile = True
    savedat = True
    
    # Specify interval to split time period up into in seconds
    interval = 8. * 3600.
    
     # Cross correlation threshold to declare an event
    threshold = 0.65
    
    # Specify part of signal to keep (will be applied to st1 and st2)
    before = 50. # Time to cut before event or trigger start time in s
    after = 150. # Time to cut after event or trigger start time in s
    
    # Specify new sampling rate to reduce computation time
    newsamprate = 20.
    
    # First look at known event
    print('Grabbing first known event...')
    st1 = getStreamObject(known_event_time-before,known_event_time+after,lslat,lslon,
                          radius=radius,maxradius=500.,loadfromfile=loadfromfile,
                          mintraces=4,savedat=savedat,folderdat=folderdat,
                          filenamepref=filenamepref,limit_stations=8) 
    st1 = processSignal(st1, lslat, lslon, newsamprate)
    
    # Find trigger in known event signal
    st1, trig, temp_trigger_times, temp_teleseisms = findTriggers(lslat,lslon,st1,[])
    
    # Review signal to remove any traces before doing additional processing
    # zp1 = reviewData.InteractivePlot(st1)
    
    """
    # Record deleted traces and delete from st1
    stations_to_delete = [item.split('.')[0] for item in zp1.deleted]
    for channel in st1:
        if channel.stats.station in stations_to_delete:
            st1.remove(channel)
    """
    
    # Store st1 trace IDs in list so that same ones can be evaluated later
    st1_trace_ids = {}
    for trace in st1:
        # Assigning value of 1 makes all traces equally weighted
        st1_trace_ids[str(trace).split(' | ')[0]] = 1
    
    # Search for aftershocks
    
    # Split up big date range into smaller chunks
    starts = np.arange(starttime, endtime, interval)
    ends = starts + interval
    
    # Create lists for storing and sorting triggers
    trigger_times = []
    teleseisms = []
    aftershocks = []
    
    # Create list for storing pseudo-energy information 
    pseudo_energies = []
    on = 2.0 # sta/lta threshold to determine event 'peak'
    off = 1.0 # sta/lta threshold to determine event start time
    
    # Create DataFrame for aftershocks
    aftershocks_df = pd.DataFrame()
    
    # alldat = Stream()
    
    print('Searching for aftershocks...')
    
    # Loop through smaller date ranges and add landslide events to dataframe
    for i in range(0,len(starts)):
        print('')
        print('Assessing time range %s to %s...' % (starts[i],ends[i]))
        print('')
        st2 = getStreamObject(starts[i],ends[i],lslat,lslon,radius=radius,
                              maxradius=500.,loadfromfile=loadfromfile,
                              mintraces=3,savedat=savedat,folderdat=folderdat,
                              filenamepref=filenamepref,limit_stations=8) 
        print('Processing signal...')
        st2 = processSignal(st2, lslat, lslon)
        
        # Grab triggers
        print('Grabbing triggers...')
        st2, trig, new_trigger_times, new_teleseisms = findTriggers(lslat,lslon,st2,
                                                                    trigger_times, 
                                                                    st1_trace_ids)
        # alldat += st2.copy()        
                                                       
        for trigger_time in new_trigger_times:
            trigger_times.append(trigger_time)
        for teleseism_time in new_teleseisms:
            teleseisms.append(teleseism_time)
            
        # Search triggers for aftershocks
        print('Threshold = %.2f. Searching for aftershocks...' % threshold)
        new_aftershocks, junk, af1 = detectAftershocks(st2, st1, trig, min_time = 20., 
                                                       before=before, after=after, 
                                                       threshold=threshold, 
                                                       newsamprate=newsamprate) 
        
        # Add new aftershocks to list
        for aftershock in new_aftershocks:
            aftershocks.append(aftershock)
            st_trim = st2.copy().trim(aftershock-before, aftershock+after)
            
        # Add newly determined aftershocks and pseudo-energies to dataframe and 
        # save to file
        if len(new_aftershocks) > 0:
            aftershocks_filename = filenamepref + '_aftershocks.csv'
            print('Saving aftershocks to file "%s"...' % aftershocks_filename)
            new_aftershocks_df = pd.DataFrame({'TIMESTAMP': new_aftershocks})
            aftershocks_df = pd.concat([aftershocks_df, new_aftershocks_df],
                                       ignore_index = True)
            aftershocks_df.to_csv(aftershocks_filename)
            

            
    # alldat.merge() # This keeps producing an error
    # reviewData.InteractivePlot(alldat, vlines=aftershocks)
    
    # Look for landslides in ComCat and return
    print('Looking for landslides in ComCat...')
    network = st2[0].stats.network # Seismic network of station closest to landslide
    station = st2[0].stats.station # Seismic station closest to landslide
    possible_ls = searchComCatforLandslides(starttime,endtime,lslat,lslon,
                                            network,station)  
    print('%i possible landslides found in ComCat.' % len(possible_ls))
    
    print('Done.')