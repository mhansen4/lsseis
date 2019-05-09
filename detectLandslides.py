import numpy as np
import pandas as pd
from obspy.clients.fdsn.header import URL_MAPPINGS
from obspy.signal.trigger import coincidence_trigger as ct
from obspy import UTCDateTime
from obspy.taup import TauPyModel
import obspy.signal.filter as filte
from reviewData import reviewData
from removeTeleseisms import calcCoordDistance, removeTeleseisms
from findFirstArrivals import findFirstArrivals
from sigproc import sigproc
from scipy import signal as spsignal
import traceback

import warnings
warnings.filterwarnings("ignore")

"""
Codes for retrieving obspy data streams and evaluating their signals for both
stand-alone landslides and landslides that may have occurred as part of a series
and have a known event to compare with (aka aftershocks).
"""

def getClients(starttime,endtime,lslat,lslon,radius=200.):
    """
    Returns list of valid clients/network codes to request data from in 
    getStreamObject, given landslide coordinates and a time range to search for
    data in.
    INPUTS
    starttime (UTCDateTime) - start time of stream object
    endtime (UTCDateTime) - end time of stream object
    lslat (float) - latitudinal coordinate of landslide (make negative for south
        of Equator)
    lslon (float) - longitudinal coordinate of landslide (make negative for west
        of Prime Meridian)
    radius (float) - optional; search radius in km for finding nearest seismic
        stations
    OUTPUT
    valid_clients (list of strings) - list of FDSN network codes that will return
        seismic traces
    """
    # Full list of FDSN webservice clients
    full_client_list = [key for key in sorted(URL_MAPPINGS.keys())]
    valid_clients = []
    
    # Seismic channels to search for
    channels = 'EHZ,BHZ,HHZ'
    
    # Search for data within initial radius
    print('Retrieving data from stations within %i km of event...' % int(radius))
    for i in range(0,len(full_client_list)):
        client = full_client_list[i]
        try:
            reviewData.getepidata(lslat, lslon, starttime, tstart=0.,
                                  tend=endtime-starttime, minradiuskm=0., 
                                  maxradiuskm=radius, chanuse=channels, 
                                  location='*', clientnames=client)
            valid_clients.append(client)
        except:
            pass
        
    return(valid_clients)

def getStreamObject(starttime,endtime,lslat,lslon,radius=100.,maxradius=300.,
                    client=['IRIS'],mintraces=7,loadfromfile=False,savedat=False,
                    folderdat='Data',filenamepref='Data_',limit_stations=10):
    """
    Uses seisk reviewData module to grab seismic data using FDSN webservices
    for stations within a specified radius of the landslide as a stream object.
    Increments radius until minimum number of traces or maximum search radius
    is achieved.
    INPUTS
    starttime (UTCDateTime) - start time of stream object
    endtime (UTCDateTime) - end time of stream object
    lslat (float) - latitudinal coordinate of landslide (make negative for south
        of Equator)
    lslon (float) - longitudinal coordinate of landslide (make negative for west
        of Prime Meridian)
    radius (float) - optional; search radius in km for finding nearest seismic
        stations
    maxradius (float) - optional; max radius in km that function will increment 
        to in its  data search
    client (list of strings) - optional; list of FDSN network codes to request
        data from
    mintraces (int) - optional; lowest number of seismic traces that can be 
        returned without the function throwing an error
    loadfromfile (Boolean) - optional; gives function permission to look for 
        seismic data in current working directory
    folderdat (string) - optional; name of folder to search in for seismic data.
        Must be located in current working directory.
    filenamepref (string) - optional; beginning string of data file names, 
        usually name of landslide in lowercase 
    limit_stations (int) - optional; maximum number of stations to return in
        stream object
    OUTPUT
    st - obspy stream object with seismic data
    """
    
    # Seismic channels to search for
    channels = 'EHZ,BHZ,HHZ'
    
    # Search for signal with initial search radius
    st = reviewData.getepidata(lslat, lslon, starttime, tstart=0.,
                               tend=endtime-starttime, minradiuskm=0., 
                               maxradiuskm=radius, chanuse=channels, 
                               location='*', clientnames=['IRIS'],
                               savedat=savedat, folderdat=folderdat,
                               filenamepref=filenamepref,
                               loadfromfile=loadfromfile,detrend='demean')
    
    # Count stations returned
    station_count = 0
    if st is not None:
        for trace in st:
            station_count += 1
    
    # Check if number of traces in stream object less than minimum; if so,
    # increment radius by 50 km and search for data again    
    while station_count < limit_stations and radius <= maxradius:
        if len(st) > 0:
            st.clear() # Clear stream object
            
        radius += 50. # km
        print('Incrementing radius to %i km and retrieving data...' % int(radius))
        st = reviewData.getepidata(lslat, lslon, starttime, tstart=0.,
                                   tend=endtime-starttime, minradiuskm=0., 
                                   maxradiuskm=radius, chanuse=channels, 
                                   location='*', clientnames=['IRIS'],
                                   savedat=savedat, folderdat=folderdat,
                                   filenamepref=filenamepref,
                                   loadfromfile=loadfromfile,detrend='demean')
        
        # Count stations returned
        station_count = 0
        for trace in st:
            station_count += 1    
            
        print('Number of stations at this radius = %i' % station_count)
        
    print('%i stations within %i km of landslide.' % (len(st),int(radius)))
    
    # Limit number of stations returned
    if len(st) > limit_stations:
        st = st[:limit_stations]
        print('Returning only %i closest stations.' % len(st))
    if len(st) < mintraces:
        raise Exception('Less than %i stations returned.' % mintraces) 
    
    return(st)
      
def findTriggers(lslat, lslon, st, trigger_times, trace_ids=None):
    """
    Finds spikes/triggers in stream st using the obspy coincidence_trigger function 
    and stores in a list called trigger_times.
    INPUT
    lslat (float) - latitudinal coordinate of landslide (make negative for south
        of Equator)
    lslon (float) - longitudinal coordinate of landslide (make negative for west
        of Prime Meridian)
    st - obspy stream object with seismic data
    trigger_times (list of UTCDateTimes) - list (empty or not) to append trigger
        times to 
    trace_ids (dict) - optional; dictionary of trace IDs to include in stream 
        object, with weights as their values
        Example: {'CC.OBSR..BHZ': 1,'CC.PANH..BHZ': 1,'PB.B941..EHZ': 1}
    OUTPUT
    st - stream object with updated distances
    trig - list of coicidence_trigger objects
    trigger_times
    removed_triggers (list of UTCDateTimes) - list of trigger times determined
        to be teleseisms and removed from trigger list
    """
    
    network = st[0].stats.network
    station = st[0].stats.station # first station in stream
    
    print('Looking for triggers...')
    
    # play with these parameters
    init_trig = ct("recstalta", 2.5, 1, st, 3, trace_ids=trace_ids,
                   sta=4., lta=20.)
    print('%i triggers initially returned.' % len(init_trig))
    if len(init_trig) > 0:
        # Define time range to search for teleseisms over
        end = UTCDateTime(st[0].stats.starttime) + 100. # signal time
        start = end - 15*60. # 15 min before signal
        trig, removed_triggers = removeTeleseisms(start,end,network,
                                                  station,init_trig)
        
        print('%i teleseism(s) found, %i trigger(s) remaining.' % 
              (len(removed_triggers),len(trig)))
        
        print('')
        
        trigger_times = [t['time'] for t in trig]
    else:
        trig = []
        trigger_times = []
        removed_triggers = []
        
    return(st, trig, trigger_times, removed_triggers)

def predictArrivalTimes(st, lslat, lslon):     
    """
    Uses 1D Earth model TauPyModel to predict how long it took for S waves 
    generated by landslide to reach each station.
    INPUT
    st - obspy stream object with seismic data
    lslat (float) - latitudinal coordinate of landslide (make negative for south
        of Equator)
    lslon (float) - longitudinal coordinate of landslide (make negative for west
        of Prime Meridian)
    OUTPUT
    pred_arrival_secs (list of floats) - arrival times of event at each 
        station (going away from the landslide) based on obspy TauPyModel in s
        after landslide time
    """
    
    # With taup function
    model = TauPyModel(model="iasp91")
    pred_arrival_secs = [] # List of datetime objects

    # Get distance in degrees from landslide to each station
    # Use distance to find arrival time at each station using TauPyModel
    for trace in st:
        stationlat = trace.stats.coordinates['latitude']
        stationlon = trace.stats.coordinates['longitude']
        ddeg = calcCoordDistance(lslat,lslon,stationlat,stationlon)[1]

        arrival = model.get_travel_times(source_depth_in_km=0.0,
                                         distance_in_degree=ddeg,
                                         phase_list=["S"])  
        # Check if other arrival times besides first might be better?
        pred_arrival_secs.append(arrival[0].time)
    
    return(pred_arrival_secs)

def detectLandslides(st, trig, lslat, lslon, min_stations = 3, min_time1 = 1.0, 
                     min_time2 = 20., trig_i = []):
    """
    Loops through all triggers in trigger_times and finds the first arrival times 
    of all traces using the method in the Ballard paper. Predicts arrival times
    using the obspy TauPyModel. If the two sets arrival times are within 
    min_time_diff of each other for at least min_stations number of stations, a 
    landslide is detected.
    INPUT
    st - obspy stream object with seismic data
    trig - list of coicidence_trigger objects
    lslat (float) - latitudinal coordinate of landslide (make negative for south
        of Equator)
    lslon (float) - longitudinal coordinate of landslide (make negative for west
        of Prime Meridian)
    min_stations (int) - optional; minimum number of stations that need to 
        display a linear moveout in first arrival times for trigger to count 
        as event
    min_time1 (float) - optional; minimum time in seconds that first arrival 
        time at station can differ from predicted arrival time for a perfect 
        linear moveout in order for station to 'pass' and landslide to be detected
    min_time2 (float) - optional; minimum number of seconds detections must be 
        within to be counted as separate events. Events closer in time will be 
        counted as one event and only the earliest time will be returned.
    trig_i (list) - optional; list of trig indices to evaluate
    OUTPUT
    events_df - pandas DataFrame with detected landslide times
    """
    
    # If no triggers specified for evaluation, evaluate all of them
    if trig_i == []:
        trig_i = range(0,len(trig))
    
    # Create list to store event times
    event_times = []
    
    for t in trig_i:
        print('Processing trigger %i of %i...' % (t+1,len(trig)))
        print(trig[t]['time'])
        
        # Take slice of signal around trigger time
        temp = st.copy().trim(trig[t]['time']-100., 
                              trig[t]['time']+100.)
        
        # Delete problematic stations (will need to add to)
        channels_to_remove = []
        for channel in temp:
            if channel.std() < 2.0:
                channels_to_remove.append(channel)
        
        for channel in channels_to_remove:
            #if len(st.select(station = channel.stats.station)) > 0:
            temp.remove(channel)
        
        # Find first arrivals in signal
        arrival_times, arrival_inds = findFirstArrivals(temp)
        
        # Check if arrival time belongs to event        
        if len(arrival_times) > 0:  
            # Find seconds between first arrival at each station and the closest station
            arrival_secs = [time - arrival_times[0] for time in arrival_times]  
            
            # Get predicted arrival times
            print('Predicting signal arrival times using 1D Earth model...')
            pred_arrival_secs = predictArrivalTimes(temp, lslat, lslon)
            
            # Find difference in seconds between predicted and real arrival times
            arrival_secs = np.array(arrival_secs)
            pred_arrival_secs = np.array(pred_arrival_secs)
            pred_diff = arrival_secs - pred_arrival_secs
            
            print('Difference between predicted arrival times and actual:')
            print(pred_diff)
            
            # Check if predicted arrival times within some number of 
            # seconds of actual arrival times for min number of stations
            check_pass_count = 0
            for diff in pred_diff:
                if abs(diff) <= min_time1:
                    check_pass_count += 1
                    
            print('%i predicted arrival time(s) within %.1f s of actual times.' 
                  % (check_pass_count, min_time1))
             
            # Store landslide info if minimum number of stations passed
            if check_pass_count >= min_stations:                     
                print('Landslide detected at %s.' % str(arrival_times[0]))                           
                # Save event time and linear regression parameters
                event_times.append(arrival_times[0])
                    
            print('') # Print blank line between triggers
    
            
    # If any times in event_times within min_time2 s of each other, save first one   
    new_event_times = []
    if len(event_times) > 0:
        new_event_times = [event_times[0]]
        event_times.sort() # Sort times from earliest to latest
        for e in range(1,len(event_times)):
            if event_times[e] - event_times[e-1] > min_time2:
                new_event_times.append(event_times[e]) 
                
    print('%i possible landslide(s) found.' % len(new_event_times))
    print('') # Print blank line
    
    # Create dataframe for detected events
    events_df = pd.DataFrame({'Event time': new_event_times})
        
    return(events_df)
    
def detectAftershocks(st, template, trig, min_time = 20., threshold=0.75, 
                      newsamprate = 20., before=30., after=200., smoothwin=201, 
                      smoothorder=3):
    """
    Loops through all triggers in trigger_times, finds the first arrival times 
    of all traces, and compares these to those of the known event. If the arrival 
    times are within min_time_diff for at least min_stations number of stations, 
    an aftershock is detected.
    INPUT
    st - obspy stream object with seismic data
    template - obspy stream of known event
    trig - list of coicidence_trigger objects
    min_time (float) - optional; minimum time between aftershocks to count as 
        separate events (sec)
    threshold (float) - optional; cross-correlation threshold for use in 
        templateXcorrRA
    newsamprate (float) - optional; sampling rate to resample template to. Must
        match the sampling rate of the stream object the template is being 
        compared to.
    before (float) - seconds before trigger time to look for aftershocks in
    after (float) - seconds after trigger time to look for aftershocks in
    smoothwin (int) - window length in samples for Savgol smoothing of envelopes
    smoothorder (int) - polynomial order for Savgol smoothing
    OUTPUT
    aftershocks (list of UTCDateTimes) - times of detected aftershocks
    discard (list): list of discarded triggers
    st_after (list): list of obspy streams of extfacted aftershock triggers
    """
    
    # Create empty lists to store aftershock times
    aftershocks = []
    st_after = []
    discard = []
    
    # Process template
    tproc = template.copy()
    for tr in tproc:
        tr.data = filte.envelope(tr.data)
    
    # Process signal slice around each triggering time
    for t in range(0,len(trig)):
        print('Processing trigger %i of %i...' % (t+1,len(trig)))
        print(trig[t]['time'])
        
        # Take slice of signal around trigger time
        temp = st.copy().trim(trig[t]['time']-before, 
                              trig[t]['time']+after)
        
        # Resample signal
        temp.resample(newsamprate)
        
        # Process trigger
        stproc = temp.copy()
        for tr in stproc:
            tr.data = spsignal.savgol_filter(filte.envelope(tr.data), smoothwin, 
                                             smoothorder)
            
        # Check if only stations present in tproc are present in stproc
        tproc_stations = [tr.id for tr in tproc]
        for tr in stproc:
            if tr.id not in tproc_stations:
                stproc.remove(tr)
        
        # Check if tproc is not longer than stproc
        if len(tproc[0]) > len(stproc[0]):
            max_index = len(stproc[0])
            for s in range(len(stproc)):
                stproc[s].data = stproc[s].data[:max_index]
            for s in range(len(tproc)):
                tproc[s].data = tproc[s].data[:max_index]
       
        # Cross correlate
        times = []
        try:
            xcorFunc, xcorLags, ccs, times = sigproc.templateXcorrRA(stproc, 
                                                                     tproc, 
                                                                     threshold=threshold)      
        except:
            print('Error occurred within templateXcorrRA function for trigger %i:'
                  % t)
            traceback.print_exc()
        
        # Check if event meets threshold
        if len(times) == 0:
            discard.append(trig[t])
        else:
            # Take highest value returned and adjust for before time to get 
            # approximate event time
            indx = np.argmax(xcorFunc)
            aftershocks.append(stproc[0].stats.starttime + xcorLags[indx] + before)
    
    # If any times in aftershocks within min_time of each other, save first one   
    if len(aftershocks) > 1:
        new_aftershocks = [aftershocks[0]] 
        aftershocks.sort() # Sort times from earliest to latest
        for a in range(1,len(aftershocks)):
            if aftershocks[a] - aftershocks[a-1] > min_time:
                new_aftershocks.append(aftershocks[a])     
        
        aftershocks = new_aftershocks
    
    for af in aftershocks:
        st_after.append(st.copy().trim(af-before, af+after))
        
    print('%i aftershock(s) found.' % len(aftershocks))
    print('') # Print blank line
        
    return(aftershocks, discard, st_after)