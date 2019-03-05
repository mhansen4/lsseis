import numpy as np
import pandas as pd
from obspy.signal.trigger import coincidence_trigger as ct
from obspy import UTCDateTime
from obspy.taup import TauPyModel
from reviewData import reviewData
from removeTeleseisms import calcCoordDistance, removeTeleseisms
from findFirstArrivals import findFirstArrivals

import warnings
warnings.filterwarnings("ignore")

def getStreamObject(starttime,endtime,lslat,lslon,radius=100.,
                    limit_stations=12):
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
    limit_stations (int) - optional; maximum number of stations to return in
        stream object
    OUTPUT
    st - obspy stream object with seismic data
    """
    
    # Seismic channels to search for
    channels = 'EHZ,BHZ,HHZ'
    
    # Search for data within initial radius
    print('Retrieving data from stations within %i km of landslide...' % int(radius))
    station_count = 0
    st = reviewData.getepidata(lslat, lslon, starttime, tstart=0.,
                               tend=endtime-starttime, minradiuskm=0., 
                               maxradiuskm=radius, chanuse=channels, 
                               location='*', clientnames=['IRIS'],
                               savedat=False, detrend='demean')
    for trace in st:
        station_count += 1
    
    # Check if number of traces in stream object less than minimum; if so,
    # increment radius by 50 km and search for data again
    maxradius = 300 # maximum radius to search within (in km)
    mintraces = 5 # minimum number of traces accepted
    
    while station_count < mintraces and radius <= maxradius:
        radius += 50. # km
        print('Incrementing radius to %i km and retrieving data...' % int(radius))
        st = reviewData.getepidata(lslat, lslon, starttime, tstart=0.,
                                   tend=endtime-starttime, minradiuskm=0., 
                                   maxradiuskm=radius, chanuse=channels, 
                                   location='*', clientnames=['IRIS'],
                                   savedat=False, detrend='demean')
    
    print('%i stations within %i km of landslide.' % (len(st),int(radius)))
    
    # Limit number of stations returned
    if len(st) > limit_stations:
        st = st[:limit_stations]
        print('Returning only %i closest stations.' % len(st))
    
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
    
    print('Modifying stream object...')
    st = reviewData.attach_distaz_IRIS(st, lslat, lslon)
    st = st.sort(keys=['rdist', 'channel'])
    
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

def detectLandslides(st, trig, lslat, lslon, min_stations = 3, min_time_diff = 1.0, 
                     trig_i = []):
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
    min_stations (int) - minimum number of stations that need to display a linear
        moveout in first arrival times for trigger to count as event
    min_time_diff (float) - minimum time in seconds that first arrival time at
        station can differ from predicted arrival time for a perfect linear
        moveout in order for station to pass
    trig_i (list) - list of trig indices to evaluate
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
                if abs(diff) <= min_time_diff:
                    check_pass_count += 1
                    
            print('%i predicted arrival time(s) within %.1f s of actual times.' 
                  % (check_pass_count, min_time_diff))
             
            # Store landslide info if minimum number of stations passed
            if check_pass_count >= min_stations:                     
                print('Landslide detected at %s.' % str(arrival_times[0]))                           
                # Save event time and linear regression parameters
                event_times.append(arrival_times[0])
                    
            print('') # Print blank line between triggers
    
    # If any times in event_times within 1 minute of each other, save first one
    new_event_times = [event_times[0]]
    
    if len(event_times) > 0:
        event_times.sort() # Sort times from earliest to latest
        for e in range(1,len(event_times)):
            if event_times[e] - event_times[e-1] > 60:
                new_event_times.append(event_times[e])     
                
    print('%i possible landslide(s) found.' % len(new_event_times))
    print('') # Print blank line
    
    # Create dataframe for detected events
    events_df = pd.DataFrame({'Event time': new_event_times})
        
    return(events_df)
    
def detectAftershocks(st, trig, arrival_times1, min_stations = 3, 
                      min_time_diff = 1.0):
    """
    Loops through all triggers in trigger_times, finds the first arrival times 
    of all traces, and compares these to those of the known event. If the arrival 
    times are within min_time_diff for at least min_stations number of stations, 
    an aftershock is detected.
    INPUT
    st - obspy stream object with seismic data
    trig - list of coicidence_trigger objects
    arrival_times1 (list of UTCDateTimes) - arrival times for known event
    min_stations (int) - minimum number of stations that need to display a linear
        moveout in first arrival times for trigger to count as event
    min_time_diff (float) - minimum time in seconds that first arrival time at
        station can differ from predicted arrival time for a perfect linear
        moveout in order for station to pass
    trig_i (list) - list of trig indices to evaluate
    OUTPUT
    aftershocks (list of UTCDateTimes) - times of detected aftershocks
    """
    
    # Create empty list to store aftershock times
    aftershocks = []
    
    for t in range(0,len(trig)):
        print('Processing trigger %i of %i...' % (t+1,len(trig)))
        print(trig[t]['time'])
        
        # Take slice of signal around trigger time
        temp = st.copy().trim(trig[t]['time']-100., 
                              trig[t]['time']+100.)
        
        # Find first arrivals in signal
        arrival_times2, arrival_inds2 = findFirstArrivals(temp)
        
        # Compare these to arrival times from known event        
        if len(arrival_times2) > 0: 
            # Normalize arrival_times based on first arrival time
            arrival_times1 = [at - arrival_times1[0] for at in arrival_times1]
            arrival_times2 = [at - arrival_times2[0] for at in arrival_times2]
            
            # Find absolute difference between the two sets of arrival times
            time_diff = abs(np.array(arrival_times1) - np.array(arrival_times2))
            
            print('Difference between two sets of arrival times in seconds:')
            print(time_diff)
            
            # If two time differences within 1 second of each other for
            # at least min_stations stations, add time to aftershocks list
            passed_stations = 0
            for d in time_diff:
                if d <= min_time_diff:
                    passed_stations += 1
                    
            print("%i arrival time(s) within %.1f s of known event's arrival times."
                  % (passed_stations, min_time_diff))
            
            if passed_stations >= min_stations:
                aftershocks.append(trig[t]['time'])
                print('Aftershock detected at %s.' % str(aftershocks[-1]))
                        
            print('') # Print blank line between triggers
    
    # If any times in aftershocks within 1 minute of each other, save first one   
    if len(aftershocks) > 1:
        new_aftershocks = [aftershocks[0]] 
        aftershocks.sort() # Sort times from earliest to latest
        for a in range(1,len(aftershocks)):
            if aftershocks[a] - aftershocks[a-1] > 60:
                new_aftershocks.append(aftershocks[a])     
        
        aftershocks = new_aftershocks
        
    print('%i aftershock(s) found.' % len(aftershocks))
    print('') # Print blank line
        
    return(aftershocks)