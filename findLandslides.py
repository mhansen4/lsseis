import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
from obspy.core.stream import Stream
from obspy.signal.trigger import coincidence_trigger as ct
from reviewData import reviewData
from removeTeleseisms import removeTeleseisms, searchComCatforLandslides
from findFirstArrivals import findFirstArrivals, plotFirstArrivals

import warnings
warnings.filterwarnings("ignore")

def getStreamObject(starttime,endtime,lslat,lslon,radius=100.):
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
    
    # Limit number of stations returned to 12
    if len(st) > 12:
        st = st[:12]
        print('Returning only %i stations.' % len(st))
    
    network = st[0].stats.network # seismic network first stream came from
    station = st[0].stats.station # first station in stream
    
    return(st, network, station)
      
def findTriggers(st, trigger_times):
    """
    Finds spikes/triggers in stream st using the obspy coincidence_trigger function 
    and stores in a list called trigger_times.
    INPUT
    st - obspy stream object with seismic data
    trigger_times (list of UTCDateTimes) - times returned from obspy.signal.trigger
        coincidence_trigger() for stream object as potential landslide events
    OUTPUT
    trig - list of coicidence_trigger objects
    trigger_times
    """
    
    print('Modifying stream object...')
    st = reviewData.attach_distaz_IRIS(st, lslat, lslon)
    st = st.sort(keys=['rdist', 'channel'])
    st.filter('bandpass', freqmin=1.0, freqmax=5.0)
    
    network = st[0].stats.network
    station = st[0].stats.station # first station in stream
    
    print('Looking for triggers...')
    
    # play with these parameters
    init_trig = ct("recstalta", 2.5, 1, st, 3, sta=4., lta=20.)
    print('%i triggers initially returned.' % len(init_trig))
    if len(init_trig) > 0:
        trig, removed_triggers = removeTeleseisms(starttime,endtime,network,
                                                  station,init_trig)
        
        print('%i teleseism(s) found, %i trigger(s) remaining.' % 
              (len(removed_triggers),len(trig)))
        
        print('')
        
        trigger_times = [t['time'] for t in trig]
        
    return(trig, trigger_times)

def predictArrivalTimes(st, arrival_secs, stations_to_fit): 
    """
    Fit linear regression to calculated arrival times (converted to seconds)
    and predict arrival times from regression model.
    INPUT
    st - obspy stream object with seismic data
    trig - list of coicidence_trigger objects
    arrival_secs (list of floats) - list of times in seconds since arrival time 
        at station closest to landslide
    OUTPUT
    pred_arrival_secs (list of floats) - list of times in seconds since arrival 
        time at station closest to landslide, predicted from linear regression
    m (float) - linear regression slope
    b (float) - linear regression intercepts    
    """
    
    # Calculate distance from each seismic station to closest station
    station_dist = [tr.stats.rdist for tr in st]
            
    # Get linear regression using minimum number of stations
    print('Fitting linear regression to %i stations...' % stations_to_fit)
    m, b = np.polyfit(station_dist[1:stations_to_fit+1], 
                      arrival_secs[1:stations_to_fit+1], 1)         
    print('Slope of linear regression = %f' % m)  

    # Use linear regression model to predict arrival times
    pred_arrival_secs = np.dot(m, station_dist) + b       
                
    return(pred_arrival_secs, m, b)

def findLandslides(st, trig, event_times, trig_i = [], 
                   min_stations = 3, min_time_diff = 1.0):
    """
    Loops through all triggers in trigger_times and finds the first arrival times 
    of all traces. Calculates a simple linear regression using the obspy polyfit 
    function. If the arrival times predicted by the regression are within 
    min_time_diff for at least min_stations number of stations, a landslide is 
    detected.
    INPUT
    st - obspy stream object with seismic data
    trig - list of coicidence_trigger objects
    event_times (list of UTCDateTimes) - times from trigger_times which are 
        potential landslides based on various checks
    trig_i (list) - list of trig indices to evaluate
    min_stations (int) - minimum number of stations that need to display a linear
        moveout in first arrival times for trigger to count as event
    min_time_diff (float) - minimum time in seconds that first arrival time at
        station can differ from predicted arrival time for a perfect linear
        moveout in order for station to pass
    OUTPUT
    event_times
    m_list (list of floats) - list of linear regression slopes
    b_list (list of floats) - list of linear regression intercepts
    """
    
    if trig_i == []:
        trig_i = range(0,len(trig))
    
    m_list = []
    b_list = []
    
    for t in trig_i:
        print('Processing trigger %i of %i...' % (t+1,len(trig)))
        print(trig[t]['time'])
        
        # Take slice of signal around trigger time
        temp = st.copy().trim(trig[t]['time']-100., 
                              trig[t]['time']+300.)
        
        # Delete problematic stations (will need to add to)
        channels_to_remove = []
        for channel in temp:
            if channel.std() < 2.0:
                channels_to_remove.append(channel)
        
        for channel in channels_to_remove:
            #if len(st.select(station = channel.stats.station)) > 0:
            temp.remove(channel)
        
        # Find first arrivals in signal
        F4, arrival_times, arrival_inds = findFirstArrivals(trig[t]['time']-100., 
                                                            temp, min_stations,
                                                            plot_checkcalcs = False)
        
        # Check if arrival time belongs to event        
        if len(arrival_times) > 0:  
            stations_to_fit = [7,3]
            try_less_stations = True
            for stations in stations_to_fit:
                if try_less_stations:
                    # Find seconds between first arrival at each station and the closest station
                    arrival_secs = [time - arrival_times[0] for time in arrival_times]  
                    
                    # Get predicted arrival times
                    arrival_secs_pred, m, b = predictArrivalTimes(temp, arrival_secs, stations)
                    
                    # Find difference in seconds between predicted and real arrival times
                    arrival_secs = np.array(arrival_secs)
                    pred_diff = arrival_secs - arrival_secs_pred
                    
                    print('Difference between predicted arrival times and actual:')
                    print(pred_diff)
                    
                    # If predicted arrival times within some number of 
                    # seconds of actual arrival times for min number of  
                    # stations, add trigger time to event list
                    check_pass_count = 0
                    for diff in pred_diff:
                        if abs(diff) <= min_time_diff:
                            check_pass_count += 1
                            
                    print('%i predicted arrival time(s) within %.1f s of actual times.' 
                          % (check_pass_count, min_time_diff))
                            
                    if check_pass_count >= min_stations:
                        print('Landslide detected at %s.' % str(arrival_times[0]))
                        
                        # Save event time and linear regression parameters
                        event_times.append(arrival_times[0])
                        m_list.append(m)
                        b_list.append(b)
                        
                        try_less_stations = False # Don't try any new regressions
                                
    #                    plotFirstArrivals(trig[t]['time'],temp,F4,
    #                                      arrival_times,arrival_inds)
                        
            print('') # Print blank line between triggers
                
    # If any times in event_times within 100 seconds of each other, save first one
    if len(event_times) > 0:
        event_times.sort() # Sort times from earliest to latest
        old_event_times = event_times # Store original list of event times
        event_times = [event_times[0]] # Create new list for event times
        for e in range(1,len(old_event_times)):
            if old_event_times[e] - old_event_times[e-1] > 100:
                event_times.append(old_event_times[e])            
                
    print('%i possible landslide(s) found.' % len(event_times))
        
    return(event_times, m_list, b_list)
    
def viewEvent(st,lslat,lslon,event_time):
    st = reviewData.attach_distaz_IRIS(st, lslat, lslon)
    st = st.sort(keys=['rdist', 'channel'])
    st.filter('bandpass', freqmin=1.0, freqmax=5.0)
    
    # Trim signal to event time
    st_trim = st.copy().trim(event_time-100.,event_time+300.)
    
    # Plot event
    reviewData.InteractivePlot(st_trim)

    # Delete problematic stations (will need to add to)
    channels_to_remove = []
    for channel in st_trim:
        if channel.std() < 2.0:
            channels_to_remove.append(channel)
    
    for channel in channels_to_remove:
        #if len(st.select(station = channel.stats.station)) > 0:
        st_trim.remove(channel)
    
    # Plot first arrivals
    F4, arrival_times, arrival_inds = findFirstArrivals(event_time-100.,st_trim,
                                                        min_stations,
                                                        plot_checkcalcs=False)
    plotFirstArrivals(event_time-100.,st_trim,F4,arrival_times,arrival_inds)

    # Plot linear regression of arrival_times
    if len(arrival_times) > 0:
        # Get real and predicted arrival times
        arrival_secs = [time - arrival_times[0] for time in arrival_times]
        arrival_secs_pred, m, b = predictArrivalTimes(st_trim, arrival_secs,3)

        # Calculate distance from each seismic station to closest station
        station_dist = [tr.stats.rdist for tr in st]
        
        # Figure out how many arrival times to plot                                   
        if len(st_trim) >= 8:      
            f = 8 # number of first arrival times to plot
        else:
            f = len(st_trim)
            
        plt.figure()
        plt.title('Linear Moveout Prediction and First Arrival Times')
        plt.plot(station_dist[:f], arrival_secs[:f], 'ro', 
                 label='Signal arrival times')
        plt.plot(station_dist[:f], np.dot(m, station_dist[:f]) + b, '--b',
                 label='Line of best fit')
        plt.xlabel('Distance from station to landslide (km)')
        plt.ylabel('Landslide signal arrival time (s)')
        plt.legend()
        plt.show()
    
    return

##############################################################################    
# Input parameters
    
lslat = 46.843
lslon = -121.75
radius = 50.

starttime = UTCDateTime(2011,6,25,16,0)
endtime = UTCDateTime(2011,6,25,16,59)
interval = 1. * 3600.  # seconds

starts = np.arange(starttime, endtime, interval)
ends = starts + interval

trigger_times = []
event_times = []

min_stations = 3
min_time_diff = 1.0

for i in range(0,len(starts)):
    st, network, station = getStreamObject(starts[i],ends[i],lslat,lslon,radius)  
    trig, trigger_times = findTriggers(st, trigger_times)
    event_times, m_list, b_list = findLandslides(st, trig, event_times,[],
                                                 min_stations,min_time_diff) 

possible_ls = searchComCatforLandslides(starttime,endtime,lslat,
                                          lslon,network,station)    

#print('Plotting found events...')
#for event in event_times:
#    viewEvent(st,lslat,lslon,event)
    
#viewEvent(st,lslat,lslon,trigger_times[25])
