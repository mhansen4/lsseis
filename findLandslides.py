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
    print('Retrieving data for radius = %i km...' % int(radius))
    st = Stream()
    st_init = reviewData.getepidata(lslat, lslon, starttime, tstart=0.,
                               tend=endtime-starttime, minradiuskm=0., 
                               maxradiuskm=radius, chanuse=channels, 
                               location='*', clientnames=['IRIS'],
                               savedat=False, detrend='demean')
    for trace in st_init:
        st.append(trace)
    
    # Check if number of traces in stream object less than minimum; if so,
    # increment radius by 50 km and search for data again
    maxradius = 350 # maximum radius to search within (in km)
    mintraces = 5 # minimum number of traces accepted
    while (not st or len(st) < mintraces) and radius <= maxradius:
        radius += 50. # km
        print('Incrementing radius to %i km and retrieving data...' % int(radius))
        st = reviewData.getepidata(lslat, lslon, starttime, tstart=0.,
                                   tend=endtime-starttime, minradiuskm=0., 
                                   maxradiuskm=radius, chanuse=channels, 
                                   location='*', clientnames=['IRIS'],
                                   savedat=False, detrend='demean')
    
    if st:
        print('%i stations within %i km of landslide.' % (len(st),int(radius)))
    
    return(st)

# Get seismic signal files from specified seismic network        
def findLandslides(st,useTrig=True,plotTrig=False):
    """
    Uses seisk reviewData module to grab seismic data using FDSN webservices
    for stations within a specified radius of the landslide as a stream object.
    Increments radius until minimum number of traces or maximum search radius
    is achieved.
    INPUTS
    st - obspy stream object with seismic data
    useTrig (boolean) - optional, set to False to skip checking for landslides 
                        and just plot stream object
    plotTrig (boolean) - optional, set to False to skip plotting and just check
                         for landslides in stream object
    OUTPUTS
    trigger_times (list of UTCDateTimes) - times returned from obspy.signal.trigger
        coincidence_trigger() for stream object as potential landslide events
    event_times (list of UTCDateTimes) - times from trigger_times which are 
        potential landslides based on various checks
    """
    
    print('Modifying stream object...')
    st = reviewData.attach_distaz_IRIS(st, lslat, lslon)
    st = st.sort(keys=['rdist', 'channel'])
    st.filter('bandpass', freqmin=1.0, freqmax=5.0)
    
    network = st[0].stats.network
    station = st[0].stats.station # first station in stream
    
    trigger_times = []
    event_times = []
    
    print('Looking for triggers...')
    if useTrig:
        # play with these parameters
        init_trig = ct("recstalta", 2.5, 1, st, 3, sta=4., lta=20.)
        print('%i triggers initially returned.' % len(init_trig))
        if len(init_trig) > 0:
            trig, removed_triggers = removeTeleseisms(starttime,endtime,network,
                                                      station,init_trig)
            trigger_times = [t['time'] for t in trig]
            
            for t in range(0,len(trig)):
                # Select part of signal around trigger
                # If multiple triggers, cut signal so that two triggers aren't in same slice
                if t > 1 and ((t < (len(trig)-1)) and (trig[t+1]['time']-trig[t]['time'] < 300)):
                    temp = st.copy().trim(trig[t]['time'] - 100., trig[t+1]['time'] - 100.)
                    # If traces in temp are too short, return to original length
                    if len(temp[0]) < 10000:
                        temp = st.copy().trim(trig[t]['time'] - 100., trig[t]['time'] + 300.)  
                else:
                    temp = st.copy().trim(trig[t]['time']-100., trig[t]['time']+300.)
                
                # Delete problematic stations (will need to add to)
                channels_to_remove = []
                for channel in temp:
                    if channel.std() < 1.0:
                        channels_to_remove.append(channel)
                
                for channel in channels_to_remove:
                    #if len(st.select(station = channel.stats.station)) > 0:
                    temp.remove(channel)
                
                # Find first arrivals in signal
                F4, arrival_times, arrival_inds = findFirstArrivals(trig[t]['time']-100., temp, plot_checkcalcs = False)
                
                # TO-DO: Figure out how to check if arrival time belongs to event
                diff = np.asarray(arrival_times) - arrival_times[1]
                if sum(1 for i in diff if i <= 50. and i > 0) >= 4:
                    event_times.append(arrival_times[0])
                    plotFirstArrivals(trig[t]['time']-100., temp, F4, arrival_times, arrival_inds)                    
                    
                # Plot arrival times as a function of distance away from landslide
                if 1:
                    if len(arrival_times) > 0:
                        arrival_seconds = []
                        for time in arrival_times:
                            arrival_seconds.append(time.second)
                            
                        station_dist = []
                        for tr in temp:
                            station_dist.append(tr.stats.rdist)
                        
                        m, b = np.polyfit(station_dist, arrival_seconds, 1)
                        print('m = ', m)
                        
                        plt.figure()
                        plt.plot(station_dist, arrival_seconds, 'ro', 
                                 label='Signal arrival times')
                        plt.plot(station_dist, np.dot(m, station_dist) + b, '--b',
                                 label='Line of best fit')
                        plt.xlabel('Distance from station to landslide (km)')
                        plt.ylabel('Landslide signal arrival time (s)')
                        plt.legend()
                        
                # Plot signal
                if plotTrig:
                    print('Plotting signal...')
                    reviewData.InteractivePlot(temp,xlim=(0,400))
        
    return(trigger_times, event_times)
    
# Input parameters
    
lslat = 61.226
lslon = -152.167
radius = 50.

starttime = UTCDateTime(2014,9,10,19,20,0)
endtime = UTCDateTime(2014,9,10,19,22,0)
absendtime = endtime

st = getStreamObject(starttime,endtime,lslat,lslon,radius)  
network = st[0].stats.network
station = st[0].stats.station # first station in stream
      
trigger_times, event_times = findLandslides(st,useTrig=True,plotTrig=True) 
possible_event_times = searchComCatforLandslides(starttime,absendtime,lslat,
                                                 lslon,network,station)
