#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:38:28 2018

@author: mchansen
"""
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from reviewData import reviewData
from findFirstArrivals import findFirstArrivals
from detectLandslides import getStreamObject, findTriggers, detectAftershocks
from removeTeleseisms import searchComCatforLandslides
from viewEvents import viewEvents

import warnings
warnings.filterwarnings("ignore")

"""
Takes a known landslide event and the seismic signals created by it at nearby 
stations to signals later on to find potential 'aftershock' events.
"""
# Input parameters
lslat = 46.843 # latitude of landslide in deg
lslon = -121.75 # longitude of landslide in deg
radius = 50. # search radius for nearest seismic stations in km
# Define date range to look for landslides in
starttime = UTCDateTime(2011,6,24,16,0)
endtime = UTCDateTime(2011,6,24,23,0)
interval = 7. * 3600.  # interval to split time period up into in seconds

# Create dataframe to store landslide events in
aftershocks_df = pd.DataFrame()

# Set thresholds for landslide search
min_time_diff = 1.0 # number of seconds arrival times can differ from known event by
min_stations = 4 # number of stations that must be within min_time_diff of known event
fit_stations = range(7,2,-1) # Number of closest stations to fit model to

# First look at known event
print('Grabbing known event...')
known_event_time = UTCDateTime('2011-06-24T16:40:46.019900Z')
st1, network, station = getStreamObject(known_event_time-100.,
                                        known_event_time+100.,
                                        lslat,lslon,radius)  
st1.filter('bandpass', freqmin=1.0, freqmax=5.0)
st1, trig, temp_trigger_times, temp_teleseisms = findTriggers(lslat,lslon,st1,[])
"""
# Review signal to remove any traces before doing additional processing
zp1 = reviewData.InteractivePlot(st1)

# Record deleted traces and delete from st1
stations_to_delete = [item.split('.')[0] for item in zp1.deleted]
for channel in st1:
    if channel.stats.station in stations_to_delete:
        st1.remove(channel)
"""
# Store st1 stations in list so that same stations can be evaluated later
st1_stations = [station.stats.station for station in st1]
# Find arrival times for known event
print('Getting arrival times from known event...')
arrival_times1, arrival_inds1 = findFirstArrivals(st1)

# Search for aftershocks

# Split up big date range into smaller chunks
starts = np.arange(starttime, endtime, interval)
ends = starts + interval

# Create lists for storing and sorting triggers
trigger_times = []
teleseisms = []
aftershocks = []

print('Searching for aftershocks...')

# Loop through smaller date ranges and find add landslide events to dataframe
for i in range(0,len(starts)):
    print('')
    print('Assessing time range %s to %s...' % (starts[i],ends[i]))
    print('')
    st2, network, station = getStreamObject(starts[i],ends[i],lslat,lslon,radius)  
    st2.filter('bandpass', freqmin=1.0, freqmax=5.0)
    # Grab triggers
    st2, trig, new_trigger_times, new_teleseisms = findTriggers(lslat,lslon,st2,trigger_times)
    # Check that new stream object has same stations as known event's object
    st2_stations = [station.stats.station for station in st2]
    if st2_stations != st1_stations:
        print('Station mismatch. Please fix.')
        break
    for trigger_time in new_trigger_times:
        trigger_times.append(trigger_time)
    for teleseism_time in new_teleseisms:
        teleseisms.append(teleseism_time)
    # Search triggers for aftershocks
    new_aftershocks = detectAftershocks(st2,trig,arrival_times1,min_stations,
                                        min_time_diff)
    for aftershock in new_aftershocks:
        aftershocks.append(aftershock)
        
# TO-DO: Save aftershocks to csv file

# Look for landslides in ComCat and return
#possible_ls = searchComCatforLandslides(starttime,endtime,lslat,lslon,
#                                        network,station)  

# Look at specific event
"""
eventrow = filt_events_df.iloc[26]
viewEvents(eventrow,lslat,lslon,radius,plot_arrival_times=True,plot_predictions=True)
"""
"""
for index, eventrow in events_df.iterrows():
    viewEvents(eventrow,lslat,lslon,radius)
"""