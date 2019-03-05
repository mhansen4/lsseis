#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:39:03 2019

@author: mchansen
"""

"""
Use this file to specify input parameters and run the other functions in this
package.
"""


# Imports
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from detectLandslides import getStreamObject, findTriggers, detectLandslides
from removeTeleseisms import searchComCatforLandslides
from viewEvents import viewEvents

# Input parameters
lslat = 46.843 # latitude of landslide in deg
lslon = -121.75 # longitude of landslide in deg
radius = 50. # search radius for nearest seismic stations in km
# Define date range to look for landslides in
starttime = UTCDateTime(2011,6,24,0,0)
endtime = UTCDateTime(2011,6,24,23,59)
interval = 6. * 3600.  # interval to split time period up into in seconds

# Create list to store trigger times from seismic signals in
trigger_times = []

# Create list to store teleseisms
teleseisms = []

# Create dataframe to store landslide events in
events_df = pd.DataFrame()

# Set thresholds for landslide search
min_stations = 3 # number of stations that must fit model
min_time_diff = 1.0 # number of seconds first arrival times can differ from model by
fit_stations = range(7,2,-1) # Number of closest stations to fit model to

# Split up big date range into smaller chunks
starts = np.arange(starttime, endtime, interval)
ends = starts + interval

# Loop through smaller date ranges and find add landslide events to dataframe
for i in range(0,len(starts)):
    print('')
    print('Assessing time range %s to %s...' % (starts[i],ends[i]))
    print('')
    st = getStreamObject(starts[i],ends[i],lslat,lslon,radius)  
    st.filter('bandpass', freqmin=1.0, freqmax=5.0)
    st, trig, new_trigger_times, new_teleseisms = findTriggers(lslat,lslon,st,
                                                               trigger_times)
    for trigger_time in new_trigger_times:
        trigger_times.append(trigger_time)
    for teleseism_time in new_teleseisms:
        teleseisms.append(teleseism_time)
    new_events_df = detectLandslides(st,trig,lslat,lslon,min_stations,
                                     min_time_diff,[])
    events_df = events_df.append(new_events_df)
    
# Create dataframe for signal trigger times
triggers_df = pd.DataFrame({'Trigger times': trigger_times}) 

# Reindex dataframe
events_df = events_df.reset_index(drop=True)

# Save dataframes to file
save_dfs = False # Set to True to save event and trigger time dataframes to CSVs
if save_dfs:
    events_df.to_csv('nisqually_predicted_events.csv')
    triggers_df.to_csv('nisqually_trigger_times.csv')

# Look for landslides in ComCat and return
network = st[0].stats.network # Seismic network of station closest to landslide
station = st[0].stats.station # Seismic station closest to landslide
possible_ls = searchComCatforLandslides(starttime,endtime,lslat,lslon,
                                        network,station)  

# Look at specific event
"""
eventrow = filt_events_df.iloc[26]
viewEvents(eventrow,lslat,lslon,radius,plot_arrival_times=True,plot_predictions=True)
"""
#for index, eventrow in events_df.iterrows():
#    viewEvents(eventrow,lslat,lslon,radius)