#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:39:03 2019

@author: mchansen
"""
# Imports
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from findLandslides import getStreamObject, findTriggers, findLandslides
from removeTeleseisms import searchComCatforLandslides
from viewEvents import viewEvents

# Input parameters
lslat = 46.843 # latitude of landslide in deg
lslon = -121.75 # longitude of landslide in deg
radius = 50. # search radius for nearest seismic stations in km
# Define date range to look for landslides in
starttime = UTCDateTime(2011,6,27,16,0)
#endtime = UTCDateTime(2011,7,7,4,59)
endtime = UTCDateTime(2011,6,27,22,0)

# Create list to store trigger times from seismic signals in
trigger_times = []

# Create dataframe to store landslide events in
events_df = pd.DataFrame()

# Set thresholds for landslide search
min_stations = 3 # number of stations that must fit model
min_time_diff = 5.0 # number of seconds first arrival times can differ from model by
fit_stations = range(7,2,-1) # Number of closest stations to fit model to

# Split up big date range into smaller chunks
interval = 6. * 3600.  # seconds
starts = np.arange(starttime, endtime, interval)
ends = starts + interval

# Loop through smaller date ranges and find add landslide events to dataframe
for i in range(0,len(starts)):
    print('')
    print('Assessing time range %s to %s...' % (starts[i],ends[i]))
    print('')
    st, network, station = getStreamObject(starts[i],ends[i],lslat,lslon,radius)  
    trig, new_trigger_times = findTriggers(st,trigger_times)
    for trigger_time in new_trigger_times:
        trigger_times.append(trigger_time)
    new_events_df = findLandslides(st,trig,fit_stations,min_stations,min_time_diff,[])
    events_df = events_df.append(new_events_df)
    
# Create dataframe for signal trigger times
triggers_df = pd.DataFrame({'Trigger times': trigger_times}) 
    
# Filter events dataframe
min_mvout = -1.0 # lowest slope that model can have
max_intcpt = 50.0 # greatest y-intercept that model can have
filt_events_df = events_df[events_df['Moveout intercept'] >= min_mvout]
filt_events_df = filt_events_df[abs(events_df['Moveout intercept']) <= max_intcpt]

# Reindex dataframe
events_df = events_df.reset_index(drop=True)
filt_events_df = filt_events_df.reset_index(drop=True)

# Save dataframes to file
save_dfs = False # Set to True to save event and trigger time dataframes to CSVs
if save_dfs:
    filt_events_df.to_csv('nisqually_predicted_events.csv')
    triggers_df.to_csv('nisqually_trigger_times.csv')

# Look for landslides in ComCat and return
possible_ls = searchComCatforLandslides(starttime,endtime,radius,lslat,
                                          lslon,network,station)  

# Read in dataframe
events_df = pd.read_csv('nisqually_predicted_events.csv')
events_df = events_df.drop(['Unnamed: 0'], axis=1)

# Look at specific event
eventrow = events_df.iloc[1]
for index, eventrow in events_df.iterrows():
    viewEvents(eventrow,lslat,lslon,radius,min_stations=8,min_time_diff=5.0)