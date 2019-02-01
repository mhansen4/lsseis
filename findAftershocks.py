#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:38:28 2018

@author: mchansen
"""
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
import csv
from obspy import UTCDateTime
from obspy.signal.trigger import coincidence_trigger as ct
import obspy.signal.filter as filte
import obspy.signal.cross_correlation as cc
from reviewData import reviewData
from removeTeleseisms import removeTeleseisms
from findFirstArrivals import findFirstArrivals, plotFirstArrivals

import warnings
warnings.filterwarnings("ignore")

"""
Takes a known landslide event and the seismic signals created by it at nearby 
stations to signals later on to find potential 'aftershock' events.
"""
    
def calculateMismatch(arrival_times1, arrival_times2, anchor_station):
    """ 
    Calculates time difference between arrivals at all stations with respect to
    'anchor' station (usually station closest to landslide, or next closest one
    if signal at closest station is too noisy). Finds mismatch between these 
    time lags at the current trigger and the ones for the known landslide event.
    INPUTS
    arrival_times1 (list of floats) - arrival times at each station, where first
        time in list corresponds to closest station, for known event
    arrival_times2 (list of floats) - arrival times at each station, where first
        time in list corresponds to closest station, for trigger being checked
    anchor_station (int) - list index of station being used as anchor station 
        (if signals from 8 stations are being analyzed, 0 would be the closest 
        station to the landslide and 7 would be the farthest)
    OUTPUT     
    mismatch (list of floats) - difference in seconds between time lag lists for
        known landslide event and event being checked
    """
    
    # Calculate time lag at each station with arrival time at an anchor station
    anchor_station = 1
    diff1 = []
    for at1 in arrival_times1:
        diff1.append(at1 - arrival_times1[anchor_station])
     
    # Calculate time lag at each station for check event using same anchor station
    diff2 = []
    for at2 in arrival_times2:
        diff2.append(at2 - arrival_times2[anchor_station])
    
    # Compare time lags
    mismatch = []
    for i in range(0,len(diff1)):
        mismatch.append(diff2[i] - diff1[i])
        
    return(mismatch)
    
def take1DXCorr(signal1, signal2, shift=20000, plotcc=True):
    """ 
    Takes 1D cross-correlation between two signals and returns shift and correlation
    coefficient for when signals match best.
    INPUTS
    signal1 (obspy trace object) - trace 1
    signal2 (obspy trace object) - trace 2 to correlate with trace 1
    shift (int) - optional, total length of samples to shift for cross-correlation
    plotcc (boolean) - optional, set to True to visualize cross-correlation
    OUTPUT     
    best_shift (int) - index of max cross-correlation value
    maxcoeff (float) - max cross-correlation value
    """
    
    cross_cor = cc.correlate(signal1, signal2, shift, demean=True, 
                             normalize=True, domain='time')
    best_shift, maxcoeff = cc.xcorr_max(cross_cor)
    # zero_shift_coeff = cross_cor[shift]

    if plotcc:
        plt.figure()

        plt.subplot(411)
        plt.plot(signal1)
        plt.ylabel('Signal 1')
        x1 = np.arange(0,len(signal1))

        plt.subplot(412)
        plt.plot(signal2)
        plt.ylabel('Signal 2')
        x2 = np.arange(0,len(signal2))
        
        plt.subplot(413)
        shift = np.median(x1) - np.median(x2)
        plt.plot(x1,signal1)
        plt.plot(x2+shift,signal2)
        plt.ylabel('Signal comparison')

        plt.subplot(414)
        plt.plot(cross_cor)
        plt.ylabel('Cross-correlation\ncoefficient')
        xloc = plt.xticks()[0]
        xshifts = [int(x) - shift for x in xloc]
        plt.xticks(xloc, xshifts)
        plt.show() 
    
    # return(zero_shift_coeff)
    return(best_shift, maxcoeff)
    
def getEventDF(found_events, time_diff = 100.0):
    """ 
    Filters found_events array for duplicate events and puts into a Pandas
    dataframe without the duplicates.
    INPUTS
    found_events (2D numpy array) - possible landslide event times from
        seismic data with the max mismatch, RMS error, and max cross-
        correlation coefficient
    time_diff (float) -  max amount of time in seconds two events can differ
        before being considered separate events
    OUTPUT     
    df (pandas dataframe) - has unique events from found_events 
    """
    
    header_list = ['Event time','Max mismatch','RMS error','Max xcorr coeff']
    df = pd.DataFrame(found_events, columns=header_list)
        
    # Filter found_events for duplicates
    df = df.sort_values(by=['Event time']) # Reorder dataframe by increasing times
    header_list.append('Drop?')
    df = df.reindex(columns = header_list) # Add column for marking which rows to drop  
    df.index = np.arange(0, len(df)) # Set index to match rows
    
    for i in range(1,len(df)):
        # If two successive times are within 100 s of each other
        if df.iloc[i]['Event time'] - df.iloc[i-1]['Event time'] < time_diff:
            # Mark the row with the larger rms error as a drop
            if df.iloc[i]['RMS error'] > df.iloc[i-1]['RMS error']:
                 df['Drop?'][i-1] = True
            else: # Keep the earliest time
                df['Drop?'][i] = True
                
    df = df[df['Drop?'] != True] # Create new dataframe without drops
    df = df.drop('Drop?', axis=1) # Drop drops column
    df.index = np.arange(0, len(df)) # Set index to match rows
    
    return(df)
    
###############################################################################
# USER DEFINED VARIABLES

# Landslide coordinates
# Nisqually
lat = 46.844618
lon = -121.752262 

# Read in event times
f = open('../Events/Nisqually/Nisqually_event_times.csv')
csv_f = csv.reader(f)
event_times = []
for row in csv_f:
    event_times.append(row[6])
del event_times[0] # Delete header
event_times = [UTCDateTime(date) for date in event_times]

# Establish time window to search over for known event
starttime1 = event_times[0] - timedelta(seconds=100)
endtime1 = starttime1 + timedelta(seconds=400)

# Define time window to search over for aftershocks
starttime2 = UTCDateTime(2011,6,24,16,30)
endtime2 = UTCDateTime(2011,6,24,17,30)
interval = 3600. # seconds, how big seismic signal segements will be

# User-defined thresholds for narrowing down aftershock events
# TO-DO: determine thresholds for cross-correlation
rms_thresh = 2.0 # RMS error threshold, all events with higher error excluded
stations_to_match = 4 # No. of stations which must have similar arrival times

# Frequency bounds for bandpass filter
fmin = 1.0 # Hz
fmax = 5.0 # Hz

# Don't change anything below this line unless something breaks!
###############################################################################
## LOOK AT FIRST EVENT

# Read in data from IRIS
print('Analyzing known event...')

tstart = 0.
channels = 'EHZ,BHZ,HHZ'
st1 = reviewData.getepidata(lat, lon, starttime1, tstart=tstart,
                           tend=endtime1-starttime1, minradiuskm=0., 
                           maxradiuskm=30., chanuse=channels, 
                           location='*', clientnames=['IRIS'],
                           savedat=False, detrend='demean')

# Get station coordinates and sort from closest to farthest
st1 = reviewData.attach_distaz_IRIS(st1, lat, lon)
st1 = st1.sort(keys=['rdist', 'channel'])

# Apply bandpass filter to signals
st1.filter('bandpass', freqmin=fmin, freqmax=fmax)
#st1.filter('highpass', freq=fmin)

# Review signal to remove any traces before doing additional processing
# zp1 = reviewData.InteractivePlot(st1)

# Record deleted traces and delete from st1
# stations_to_delete = [item.split('.')[0] for item in zp1.deleted]
stations_to_delete = ['RER','RVC','WPW']
for channel in st1:
    if channel.stats.station in stations_to_delete:
        st1.remove(channel)

# Store st1 stations in list so that same stations can be evaluated later
st1_channels = [channel.stats.station for channel in st1]

# Find landslide arrival times for each station   
F4_1, arrival_times1, arrival_inds1 = findFirstArrivals(starttime1, st1, 
                                                        plot_checkcalcs = False)
# plotFirstArrivals(starttime1, st1, F4_1, arrival_times1, arrival_inds1)

## LOOK FOR MORE EVENTS ##
print('Looking for other events...')
teleseisms = [] # Compile list of teleseisms to make sure events aren't being
                # ruled out incorrectly
found_events = [] # Compile list of potential aftershocks

starts = np.arange(starttime2, endtime2, interval)
ends = starts + interval
for start,end in zip(starts,ends):
    print('Retrieving data for times %s to %s...' % (str(start),str(end)))
    st2 = reviewData.getepidata(lat, lon, start, tstart=tstart,
                                tend=end-start, minradiuskm=0., 
                                maxradiuskm=30., chanuse=channels, 
                                location='*', clientnames=['IRIS'],
                                savedat=False, detrend='demean') 
        
    # Get station coordinates and sort by closest to farthest
    st2 = reviewData.attach_distaz_IRIS(st2, lat, lon)
    st2 = st2.sort(keys=['rdist', 'channel'])
    
    # Limit length of each trace to 40000 samples
    max_length = 40000
    for i in range(0,len(st2)):
        if len(st2[i]) > 40000:
            max_length = len(st2[i])
    
    # Apply bandpass filter to signals
    print('Applying filters and removing problematic channels...')
    st2.filter('bandpass', freqmin=fmin, freqmax=fmax)
    #st2.filter('highpass', freq=fmin)
        
    # Delete all channels not in st1
    for channel in st2:
        if channel.stats.station not in st1_channels:
            st2.remove(channel)
            
    # Plot data  
    # reviewData.InteractivePlot(st2) 
    
    # Search for triggers in signal
    print('Looking for triggers...')
    init_trig = ct("recstalta", 2.5, 1, st2, 3, sta=4., lta=20.)
    
    # Check that triggers aren't actually earthquakes
    network = st2[0].stats.network # seismic network first stream came from
    station = st2[0].stats.station # first station in stream
    # This was throwing error "No layer contains this depth" from TauPi module
    trig, new_teleseisms = removeTeleseisms(start,end,network,station,init_trig)
    for newt in new_teleseisms:
        teleseisms.append(newt)
    trig = init_trig # Delete this when above lines are fixed and uncommented
    
    # Take envelope of signal 1  
    st1_env = st1.copy()
    for trace in st1_env:
        trace.data = filte.envelope(trace.data) 
        
    # Loop through returned triggers
    print('Searching list of triggers for landslides...')
    for t in range(0,len(trig)):
        print('Determining if trigger %i of %i is another landslide...' % (t,len(trig)))
        
        # Select part of signal around trigger
        # If multiple triggers, cut signal so that two triggers aren't in same slice
        # Take envelope of signal
        if t > 1 and ((t < (len(trig)-1)) and (trig[t+1]['time']-trig[t]['time'] < 300)):
            temp = st2.copy().trim(trig[t]['time'] - 100., trig[t+1]['time'] - 100.)
            # If traces in temp are too short, return to original length
            if len(temp[0]) < 10000:
                temp = st2.copy().trim(trig[t]['time'] - 100., trig[t]['time'] + 300.)  
            temp_env = temp.copy()
            for trace in temp_env:
                trace.data = filte.envelope(trace.data) 
    
        else:
            temp = st2.copy().trim(trig[t]['time'] - 100., trig[t]['time'] + 300.)  
            temp_env = temp.copy()
            for trace in temp_env:
                trace.data = filte.envelope(trace.data) 
        
        # Get number of traces in signal
        numtraces = len(temp)
                
        # Plot data  
        # reviewData.InteractivePlot(temp) 
        
        # Proceed only if there is more than one trace available 
        if len(temp) > 0:
            bestshift, maxcoeff = take1DXCorr(st1_env[0], temp_env[0], plotcc= False)           

            F4_2, arrival_times2, arrival_inds2 = findFirstArrivals(trig[t]['time'] - \
                                                  100., temp, plot_checkcalcs = False) 
            
            plotFirstArrivals(trig[t]['time'], temp, F4_2, arrival_times2, arrival_inds2)
            
            # Find time difference between corresponding stations
            mismatch = calculateMismatch(arrival_times1, arrival_times2, 1)
            
            # Check for outlier stations
            st_outliers = []
            for i in range(0,len(mismatch)):
                if abs(mismatch[i]) > 5.0:
                    st_outliers.append(i)
            
            # Find RMS error, ignore outliers from calculation
            rmserror= 0.0
            new_mismatch = []
            for i in range(0, numtraces):
                if (numtraces - len(st_outliers)) >= stations_to_match:
                    if i not in st_outliers:
                        rmserror += np.abs(mismatch[i])
                        new_mismatch.append(mismatch[i])
                    n = numtraces-len(st_outliers)
                else:
                    rmserror += np.abs(mismatch[i])
                    new_mismatch.append(mismatch[i])
                    n = numtraces
            rmserror = np.sqrt(rmserror/n)
            
            # Find mismatch with largest absolute value
            max_mismatch = 0
            for i in range(0,len(new_mismatch)):
                if abs(new_mismatch[i]) > abs(max_mismatch):
                    max_mismatch = new_mismatch[i]
            
            # Check that rmserror is below threshold
            if rmserror <= rms_thresh:
                found_events.append([trig[t]['time'], max_mismatch, rmserror, maxcoeff])
                reviewData.InteractivePlot(temp) 
              
# Create pandas dataframe with found_events
print('Compiling information into dataframe df...')

df = getEventDF(found_events)

print('Done.')