#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 09:10:50 2019

@author: mchansen
"""

import numpy as np
import pandas as pd
from obspy.signal.trigger import coincidence_trigger as ct
from obspy import UTCDateTime
from reviewData import reviewData
from removeTeleseisms import removeTeleseisms
from findFirstArrivals import transformSignals, findFirstArrivals

# Get stream object for known event
lslat = 46.843 # latitude of landslide in deg
lslon = -121.75 # longitude of landslide in deg
radius = 50. # search radius for nearest seismic stations in km
eventtime = UTCDateTime(2011,6,25,15,20)
channels = 'EHZ,BHZ,HHZ'
st1 = reviewData.getepidata(lslat, lslon, eventtime - 100, tstart=0.,
                            tend=200., minradiuskm=0., 
                            maxradiuskm=radius, chanuse=channels, 
                            location='*', clientnames=['IRIS'],
                            savedat=False, detrend='demean')

# Keep only first 8 stations of st1
if len(st1) > 8:
    st1 = st1[:8]

st1.filter('bandpass', freqmin=1.0, freqmax=5.0)
st1 = reviewData.attach_distaz_IRIS(st1, lslat, lslon)
st1 = st1.sort(keys=['rdist', 'channel'])
sampling_rates1 = [trace.stats.sampling_rate for trace in st1]

# Highlight landslide onset time using kurtosis method, create new stream object
kurt1 = transformSignals(st1)
st1_kurt = st1.copy()
for i in range(0,len(st1_kurt)):
    st1_kurt[i].data = kurt1[i]
    st1_kurt[i].stats.sampling_rate = max(sampling_rates1)

# Create event template using known event
# Set similarity thresholds for each station
# Define list of trace ids trigger detection will be limited to
event_template = {}
similarity_thresholds = {}
threshold = 0.8 # Threshold for nearest station
for i in range(0,4):
    station = st1[i].stats.station
    
    # Stuff kurtosis signal from each station into event template
    st1_ = st1.select(station=station)
    event_template[station] = []
    event_template[station].append(st1_) 
    
    # Progressively lower similarity threshold by 1/8 for stations farther away
    similarity_thresholds[station] = threshold
    threshold = threshold - threshold/8
    
# Define list of trace ids trigger detection will be limited to  
trace_ids = {}
for trace in st1:
    # Assigning value of 1 makes all traces equally weighted
    trace_ids[str(trace).split(' | ')[0]] = 1

# Get new stream object to search over
# Use kurtosis method to find arrival times
starttime = UTCDateTime(2011,6,27,1,20)
#endtime = UTCDateTime(2011,6,26,23,59,59)
endtime = starttime + 10.*60 # 10 minutes later
st2 = reviewData.getepidata(lslat, lslon, starttime, tstart=0.,
                            tend=endtime-starttime, minradiuskm=0., 
                            maxradiuskm=radius, chanuse=channels, 
                            location='*', clientnames=['IRIS'],
                            savedat=False, detrend='demean')
st2.filter('bandpass', freqmin=1.0, freqmax=5.0)
st2 = reviewData.attach_distaz_IRIS(st2, lslat, lslon)
st2 = st2.sort(keys=['rdist', 'channel'])
sampling_rates2 = [trace.stats.sampling_rate for trace in st2]

# Find initial triggers and use to split up new stream into smaller chunks
#trig = ct("recstalta", 2.5, 1, st2, 3, sta=4., lta=20., trace_ids=trace_ids)
#t = 0
#st2_trim = st2.copy().trim(trig[t]['time']-100., trig[t]['time']+100.)
st2_trim = st2.copy()

kurt2 = transformSignals(st2_trim)
st2_kurt = st2.copy()
for i in range(0,len(st2_kurt)):
    st2_kurt[i].data = kurt2[i]
    st2_kurt[i].stats.sampling_rate = max(sampling_rates2)

# Use event template to find triggers
#trig = ct("recstalta", 2.5, 1, st, 3, sta=4., lta=20.)
#trig = ct("classicstalta",5,1,st2_kurt,4,sta=0.5,lta=10,trace_ids=trace_ids,
#          event_templates=event_template,
#          similarity_threshold=similarity_thresholds)
#refined_trig = ct("classicstalta", 2.5, 1, st2_kurt, 3, sta=4., lta=20.,trace_ids=trace_ids,
#                  event_templates=event_template,
#                  similarity_threshold=similarity_thresholds)
refined_trig = ct("classicstalta", 5, 1, st2_kurt, 3, sta=0.5, lta=10.,trace_ids=trace_ids,
                  event_templates=event_template,
                  similarity_threshold=similarity_thresholds)