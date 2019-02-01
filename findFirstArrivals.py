#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 13:02:25 2018

@author: mchansen
"""

import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
import obspy.realtime.signal as signal
from scipy import signal as spsignal

import warnings
warnings.filterwarnings("ignore")

"""
findFirstArrivals - Transforms obspy stream objects in order to locate triggering
    events in seismic signals, specifically landslides.
"""

def findFirstArrivals(starttime, st, plot_checkcalcs = False):
    """
    Transforms each trace in obspy stream object to determine the "first arrival",
    or onset of the trigger that might be a landslide. Returns time of this onset
    for each trace, as well as the signal index corresponding to this time and 
    the transformed traces for plotting. Adapts methodology from Baillard et al 
    (2014) paper on using kurtosis to pick P- and S-wave onset-times.
    INPUTS
    starttime (UTCDateTime) - starting time of stream object
    st - obspy stream object with seismic information
    plot_checkcalcs (boolean) - optional, set to True to visualize calculations
        at each step of signal transformation process
    OUTPUTS
    F4_traces (2D numpy array) - traces from stream object after fourth and final transformation 
        step, useful for plotting first arrival times
    arrival_times (list of UTCDateTimes) - first arrival times for each trace;
        if multiple arrivals, algorithm selects arrival closest to arrival time
        of first trace (first, first arrival time picked for first trace)
    arrival_i (list of numpy int64s) - indices of first arrival times within
        traces
    """
    
    arrival_times = []
    arrival_i = []
    
    max_length = 0
    for trace in st:
        if len(trace) > max_length:
            max_length = len(trace)
    
    F4_traces = np.zeros((len(st),max_length))
    
    # Iterate through every channel in stream
    for t in range(0,len(st)):     
        # Take kurtosis of trace
        F1 = signal.kurtosis(st[t], win=5000.0)
            
        # Change first part of signal to avoid initial noise
        F1[:1000] = F1[1000]
    
        # Remove negative slopes
        F2 = np.zeros(len(F1))
        F2[0] = F1[0]
        for i in range(0,len(F1)):
            dF = F1[i] - F1[i-1]
            if dF >= 0:
                d = 1
            else:
                d = 0
            F2[i] = F2[i-1] + (d * dF)

        # Remove linear trend
        F3 = np.zeros(len(F2))
        b = F2[0]
        a = (F2[-1] - b)/(len(F2) - 1)
        for i in range(0,len(F2)):
            F3[i] = F2[i] - (a*i + b)
            
        # Smooth F3 curve
        F3smooth = spsignal.savgol_filter(F3,501,1)
        # Define lists of maxima and minima
        M = [] # maxima
        M_i = [] # maxima indices
        m = [] # minima
        m_i = [] # minima indices
        for i in range(1,len(F3smooth)-1):
            if F3smooth[i] > F3smooth[i-1] and F3smooth[i] > F3smooth[i+1]:
                M.append(F3smooth[i])
                M_i.append(i)
            if F3smooth[i] < F3smooth[i-1] and F3smooth[i] < F3smooth[i+1]:
                m.append(F3smooth[i])
                m_i.append(i)
        M.append(0)
        M_i.append(len(F3smooth))
        if len(m_i) == 0:
            m_i.append(np.argmin(F3smooth))
           
        # Scale amplitudes based on local maxima
        F4 = np.zeros(len(F3smooth))
        Mlist = []
        for i in range(0,len(F3smooth)):
            # Find next maximum
            for j in reversed(range(0,len(M))):
                if i <= M_i[j]:
                    thisM = M[j]
            if i < m_i[0]:
                thisM = F3[i]
            Mlist.append(thisM)
            
            # Calculate difference between F3 value and next maximum
            T = F3smooth[i] - thisM
            
            # Calculate new signal
            if T < 0:
                F4[i] = T
            else:
                F4[i] = 0 
                
            if len(M) > 1:
                for j in range(1,len(m)):
                    if i < m_i[j] and i > M_i[j-1]:
                        F4[i] = 0
                
        # Plot each step 
        if plot_checkcalcs:
            plt.figure()
                
            plt.subplot(511)
            plt.title('Station = ' + st[t].stats.station)
            plt.plot(st[t].data)
                
            plt.subplot(512)
            plt.plot(F1)
    
            plt.subplot(513)
            plt.plot(F2)
                
            plt.subplot(514)
            plt.plot(F3)
            plt.plot(F3smooth,'r')
    
            plt.subplot(515)
            plt.plot(F4)
                
            plt.show()   
        
        sample_rate = st[t].stats.sampling_rate
        
        # Find first arrival time
        
        # First find how many spikes were detected 
        spike_values = np.where(F4 < min(F4)*.2)[0]
        mins = [spike_values[0]]
        not_mins = [spike_values[1]]
        for i in range(2,len(spike_values)):
            # Find next nonconsecutive index and store in list
            if spike_values[i] != (mins[-1] + 1) and spike_values[i] != (not_mins[-1] + 1):
                mins.append(spike_values[i])
            else: 
                not_mins.append(spike_values[i])

        # Set minimum that is closest in time to previous station's arrival time 
        # but AFTER it as arrival time
        if len(mins) > 1 and t > 0:
            closest_min = mins[0]
            for i in range(1,len(mins)):
                if abs(arrival_times[t-1] - UTCDateTime(mins[i]/sample_rate + \
                   starttime.timestamp)) < abs(arrival_times[t-1] - \
                       UTCDateTime(closest_min/sample_rate + starttime.timestamp)):
                    closest_min = mins[i]
            arrival_index = closest_min
        else:
            arrival_index = mins[0]
    
        F4_traces[t] = np.interp(range(0,max_length),range(0,len(F4)),F4)
        arrival_i.append(arrival_index) # Index of first arrival time
        arrival_timedelta = arrival_index/sample_rate
        arrival_times.append(UTCDateTime(arrival_timedelta + starttime.timestamp))      
        
    return(F4_traces, arrival_times, arrival_i)

def plotFirstArrivals(signal_time, st, F4_traces, arrival_times, arrival_i):   
    """
    Plot first arrival times as vertical dashed lines on transformed stream 
    object traces. Useful for checking if signal displaying moveout consistent
    with a landslide, and if right arrival time is being selected as the 
    first arrival.
    INPUTS
    starttime (UTCDateTime) - starting time of stream object
    st - obspy stream object with seismic information
    F4_traces (2D numpy array) - traces from stream object after fourth and final transformation 
        step, useful for plotting first arrival times
    arrival_times (list of UTCDateTimes) - first arrival times for each trace;
        if multiple arrivals, algorithm selects arrival closest to arrival time
        of first trace (first, first arrival time picked for first trace)
    arrival_i (list of numpy int64s) - indices of first arrival times within
        traces
    """
    plt.figure()
    
    numstations = len(st)
    if len(st) > 7:
        numstations = 7
        
    for i in range(0,numstations):
        subplot_id = numstations*100 + 10 + i+1
        plt.subplot(subplot_id)
        plt.ylabel(st[i].stats.station)
        plt.xlabel(signal_time)
        plt.xlim((0,len(st[i])))
        plt.plot(F4_traces[i])
        plt.axvline(x=arrival_i[i], color = 'r', linestyle = '--') 
        
    plt.show()