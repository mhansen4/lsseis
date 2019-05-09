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

def transformSignals(st, smoothwin=501, smoothorder=1, plot_checkcalcs=False):
    """
    Transforms each trace in obspy stream object to determine the "first arrival",
    or onset of the trigger that might be a landslide. Returns time of this onset
    for each trace, as well as the signal index corresponding to this time and 
    the transformed traces for plotting. Adapts methodology from Baillard et al 
    (2014) paper on using kurtosis to pick P- and S-wave onset-times.
    INPUTS
    st - obspy stream object with seismic information
    smoothwin (int) - Window length in samples for Savgol smoothing of 
        envelopes or kurtosis F3
    smoothorder (int) - Polynomial order for Savgol smoothing
    plot_checkcalcs (boolean) - optional, set to True to visualize calculations
        at each step of signal transformation process
    OUTPUT
    F4_traces (2D numpy array) - traces from stream object after fourth and 
        final transformation step, useful for plotting first arrival times
    """    
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
        # F1[:1000] = F1[1000]
    
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
        F3smooth = spsignal.savgol_filter(F3,smoothwin,smoothorder)
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
    
        # Interpolate transformed signals so they are all the same length
        # NOTE: Does not affect sampling rate -- still same as stream object
        F4_traces[t] = np.interp(range(0,max_length),range(0,len(F4)),F4)
        
    return(F4_traces)
    
def findFirstArrivals(st):
    """
    Transforms each trace in obspy stream object to determine the "first arrival",
    or onset of the trigger that might be a landslide. Returns time of this onset
    for each trace, as well as the signal index corresponding to this time and 
    the transformed traces for plotting. Adapts methodology from Baillard et al 
    (2014) paper on using kurtosis to pick P- and S-wave onset-times.
    INPUT
    st - obspy stream object with seismic information
    OUTPUTS
    arrival_times (list of UTCDateTimes) - first arrival times for each trace;
        if multiple arrivals, algorithm selects arrival closest to arrival time
        of first trace (first, first arrival time picked for first trace)
    arrival_i (list of numpy int64s) - indices of first arrival times within
        traces
    """
    # Pull in signal with kurtosis computed
    F4_traces = transformSignals(st)
    
    # Get stream object start time
    starttime = st[0].stats.starttime
    
    # Compare local minima of transformed signal to find landslide arrival
    # time at each station   
    
    # Get minima for closest station
    t1 = F4_traces[0]
    all_first_mins = spsignal.argrelextrema(t1, np.less)[0]
    first_mins = all_first_mins[np.where(t1[all_first_mins] < min(t1)*.05)[0]]
    
    sample_rate = st[0].stats.sampling_rate
    
    # Loop through mins for closest station, select minima in other signals that
    # is closest in time, and select min from closest station that produces most
    # regular moveout
    arrival_i = [] # Indices of first arrival times in signal traces
    arrival_td = [] # Timedeltas of first arrival times (in s) after starttime
    for m in range(0,len(first_mins)):
        temp_arrival_i = [] # Indices of first arrival times in signal traces
        temp_arrival_td = [] # Timedeltas of first arrival times (in s) after starttime 
        compare_time = UTCDateTime(first_mins[m]/sample_rate + starttime.timestamp)
        
        for t in range(0,len(F4_traces)):   
            F4 = F4_traces[t]
               
            # Find minima in signal (must be at least 5% of size of smallest minimum)
            all_mins = spsignal.argrelextrema(F4, np.less)[0]
            mins = all_mins[np.where(F4[all_mins] < min(F4)*.05)[0]]
            
            # Get sampling rate of this trace
            sample_rate = st[t].stats.sampling_rate
        
            # Set minimum that is closest in time to overall arrival time as
            # signal's arrival time
            if len(mins) > 0:
                arrival_index = mins[0]
                if len(mins) > 1:
                    td1 = abs(compare_time - \
                              UTCDateTime(mins[0]/sample_rate + starttime.timestamp))
                    closest_min = mins[0]
                    for i in range(1,len(mins)):
                        temp_arrival_time = UTCDateTime(mins[i]/sample_rate + starttime.timestamp)
                        td2 = abs(compare_time - temp_arrival_time) 
                        if td2 < td1:
                            td1 = abs(compare_time - temp_arrival_time)
                            closest_min = mins[i]
                                # Pull in signal with kurtosis computed
    F4_traces = transformSignals(st)
    
    # Get stream object start time
    starttime = st[0].stats.starttime
    
    # Compare local minima of transformed signal to find landslide arrival
    # time at each station   
    
    # Get minima for closest station
    t1 = F4_traces[0]
    all_first_mins = spsignal.argrelextrema(t1, np.less)[0]
    first_mins = all_first_mins[np.where(t1[all_first_mins] < min(t1)*.05)[0]]
    
    sample_rate = st[0].stats.sampling_rate
    
    # Loop through mins for closest station, select minima in other signals that
    # is closest in time, and select min from closest station that produces most
    # regular moveout
    arrival_i = [] # Indices of first arrival times in signal traces
    arrival_td = [] # Timedeltas of first arrival times (in s) after starttime
    for m in range(0,len(first_mins)):
        temp_arrival_i = [] # Indices of first arrival times in signal traces
        temp_arrival_td = [] # Timedeltas of first arrival times (in s) after starttime 
        compare_time = UTCDateTime(first_mins[m]/sample_rate + starttime.timestamp)
        
        for t in range(0,len(F4_traces)):   
            F4 = F4_traces[t]
               
            # Find minima in signal (must be at least 5% of size of smallest minimum)
            all_mins = spsignal.argrelextrema(F4, np.less)[0]
            mins = all_mins[np.where(F4[all_mins] < min(F4)*.05)[0]]
            
            # Get sampling rate of this trace
            sample_rate = st[t].stats.sampling_rate
        
            # Set minimum that is closest in time to overall arrival time as
            # signal's arrival time
            if len(mins) > 0:
                arrival_index = mins[0]
                if len(mins) > 1:
                    td1 = abs(compare_time - \
                              UTCDateTime(mins[0]/sample_rate + starttime.timestamp))
                    closest_min = mins[0]
                    for i in range(1,len(mins)):
                        temp_arrival_time = UTCDateTime(mins[i]/sample_rate + starttime.timestamp)
                        td2 = abs(compare_time - temp_arrival_time) 
                        if td2 < td1:
                            td1 = abs(compare_time - temp_arrival_time)
                            closest_min = mins[i]
                            compare_time = UTCDateTime(first_mins[m]/sample_rate + starttime.timestamp)
                    arrival_index = closest_min 
            else:
                arrival_index = 0 # Stops error from being thrown, may result in errors later
       
            temp_arrival_i.append(arrival_index) # Index of first arrival time
    
            # Calculate first arrival timedeltas
            temp_arrival_td.append(arrival_index/sample_rate) # Time in s after starttime
        
        # Determine signal moveout by differing timedeltas
        # Save arrival_i and and arrival_td if deviation of timedeltas is small
        signal_diff = np.std(np.diff(temp_arrival_td))
        
        if m == 0:
            old_signal_diff = signal_diff
            
        if m == 0 or signal_diff < old_signal_diff:
            old_signal_diff = signal_diff
            arrival_i = temp_arrival_i
            arrival_td = temp_arrival_td
                
    # Calculate first arrival times of each individual signal from timedeltas
    arrival_times = [UTCDateTime(td + starttime.timestamp) for td in arrival_td]
        
    return(arrival_times, arrival_i)

def plotFirstArrivals(st, arrival_i):   
    """
    Plot first arrival times as vertical dashed lines on transformed stream 
    object traces. Useful for checking if signal displaying moveout consistent
    with a landslide, and if right arrival time is being selected as the 
    first arrival.
    INPUTS
    st - obspy stream object with seismic information
    arrival_i (list of numpy int64s) - indices of first arrival times within
        traces
    OUTPUT 
    plot
    """
    # Get starttime from stream object
    starttime = st[0].stats.starttime
    
    # Get transformed signal
    F4_traces = transformSignals(st)
    
    plt.figure()
    
    numstations = len(st)
    if len(st) > 8:
        numstations = 8
        
    for i in range(0,numstations):
        subplot_id = numstations*100 + 10 + i+1
        plt.subplot(subplot_id)
        plt.ylabel(st[i].stats.station)
        plt.xlabel(starttime)
        plt.xlim((0,len(st[i])))
        plt.plot(F4_traces[i])
        plt.axvline(x=arrival_i[i], color = 'r', linestyle = '--') 
        
    plt.show()