import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from obspy.signal.trigger import classic_sta_lta
import obspy.signal.filter as filte
from obspy import UTCDateTime
import scipy.signal as spsignal

from sigproc import sigproc

"""
Functions for calculating and saving the pseudo-energies of landslide signals.
"""
    
def getEnergies(signal, energies_df, before, stalta_lowerlim=1.0, 
                stalta_upperlim=2.0, taper_length=0.0, smoothwin=301, 
                smoothorder=3, check_calcs=False):
    """
    Calculates the pseudo-energy for an input signal by determining the event
    start and end times using a comination of the signal envelope and the
    STA/LTA of the signal, and computing the area below the envelope between
    these two times.
    INPUTS
    signal (numpy array) - seismic signal trace from obspy stream object,
        cropped around event
    energies_df (pandas DataFrame) - table of pseudo-energies returned from previous 
        events in list, if evaluating landslide sequences, or empty table if 
        evaluating only one event
    before (float) - seconds before trigger time that signal was cropped; used
        to get initial event time
    stalta_lowerlim (float) - optional; lower STA/LTA threshold for locating 
        signal minima, used for determining event start and end times
    stalta_upperlim (float) - optional; upper STA/LTA threshold for locating 
        event and determining event start and end times
    taper_length (float) - optional; length in seconds of taper at start of signal
    smoothwin (int) - optional; window length in samples for Savgol smoothing 
        of envelopes
    smoothorder (int) - optional; polynomial order for Savgol smoothing
    check_calcs (boolean) - optional; setting to True plots the signal with the 
        calculated start and end times, as well as all minima, maxima, and event
        time that were used to perform the calculations
    OUTPUT
    energies_df (pandas DataFrame) - updated table of pseudo-energies for all 
        events evaluated in sequence
    """
    # Calculate envelope of signal
    signal_envelope = signal.copy()
    signal_envelope.data = filte.envelope(signal_envelope.data)
    signal_envelope.data = spsignal.savgol_filter(signal_envelope.data, 
                                                  smoothwin, smoothorder)

    # Calculate STA/LTA of signal
    samp_rate = signal_envelope.stats.sampling_rate # Hz
    sta_lta = classic_sta_lta(signal_envelope, int(3*samp_rate), 
                              int(10*samp_rate))
    sta_lta = spsignal.savgol_filter(sta_lta, smoothwin, smoothorder)
    
    # Get index of event time in signal
    eventtime_index = int(before*samp_rate)
    event_time = signal_envelope.stats.starttime + before

    # Find peaks in sta_lta
    peaks, mins = sigproc.peakdet(sta_lta, stalta_lowerlim)
    peak_inds = [int(peak[0]) for peak in peaks]
    
    # Find minima in sta_lta signal
    peaks, mins = sigproc.peakdet(sta_lta, 0.05)
    min_inds = [int(mini[0]) for mini in mins]

    # Only keep peaks that are within desired range
    # (Not in taper, not at end of signal, not before last detected event's end time) 
    lower_search_indx = int(taper_length*samp_rate)
    higher_search_indx = int(2*before*samp_rate)
    
    # Convert search limits to UTCDateTimes
    lower_search_time = signal_envelope.stats.starttime + lower_search_indx/samp_rate
    higher_search_time = signal_envelope.stats.starttime + higher_search_indx/samp_rate
    
    # Get end time of last event
    if len(energies_df) == 0:
        previous_end_time = signal_envelope.stats.starttime
    else:
        previous_end_time = UTCDateTime(energies_df['Trigger times'].values[-1]) - \
                            before + energies_df['End times'].values[-1]/samp_rate 
    
    # If end time of last event outside of search window, return dataframe
    # without updating                        
    if previous_end_time > higher_search_time:
        return(energies_df)
    
    # If end time of last event is within search window but greater than
    # lower search limit, match lower search limit to previous end time
    if previous_end_time > lower_search_time and previous_end_time < higher_search_time:
        time_diff = previous_end_time - lower_search_time
        lower_search_indx += int(time_diff*samp_rate)
        higher_search_indx += int(time_diff*samp_rate)
        
    # If triggering event time is before lower limit of search window, 
    # return dataframe without updating
    if lower_search_indx >= (eventtime_index - int(samp_rate)):
        return(energies_df)
        
    # Find peaks in search window, look for peaks greater tha upper sta/lta limit
    peak_inds1 = []
    peak_inds2 = []
    for peak in peak_inds:
        if peak > lower_search_indx and peak < higher_search_indx:
            peak_inds1.append(peak)  
            if sta_lta[peak] >= stalta_upperlim:
                peak_inds2.append(peak)                
    
    # Find largest peak in range
    signal_peak = lower_search_indx +\
                  np.argmax(signal_envelope[lower_search_indx:higher_search_indx])
                  
    # If there are peaks above sta/lta threshold and within search window, 
    # find peak that is closest to max amplitude in search window
    if len(peak_inds2) > 0:
        stalta_peak = peak_inds2[0]
        for peak in peak_inds2:
            min_diff = abs(signal_peak - stalta_peak)
            if abs(signal_peak - peak) < min_diff:
                stalta_peak = peak 
                
    # If there are peaks within search window but not above threshold,
    # find peak that is closest to max amplitude in search window
    elif len(peak_inds1) > 0:
        stalta_peak = peak_inds1[0]
        for peak in peak_inds1:
            min_diff = abs(signal_peak - signal_envelope[stalta_peak])
            if abs(signal_peak - peak) < min_diff:
                stalta_peak = peak  
                
    # If there aren't any peaks within search window, pick the biggest peak 
    else:
        stalta_peak = np.argmax(sta_lta[peak_inds])
        lower_search_indx = 0
    
    # Find minima with sta_lta < min threshold
    min_inds1 = []
    for mini in min_inds:
        if sta_lta[mini] <= stalta_lowerlim:
            min_inds1.append(mini)
            
    # Locate min above lower sta/lta threshold that is immediately before 
    # first peak above upper sta/lta threshold for event start time
    # If no peaks above upper sta/lta threshold, select smallest min immediately
    # preceding stalta_peak
    # If neither conditions are met, choose lower_search_indx
    if len(min_inds1) > 0:
        peak_list = []
        if len(peak_inds2) > 0:
            for peak in peak_inds2:
                if peak < stalta_peak:
                    peak_list.append(peak)
        if len(peak_list) == 0:
            peak_list = [stalta_peak]

        compare_peak = max(peak_list)
        
        reverse_min_inds1 = min_inds1
        reverse_min_inds1.sort(reverse=True)

        checkmin = compare_peak
        for mini in reverse_min_inds1:
            if mini < checkmin:
                checkmin = mini
            else:
                starttime = checkmin
                break
    else:
        starttime = lower_search_indx
                
    # Find min where signal returns to starttime amplitude to get end time
    # If more peaks above upper sta/lta threshold after stalta_peak and before 
    # upper_search_indx, endtime must be after this peak
    # If no endtime found, choose last index of signal
    last_peak = stalta_peak
    for peak in peak_inds2:
        if peak > stalta_peak:
            last_peak = peak
    endtime = len(signal_envelope) - 1
    noise_level = 0.10 # percent of signal amplitude that is noise
    for i in range(last_peak+1, len(signal_envelope) - 1):
        if signal_envelope[i] <= (1+noise_level)*signal_envelope[starttime]:
            endtime = i
            break
     
    # Integrate envelope between signal start and end times to get pseudo-energy
    try:
        integral = sum(signal_envelope[starttime:endtime])/samp_rate #  m/s^2
        signal_length = (endtime - starttime)/samp_rate # units are s
        max_amp = max(signal_envelope[starttime:endtime]) # units are m/s
    except:
        integral = -999.
        signal_length = -999.
        max_amp = -999.
        
    # Visualize signal to verify start and end time calculations
    if check_calcs:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(signal)
        plt.plot(signal_envelope, 'r', label = 'Envelope')
        line1 = starttime
        line2 = endtime
        plt.axvline(line1, color='k', linestyle='--', label = 'Start and end times')
        plt.axvline(line2, color='k', linestyle='-.')
        plt.title(str(event_time) + '\nSignal with Envelope')
        plt.ylabel('Amplitude (m/s)')
        plt.xlabel('Signal Index')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(sta_lta)
        plt.axvline(line1, color='k', linestyle='--')
        plt.axvline(line2, color='k', linestyle='-.')
        plt.axvline(lower_search_indx, color='g', linestyle=':', 
                    label = 'Search window')
        plt.axvline(higher_search_indx, color='g', linestyle=':')
        plt.axvline(eventtime_index, color='b', linestyle=':', 
                    label = 'Event triggering time')
        plt.plot(peak_inds2, sta_lta[peak_inds2], 'r*')
        plt.plot(peak_inds1, sta_lta[peak_inds1], 'r.')
        plt.plot(min_inds, sta_lta[min_inds], 'k.')
        plt.plot(min_inds1, sta_lta[min_inds1], 'k*')
        plt.axvline(stalta_peak, color='r', label = 'Signal peak')
        plt.title('STA/LTA')
        plt.ylabel('Amplitude')
        plt.xlabel('Signal Index')
        plt.subplots_adjust(hspace=0.5)
        plt.show()
    
    # Organize return info into dataframe
    col_names = ['Trigger times', 'Start times', 'End times', 'Peak times', 
                 'Integrals (m/s^2)', 'Max amplitudes (m/s)', 'Signal lengths (s)',
                 'Sampling rates']
    new_energies_df = pd.DataFrame({col_names[0]: [event_time],
                                    col_names[1]: [starttime],
                                    col_names[2]: [endtime],
                                    col_names[3]: [stalta_peak],
                                    col_names[4]: [integral],
                                    col_names[5]: [max_amp],
                                    col_names[6]: [signal_length],
                                    col_names[7]: [samp_rate]}, 
                                   columns = col_names)
    energies_df = pd.concat([energies_df, new_energies_df], ignore_index = True)
    
    return(energies_df)
    
def saveEnergies(energies_df, filepath, before):
    """
    Converts returned start, end, and peak times from getPseudoEnergies() into
    UTCDateTime timestamps and saves energies_df to csv file.
    INPUTS
    energies_df (pandas DataFrame) - table of pseudo-energies returned from previous 
        events in list, if evaluating landslide sequences, or empty table if 
        evaluating only one event
    filepath (string) - path that energies table will be saved to
    before (float) - seconds before trigger time that signal was cropped; used
        to get initial event time
    OUTPUT
    energies_df (pandas DataFrame) - updated table of pseudo-energies for all 
        events evaluated in sequence
    """
    # Convert indices of returned star, end, and signal peak times into 
    # UTCDateTimes for each row in DataFrame
    for index, row in energies_df.iterrows():
        energies_df.at[index, 'Start times'] = UTCDateTime(row['Trigger times']) - \
                                               before + row['Start times']/row['Sampling rates']
        energies_df.at[index, 'End times'] = UTCDateTime(row['Trigger times']) - \
                                               before + row['End times']/row['Sampling rates']
        energies_df.at[index, 'Peak times'] = UTCDateTime(row['Trigger times']) - \
                                               before + row['Peak times']/row['Sampling rates']
    
    # TO DO: get rid of duplicate rows
    
    # Save DataFrame to CSV
    energies_df.to_csv(filepath)
    
    return(energies_df)