"""
Created on Thu Mar 28 15:47:08 2019

@author: mchansen
"""

import numpy as np
import pandas as pd
import glob
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta

from reviewData import reviewData

from detectLandslides import getStreamObject
from findAftershocks import processSignal
from getPseudoEnergies import getPseudoEnergies, savePseudoEnergies

# Plot scatter plots of event pseudo-energies, max amplitudes, and signal lengths
def plotEnergyOverview(energies_df, filenamepref, save_plot=False):
    event_times = pd.to_datetime(energies_df['Start times'].apply(str), 
                                 format='%Y-%m-%d %H:%M:%S.%f')
    """
    Plots pseudo-energies, max amplitudes, and signal lengths returned in
    energies_df as separate scatter plots.
    INPUTS
    energies_df (pandas DataFrame) - table of pseudo-energies and other information
        for a sequence of events  
    filenamepref (string) - file prefix that has been used for all files
        associated with this event; usually lowercase, one-word event name
    save_plot (boolean) - optional; if set to True, will save plot as PNG to 
        current directory
    OUTPUT
    plot
    """
    energies = energies_df['Integrals (m/s^2)'].values
    max_amps = energies_df['Max amplitudes (m/s)'].values
    signal_lengths = energies_df['Signal lengths (s)'].values
    
    fig = plt.figure(figsize=(6,9))
    ax1 = fig.add_subplot(311)
    ax1.set_title('Aftershock Pseudo-Energies')
    ax1.plot(event_times, np.log10(energies),'.')
    ax1.set_ylabel('Log(Integral) (m/s^2)')
    ax2 = fig.add_subplot(312)
    ax2.plot(event_times, np.log10(max_amps), '.')
    ax2.set_ylabel('Log(Max Amplitude) (m/s)')
    ax3 = fig.add_subplot(313)
    ax3.plot(event_times, signal_lengths, '.')
    ax3.set_ylabel('Signal Length (s)')
    myFmt = mdates.DateFormatter('%m/%d %H:%M:%S') # Format date strings for tick labels
    ax3.xaxis.set_major_formatter(myFmt)
    fig.autofmt_xdate() # Rotate tick labels automatically
    ax3.set_xlabel('Event Time (UTC)') # Label axis
    plt.show()
    
    if save_plot:
        fig.savefig('./%s_energies_overview.png' % filenamepref, bbox_inches='tight')
        
    return

# Plot histogram of event energies
def plotEnergyHistogram(energies_df, filenamepref, num_bins=10, save_plot=False):
    """
    Plots histogram of pseudo-energies for events in events_df.
    INPUTS
    energies_df (pandas DataFrame) - table of pseudo-energies and other information
        for a sequence of events  
    filenamepref (string) - file prefix that has been used for all files
        associated with this event; usually lowercase, one-word event name
    num_bins (int) - optional; number of bins to plot in histogram
    save_plot (boolean) - optional; if set to True, will save plot as PNG to 
        current directory
    OUTPUT
    plot
    """
    energies = energies_df['Integrals (m/s^2)'].values
    
    energyHistogram = plt.figure()
    plt.hist(energies, bins=num_bins)
    plt.xlabel('Energy (Integral of Signal Envelope, m/s^2)')
    plt.ylabel('Landslide Frequency')
    plt.title('Histogram of Aftershock Energies')
    plt.show()
    
    if save_plot:
        energyHistogram.savefig('./%s_energy_histogram.png' % filenamepref, 
                                bbox_inches='tight')
        
    return

# Group energies by pseudo-energy
def getEnergyGroups(energies_df, num_bins = 5):
    """
    Groups pseudo-energies into num_bins number of groups and creates a new
    column in energies_df with each event's group number.
    INPUTS
    energies_df (pandas DataFrame) - table of pseudo-energies and other information
        for a sequence of events  
    num_bins (int) - optional; number of groups to create
    OUTPUT
    energies_df (pandas DataFrame) - updated table of pseudo-energies\
    energy_bins (list of tuples) - list of minimum and maximum energies used 
        to group each event, in order of increasing group number
    """
    energies = energies_df['Integrals (m/s^2)'].values
    energy_bins = []
    i = 0
    incrmt = np.ceil(max(energies)/num_bins/1000)*1000
    while i < max(energies):
        energy_bins.append((i,i+incrmt))
        i += incrmt
    
    energy_groups = []
    for i in range(len(energies)):
        for j in range(len(energy_bins)):
            if energies[i] <= energy_bins[j][1] and energies[i] >= energy_bins[j][0]:
                energy_groups.append(j+1)
                
    energies_df['Group'] = energy_groups
    
    return(energies_df, energy_bins)
    
# Plot energies vs. time, with different colors and sizes of dots for 
# different energy ranges
def plotEnergyGroups(energies_df, energy_bins, filenamepref, x_axis = 'event times', 
                     dot_multiplier=3, save_plot=False):
    """
    Create scatter plot of event pseudo-energies where each dot's size and color
    corresponds to their grouping created in getEnergyGroups(). You can plot
    either event times or signal lengths on the x-axis.
    INPUTS
    energies_df (pandas DataFrame) - table of pseudo-energies and energy groupings
        for a sequence of events, returned by getEnergyGroups()
    energy_bins (list of tuples) - list of minimum and maximum energies used 
        to group each event, in order of increasing group number, returned by
        getEnergyGroups()
    filenamepref (string) - file prefix that has been used for all files
        associated with this event; usually lowercase, one-word event name
    x_axis (string) - what to plot on the x-axis. Acceptable options are 'event
        times' and 'signal lengths'
    dot_multiplier (int) - optional; scalar describing how much larger each 
        energy group's markers will be than the previous group's
    save_plot (boolean) - optional; if set to True, will save plot as PNG to 
        current directory
    OUTPUT
    plot
    """        
    # Plot energies with different sized dots 
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for i in range(1,len(energy_bins)+1):
        # Select data for x-axis based on label user chose
        if x_axis == 'event times':
            x_data = pd.to_datetime(energies_df[energies_df.Group == i]['Start times'].apply(str), 
                                    format='%Y-%m-%d %H:%M:%S.%f')
        if x_axis == 'signal lengths':
            x_data = energies_df[energies_df.Group == i]['Signal lengths (s)'].values
            
        ax1.plot(x_data,
                 energies_df['Integrals (m/s^2)'][energies_df.Group == i], 'o', 
                 markersize=i*dot_multiplier, 
                 label='%i - %i' % (energy_bins[i-1][0], energy_bins[i-1][1]))
    # Pick corresponding axes labels and plot title
    if x_axis == 'event times':
        xlabel_string = 'Event Time (UTC)'
        plot_title = 'Landslide Energy Over Time'
    if x_axis == 'signal lengths':
        xlabel_string = 'Signal length (s)'
        plot_title = 'Landslide Energy vs. Signal Length'
        
    ax1.set_xlabel(xlabel_string)
    ax1.set_ylabel('Energy (Integral of Signal Envelope, m/s^2)')
    ax1.set_title(plot_title)
    if x_axis == 'event times':
        myFmt = mdates.DateFormatter('%m/%d %H:%M:%S')
        ax1.xaxis.set_major_formatter(myFmt)
        fig.autofmt_xdate() # Rotate tick labels automatically
    ax1.legend()
    plt.show()
    
    if save_plot:
        if x_axis == 'event times':
            plot_filename = './%s_energygroups_vs_eventtime.png' % filenamepref
        if x_axis == 'signal lengths':
            plot_filename = './%s_energygroups_vs_signallength.png' % filenamepref
        
        fig.savefig(plot_filename, bbox_inches='tight')
    
    return

# Plot number of aftershocks per day
def plotAftershockBarChart(energies_df, filenamepref, save_plot=False):
    """
    Plot vertical bar chart of number of events that happened each day in an 
    event sequence. 
    INPUTS
    energies_df (pandas DataFrame) - table of pseudo-energies and energy groupings
        for a sequence of events, returned by getEnergyGroups()
    filenamepref (string) - file prefix that has been used for all files
        associated with this event; usually lowercase, one-word event name
    save_plot (boolean) - optional; if set to True, will save plot as PNG to 
        current directory
    OUTPUT
    plot
    """        
    event_times = pd.to_datetime(energies_df['Start times'].apply(str), 
                                 format='%Y-%m-%d %H:%M:%S.%f')
    event_days = event_times.map(pd.Timestamp.date).unique()
    
    event_count = []
    for event_day in event_days:
        day = event_day.day
        day_count = 0
        for event in event_times:
            if event.day == day:
                day_count += 1
        event_count.append(day_count)
    
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.bar(event_days, event_count)
    ax2.set_ylabel('Number of Aftershocks')
    ax2.set_xlabel('Day (UTC)')
    ax2.set_title('Number of Aftershocks Per Day')
    myFmt = mdates.DateFormatter('%m/%d %H:%M:%S')
    ax2.xaxis.set_major_formatter(myFmt)
    fig.autofmt_xdate() # Rotate tick labels automatically
    plt.show()
    
    if save_plot:
        fig.savefig('./%s_events_barchart.png' % filenamepref, 
                    bbox_inches='tight')
    
    return

# Plot individual events
def plotEvent(energies_df, event_id, station_indx, filenamepref, 
              before=100., after=300., save_plot=False):
    """
    Plot event signal that was used to determine its pseudo-energy.
    INPUTS
    energies_df (pandas DataFrame) - table of pseudo-energies and energy groupings
        for a sequence of events, returned by getEnergyGroups()
    event_id (int) - ID of event in energies_df
    station_indx (int) - index of signal trace in obspy stream object. If closest 
        signal was used for calculations, set this to 0
    filenamepref (string) - file prefix that has been used for all files
        associated with this event; usually lowercase, one-word event name
    before (float) - seconds before trigger time that signal was cropped
    after (float) - seconds after trigger time that signal was cropped
    save_plot (boolean) - optional; if set to True, will save plot as PNG to 
        current directory
    OUTPUT
    plot
    """        
    # Pull important info from event's Pandas series
    energies_series = energies_df.iloc[event_id]
    event_time = UTCDateTime(energies_series['Trigger times'])
    event_energy = energies_series['Integrals (m/s^2)']
    samp_rate = energies_series['Sampling rates']
    starttime_s = UTCDateTime(energies_series['Start times']) - (event_time - before)
    endtime_s = UTCDateTime(energies_series['End times']) - (event_time - before)
    
    # Get seismic signal stream object
    st1 = getStreamObject(event_time-before,event_time+after,lslat,lslon,
                          radius=radius,loadfromfile=loadfromfile,
                          savedat=savedat,folderdat=folderdat,
                          filenamepref=filenamepref,limit_stations=8) 
    st1 = processSignal(st1, lslat, lslon)  
    signal = st1[station_indx]
    
    # Plot signal for station used to calculate pseudo-energies
    eventPlot = plt.figure()
    plt.plot(range(0,len(signal))/samp_rate, signal)
    line1 = starttime_s
    line2 = endtime_s
    plt.axvline(line1, color='k', linestyle='--', label = 'Start and end times')
    plt.axvline(line2, color='k', linestyle='-.')
    plt.title('Seismic Signal for Event %i\nPseudo-Energy = %i' % 
              (event_id, int(np.floor(event_energy))))
    plt.xlabel('Time after %s (s)' % str(event_time - before))
    plt.ylabel('Amplitude (m/s)')
    plt.legend()
    plt.show()
    
    if save_plot:
        eventPlot.savefig('./%s_event%i_signal.png' % (filenamepref, event_id), 
                          bbox_inches='tight')
    
    return

if __name__ == '__main__': 
    # Input parameters
    lslat = 58.77918 # latitude of landslide in deg
    lslon = -136.88827 # longitude of landslide in deg
    radius = 150. # search radius for nearest seismic stations in km
    
    # Data file info
    filenamepref = 'lamplugh'
    folderdat = filenamepref + '_data'
    loadfromfile = True
    savedat = True
    
    # Import event times
    excel_file = pd.read_csv('./event_search/%s_detected_aftershocks.csv' % filenamepref)
    event_times = excel_file[excel_file['Verified event'] == 'Y']['TIMESTAMP'].values

    # Define time in seconds before and after event time to grab signal for  
    before = 100.
    after = 300.
    
    # Station to calculate pseudoenergies on
    station_indx = 0
    
    # Create empty dataframe to store pseudo-energies in
    energies_df = pd.DataFrame()
    
    # Look for CSV with pseudo-energies
    # If file not found, find pseudo-energies for all events in list
    csv_to_find = filenamepref + '_verified_events.csv'
    filenames = glob.glob(csv_to_find)
    filenames.sort()
        
    if len(filenames) != 0:
        energies_df = pd.read_csv(csv_to_find)
    else:
        pseudo_energies = []
        eventno = 0
        # Loop through all event times and calculate pseudo-energies
        for event_time in event_times[eventno:]:
            event_time = UTCDateTime(event_time)
            eventno += 1
            
            print('')
            print('Event %i of %i...' % (eventno, len(event_times)))
        
            # Get seismic signal stream object
            st1 = getStreamObject(event_time-before,event_time+after,lslat,lslon,
                                  radius=radius,maxradius=500.,loadfromfile=loadfromfile,
                                  mintraces=7,savedat=savedat,folderdat=folderdat,
                                  filenamepref=filenamepref,limit_stations=8) 
            st1 = processSignal(st1, lslat, lslon)
            
            energies_df = getPseudoEnergies(energies_df, st1[station_indx], 
                                            event_time, before, 
                                            sta_lta_lims=[1.0,2.0], 
                                            taper_length=15., smoothwin=501, 
                                            check_calcs=True)
            # See signal
            # zp1 = reviewData.InteractivePlot(st1)

        # Save pseudo-energies to CSV           
        energies_df = savePseudoEnergies(energies_df, 
                                         filenamepref + '_verified_events.csv', 
                                         before)
    
    # Create different plots of pseudo-energies
    plotEnergyOverview(energies_df, filenamepref, save_plot=True)    
    plotEnergyHistogram(energies_df, filenamepref, save_plot=True)
    plotAftershockBarChart(energies_df, filenamepref, save_plot=True)
    
    energies_df, energy_bins = getEnergyGroups(energies_df)
    plotEnergyGroups(energies_df, energy_bins, filenamepref,
                     x_axis='signal lengths', save_plot=True)
    plotEnergyGroups(energies_df, energy_bins, filenamepref,
                     x_axis='event times', save_plot=True)
    for i in energies_df[energies_df.Group == 2].index.values:
        plotEvent(energies_df, i, station_indx, filenamepref, before=before, 
                  after=after, save_plot=True)