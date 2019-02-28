# Imports
import matplotlib.pyplot as plt
import numpy as np
from reviewData import reviewData
from obspy import UTCDateTime
from findFirstArrivals import transformSignals, findFirstArrivals, plotFirstArrivals
from detectLandslides import getStreamObject

def viewEvents(eventrow,lslat,lslon,radius,fit_stations=8,min_time_diff=5.0,
              plot_arrival_times=False,plot_predictions=False):  
    """
    Visualize snippets of signal around returned event or trigger times, and,
    if desired, view first arrival times of signals from closest stations and
    linear regression of first arrival times that was used to determine if
    signal belongs to a landslide event.
    INPUTS
    eventrow (Pandas Series) - row from events dataframe corresponding
    lslat (float) - latitudinal coordinate of landslide (make negative for south
        of Equator)
    lslon (float) - longitudinal coordinate of landslide (make negative for west
        of Prime Meridian)
    radius (float) - optional; search radius in km for finding nearest seismic
        stations and plotting their signals
    fit_stations (int) - number of stations to use for picking first arrival
        times
    min_time_diff (float) - optional; number of seconds first arrival times can differ 
        from model times by
    plot_arrival_times (boolean) - optional; set as True to plot first arrival
        times on each seismic signal
    plot_predictions (boolean) - optional; set as True to plot first arrival
        times as a function of distance away from landslide and compare to 
        predicted arrival times from model
    OUTPUTS
    None
    """
    # Get event info
    eventtime = UTCDateTime(eventrow['Event time'])
    m = eventrow['Moveout slope (s/km)']
    b = eventrow['Moveout intercept']
    
    # Get data
    st, network, station = getStreamObject(eventtime-100.,eventtime+300.,
                                           lslat,lslon,radius)
    st = reviewData.attach_distaz_IRIS(st, lslat, lslon)
    st = st.sort(keys=['rdist', 'channel'])
    st.filter('bandpass', freqmin=1.0, freqmax=5.0)
    
    # Plot event
    reviewData.InteractivePlot(st)
    
    # Delete problematic stations (will need to add to)
    channels_to_remove = []
    for channel in st:
        if channel.std() < 2.0:
            channels_to_remove.append(channel)
    
    for channel in channels_to_remove:
        #if len(st.select(station = channel.stats.station)) > 0:
        st.remove(channel)
    
    # Plot first arrival times
    if plot_arrival_times:
        arrival_times, arrival_inds = findFirstArrivals(st)
        plotFirstArrivals(st,arrival_times,arrival_inds)
    
    # Plot linear regression of arrival_times
    if plot_predictions:
        # Get first arrival times
        arrival_times, arrival_inds = findFirstArrivals(st)
        if len(arrival_times) > 0:  
            # Find seconds between first arrival at each station and the closest station
            arrival_secs = [time - arrival_times[0] for time in arrival_times] 
        
            # Calculate distance from each seismic station to closest station
            station_dist = [tr.stats.rdist for tr in st]
                    
            # Get predicted arrival times
            arrival_secs_pred = np.dot(m, station_dist) + b  
                
            plt.figure()
            plt.title('Linear Moveout Prediction and First Arrival Times')
            plt.plot(station_dist, arrival_secs, 'ro', 
                     label='Signal arrival times')
            # Plot errorbar of 5 seconds around each predicted arrival time
            plt.errorbar(station_dist, arrival_secs_pred, yerr=min_time_diff, 
                      fmt='bo', label='Predicted arrival times')
            plt.xlabel('Distance from station to landslide (km)')
            plt.ylabel('Landslide signal arrival time (s)')
            plt.legend()
            plt.show()

    return