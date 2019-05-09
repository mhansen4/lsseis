import matplotlib.pyplot as plt

"""
plotStations - Plots simple scatter plot of station locations with respect to 
the landslide coordinates. May be helpful when trying to determine if the 
correct moveout is being observed at different stations.
"""

def plotStations(st, ls_lat, ls_lon):
    """
    Creates scatter plot of landslide and nearby seismic station locations.
    INPUTS
    st - obspy stream object with seismic data
    lslat (float) - latitudinal coordinate of landslide (make negative for south
        of Equator)
    lslon (float) - longitudinal coordinate of landslide (make negative for west
        of Prime Meridian)
    OUTPUTS
    plot
    """   
    station_names = []
    station_lats = []
    station_lons = []
    
    for trace in st:
        station_names.append(trace.stats.station)
        station_lats.append(trace.stats.coordinates['latitude'])
        station_lons.append(trace.stats.coordinates['longitude'])
        
    fig, ax = plt.subplots()
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Locations of Closest Seismic Stations to Landslide')
    ax.plot(ls_lon, ls_lat, 'r*')
    ax.annotate('Landslide', (ls_lon, ls_lat))
    ax.plot(station_lons, station_lats, 'bo')
    for i, txt in enumerate(station_names):
        ax.annotate(txt, (station_lons[i], station_lats[i]))
    plt.show()