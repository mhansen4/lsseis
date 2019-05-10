import pandas as pd
from obspy import UTCDateTime

from reviewData import reviewData

from detectLandslides import getStreamObject, findTriggers

import warnings
warnings.filterwarnings("ignore")

"""
Code for visually verifying detected aftershocks stored in a CSV file produced 
by findAftershocks().
"""

def processSignal(st, lslat, lslon, newsamprate=0.):
    """
    Ensures template and target signals are processed the same way before 
    cross-correlation.
    INPUTS
    st - obspy stream object with seismic data
    lslat (float) - latitudinal coordinate of landslide (make negative for south
        of Equator)
    lslon (float) - longitudinal coordinate of landslide (make negative for west
        of Prime Meridian)
    newsamprate (float) - optional; sampling rate to resample template to. Must
        match the sampling rate of the stream object the template is being 
        compared to.
    OUTPUT
    st - processed stream object
    """
    # Demean signal
    st.detrend('demean')
    
    # Apply bandpass filter
    fmin = 1.0 # lower frequency limit, Hz'
    fmax = 3.0 # upper frequency limit, Hz
    st.filter('bandpass', freqmin=fmin, freqmax=fmax)
    
    # Apply signal taper
    #st.taper(max_percentage=0.05, type='cosine')
    
    """
    # Apply cosine filter
    prefilt = [0.01, 0.02, 10, 20]
    st.remove_response(pre_filt=prefilt, output='VEL')
    """
    
    # Attach station coordinates to stream object
    st = reviewData.attach_distaz_IRIS(st, lslat, lslon)
    st = st.sort(keys=['rdist', 'channel'])
    
    # Resample signal
    if newsamprate > 0.:
        st.resample(newsamprate)
    
    return(st)
    
if __name__ == '__main__':  
    # Input parameters
    lslat = 58.77918 # latitude of landslide in deg
    lslon = -136.88827 # longitude of landslide in deg
    radius = 150. # search radius for nearest seismic stations in km
    filenamepref = 'lamplugh'


    folderdat = filenamepref + '_data'
    loadfromfile = True
    savedat = True
    
    # Import detected aftershocks from saved CSV
    aftershocks_df = pd.read_csv('event_search/' + filenamepref+'_detected_aftershocks.csv')
    aftershocks_df = aftershocks_df[aftershocks_df['Verified event'] == 'Y']
    aftershocks = aftershocks_df.TIMESTAMP.values
    
    # Specify part of signal to keep (will be applied to st1 and st2)
    before = 50. # Time to cut before event or trigger start time in s
    after = 250. # Time to cut after event or trigger start time in s
    maxstations = 8 # number of nearest stations to include in analysis

    # Check signals of each event in aftershock list
    for a in range(len)aftershocks)):
        print('Checking aftershock %i of %i...' % (a+1,len(aftershocks)))
        as_time = UTCDateTime(aftershocks[a])
        st1 = getStreamObject(as_time-before,as_time+after,lslat,lslon,
                              radius=radius,maxradius=500.,loadfromfile=loadfromfile,
                              mintraces=4,savedat=savedat,folderdat=folderdat,
                              filenamepref=filenamepref,limit_stations=8) 
        st1 = processSignal(st1, lslat, lslon)
        st1, trig, new_trigger_times, teleseisms = findTriggers(lslat,lslon,st1,[])
        zp1 = reviewData.InteractivePlot(st1)