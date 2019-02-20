#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:23:44 2018

@author: mchansen
"""
import math
import pandas as pd
from datetime import timedelta
from obspy.taup import TauPyModel
from libcomcat.search import search
import urllib.request

"""
removeTeleseisms - Uses libcomcat module to search UGSS ComCat and remove 
seismic signals returned in seisk stream object which correspond to earthquakes.
"""

def getStationCoordinates(network,station):
    """
    Uses IRIS FDSN webservices to find the lat/lon coordinates of a specified
    seismic station.
    INPUTS
    network (str) - seismic network code corresponding to station closest to 
        landslide, comma separated in a single string
        Example: 'NF,IW,RE,TA,UU'
    station (str) - name of seismic station closest to landslide, three letter
        string that is not case-sensitive
        Example: 'BFR,WOY,TCR,WTM'
    OUTPUTS
    stationlat (float) - latitudinal coordinate of seismic station
    stationlon (float) - longitudinal coordinate of seismic station
    """
    url = 'http://service.iris.edu/fdsnws/station/1/query?net=' + \
          network + '&format=text&level=sta'
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request) as response:
        content = response.read()

    stationlist = content.decode().split("\n" + network)
    for item in stationlist:
        if station in item:
            closeststation = item

    stationlat = float(closeststation.split("|")[2])
    stationlon = float(closeststation.split("|")[3])
    
    return(stationlat,stationlon)
    
def getQuakeDict(starttime,endtime,network,station):
    """
    Determines when earthquakes from ComCat arrived at seismic station closest
    to landslide, returns dictionary containing earthquake arrival times and
    other identifying information.
    INPUTS
    starttime (UTCDateTime) - earliest time earthquakes in ComCat must 
        have occurred to be returned
    endtime (UTCDateTime) - latest time earthquakes in ComCat must have 
        occurred to be returned
    network (str) - seismic network code corresponding to station closest to 
        landslide, comma separated in a single string
        Example: 'NF,IW,RE,TA,UU'
    station (str) - name of seismic station closest to landslide, three letter
        string that is not case-sensitive
        Example: 'BFR,WOY,TCR,WTM'
    OUTPUT
    quakedict - dictionary containing full earthquake summaries from ComCat
        ('summary'), earthquake IDs in ComCat ('id'), dates and times ('date'),
        coordinates of earthquake epicenter ('latitude' and 'longitude'), depth
        to earthquake epicenter ('depth'), distance of earthquake to landslide 
        in km ('distance'), earthquake magnitude ('magnitude'), calculated 
        arrival time of earthquake at seismic station closest to landslide
        ('arrival time'), and event type ('event type') -- 'earthquake' for 
        all events here.
    """

    # Get coordinates of nearest station to landslide
    stationlat, stationlon = getStationCoordinates(network, station)

    # Get list of earthquakes that happened between starttime and endtime 
    # from ComCat
    eqstartdt = starttime.datetime
    eqenddt = endtime.datetime
    tempquakes = search(starttime=eqstartdt,endtime=eqenddt,minmagnitude=0.5,
                        mindepth=0.0)

    quakeids = []
    quakedates = []
    quakelats = []
    quakelons = []
    quakedepths = []
    quakemags = []
   
    # Save attributes to lists
    for quake in tempquakes:
        quakeids.append(quake.id)
        quakedates.append(quake.time)
        quakelats.append(quake.latitude)
        quakelons.append(quake.longitude)
        quakedepths.append(quake.depth)
        quakemags.append(quake.magnitude)

    # Create dictionary for attributes
    quakedict = {}
    eqkeys = ['id','date','latitude','longitude','depth','distance',
              'magnitude','arrival time','event type']
    for key in eqkeys:
        quakedict[key] = []

    quakedict['id'] = [quakeid for quakeid in quakeids]
    quakedict['date'] = [quakedate for quakedate in quakedates]
    quakedict['latitude'] = [quakelat for quakelat in quakelats]
    quakedict['longitude'] = [quakelon for quakelon in quakelons]
    quakedict['depth'] = [quakedepth for quakedepth in quakedepths]
    quakedict['magnitude'] = [quakemag for quakemag in quakemags]
    quakedict['event type'] = ['earthquake'] * len(quakeids)

    quakedistskm, arrivaltimes = findEQArrivalTimes(network, station, quakedict)
    quakedict['distance'] = [quakedist for quakedist in quakedistskm]
    quakedict['arrival time'] = [arrivaltime for arrivaltime in arrivaltimes]
        
    return(quakedict)
    
def calcCoordDistance(lat1,lon1,lat2,lon2):
    """
    Uses Great Circle Method and Euclidean Distance Method to get distances 
    between two coordinate pairs in kilometers and degrees, respectively.
    INPUTS
    lat1 (float) - latitude of first coordinate pair
    lon1 (float) - longitude of first coordinate pair
    lat2 (float) - latitude of second coordinate pair
    lon2 (float) - longitude of second coordinate pair
    OUTPUTS
    dkm (float) - distance in kilometers
    ddeg (float) - distance in degrees
    """
    
    R = 6371 # Earth's radius in km
        
    # Use great circle method to get distances in km
    angle1 = math.radians(lat2) # Latitude of nearest station
    angle2 = math.radians(lat1) # Earthquake latitude
    latdiff = math.radians(lat1-lat2)
    londiff = math.radians(lon1-lon2)
    
    a = math.sin(latdiff/2)**2 + \
        math.cos(angle1)*math.cos(angle2) * \
        math.sin(londiff/2)**2
    c = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    dkm = R*c
    
    # Use simple distance equation to get distances in degrees
    ddeg = math.sqrt((lat1-lat2)**2+(lon1-lon2)**2)
    
    return(dkm,ddeg)
    
def findEQArrivalTimes(network,station,quakedict):
    """
    Determines distance from earthquake epicenter to landslide and when 
    earthquakes from ComCat arrived at seismic station closest to landslide.
    INPUTS
    network (str) - seismic network code corresponding to station closest to 
        landslide, comma separated in a single string
        Example: 'NF,IW,RE,TA,UU'
    station (str) - name of seismic station closest to landslide, three letter
        string that is not case-sensitive
        Example: 'BFR,WOY,TCR,WTM'
    quakedict - dictionary containing full earthquake summaries from ComCat
        ('summary'), earthquake IDs in ComCat ('id'), dates and times ('date'),
        coordinates of earthquake epicenter ('latitude' and 'longitude'), depth
        to earthquake epicenter ('depth'), distance of earthquake to landslide 
        in km ('distance'), earthquake magnitude ('magnitude'), calculated 
        arrival time of earthquake at seismic station closest to landslide
        ('arrival time'), and event type ('event type') -- 'earthquake' for 
        all events here
    OUTPUTS
    quakedistskm (list of floats) - list of distances in kilometers from 
        earthquake location to seismic station closest to landslide, in the 
        same order as the events in quakedict
    arrivaltimes (list of UTCDateTimes) - estimated time of arrival of seismic
        waves from each earthquake in quakedict at the seismic station closest
        to the landslide
    """
    
    # Get coordinates of nearest station to landslide
    stationlat, stationlon = getStationCoordinates(network, station)
    
    # Get lat, lon, depth, date information from quakedict
    quakelats = quakedict['latitude']
    quakelons = quakedict['longitude']
    quakedepths = quakedict['depth']
    quakedates = quakedict['date']

    # Compute earthquake distances away from this station
    quakedistskm = [] # in km
    quakedistsdeg = []
    for eqlat,eqlon in zip(quakelats,quakelons):
        dkm, ddeg = calcCoordDistance(eqlat,eqlon,stationlat,stationlon)
        quakedistskm.append(dkm)
        quakedistsdeg.append(ddeg)

    # Find S traveltimes (in seconds) and arrival times (UTCDateTime objects)
    # (Could probably improve by using station coordinates instead of 
    #  landslide coordinates)
    model = TauPyModel(model="iasp91")
    arrivaltimes = [] # List of datetime objects

    for i in range(0,len(quakedepths)):
        arrival = model.get_travel_times(source_depth_in_km=quakedepths[i],
                                         distance_in_degree=quakedistsdeg[i],
                                         phase_list=["S"])
        if len(arrival) == 0:
            # Make arrival time something far in the future so it's not
            # flagged as teleseism
            arrivaltimes.append(quakedates[i] + timedelta(days=365))
        else:
            t = arrival[0].time
            arrivaltimes.append(quakedates[i] + timedelta(seconds=t))
        
    return(quakedistskm, arrivaltimes)

def removeTeleseisms(starttime,endtime,network,station,trig):
    """
    Uses above methods to get arrival times of earthquakes at seismic station
    closest to landslide, compares these times to potential landslide times in
    trig and sorts them into teleseisms (teleseisms) and non-teleseisms 
    (new_trig).
    INPUTS
    starttime (UTCDateTime) - earliest time earthquakes in ComCat must 
        have occurred to be returned
    endtime (UTCDateTime) - latest time earthquakes in ComCat must have 
        occurred to be returned
    network (str) - seismic network code corresponding to station closest to 
        landslide, comma separated in a single string
        Example: 'NF,IW,RE,TA,UU'
    station (str) - name of seismic station closest to landslide, three letter
        string that is not case-sensitive
        Example: 'BFR,WOY,TCR,WTM'
    trig (list of UTCDateTimes) - times of potential landslides, returned from
        seisk stream object in findLandslides() using obspy.signal.trigger
        coincidence_trigger()
    OUTPUTS
    new_trig (list of UTCDateTimes) - times of potential landslides with 
        teleseisms removed
    teleseisms (list of UTCDateTimes) - times of teleseisms
    """
    
    # Get dictionary of earthquake info from ComCat, find earthquake arrival 
    # times at first station
    quakedict = getQuakeDict(starttime,endtime,network,station)
    
    # Compare earthquake arrival times to signal time
    
    # Define two lists for sorting trigger times into
    new_trig = []
    teleseisms = []
    
    # Loop through times and trig and sort as teleseisms or triggers
    for t in trig:
        comparetime = t['time'].datetime
        # print('\nTime to compare earthquake arrivals against: ', comparetime)
        
        maxtimediff = 20. # How close EQ arrival time should be for match (in s)
    
        matchfound = False
        tseism = 0
        for i in range(0,len(quakedict['arrival time'])):
            eqtime = quakedict['arrival time'][i]
            if abs((eqtime - comparetime).total_seconds()) <= maxtimediff:
                matchfound = True
                tseism = i
        
        if matchfound:
            print('\nFound teleseism.')
            for key in ['id','latitude','longitude','magnitude','arrival time']:
                print(key+': '+str(quakedict[key][tseism]))
            print('trigger match:' + str(t['time']))
            print('Removing teleseism from list of triggers.\n')
            # Record match in list of teleseisms
            teleseisms.append(t)
        else:
            # Add to new list of triggers to return
            new_trig.append(t)
        
            
    if len(teleseisms) == 0:
        print('No teleseisms found in trigger list.')
    
    # Return lists
    return(new_trig, teleseisms)
    
def searchComCatforLandslides(starttime,endtime,lslat,lslon,network,station):
    """
    Returns dataframe of landslides that were recorded in ComCat between 
    starttime and endtime, in addition to other ComCat info.
    INPUTS
    starttime (UTCDateTime) - earliest time earthquakes in ComCat must 
        have occurred to be removed from stream object
    endtime (UTCDateTime) - latest time earthquakes in ComCat must have occurred
        to be removed from stream object
    lslat (float) - latitudinal coordinate of landslide (make negative for south
        of Equator)
    lslon (float) - longitudinal coordinate of landslide (make negative for west
        of Prime Meridian)
    network (str) - seismic network code corresponding to station closest to 
        landslide, comma separated in a single string
        Example: 'NF,IW,RE,TA,UU'
    station (str) - name of seismic station closest to landslide, three letter
        string that is not case-sensitive
        Example: 'BFR,WOY,TCR,WTM'
    OUTPUT
    lsdf - pandas dataframe containing event IDs of landslides and potential
        landslides in ComCat ('id'), event dates and times ('date'), event 
        coordinates ('latitude' and 'longitude'), magnitude ('magnitude'), depth 
        ('depth'), distance to landslide we are searching for in km ('distance'),
        and event type in ComCat ('eventtype'). 
    """
    
    # Convert UTCDateTimes into datetimes
    eqstartdt = starttime.datetime
    eqenddt = endtime.datetime
    
    # Get list of earthquakes during time window from ComCat
    tempquakes = search(starttime=eqstartdt,endtime=eqenddt,minmagnitude=2.0)

    quakeids = []
    quakedates = []
    quakelats = []
    quakelons = []
    quakedepths = []
    quakemags = []
    quakedistskm = []
    quaketypes = []
   
    # Save attributes to lists
    for quake in tempquakes:
        quakeids.append(quake.id)
        quakedates.append(quake.time)
        quakelats.append(quake.latitude)
        quakelons.append(quake.longitude)
        quakedepths.append(quake.depth)
        quakemags.append(quake.magnitude)
        quaketypes.append('earthquake')
        
        # Calculate distance from landslide to earthquakes
        dkm, ddeg = calcCoordDistance(lslat,lslon,quake.latitude,quake.longitude)
        quakedistskm.append(dkm)

    # Get list of landslides during time window from ComCat
    tempslides = search(starttime=eqstartdt,endtime=eqenddt,minmagnitude=2.0,
                        eventtype = 'landslide')
   
    # Save attributes to lists
    for slide in tempslides:
        quakeids.append(slide.id)
        quakedates.append(slide.time)
        quakelats.append(slide.latitude)
        quakelons.append(slide.longitude)
        quakedepths.append(slide.depth)
        quakemags.append(slide.magnitude)
        quaketypes.append('landslide')
        
        # Calculate distance between landslides
        dkm, ddeg = calcCoordDistance(lslat,lslon,slide.latitude,slide.longitude)
        quakedistskm.append(dkm)
         
    # Combine quake lists into pandas dataframe
    lsdf = pd.DataFrame({'id': quakeids,
                            'date': quakedates,
                            'latitude': quakelats,
                            'longitude': quakelons,
                            'depth': quakedepths,
                            'magnitude': quakemags,
                            'distance': quakedistskm,
                            'event type': quaketypes})
    
    if len(lsdf) > 0:
        # Filter events by magnitude
        minmag = 2.0
        maxmag = 5.0
        # Create dataframe to hold quakes we don't think are landslides
        remove_lsdf = lsdf[(lsdf.magnitude > maxmag) | (lsdf.magnitude < minmag)]
        lsdf = lsdf[(lsdf.magnitude <= maxmag) & (lsdf.magnitude >= minmag)]
        
        # Filter events by depth (looking for shallow earthquakes)
        maxdepth = 7.0 # km
        remove_lsdf = remove_lsdf.append(lsdf[lsdf.depth > maxdepth])
        lsdf = lsdf[lsdf.depth <= maxdepth]
        
        # Filter events by distance from landslide 
        maxdist = 10.0 # km
        remove_lsdf = remove_lsdf.append(lsdf[lsdf.distance > maxdist])
        lsdf = lsdf[lsdf.distance <= maxdist]
        
        # Include events if event type is 'landslide', regardless of magnitude, 
        # depth, or distance
        lsdf = lsdf.append(remove_lsdf[remove_lsdf['event type'] == 'landslide'])
    
    return(lsdf)