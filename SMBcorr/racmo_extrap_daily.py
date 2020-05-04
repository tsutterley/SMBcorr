#!/usr/bin/env python
u"""
racmo_extrap_daily.py
Written by Tyler Sutterley (04/2020)
Interpolates and extrapolates daily RACMO products to times and coordinates

Uses fast nearest-neighbor search algorithms
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
and inverse distance weighted interpolation to extrapolate spatially

INPUTS:
    base_dir: working data directory
    EPSG: projection of input spatial coordinates
    MODEL: daily model outputs to interpolate
        FGRN055: 5.5km Greenland RACMO2.3p2
        FGRN11: 11km Greenland RACMO2.3p2
        XANT27: 27km Antarctic RACMO2.3p2
        ASE055: 5.5km Amundsen Sea Embayment RACMO2.3p2
        XPEN055: 5.5km Antarctic Peninsula RACMO2.3p2
    tdec: dates to interpolate in year-decimal
    X: x-coordinates to interpolate in projection EPSG
    Y: y-coordinates to interpolate in projection EPSG

OPTIONS:
    VARIABLE: RACMO product to calculate
        smb: Surface Mass Balance
        hgtsrf: Change of Surface Height
    SEARCH: nearest-neighbor search algorithm (BallTree or KDTree)
    NN: number of nearest-neighbor points to use
    POWER: inverse distance weighting power
    FILL_VALUE: output fill_value for invalid points

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        http://www.numpy.org
        http://www.scipy.org/NumPy_for_Matlab_Users
    scipy: Scientific Tools for Python
        http://www.scipy.org/
    netCDF4: Python interface to the netCDF C library
         https://unidata.github.io/netcdf4-python/netCDF4/index.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/

PROGRAM DEPENDENCIES:
    convert_calendar_decimal.py: converts from calendar dates to decimal years
    convert_julian.py: returns the calendar date and time given a Julian date
    regress_model.py: models a time series using least-squares regression

UPDATE HISTORY:
    Written 04/2020
"""
from __future__ import print_function

import sys
import os
import re
import pyproj
import netCDF4
import numpy as np
import scipy.spatial
import scipy.interpolate
from SMBcorr.convert_calendar_decimal import convert_calendar_decimal
from SMBcorr.convert_julian import convert_julian
from SMBcorr.regress_model import regress_model

#-- PURPOSE: read and interpolate daily RACMO2.3 outputs
def extrapolate_racmo_daily(base_dir, EPSG, MODEL, tdec, X, Y,
    VARIABLE='smb', SEARCH='BallTree', NN=10, POWER=2.0, FILL_VALUE=None):

    #-- start and end years to read
    SY,EY = (np.min(np.floor(tdec)),np.max(np.floor(tdec)))
    #-- input list of files
    if (MODEL == 'FGRN055'):
        #-- filename and directory for input FGRN055 files
        file_pattern = 'RACMO2.3p2_FGRN055_{0}_daily_{1:4d}.nc'
        DIRECTORY = os.path.join(base_dir,'RACMO','GL','RACMO2.3p2_FGRN055')

    #-- create list of files to read
    input_files = [file_pattern.format(VARIABLE,YEAR) for YEAR in range(SY,EY+1)
        if file_pattern.format(VARIABLE,YEAR) in os.listdir(DIRECTORY)]

    #-- calculate number of time steps to read
    nt = 0
    for FILE in input_files:
        #-- Open the RACMO NetCDF file for reading
        with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
            nx = len(fileID.variables['rlon'][:])
            ny = len(fileID.variables['rlat'][:])
            nt += len(fileID.variables['time'][:])
            #-- invalid data value
            fv = np.float(fileID.variables[VARIABLE]._FillValue)

    #-- create a masked array with all data
    fd = {}
    fd[VARIABLE] = np.ma.zeros((nt,ny,nx),fill_value=fv)
    fd[VARIABLE].mask = np.zeros((nt,ny,nx),dtype=np.bool)
    fd['time'] = np.zeros((nt))
    #-- create a counter variable for filling variables
    c = 0
    #-- for each file in the list
    for FILE in input_files:
        #-- Open the RACMO NetCDF file for reading
        with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
            #-- number of time variables within file
            t = len(fileID.variables['time'][:])
            #-- Get data from netCDF variable and remove singleton dimensions
            fd[VARIABLE][c:c+t,:,:] = np.squeeze(fileID.variables[VARIABLE][:])
            #-- verify mask object for interpolating data
            fd[VARIABLE].mask[c:c+t,:,:] |= (fd[VARIABLE].data[c:c+t,:,:] == fv)
            #-- racmo coordinates
            fd['lon'] = fileID.variables['lon'][:,:].copy()
            fd['lat'] = fileID.variables['lat'][:,:].copy()
            fd['x'] = fileID.variables['rlon'][:].copy()
            fd['y'] = fileID.variables['rlat'][:].copy()
            #-- rotated pole parameters
            proj4_params = fileID.variables['rotated_pole'].proj4_params
            #-- extract delta time and epoch of time
            delta_time = fileID.variables['time'][:].copy()
            time_units = fileID.variables['time'].units
            #-- convert epoch of time to Julian days
            Y,M,D,h,m,s = [float(d) for d in re.findall('\d+\.\d+|\d+',units)]
            epoch_julian = calc_julian_day(Y,M,D,HOUR=h,MINUTE=m,SECOND=s)
            #-- calculate time array in Julian days
            YY,MM,DD,hh,mm,ss = convert_julian(epoch_julian + delta_time)
            #-- calculate time in year-decimal
            fd['time'][c:c+t] = convert_calendar_decimal(YY,MM,DD,
                HOUR=hh,MINUTE=mm,SECOND=ss)

    #-- indices of specified ice mask
    i,j = np.nonzero(fd[VARIABLE][0,:,:] != fv)

    #-- convert RACMO latitude and longitude to input coordinates (EPSG)
    proj1 = pyproj.Proj("+init={0}".format(EPSG))
    proj2 = pyproj.Proj("+init=EPSG:{0:d}".format(4326))
    xg,yg = pyproj.transform(proj2, proj1, fd['lon'], fd['lat'])

    #-- construct search tree from original points
    #-- can use either BallTree or KDTree algorithms
    xy1 = np.concatenate((xg[i,j,None],yg[i,j,None]),axis=1)
    tree = BallTree(xy1) if (SEARCH == 'BallTree') else KDTree(xy1)

    #-- output interpolated arrays of firn variable (height or firn air content)
    npts = len(tdec)
    extrap_data = np.ma.zeros((npts),fill_value=fv,dtype=np.float)
    #-- type designating algorithm used (1:interpolate, 2:backward, 3:forward)
    extrap_data.interpolation = np.zeros((npts),dtype=np.uint8)

    #-- find days that can be interpolated
    if np.any((tdec >= fd['time'].min()) & (tdec < fd['time'].max())):
        #-- indices of dates for interpolated days
        ind,=np.nonzero((tdec >= fd['time'].min()) & (tdec < fd['time'].max()))
        f = scipy.interpolate.interp1d(fd['time'], np.arange(nt), kind='linear')
        date_indice = f(tdec[ind]).astype(np.int)
        #-- for each unique firn date
        #-- linearly interpolate in time between two firn maps
        #-- then then inverse distance weighting to extrapolate in space
        for k in np.unique(date_indice):
            kk, = np.nonzero(date_indice==k)
            count = np.count_nonzero(date_indice==k)
            #-- query the search tree to find the NN closest points
            xy2 = np.concatenate((X[kk,None],Y[kk,None]),axis=1)
            dist,indices = tree.query(xy2, k=NN, return_distance=True)
            #-- normalized weights if POWER > 0 (typically between 1 and 3)
            #-- in the inverse distance weighting
            power_inverse_distance = dist**(-POWER)
            s = np.sum(power_inverse_distance, axis=1)
            w = power_inverse_distance/np.broadcast_to(s[:,None],(count,NN))
            #-- variable for times before and after tdec
            var1 = fd[VARIABLE][k,i,j]
            var2 = fd[VARIABLE][k+1,i,j]
            #-- linearly interpolate to date
            dt = (tdec[kk] - fd['time'][k])/(fd['time'][k+1] - fd['time'][k])
            #-- spatially extrapolate using inverse distance weighting
            extrap_data[kk] = (1.0-dt)*np.sum(w*var1[indices],axis=1) + \
                dt*np.sum(w*var2[indices], axis=1)
        #-- set interpolation type (1: interpolated in time)
        extrap_data.interpolation[ind] = 1

    #-- check if needing to extrapolate backwards in time
    count = np.count_nonzero(tdec < fd['time'].min())
    if (count > 0):
        #-- indices of dates before firn model
        ind, = np.nonzero(tdec < fd['time'].min())
        #-- query the search tree to find the NN closest points
        xy2 = np.concatenate((X[ind,None],Y[ind,None]),axis=1)
        dist,indices = tree.query(xy2, k=NN, return_distance=True)
        #-- normalized weights if POWER > 0 (typically between 1 and 3)
        #-- in the inverse distance weighting
        power_inverse_distance = dist**(-POWER)
        s = np.sum(power_inverse_distance, axis=1)
        w = power_inverse_distance/np.broadcast_to(s[:,None],(count,NN))
        #-- calculate a regression model for calculating values
        #-- spatially interpolate firn elevation or air content to coordinates
        FIRN = np.zeros((count,N))
        #-- create interpolated time series for calculating regression model
        for k in range(nt):
            #-- spatially extrapolate firn elevation or air content
            firn1 = fd[VARIABLE][k,i,j]
            FIRN[:,k] = np.sum(w*firn1[indices],axis=1)
        #-- calculate regression model
        for n,v in enumerate(ind):
            extrap_data[v] = regress_model(fd['time'], FIRN[n,:], tdec[v],
                ORDER=2, CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[0])
        #-- set interpolation type (2: extrapolated backwards in time)
        extrap_data.interpolation[ind] = 2

    #-- check if needing to extrapolate forward in time
    count = np.count_nonzero(tdec >= fd['time'].max())
    if (count > 0):
        #-- indices of dates after firn model
        ind, = np.nonzero(tdec >= fd['time'].max())
        #-- query the search tree to find the NN closest points
        xy2 = np.concatenate((X[ind,None],Y[ind,None]),axis=1)
        dist,indices = tree.query(xy2, k=NN, return_distance=True)
        #-- normalized weights if POWER > 0 (typically between 1 and 3)
        #-- in the inverse distance weighting
        power_inverse_distance = dist**(-POWER)
        s = np.sum(power_inverse_distance, axis=1)
        w = power_inverse_distance/np.broadcast_to(s[:,None],(count,NN))
        #-- calculate a regression model for calculating values
        #-- spatially interpolate firn elevation or air content to coordinates
        FIRN = np.zeros((count,N))
        #-- create interpolated time series for calculating regression model
        for k in range(nt):
            #-- spatially extrapolate firn elevation or air content
            firn1 = fd[VARIABLE][k,i,j]
            FIRN[:,k] = np.sum(w*firn1[indices],axis=1)
        #-- calculate regression model
        for n,v in enumerate(ind):
            extrap_data[v] = regress_model(fd['time'], FIRN[n,:], tdec[v],
                ORDER=2, CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[-1])
        #-- set interpolation type (3: extrapolated forward in time)
        extrap_data.interpolation[ind] = 3

    #-- complete mask if any invalid in data
    invalid, = np.nonzero(extrap_data.data == extrap_data.fill_value)
    extrap_data.mask[invalid] = True
    #-- replace fill value if specified
    if FILL_VALUE:
        extrap_data.fill_value = FILL_VALUE
        extrap_data.data[extrap_data.mask] = extrap_data.fill_value

    #-- return the interpolated values
    return extrap_data

#-- PURPOSE: calculate the Julian day from the calendar date
def calc_julian_day(YEAR, MONTH, DAY, HOUR=0, MINUTE=0, SECOND=0):
    JD = 367.*YEAR - np.floor(7.*(YEAR + np.floor((MONTH+9.)/12.))/4.) - \
        np.floor(3.*(np.floor((YEAR + (MONTH - 9.)/7.)/100.) + 1.)/4.) + \
        np.floor(275.*MONTH/9.) + DAY + 1721028.5 + HOUR/24. + MINUTE/1440. + \
        SECOND/86400.
    return JD
