#!/usr/bin/env python
u"""
mar_interp_daily.py
Written by Tyler Sutterley (05/2020)
Interpolates and extrapolates daily MAR products to times and coordinates

INPUTS:
    DIRECTORY: full path to the MAR data directory
        <path_to_mar>/MARv3.11/Greenland/ERA_1958-2019-15km/daily_15km
        <path_to_mar>/MARv3.11/Greenland/NCEP1_1948-2020_20km/daily_20km
        <path_to_mar>/MARv3.10/Greenland/NCEP1_1948-2019_20km/daily_20km
        <path_to_mar>/MARv3.9/Greenland/ERA_1958-2018_10km/daily_10km
    EPSG: projection of input spatial coordinates
    tdec: dates to interpolate in year-decimal
    X: x-coordinates to interpolate
    Y: y-coordinates to interpolate

OPTIONS:
    VARIABLE: MAR product to interpolate
    SIGMA: Standard deviation for Gaussian kernel
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
    Updated 05/2020: Gaussian average fields before interpolation
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
import scipy.ndimage
import scipy.interpolate
from SMBcorr.convert_calendar_decimal import convert_calendar_decimal
from SMBcorr.convert_julian import convert_julian
from SMBcorr.regress_model import regress_model

#-- PURPOSE: read and interpolate daily MAR outputs
def interpolate_mar_daily(DIRECTORY, EPSG, VERSION, tdec, X, Y,
    VARIABLE='SMB', SIGMA=1.5, FILL_VALUE=None):

    #-- start and end years to read
    SY = np.nanmin(np.floor(tdec)).astype(np.int)
    EY = np.nanmax(np.floor(tdec)).astype(np.int)
    YRS = '|'.join(['{0:4d}'.format(Y) for Y in range(SY,EY+1)])
    #-- regular expression pattern for MAR dataset
    rx = re.compile('{0}-(.*?)-({1})(_subset)?.nc$'.format(VERSION,YRS))

    #-- variable coordinates
    XNAME,YNAME,TIMENAME = ('X10_105','Y21_199','TIME')
    #-- MAR model projection: Polar Stereographic (Oblique)
    #-- Earth Radius: 6371229 m
    #-- True Latitude: 0
    #-- Center Longitude: -40
    #-- Center Latitude: 70.5
    proj4_params = ("+proj=sterea +lat_0=+70.5 +lat_ts=0 +lon_0=-40.0 "
        "+a=6371229 +no_defs")

    #-- create list of files to read
    input_files = [fi for fi in os.listdir(DIRECTORY) if rx.match(fi)]

    #-- calculate number of time steps to read
    nt = 0
    for FILE in input_files:
        #-- Open the MAR NetCDF file for reading
        with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
            nx = len(fileID.variables[XNAME][:])
            ny = len(fileID.variables[YNAME][:])
            nt += len(fileID.variables[TIMENAME][:])

    #-- create a masked array with all data
    fd = {}
    fd[VARIABLE] = np.ma.zeros((nt,ny,nx),fill_value=FILL_VALUE)
    fd[VARIABLE].mask = np.zeros((nt,ny,nx),dtype=np.bool)
    fd['TIME'] = np.zeros((nt))
    #-- create a counter variable for filling variables
    c = 0
    #-- for each file in the list
    for FILE in input_files:
        #-- Open the MAR NetCDF file for reading
        with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
            #-- number of time variables within file
            t=len(fileID.variables['TIME'][:])
            #-- surface type
            SRF=fileID.variables['SRF'][:]
            #-- ice fraction
            FRA=fileID.variables['FRA'][:]/100.0            
            #-- Get data from netCDF variable and remove singleton dimensions
            tmp=np.squeeze(fileID.variables[VARIABLE][:])
            #-- combine sectors for multi-layered data
            if (np.ndim(tmp) == 4):
                #-- create mask for combining data
                i,j=np.nonzero(SRF == 4)
                MASK=np.zeros((nt,ny,nx))
                MASK[:,i,j]=FRA[:,0,i,j]
                #-- combine data
                fd[VARIABLE][c:c+t,:,:]=MASK*tmp[:,0,:,:] + \
                    (1.0-MASK)*tmp[:,1,:,:]
            else:
                #-- copy data
                fd[VARIABLE][c:c+t,:,:]=tmp.copy()                
            #-- verify mask object for interpolating data
            surf_mask = np.broadcast_to(SRF, (nt,ny,nx))
            fd[VARIABLE].mask[c:c+t,:,:] |= (surf_mask != 4)
            #-- MAR coordinates
            fd['LON']=fileID.variables['LON'][:,:].copy()
            fd['LAT']=fileID.variables['LAT'][:,:].copy()
            fd['x']=fileID.variables[XNAME][:].copy()
            fd['y']=fileID.variables[YNAME][:].copy()
            #-- extract delta time and epoch of time
            delta_time=fileID.variables[TIMENAME][:].copy()
            units=fileID.variables[TIMENAME].units
            #-- convert epoch of time to Julian days
            Y1,M1,D1,h1,m1,s1=[float(d) for d in re.findall('\d+\.\d+|\d+',units)]
            epoch_julian=calc_julian_day(Y1,M1,D1,HOUR=h1,MINUTE=m1,SECOND=s1)
            #-- calculate time array in Julian days
            Y2,M2,D2,h2,m2,s2=convert_julian(epoch_julian + delta_time)
            #-- calculate time in year-decimal
            fd['TIME'][c:c+t]=convert_calendar_decimal(Y2,M2,D2,
                HOUR=h2,MINUTE=m2,SECOND=s2)

    #-- indices of specified ice mask
    i,j = np.nonzero(SRF == 4)
    
    #-- combine mask object through time to create a single mask
    fd['MASK']=1.0-np.any(fd[VARIABLE].mask,axis=0).astype(np.float)
    #-- use a gaussian filter to smooth mask
    gs = {}
    gs['MASK']=scipy.ndimage.gaussian_filter(fd['MASK'],SIGMA,
        mode='constant',cval=0)
    #-- indices of smoothed ice mask
    ii,jj = np.nonzero(np.ceil(gs['MASK']) == 1.0)
    #-- use a gaussian filter to smooth each model field
    gs[VARIABLE] = np.ma.zeros((nt,ny,nx), fill_value=FILL_VALUE)
    gs[VARIABLE].mask = np.ma.zeros((nt,ny,nx), dtype=np.bool)
    for t in range(nt):
        #-- replace fill values before smoothing data
        temp1 = np.zeros((ny,nx))
        i,j = np.nonzero(~fd[VARIABLE].mask[t,:,:])
        temp1[i,j] = fd[VARIABLE][t,i,j].copy()
        #-- smooth spatial field
        temp2 = scipy.ndimage.gaussian_filter(temp1, SIGMA,
            mode='constant', cval=0)
        #-- scale output smoothed field
        gs[VARIABLE][t,ii,jj] = temp2[ii,jj]/gs['MASK'][ii,jj]
        #-- replace valid values with original
        gs[VARIABLE][t,i,j] = temp1[i,j]
        #-- set mask variables for time
        gs[VARIABLE].mask[t,:,:] = (gs['MASK'] == 0.0)

    #-- convert projection from input coordinates (EPSG) to model coordinates
    proj1 = pyproj.Proj("+init={0}".format(EPSG))
    proj2 = pyproj.Proj(proj4_params)
    #-- calculate projected coordinates of input coordinates
    ix,iy = pyproj.transform(proj1, proj2, X, Y)

    #-- check that input points are within convex hull of valid model points
    gs['x'],gs['y'] = np.meshgrid(fd['x'],fd['y'])
    points = np.concatenate((gs['x'][ii,jj,None],gs['y'][ii,jj,None]),axis=1)
    triangle = scipy.spatial.Delaunay(points.data, qhull_options='Qt Qbb Qc Qz')
    interp_points = np.concatenate((ix[:,None],iy[:,None]),axis=1)
    valid = (triangle.find_simplex(interp_points) >= 0)

    #-- output interpolated arrays of model variable
    npts = len(tdec)
    interp_data = np.ma.zeros((npts),fill_value=FILL_VALUE,dtype=np.float)
    interp_data.mask = np.zeros((npts),dtype=np.bool)
    #-- type designating algorithm used (1:interpolate, 2:backward, 3:forward)
    interp_data.interpolation = np.zeros((npts),dtype=np.uint8)

    #-- find days that can be interpolated
    if np.any((tdec >= fd['TIME'].min()) & (tdec <= fd['TIME'].max()) & valid):
        #-- indices of dates for interpolated days
        ind, = np.nonzero((tdec >= fd['TIME'].min()) &
            (tdec <= fd['TIME'].max()) & valid)
        #-- create an interpolator for model variable
        RGI = scipy.interpolate.RegularGridInterpolator(
            (fd['TIME'],fd['y'],fd['x']), gs[VARIABLE].data)
        #-- create an interpolator for input mask
        MI = scipy.interpolate.RegularGridInterpolator(
            (fd['TIME'],fd['y'],fd['x']), gs[VARIABLE].mask)

        #-- interpolate to points
        interp_data.data[ind] = RGI.__call__(np.c_[tdec[ind],iy[ind],ix[ind]])
        interp_data.mask[ind] = MI.__call__(np.c_[tdec[ind],iy[ind],ix[ind]])
        #-- set interpolation type (1: interpolated)
        interp_data.interpolation[ind] = 1

    #-- check if needing to extrapolate backwards in time
    count = np.count_nonzero((tdec < fd['TIME'].min()) & valid)
    if (count > 0):
        #-- indices of dates before model
        ind, = np.nonzero((tdec < fd['TIME'].min()) & valid)
        #-- calculate a regression model for calculating values
        #-- spatially interpolate model variable to coordinates
        DATA = np.zeros((count,nt))
        MASK = np.zeros((count,nt),dtype=np.bool)
        #-- create interpolated time series for calculating regression model
        for k in range(nt):
            #-- spatially interpolate model variable
            S1 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE].data[k,:,:].T, kx=1, ky=1)
            S2 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE].mask[k,:,:].T, kx=1, ky=1)
            #-- create numpy masked array of interpolated values
            DATA[:,k] = S1.ev(ix[ind],iy[ind])
            MASK[:,k] = S2.ev(ix[ind],iy[ind])
        #-- calculate regression model
        for n,v in enumerate(ind):
            interp_data.data[v] = regress_model(fd['TIME'], DATA[n,:], tdec[v],
                ORDER=2, CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=fd['TIME'][0])
        #-- mask any invalid points
        interp_data.mask[ind] = np.any(MASK, axis=1)
        #-- set interpolation type (2: extrapolated backward)
        interp_data.interpolation[ind] = 2

    #-- check if needing to extrapolate forward in time
    count = np.count_nonzero((tdec > fd['TIME'].max()) & valid)
    if (count > 0):
        #-- indices of dates after model
        ind, = np.nonzero((tdec > fd['TIME'].max()) & valid)
        #-- calculate a regression model for calculating values
        #-- spatially interpolate model variable to coordinates
        DATA = np.zeros((count,nt))
        MASK = np.zeros((count,nt),dtype=np.bool)
        #-- create interpolated time series for calculating regression model
        for k in range(nt):
            #-- spatially interpolate model variable
            S1 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE].data[k,:,:].T, kx=1, ky=1)
            S2 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE].mask[k,:,:].T, kx=1, ky=1)
            #-- create numpy masked array of interpolated values
            DATA[:,k] = S1.ev(ix[ind],iy[ind])
            MASK[:,k] = S2.ev(ix[ind],iy[ind])
        #-- calculate regression model
        for n,v in enumerate(ind):
            interp_data.data[v] = regress_model(fd['TIME'], DATA[n,:], tdec[v],
                ORDER=2, CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=fd['TIME'][-1])
        #-- mask any invalid points
        interp_data.mask[ind] = np.any(MASK, axis=1)
        #-- set interpolation type (3: extrapolated forward)
        interp_data.interpolation[ind] = 3

    #-- complete mask if any invalid in data
    invalid, = np.nonzero(interp_data.data == interp_data.fill_value)
    interp_data.mask[invalid] = True

    #-- return the interpolated values
    return interp_data

#-- PURPOSE: calculate the Julian day from the calendar date
def calc_julian_day(YEAR, MONTH, DAY, HOUR=0, MINUTE=0, SECOND=0):
    JD = 367.*YEAR - np.floor(7.*(YEAR + np.floor((MONTH+9.)/12.))/4.) - \
        np.floor(3.*(np.floor((YEAR + (MONTH - 9.)/7.)/100.) + 1.)/4.) + \
        np.floor(275.*MONTH/9.) + DAY + 1721028.5 + HOUR/24. + MINUTE/1440. + \
        SECOND/86400.
    return JD
