#!/usr/bin/env python
u"""
racmo_interp_daily.py
Written by Tyler Sutterley (06/2020)
Interpolates and extrapolates daily RACMO products to times and coordinates

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
    X: x-coordinates to interpolate
    Y: y-coordinates to interpolate

OPTIONS:
    VARIABLE: RACMO product to interpolate
        smb: Surface Mass Balance
        hgtsrf: Change of Surface Height
    SIGMA: Standard deviation for Gaussian kernel
    FILL_VALUE: output fill_value for invalid points
    EXTRAPOLATE: create a regression model to extrapolate out in time

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    netCDF4: Python interface to the netCDF C library
         https://unidata.github.io/netcdf4-python/netCDF4/index.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/

PROGRAM DEPENDENCIES:
    convert_calendar_decimal.py: converts from calendar dates to decimal years
    convert_julian.py: returns the calendar date and time given a Julian date
    regress_model.py: models a time series using least-squares regression

UPDATE HISTORY:
    Updated 06/2020: set all values initially to fill_value
    Updated 05/2020: Gaussian average model fields before interpolation
        accumulate variable over all available dates
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

#-- PURPOSE: read and interpolate daily RACMO2.3 outputs
def interpolate_racmo_daily(base_dir, EPSG, MODEL, tdec, X, Y, VARIABLE='smb',
    SIGMA=1.5, FILL_VALUE=None, EXTRAPOLATE=False):

    #-- start and end years to read
    SY = np.nanmin(np.floor(tdec)).astype(np.int)
    EY = np.nanmax(np.floor(tdec)).astype(np.int)
    YRS = '|'.join(['{0:4d}'.format(Y) for Y in range(SY,EY+1)])
    #-- input list of files
    if (MODEL == 'FGRN055'):
        #-- filename and directory for input FGRN055 files
        file_pattern = 'RACMO2.3p2_FGRN055_{0}_daily_(\d+).nc'
        DIRECTORY = os.path.join(base_dir,'RACMO','GL','RACMO2.3p2_FGRN055')

    #-- create list of files to read
    rx = re.compile(file_pattern.format(VARIABLE,YRS),re.VERBOSE)
    input_files=sorted([f for f in os.listdir(DIRECTORY) if rx.match(f)])

    #-- calculate number of time steps to read
    nt = 0
    for f,FILE in enumerate(input_files):
        #-- Open the RACMO NetCDF file for reading
        with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
            nx = len(fileID.variables['rlon'][:])
            ny = len(fileID.variables['rlat'][:])
            nt += len(fileID.variables['time'][:])
            #-- invalid data value
            fv = np.float(fileID.variables[VARIABLE]._FillValue)

    #-- scaling factor for converting units
    if (VARIABLE == 'hgtsrf'):
        scale_factor = 86400.0
    elif (VARIABLE == 'smb'):
        scale_factor = 1.0

    #-- python dictionary with file variables
    fd = {}
    fd['time'] = np.zeros((nt))
    #-- python dictionary with gaussian filtered variables
    gs = {}
    #-- calculate cumulative sum of gaussian filtered values
    cumulative = np.zeros((ny,nx))
    gs['cumulative'] = np.ma.zeros((nt,ny,nx), fill_value=fv)
    gs['cumulative'].mask = np.zeros((nt,ny,nx), dtype=np.bool)
    #-- create a counter variable for filling variables
    c = 0
    #-- for each file in the list
    for f,FILE in enumerate(input_files):
        #-- Open the RACMO NetCDF file for reading
        with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
            #-- number of time variables within file
            t=len(fileID.variables['time'][:])
            fd[VARIABLE] = np.ma.zeros((t,ny,nx),fill_value=fv)
            fd[VARIABLE].mask = np.ones((t,ny,nx),dtype=np.bool)
            #-- Get data from netCDF variable and remove singleton dimensions
            tmp=np.squeeze(fileID.variables[VARIABLE][:])
            fd[VARIABLE][:] = scale_factor*tmp
            #-- indices of specified ice mask
            i,j = np.nonzero(tmp[0,:,:] != fv)
            fd[VARIABLE].mask[:,i,j] = False
            #-- combine mask object through time to create a single mask
            fd['mask']=1.0-np.any(fd[VARIABLE].mask,axis=0).astype(np.float)
            #-- racmo coordinates
            fd['lon']=fileID.variables['lon'][:,:].copy()
            fd['lat']=fileID.variables['lat'][:,:].copy()
            fd['x']=fileID.variables['rlon'][:].copy()
            fd['y']=fileID.variables['rlat'][:].copy()
            #-- rotated pole parameters
            proj4_params=fileID.variables['rotated_pole'].proj4_params
            #-- extract delta time and epoch of time
            delta_time=fileID.variables['time'][:].astype(np.float)
            units=fileID.variables['time'].units
        #-- convert epoch of time to Julian days
        Y1,M1,D1,h1,m1,s1=[float(d) for d in re.findall('\d+\.\d+|\d+',units)]
        epoch_julian=calc_julian_day(Y1,M1,D1,HOUR=h1,MINUTE=m1,SECOND=s1)
        #-- calculate time array in Julian days
        Y2,M2,D2,h2,m2,s2=convert_julian(epoch_julian + delta_time)
        #-- calculate time in year-decimal
        fd['time'][c:c+t]=convert_calendar_decimal(Y2,M2,D2,
            HOUR=h2,MINUTE=m2,SECOND=s2)
        #-- use a gaussian filter to smooth mask
        gs['mask']=scipy.ndimage.gaussian_filter(fd['mask'],SIGMA,
            mode='constant',cval=0)
        #-- indices of smoothed ice mask
        ii,jj = np.nonzero(np.ceil(gs['mask']) == 1.0)
        #-- use a gaussian filter to smooth each model field
        gs[VARIABLE] = np.ma.zeros((t,ny,nx), fill_value=fv)
        gs[VARIABLE].mask = np.ones((t,ny,nx), dtype=np.bool)
        #-- for each time
        for tt in range(t):
            #-- replace fill values before smoothing data
            temp1 = np.zeros((ny,nx))
            i,j = np.nonzero(~fd[VARIABLE].mask[tt,:,:])
            temp1[i,j] = fd[VARIABLE][tt,i,j].copy()
            #-- smooth spatial field
            temp2 = scipy.ndimage.gaussian_filter(temp1, SIGMA,
                mode='constant', cval=0)
            #-- scale output smoothed field
            gs[VARIABLE][tt,ii,jj] = temp2[ii,jj]/gs['mask'][ii,jj]
            #-- replace valid values with original
            gs[VARIABLE][tt,i,j] = temp1[i,j]
            #-- set mask variables for time
            gs[VARIABLE].mask[tt,ii,jj] = False
            #-- calculate cumulative
            cumulative[ii,jj] += gs[VARIABLE][tt,ii,jj]
            gs['cumulative'].data[c+tt,ii,jj] = np.copy(cumulative[ii,jj])
            gs['cumulative'].mask[c+tt,ii,jj] = False
        #-- add to counter
        c += t

    #-- convert projection from input coordinates (EPSG) to model coordinates
    #-- RACMO models are rotated pole latitude and longitude
    proj1 = pyproj.Proj("+init={0}".format(EPSG))
    proj2 = pyproj.Proj(proj4_params)
    #-- calculate rotated pole coordinates of input coordinates
    ix,iy = pyproj.transform(proj1, proj2, X, Y)

    #-- check that input points are within convex hull of valid model points
    gs['x'],gs['y'] = np.meshgrid(fd['x'],fd['y'])
    points = np.concatenate((gs['x'][ii,jj,None],gs['y'][ii,jj,None]),axis=1)
    triangle = scipy.spatial.Delaunay(points.data, qhull_options='Qt Qbb Qc Qz')
    interp_points = np.concatenate((ix[:,None],iy[:,None]),axis=1)
    valid = (triangle.find_simplex(interp_points) >= 0)

    #-- output interpolated arrays of model variable
    npts = len(tdec)
    interp = np.ma.zeros((npts),fill_value=fv,dtype=np.float)
    interp.mask = np.ones((npts),dtype=np.bool)
    #-- initially set all values to fill value
    interp.data[:] = interp.fill_value
    #-- type designating algorithm used (1:interpolate, 2:backward, 3:forward)
    interp.interpolation = np.zeros((npts),dtype=np.uint8)

    #-- find days that can be interpolated
    if np.any((tdec >= fd['time'].min()) & (tdec <= fd['time'].max()) & valid):
        #-- indices of dates for interpolated days
        ind, = np.nonzero((tdec >= fd['time'].min()) &
            (tdec <= fd['time'].max()) & valid)
        #-- create an interpolator for model variable
        RGI = scipy.interpolate.RegularGridInterpolator(
            (fd['time'],fd['y'],fd['x']), gs['cumulative'].data)
        #-- create an interpolator for input mask
        MI = scipy.interpolate.RegularGridInterpolator(
            (fd['time'],fd['y'],fd['x']), gs['cumulative'].mask)

        #-- interpolate to points
        interp.data[ind] = RGI.__call__(np.c_[tdec[ind],iy[ind],ix[ind]])
        interp.mask[ind] = MI.__call__(np.c_[tdec[ind],iy[ind],ix[ind]])
        #-- set interpolation type (1: interpolated)
        interp.interpolation[ind] = 1

    #-- check if needing to extrapolate backwards in time
    count = np.count_nonzero((tdec < fd['time'].min()) & valid)
    if (count > 0) and EXTRAPOLATE:
        #-- indices of dates before model
        ind, = np.nonzero((tdec < fd['time'].min()) & valid)
        #-- read the first year of data to create regression model
        N = 365
        #-- calculate a regression model for calculating values
        #-- spatially interpolate model variable to coordinates
        DATA = np.zeros((count,N))
        MASK = np.zeros((count,N),dtype=np.bool)
        TIME = np.zeros((N))
        #-- create interpolated time series for calculating regression model
        for k in range(N):
            #-- time at k
            TIME[k] = fd['time'][k]
            #-- spatially interpolate model variable
            S1 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs['cumulative'].data[k,:,:].T, kx=1, ky=1)
            S2 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs['cumulative'].mask[k,:,:].T, kx=1, ky=1)
            #-- create numpy masked array of interpolated values
            DATA[:,k] = S1.ev(ix[ind],iy[ind])
            MASK[:,k] = S2.ev(ix[ind],iy[ind])
        #-- calculate regression model
        for n,v in enumerate(ind):
            interp.data[v] = regress_model(TIME, DATA[n,:], tdec[v],
                ORDER=2, CYCLES=[0.25,0.5,1.0], RELATIVE=TIME[0])
        #-- mask any invalid points
        interp.mask[ind] = np.any(MASK, axis=1)
        #-- set interpolation type (2: extrapolated backward)
        interp.interpolation[ind] = 2

    #-- check if needing to extrapolate forward in time
    count = np.count_nonzero((tdec > fd['time'].max()) & valid)
    if (count > 0) and EXTRAPOLATE:
        #-- indices of dates after model
        ind, = np.nonzero((tdec > fd['time'].max()) & valid)
        #-- read the last year of data to create regression model
        N = 365
        #-- calculate a regression model for calculating values
        #-- spatially interpolate model variable to coordinates
        DATA = np.zeros((count,N))
        MASK = np.zeros((count,N),dtype=np.bool)
        TIME = np.zeros((N))
        #-- create interpolated time series for calculating regression model
        for k in range(N):
            kk = nt - N + k
            #-- time at kk
            TIME[k] = fd['time'][kk]
            #-- spatially interpolate model variable
            S1 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs['cumulative'].data[kk,:,:].T, kx=1, ky=1)
            S2 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs['cumulative'].mask[kk,:,:].T, kx=1, ky=1)
            #-- create numpy masked array of interpolated values
            DATA[:,k] = S1.ev(ix[ind],iy[ind])
            MASK[:,k] = S2.ev(ix[ind],iy[ind])
        #-- calculate regression model
        for n,v in enumerate(ind):
            interp.data[v] = regress_model(TIME, DATA[n,:], tdec[v],
                ORDER=2, CYCLES=[0.25,0.5,1.0], RELATIVE=TIME[-1])
        #-- mask any invalid points
        interp.mask[ind] = np.any(MASK, axis=1)
        #-- set interpolation type (3: extrapolated forward)
        interp.interpolation[ind] = 3

    #-- complete mask if any invalid in data
    invalid, = np.nonzero((interp.data == interp.fill_value) |
        np.isnan(interp.data))
    interp.mask[invalid] = True
    #-- replace fill value if specified
    if FILL_VALUE:
        interp.fill_value = FILL_VALUE
        interp.data[interp.mask] = interp.fill_value

    #-- return the interpolated values
    return interp

#-- PURPOSE: calculate the Julian day from the calendar date
def calc_julian_day(YEAR, MONTH, DAY, HOUR=0, MINUTE=0, SECOND=0):
    JD = 367.*YEAR - np.floor(7.*(YEAR + np.floor((MONTH+9.)/12.))/4.) - \
        np.floor(3.*(np.floor((YEAR + (MONTH - 9.)/7.)/100.) + 1.)/4.) + \
        np.floor(275.*MONTH/9.) + DAY + 1721028.5 + HOUR/24. + MINUTE/1440. + \
        SECOND/86400.
    return JD
