#!/usr/bin/env python
u"""
mar_interp_daily.py
Written by Tyler Sutterley (01/2021)
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
    XNAME: x-coordinate variable name in MAR netCDF4 file
    YNAME: x-coordinate variable name in MAR netCDF4 file
    TIMENAME: time variable name in MAR netCDF4 file
    VARIABLE: MAR product to interpolate
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
    regress_model.py: models a time series using least-squares regression
    time.py: utilities for calculating time operations

UPDATE HISTORY:
    Updated 01/2021: using conversion protocols following pyproj-2 updates
        https://pyproj4.github.io/pyproj/stable/gotchas.html
        using utilities from time module for conversions
    Updated 08/2020: attempt delaunay triangulation using different options
    Updated 06/2020: set all values initially to fill_value
    Updated 05/2020: Gaussian average fields before interpolation
        accumulate variable over all available dates. add coordinate options
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
from SMBcorr.regress_model import regress_model
import SMBcorr.time

#-- PURPOSE: find a valid Delaunay triangulation for coordinates x0 and y0
#-- http://www.qhull.org/html/qhull.htm#options
#-- Attempt 1: standard qhull options Qt Qbb Qc Qz
#-- Attempt 2: rescale and center the inputs with option QbB
#-- Attempt 3: joggle the inputs to find a triangulation with option QJ
#-- if no passing triangulations: exit with empty list
def find_valid_triangulation(x0,y0):
    #-- Attempt 1: try with standard options Qt Qbb Qc Qz
    #-- Qt: triangulated output, all facets will be simplicial
    #-- Qbb: scale last coordinate to [0,m] for Delaunay triangulations
    #-- Qc: keep coplanar points with nearest facet
    #-- Qz: add point-at-infinity to Delaunay triangulation

    #-- Attempt 2 in case of qhull error from Attempt 1 try Qt Qc QbB
    #-- Qt: triangulated output, all facets will be simplicial
    #-- Qc: keep coplanar points with nearest facet
    #-- QbB: scale input to unit cube centered at the origin

    #-- Attempt 3 in case of qhull error from Attempt 2 try QJ QbB
    #-- QJ: joggle input instead of merging facets
    #-- QbB: scale input to unit cube centered at the origin

    #-- try each set of qhull_options
    points = np.concatenate((x0[:,None],y0[:,None]),axis=1)
    for i,opt in enumerate(['Qt Qbb Qc Qz','Qt Qc QbB','QJ QbB']):
        try:
            triangle = scipy.spatial.Delaunay(points.data, qhull_options=opt)
        except scipy.spatial.qhull.QhullError:
            pass
        else:
            return (i+1,triangle)

    #-- if still errors: set triangle as an empty list
    triangle = []
    return (None,triangle)

#-- PURPOSE: read and interpolate daily MAR outputs
def interpolate_mar_daily(DIRECTORY, EPSG, VERSION, tdec, X, Y,
    XNAME=None, YNAME=None, TIMENAME='TIME', VARIABLE='SMB',
    SIGMA=1.5, FILL_VALUE=None, EXTRAPOLATE=False):

    #-- start and end years to read
    SY = np.nanmin(np.floor(tdec)).astype(np.int)
    EY = np.nanmax(np.floor(tdec)).astype(np.int)
    YRS = '|'.join(['{0:4d}'.format(Y) for Y in range(SY,EY+1)])
    #-- regular expression pattern for MAR dataset
    rx = re.compile(r'{0}-(.*?)-(\d+)(_subset)?.nc$'.format(VERSION,YRS))

    #-- MAR model projection: Polar Stereographic (Oblique)
    #-- Earth Radius: 6371229 m
    #-- True Latitude: 0
    #-- Center Longitude: -40
    #-- Center Latitude: 70.5
    proj4_params = ("+proj=sterea +lat_0=+70.5 +lat_ts=0 +lon_0=-40.0 "
        "+a=6371229 +no_defs")

    #-- create list of files to read
    try:
        input_files=sorted([f for f in os.listdir(DIRECTORY) if rx.match(f)])
    except Exception as e:
        print(f"failed to find files matching {VERSION} in {DIRECTORY}")
        raise(e)

    #-- calculate number of time steps to read
    nt = 0
    for f,FILE in enumerate(input_files):
        #-- Open the MAR NetCDF file for reading
        with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
            nx = len(fileID.variables[XNAME][:])
            ny = len(fileID.variables[YNAME][:])
            TIME = fileID.variables[TIMENAME][:]
            try:
                nt += np.count_nonzero(TIME.data != TIME.fill_value)
            except AttributeError:
                nt += len(TIME)

    #-- python dictionary with file variables
    fd = {}
    fd['TIME'] = np.zeros((nt))
    #-- python dictionary with gaussian filtered variables
    gs = {}
    #-- calculate cumulative sum of gaussian filtered values
    cumulative = np.zeros((ny,nx))
    gs['CUMULATIVE'] = np.ma.zeros((nt,ny,nx), fill_value=FILL_VALUE)
    gs['CUMULATIVE'].mask = np.ones((nt,ny,nx), dtype=np.bool)
    #-- create a counter variable for filling variables
    c = 0
    #-- for each file in the list
    for f,FILE in enumerate(input_files):
        #-- Open the MAR NetCDF file for reading
        with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
            #-- number of time variables within file
            TIME = fileID.variables['TIME'][:]
            try:
                t = np.count_nonzero(TIME.data != TIME.fill_value)
            except AttributeError:
                t = len(TIME)
            #-- create a masked array with all data
            fd[VARIABLE] = np.ma.zeros((t,ny,nx),fill_value=FILL_VALUE)
            fd[VARIABLE].mask = np.zeros((t,ny,nx),dtype=np.bool)
            #-- surface type
            SRF=fileID.variables['SRF'][:]
            #-- indices of specified ice mask
            i,j=np.nonzero(SRF == 4)
            #-- ice fraction
            FRA=fileID.variables['FRA'][:]/100.0
            #-- Get data from netCDF variable and remove singleton dimensions
            tmp=np.squeeze(fileID.variables[VARIABLE][:])
            #-- combine sectors for multi-layered data
            if (np.ndim(tmp) == 4):
                #-- create mask for combining data
                MASK=np.zeros((t,ny,nx))
                MASK[:,i,j]=FRA[:t,0,i,j]
                #-- combine data
                fd[VARIABLE][:]=MASK*tmp[:t,0,:,:] + (1.0-MASK)*tmp[:t,1,:,:]
            else:
                #-- copy data
                fd[VARIABLE][:]=tmp[:t,:,:].copy()
            #-- verify mask object for interpolating data
            surf_mask = np.broadcast_to(SRF, (t,ny,nx))
            fd[VARIABLE].mask = fd[VARIABLE].data == fd[VARIABLE].fill_value
            fd[VARIABLE].mask[:,:,:] |= (surf_mask != 4)
            #-- combine mask object through time to create a single mask
            fd['MASK']=1.0-np.any(fd[VARIABLE].mask,axis=0).astype(np.float)
            #-- MAR coordinates
            fd['LON']=fileID.variables['LON'][:,:].copy()
            fd['LAT']=fileID.variables['LAT'][:,:].copy()
            #-- convert x and y coordinates to meters
            fd['x']=1000.0*fileID.variables[XNAME][:].copy()
            fd['y']=1000.0*fileID.variables[YNAME][:].copy()
            #-- extract delta time and epoch of time
            delta_time=fileID.variables[TIMENAME][:t].astype(np.float)
            date_string=fileID.variables[TIMENAME].units
        #-- extract epoch and units
        epoch,to_secs = SMBcorr.time.parse_date_string(date_string)
        #-- calculate time array in Julian days
        JD = SMBcorr.time.convert_delta_time(delta_time*to_secs, epoch1=epoch,
            epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0) + 2400000.5
        #-- convert from Julian days to calendar dates
        YY,MM,DD,hh,mm,ss = SMBcorr.time.convert_julian(JD)
        #-- calculate time in year-decimal
        fd['TIME'][c:c+t] = SMBcorr.time.convert_calendar_decimal(YY,MM,
            day=DD,hour=hh,minute=mm,second=ss)
        #-- use a gaussian filter to smooth mask
        gs['MASK'] = scipy.ndimage.gaussian_filter(fd['MASK'],SIGMA,
            mode='constant',cval=0)
        #-- indices of smoothed ice mask
        ii,jj = np.nonzero(np.ceil(gs['MASK']) == 1.0)
        #-- use a gaussian filter to smooth each model field
        gs[VARIABLE] = np.ma.zeros((t,ny,nx), fill_value=FILL_VALUE)
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
            gs[VARIABLE].data[tt,ii,jj] = temp2[ii,jj]/gs['MASK'][ii,jj]
            #-- replace valid values with original
            gs[VARIABLE].data[tt,i,j] = temp1[i,j]
            #-- set mask variables for time
            gs[VARIABLE].mask[tt,ii,jj] = False
            #-- calculate cumulative
            cumulative[ii,jj] += gs[VARIABLE][tt,ii,jj]
            gs['CUMULATIVE'].data[c+tt,ii,jj] = np.copy(cumulative[ii,jj])
            gs['CUMULATIVE'].mask[c+tt,ii,jj] = False
        #-- add to counter
        c += t

    #-- convert projection from input coordinates (EPSG) to model coordinates
    crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(EPSG))
    crs2 = pyproj.CRS.from_string(proj4_params)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    #-- calculate projected coordinates of input coordinates
    ix,iy = transformer.transform(X, Y)

    #-- check that input points are within convex hull of valid model points
    gs['x'],gs['y'] = np.meshgrid(fd['x'],fd['y'])
    v,triangle = find_valid_triangulation(gs['x'][ii,jj],gs['y'][ii,jj])
    #-- check if there is a valid triangulation
    if v:
        #-- check where points are within the complex hull of the triangulation
        interp_points = np.concatenate((ix[:,None],iy[:,None]),axis=1)
        valid = (triangle.find_simplex(interp_points) >= 0)
    else:
        #-- Check ix and iy against the bounds of x and y
        valid = (ix >= fd['x'].min()) & (ix <= fd['x'].max()) & \
            (iy >= fd['y'].min()) & (iy <= fd['y'].max())

    #-- output interpolated arrays of model variable
    npts = len(tdec)
    interp = np.ma.zeros((npts),fill_value=FILL_VALUE,dtype=np.float)
    interp.mask = np.ones((npts),dtype=np.bool)
    #-- initially set all values to fill value
    interp.data[:] = interp.fill_value
    #-- type designating algorithm used (1:interpolate, 2:backward, 3:forward)
    interp.interpolation = np.zeros((npts),dtype=np.uint8)

    #-- find days that can be interpolated
    if np.any((tdec >= fd['TIME'].min()) & (tdec <= fd['TIME'].max()) & valid):
        #-- indices of dates for interpolated days
        ind, = np.nonzero((tdec >= fd['TIME'].min()) &
            (tdec <= fd['TIME'].max()) & valid)
        #-- create an interpolator for model variable
        RGI = scipy.interpolate.RegularGridInterpolator(
            (fd['TIME'],fd['y'],fd['x']), gs['CUMULATIVE'].data)
        #-- create an interpolator for input mask
        MI = scipy.interpolate.RegularGridInterpolator(
            (fd['TIME'],fd['y'],fd['x']), gs['CUMULATIVE'].mask)

        #-- interpolate to points
        interp.data[ind] = RGI.__call__(np.c_[tdec[ind],iy[ind],ix[ind]])
        interp.mask[ind] = MI.__call__(np.c_[tdec[ind],iy[ind],ix[ind]])
        #-- set interpolation type (1: interpolated)
        interp.interpolation[ind] = 1

    #-- check if needing to extrapolate backwards in time
    count = np.count_nonzero((tdec < fd['TIME'].min()) & valid)
    if (count > 0) and EXTRAPOLATE:
        #-- indices of dates before model
        ind, = np.nonzero((tdec < fd['TIME'].min()) & valid)
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
            TIME[k] = fd['TIME'][k]
            #-- spatially interpolate model variable
            S1 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs['CUMULATIVE'].data[k,:,:].T, kx=1, ky=1)
            S2 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs['CUMULATIVE'].mask[k,:,:].T, kx=1, ky=1)
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
    count = np.count_nonzero((tdec > fd['TIME'].max()) & valid)
    if (count > 0) and EXTRAPOLATE:
        #-- indices of dates after model
        ind, = np.nonzero((tdec > fd['TIME'].max()) & valid)
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
            TIME[k] = fd['TIME'][kk]
            #-- spatially interpolate model variable
            S1 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs['CUMULATIVE'].data[kk,:,:].T, kx=1, ky=1)
            S2 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs['CUMULATIVE'].mask[kk,:,:].T, kx=1, ky=1)
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

    #-- return the interpolated values
    return interp
