#!/usr/bin/env python
u"""
mar_interp_seasonal.py
Written by Tyler Sutterley (06/2020)
Interpolates and extrapolates seasonal MAR products to times and coordinates
Seasonal files are climatology files for each day of the year

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
    RANGE: start year and end year of seasonal file
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

UPDATE HISTORY:
    Written 06/2020
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

#-- PURPOSE: read and interpolate a seasonal field of MAR outputs
def interpolate_mar_seasonal(DIRECTORY, EPSG, VERSION, tdec, X, Y,
    XNAME=None, YNAME=None, TIMENAME='TIME', VARIABLE='SMB',
    RANGE=[2000,2019], SIGMA=1.5, FILL_VALUE=None, EXTRAPOLATE=False):

    #-- MAR model projection: Polar Stereographic (Oblique)
    #-- Earth Radius: 6371229 m
    #-- True Latitude: 0
    #-- Center Longitude: -40
    #-- Center Latitude: 70.5
    proj4_params = ("+proj=sterea +lat_0=+70.5 +lat_ts=0 +lon_0=-40.0 "
        "+a=6371229 +no_defs")

    #-- regular expression pattern for MAR dataset
    rx = re.compile('MARseasonal(.*?){0}-{1}.nc$'.format(*RANGE))
    #-- find mar seasonal file for RANGE
    FILE, = [f for f in os.listdir(DIRECTORY) if rx.match(f)]
    #-- Open the MAR NetCDF file for reading
    with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
        nx = len(fileID.variables[XNAME][:])
        ny = len(fileID.variables[YNAME][:])
        #-- add 1 to use january 1st as day 366
        nt = len(fileID.variables[TIMENAME][:]) + 1

    #-- python dictionary with file variables
    fd = {}
    fd['TIME'] = np.arange(nt)/365.25
    #-- create a masked array with all data
    fd[VARIABLE] = np.ma.zeros((nt,ny,nx),fill_value=FILL_VALUE)
    fd[VARIABLE].mask = np.zeros((nt,ny,nx),dtype=np.bool)
    #-- python dictionary with gaussian filtered variables
    gs = {}
    #-- use a gaussian filter to smooth each model field
    gs[VARIABLE] = np.ma.zeros((nt,ny,nx), fill_value=FILL_VALUE)
    gs[VARIABLE].mask = np.ones((nt,ny,nx), dtype=np.bool)
    #-- calculate cumulative sum of gaussian filtered values
    cumulative = np.zeros((ny,nx))
    gs['CUMULATIVE'] = np.ma.zeros((nt,ny,nx), fill_value=FILL_VALUE)
    gs['CUMULATIVE'].mask = np.ones((nt,ny,nx), dtype=np.bool)
    #-- Open the MAR NetCDF file for reading
    with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
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
            MASK=np.zeros((nt,ny,nx))
            MASK[:,i,j]=FRA[:,0,i,j]
            #-- combine data
            fd[VARIABLE][:-1,:,:] = MASK*tmp[:,0,:,:] + \
                (1.0-MASK)*tmp[:,1,:,:]
        else:
            #-- copy data
            fd[VARIABLE][:-1,:,:] = tmp.copy()
        #-- use january 1st as time 366
        fd[VARIABLE][-1,:,:] = np.copy(fd[VARIABLE][0,:,:])
        #-- verify mask object for interpolating data
        surf_mask = np.broadcast_to(SRF, (nt,ny,nx))
        fd[VARIABLE].mask[:,:,:] |= (surf_mask != 4)
        #-- combine mask object through time to create a single mask
        fd['MASK']=1.0-np.any(fd[VARIABLE].mask,axis=0).astype(np.float)
        #-- MAR coordinates
        fd['LON']=fileID.variables['LON'][:,:].copy()
        fd['LAT']=fileID.variables['LAT'][:,:].copy()
        #-- convert x and y coordinates to meters
        fd['x']=1000.0*fileID.variables[XNAME][:].copy()
        fd['y']=1000.0*fileID.variables[YNAME][:].copy()
        #-- use a gaussian filter to smooth mask
        gs['MASK']=scipy.ndimage.gaussian_filter(fd['MASK'],SIGMA,
            mode='constant',cval=0)
        #-- indices of smoothed ice mask
        ii,jj = np.nonzero(np.ceil(gs['MASK']) == 1.0)
        #-- for each time
        for t in range(nt):
            #-- replace fill values before smoothing data
            temp1 = np.zeros((ny,nx))
            i,j = np.nonzero(~fd[VARIABLE].mask[t,:,:])
            temp1[i,j] = fd[VARIABLE][t,i,j].copy()
            #-- smooth spatial field
            temp2 = scipy.ndimage.gaussian_filter(temp1, SIGMA,
                mode='constant', cval=0)
            #-- scale output smoothed field
            gs[VARIABLE].data[t,ii,jj] = temp2[ii,jj]/gs['MASK'][ii,jj]
            #-- replace valid values with original
            gs[VARIABLE].data[t,i,j] = temp1[i,j]
            #-- set mask variables for time
            gs[VARIABLE].mask[t,ii,jj] = False
            #-- calculate cumulative
            cumulative[ii,jj] += gs[VARIABLE][t,ii,jj]
            gs['CUMULATIVE'].data[t,ii,jj] = np.copy(cumulative[ii,jj])
            gs['CUMULATIVE'].mask[t,ii,jj] = False

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

    #-- calculate the modulus of the time in year-decimal
    tmod = tdec % 1
    #-- number of output data points
    npts = len(tdec)
    #-- output interpolated arrays of model variable
    interp = np.ma.zeros((npts),fill_value=FILL_VALUE,dtype=np.float)
    interp.mask = np.ones((npts),dtype=np.bool)
    #-- initially set all values to fill value
    interp.data[:] = interp.fill_value

    #-- indices of valid spatial points
    ind, = np.nonzero(valid)
    #-- create an interpolator for model variable
    RGI = scipy.interpolate.RegularGridInterpolator(
        (fd['TIME'],fd['y'],fd['x']), gs['CUMULATIVE'].data)
    #-- create an interpolator for input mask
    MI = scipy.interpolate.RegularGridInterpolator(
        (fd['TIME'],fd['y'],fd['x']), gs['CUMULATIVE'].mask)

    #-- interpolate to points
    interp.data[ind] = RGI.__call__(np.c_[tmod[ind],iy[ind],ix[ind]])
    interp.mask[ind] = MI.__call__(np.c_[tmod[ind],iy[ind],ix[ind]])

    #-- complete mask if any invalid in data
    invalid, = np.nonzero((interp.data == interp.fill_value) |
        np.isnan(interp.data))
    interp.mask[invalid] = True

    #-- return the interpolated values
    return interp
