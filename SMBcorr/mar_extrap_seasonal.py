#!/usr/bin/env python
u"""
mar_extrap_seasonal.py
Written by Tyler Sutterley (06/2020)
Interpolates and extrapolates seasonal MAR products to times and coordinates
Seasonal files are climatology files for each day of the year

Uses fast nearest-neighbor search algorithms
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
and inverse distance weighted interpolation to extrapolate spatially

INPUTS:
    DIRECTORY: full path to the MAR data directory
        <path_to_mar>/MARv3.11/Greenland/ERA_1958-2019-15km/daily_15km
        <path_to_mar>/MARv3.11/Greenland/NCEP1_1948-2020_20km/daily_20km
        <path_to_mar>/MARv3.10/Greenland/NCEP1_1948-2019_20km/daily_20km
        <path_to_mar>/MARv3.9/Greenland/ERA_1958-2018_10km/daily_10km
    EPSG: projection of input spatial coordinates
    tdec: dates to interpolate in year-decimal
    X: x-coordinates to interpolate in projection EPSG
    Y: y-coordinates to interpolate in projection EPSG

OPTIONS:
    XNAME: x-coordinate variable name in MAR netCDF4 file
    YNAME: x-coordinate variable name in MAR netCDF4 file
    TIMENAME: time variable name in MAR netCDF4 file
    VARIABLE: MAR product to interpolate
    RANGE: start year and end year of seasonal file
    SIGMA: Standard deviation for Gaussian kernel
    SEARCH: nearest-neighbor search algorithm (BallTree or KDTree)
    NN: number of nearest-neighbor points to use
    POWER: inverse distance weighting power
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
    scikit-learn: Machine Learning in Python
        https://scikit-learn.org/stable/index.html
        https://github.com/scikit-learn/scikit-learn

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
from sklearn.neighbors import KDTree, BallTree

#-- PURPOSE: read and interpolate a seasonal field of MAR outputs
def extrapolate_mar_seasonal(DIRECTORY, EPSG, VERSION, tdec, X, Y,
    XNAME=None, YNAME=None, TIMENAME='TIME', VARIABLE='SMB',
    RANGE=[2000,2019], SIGMA=1.5, SEARCH='BallTree', NN=10, POWER=2.0,
    FILL_VALUE=None, EXTRAPOLATE=False):

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

    #-- convert MAR latitude and longitude to input coordinates (EPSG)
    proj1 = pyproj.Proj("+init={0}".format(EPSG))
    proj2 = pyproj.Proj("+init=EPSG:{0:d}".format(4326))
    xg,yg = pyproj.transform(proj2, proj1, fd['LON'], fd['LAT'])

    #-- construct search tree from original points
    #-- can use either BallTree or KDTree algorithms
    xy1 = np.concatenate((xg[i,j,None],yg[i,j,None]),axis=1)
    tree = BallTree(xy1) if (SEARCH == 'BallTree') else KDTree(xy1)

    #-- calculate the modulus of the time in year-decimal
    tmod = tdec % 1
    #-- number of output data points
    npts = len(tdec)
    #-- output interpolated arrays of output variable
    extrap = np.ma.zeros((npts),fill_value=FILL_VALUE,dtype=np.float)
    extrap.mask = np.ones((npts),dtype=np.bool)
    #-- initially set all values to fill value
    extrap.data[:] = extrap.fill_value
    #-- find indices for linearly interpolating in time
    f = scipy.interpolate.interp1d(fd['TIME'], np.arange(nt), kind='linear')
    date_indice = f(tmod).astype(np.int)
    #-- for each unique model date
    #-- linearly interpolate in time between two model maps
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
        var1 = gs['CUMULATIVE'][k,i,j]
        var2 = gs['CUMULATIVE'][k+1,i,j]
        #-- linearly interpolate to date
        dt = (tmod[kk] - fd['TIME'][k])/(fd['TIME'][k+1] - fd['TIME'][k])
        #-- spatially extrapolate using inverse distance weighting
        extrap.data[kk] = (1.0-dt)*np.sum(w*var1[indices],axis=1) + \
            dt*np.sum(w*var2[indices], axis=1)

    #-- complete mask if any invalid in data
    invalid, = np.nonzero((extrap.data == extrap.fill_value) |
        np.isnan(extrap.data))
    extrap.mask[invalid] = True

    #-- return the interpolated values
    return extrap
