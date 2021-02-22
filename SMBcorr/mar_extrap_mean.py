#!/usr/bin/env python
u"""
mar_extrap_mean.py
Written by Tyler Sutterley (01/2021)
Interpolates mean MAR products to times and coordinates

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
    RANGE: start year and end year of mean file
    SIGMA: Standard deviation for Gaussian kernel
    SEARCH: nearest-neighbor search algorithm (BallTree or KDTree)
    NN: number of nearest-neighbor points to use
    POWER: inverse distance weighting power
    FILL_VALUE: output fill_value for invalid points

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
    Updated 01/2021: using conversion protocols following pyproj-2 updates
        https://pyproj4.github.io/pyproj/stable/gotchas.html
    Written 08/2020
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

#-- PURPOSE: read and interpolate a mean field of MAR outputs
def extrapolate_mar_mean(DIRECTORY, EPSG, VERSION, tdec, X, Y,
    XNAME=None, YNAME=None, TIMENAME='TIME', VARIABLE='SMB',
    RANGE=[2000,2019], SIGMA=1.5, SEARCH='BallTree', NN=10, POWER=2.0,
    FILL_VALUE=None):

    #-- regular expression pattern for MAR dataset
    rx = re.compile('MAR_SMBavg(.*?){0}-{1}.nc$'.format(*RANGE))
    #-- find mar mean file for RANGE
    FILE, = [f for f in os.listdir(DIRECTORY) if rx.match(f)]
    #-- Open the MAR NetCDF file for reading
    with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
        nx = len(fileID.variables[XNAME][:])
        ny = len(fileID.variables[YNAME][:])

    #-- python dictionary with file variables
    fd = {}
    #-- create a masked array with all data
    fd[VARIABLE] = np.ma.zeros((ny,nx),fill_value=FILL_VALUE)
    fd[VARIABLE].mask = np.zeros((ny,nx),dtype=bool)
    #-- python dictionary with gaussian filtered variables
    gs = {}
    #-- use a gaussian filter to smooth each model field
    gs[VARIABLE] = np.ma.zeros((ny,nx), fill_value=FILL_VALUE)
    gs[VARIABLE].mask = np.ones((ny,nx), dtype=bool)
    #-- Open the MAR NetCDF file for reading
    with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
        #-- surface type
        SRF=fileID.variables['SRF'][:]
        #-- indices of specified ice mask
        i,j=np.nonzero(SRF == 4)
        #-- Get data from netCDF variable and remove singleton dimensions
        tmp=np.squeeze(fileID.variables[VARIABLE][:])
        #-- combine sectors for multi-layered data
        if (np.ndim(tmp) == 3):
            #-- ice fraction
            FRA=fileID.variables['FRA'][:]/100.0
            #-- create mask for combining data
            MASK = np.zeros((ny,nx))
            MASK[i,j] = FRA[i,j]
            #-- combine data
            fd[VARIABLE][:,:] = MASK*tmp[0,:,:] + \
                (1.0-MASK)*tmp[1,:,:]
        else:
            #-- copy data
            fd[VARIABLE][:,:] = tmp.copy()
        #-- verify mask object for interpolating data
        fd[VARIABLE].mask[:,:] |= (SRF != 4)
        #-- combine mask object through time to create a single mask
        fd['MASK']=1.0 - np.array(fd[VARIABLE].mask,dtype=np.float)
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
        #-- replace fill values before smoothing data
        temp1 = np.zeros((ny,nx))
        i,j = np.nonzero(~fd[VARIABLE].mask)
        temp1[i,j] = fd[VARIABLE][i,j].copy()
        #-- smooth spatial field
        temp2 = scipy.ndimage.gaussian_filter(temp1, SIGMA,
            mode='constant', cval=0)
        #-- scale output smoothed field
        gs[VARIABLE].data[ii,jj] = temp2[ii,jj]/gs['MASK'][ii,jj]
        #-- replace valid values with original
        gs[VARIABLE].data[i,j] = temp1[i,j]
        #-- set mask variables for time
        gs[VARIABLE].mask[ii,jj] = False

    #-- convert MAR latitude and longitude to input coordinates (EPSG)
    crs1 = pyproj.CRS.from_string(EPSG)
    crs2 = pyproj.CRS.from_string("epsg:{0:d}".format(4326))
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    direction = pyproj.enums.TransformDirection.INVERSE
    #-- convert projection from model coordinates
    xg,yg = transformer.transform(fd['LON'], fd['LAT'], direction=direction)

    #-- construct search tree from original points
    #-- can use either BallTree or KDTree algorithms
    xy1 = np.concatenate((xg[i,j,None],yg[i,j,None]),axis=1)
    tree = BallTree(xy1) if (SEARCH == 'BallTree') else KDTree(xy1)

    #-- number of output data points
    npts = len(tdec)
    #-- output interpolated arrays of output variable
    extrap = np.ma.zeros((npts),fill_value=FILL_VALUE,dtype=np.float)
    extrap.mask = np.ones((npts),dtype=bool)

    #-- query the search tree to find the NN closest points
    xy2 = np.concatenate((X[:,None],Y[:,None]),axis=1)
    dist,indices = tree.query(xy2, k=NN, return_distance=True)
    #-- normalized weights if POWER > 0 (typically between 1 and 3)
    #-- in the inverse distance weighting
    power_inverse_distance = dist**(-POWER)
    s = np.sum(power_inverse_distance)
    w = power_inverse_distance/s
    #-- variable for valid points
    var1 = gs[VARIABLE][i,j]
    #-- spatially extrapolate using inverse distance weighting
    extrap.data[:] = np.sum(w*var1[indices],axis=1)

    #-- complete mask if any invalid in data
    invalid, = np.nonzero((extrap.data == extrap.fill_value) |
        np.isnan(extrap.data))
    extrap.mask[invalid] = True

    #-- return the interpolated values
    return extrap
