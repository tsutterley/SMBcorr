#!/usr/bin/env python
u"""
merra_hybrid_extrap.py
Written by Tyler Sutterley (10/2019)
Interpolates and extrapolates MERRA-2 hybrid variables to times and coordinates
    MERRA-2 Hybrid firn model outputs provided by Brooke Medley at GSFC

CALLING SEQUENCE:
    python merra_hybrid_extrap.py --directory=<path> --region=gris \
        --coordinate=[-39e4,-133e4],[-39e4,-133e4] --date=2016.1,2018.1

COMMAND LINE OPTIONS:
    -D X, --directory=X: Working data directory
    -R X, --region=X: Region to interpolate (gris, ais)
    --coordinate=X: Polar Stereographic X and Y of point
    --date=X: Date to interpolate in year-decimal format
    --csv=X: Read dates and coordinates from a csv file
    --sigma=X: Standard deviation for Gaussian kernel
    --fill-value: Replace invalid values with fill value
        (default uses original fill values from data file)

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
    scikit-learn: Machine Learning in Python
        http://scikit-learn.org/stable/index.html
        https://github.com/scikit-learn/scikit-learn

PROGRAM DEPENDENCIES:
    regress_model.py: models a time series using least-squares regression

UPDATE HISTORY:
    Written 10/2019
"""
from __future__ import print_function

import sys
import os
import re
import pyproj
import getopt
import netCDF4
import numpy as np
import scipy.ndimage
import scipy.interpolate
from sklearn.neighbors import KDTree, BallTree
from SMBcorr.regress_model import regress_model

#-- PURPOSE: set the projection parameters based on the region name
def set_projection(REGION):
    if (REGION == 'ais'):
        projection_flag = 'EPSG:3031'
    elif (REGION == 'gris'):
        projection_flag = 'EPSG:3413'
    return projection_flag

#-- PURPOSE: read and interpolate MERRA-2 hybrid firn corrections
def extrapolate_merra_hybrid(base_dir, EPSG, REGION, tdec, X, Y,
    SEARCH='BallTree', N=10, POWER=2.0, SIGMA=1.5, VARIABLE='FAC',
    FILL_VALUE=None):

    #-- set the input netCDF4 file for the variable of interest
    if VARIABLE in ('FAC'):
        hybrid_file='gsfc_{0}_{1}.nc'.format(VARIABLE,REGION.lower())
    elif VARIABLE in ('p_minus_e','melt'):
        hybrid_file='m2_hybrid_{0}_cumul_{1}.nc'.format(VARIABLE,REGION.lower())
    #-- Open the MERRA-2 Hybrid NetCDF file for reading
    fileID = netCDF4.Dataset(os.path.join(base_dir,hybrid_file), 'r')
    #-- Get data from each netCDF variable and remove singleton dimensions
    fd = {}
    fd[VARIABLE] = np.squeeze(fileID.variables[VARIABLE][:].copy())
    xg = fileID.variables['x'][:,:].copy()
    yg = fileID.variables['y'][:,:].copy()
    fd['time'] = fileID.variables['time'][:].copy()
    #-- invalid data value
    fv = np.float(fileID.variables[VARIABLE]._FillValue)
    #-- input shape of MERRA-2 Hybrid firn data
    nt,nx,ny = np.shape(fd[VARIABLE])
    #-- close the NetCDF files
    fileID.close()
    #-- time is year decimal at time step 5 days
    time_step = 5.0/365.25

    #-- indices of specified ice mask
    i,j = np.nonzero(fd[VARIABLE][0,:,:] != fv)
    #-- create mask object for interpolating data
    fd['mask'] = np.zeros((nx,ny))
    fd['mask'][i,j] = 1.0
    #-- extract x and y coordinate arrays from grids
    fd['x'],fd['y'] = (xg[:,0],yg[0,:])

    #-- use a gaussian filter to smooth mask
    gs = {}
    gs['mask'] = scipy.ndimage.gaussian_filter(fd['mask'], SIGMA,
        mode='constant', cval=0)
    #-- indices of smoothed ice mask
    ii,jj = np.nonzero(np.ceil(gs['mask']) == 1.0)
    #-- use a gaussian filter to smooth each firn field
    gs[VARIABLE] = np.ma.zeros((nt,nx,ny), fill_value=fv)
    gs[VARIABLE].mask = (gs['mask'] == 0.0)
    for t in range(nt):
        #-- replace fill values before smoothing data
        temp1 = np.zeros((nx,ny))
        #-- reference to first firn field
        temp1[i,j] = fd[VARIABLE][t,i,j] - fd[VARIABLE][0,i,j]
        #-- smooth firn field
        temp2 = scipy.ndimage.gaussian_filter(temp1, SIGMA,
            mode='constant', cval=0)
        #-- scale output smoothed firn field
        gs[VARIABLE][t,ii,jj] = temp2[ii,jj]/gs['mask'][ii,jj]
        #-- replace valid firn values with original
        gs[VARIABLE][t,i,j] = temp1[i,j]

    #-- convert projection from input coordinates (EPSG) to model coordinates
    #-- MERRA-2 Hybrid models are rotated pole latitude and longitude
    MODEL_EPSG = set_projection(REGION)
    proj1 = pyproj.Proj("+init={0}".format(EPSG))
    proj2 = pyproj.Proj("+init={0}".format(MODEL_EPSG))
    ix,iy = pyproj.transform(proj1, proj2, X, Y)

    #-- construct search tree from original points
    #-- can use either BallTree or KDTree algorithms
    xy1 = np.concatenate((xg[ii,jj,None],yg[ii,jj,None]),axis=1)
    tree = BallTree(xy1) if (SEARCH == 'BallTree') else KDTree(xy1)

    #-- output interpolated arrays of firn variable (height or firn air content)
    extrap_firn = np.full_like(tdec,fv,dtype=np.float)
    #-- type designating algorithm used (1: interpolate, 2: backward, 3:forward)
    extrap_type = np.zeros_like(tdec,dtype=np.uint8)

    #-- find days that can be interpolated
    if np.any((tdec >= fd['time'].min()) & (tdec < fd['time'].max())):
        #-- indices of dates for interpolated days
        ind,=np.nonzero((tdec >= fd['time'].min()) & (tdec < fd['time'].max()))
        f = scipy.interpolate.interp1d(fd['time'], np.arange(nt), kind='linear')
        date_indice = f(tdec[ind]).astype(np.int)
        #-- set interpolation type (1: interpolated in time)
        extrap_type[ind] = 1
        #-- for each unique firn date
        #-- linearly interpolate in time between two firn maps
        #-- then then inverse distance weighting to extrapolate in space
        for k in np.unique(date_indice):
            kk, = np.nonzero(date_indice==k)
            count = np.count_nonzero(date_indice==k)
            #-- query the search tree to find the N closest points
            xy2 = np.concatenate((X[kk,None],Y[kk,None]),axis=1)
            dist,indices = tree.query(xy2, k=N, return_distance=True)
            #-- normalized weights if POWER > 0 (typically between 1 and 3)
            #-- in the inverse distance weighting
            power_inverse_distance = dist**(-POWER)
            s = np.sum(power_inverse_distance, axis=1)
            w = power_inverse_distance/np.broadcast_to(s[:,None],(count,N))
            #-- firn height or air content for times before and after tdec
            firn1 = gs[VARIABLE][k,ii,jj]
            firn2 = gs[VARIABLE][k+1,ii,jj]
            #-- linearly interpolate to date
            dt = (tdec[kk] - fd['time'][k])/(fd['time'][k+1] - fd['time'][k])
            #-- spatially extrapolate using inverse distance weighting
            extrap_firn[kk] = (1.0-dt)*np.sum(w*firn1[indices],axis=1) + \
                dt*np.sum(w*firn2[indices], axis=1)

    #-- check if needing to extrapolate backwards in time
    count = np.count_nonzero(tdec < fd['time'].min())
    if (count > 0):
        #-- indices of dates before firn model
        ind, = np.nonzero(tdec < fd['time'].min())
        #-- set interpolation type (2: extrapolated backwards in time)
        extrap_type[ind] = 2
        #-- query the search tree to find the N closest points
        xy2 = np.concatenate((X[ind,None],Y[ind,None]),axis=1)
        dist,indices = tree.query(xy2, k=N, return_distance=True)
        #-- normalized weights if POWER > 0 (typically between 1 and 3)
        #-- in the inverse distance weighting
        power_inverse_distance = dist**(-POWER)
        s = np.sum(power_inverse_distance, axis=1)
        w = power_inverse_distance/np.broadcast_to(s[:,None],(count,N))
        #-- calculate a regression model for calculating values
        #-- read first 10 years of data to create regression model
        N = np.int(10.0/time_step)
        #-- spatially interpolate firn elevation or air content to coordinates
        FIRN = np.zeros((count,N))
        T = np.zeros((N))
        #-- create interpolated time series for calculating regression model
        for k in range(N):
            #-- time at k
            T[k] = fd['time'][k]
            #-- spatially extrapolate firn elevation or air content
            firn1 = gs[VARIABLE][k,ii,jj]
            FIRN[:,k] = np.sum(w*firn1[indices],axis=1)

        #-- calculate regression model
        for n,v in enumerate(ind):
            extrap_firn[v] = regress_model(T, FIRN[n,:], tdec[v], ORDER=2,
                CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[0])

    #-- check if needing to extrapolate forward in time
    count = np.count_nonzero(tdec >= fd['time'].max())
    if (count > 0):
        #-- indices of dates after firn model
        ind, = np.nonzero(tdec >= fd['time'].max())
        #-- set interpolation type (3: extrapolated forward in time)
        extrap_type[ind] = 3
        #-- query the search tree to find the N closest points
        xy2 = np.concatenate((X[ind,None],Y[ind,None]),axis=1)
        dist,indices = tree.query(xy2, k=N, return_distance=True)
        #-- normalized weights if POWER > 0 (typically between 1 and 3)
        #-- in the inverse distance weighting
        power_inverse_distance = dist**(-POWER)
        s = np.sum(power_inverse_distance, axis=1)
        w = power_inverse_distance/np.broadcast_to(s[:,None],(count,N))
        #-- calculate a regression model for calculating values
        #-- read last 10 years of data to create regression model
        N = np.int(10.0/time_step)
        #-- spatially interpolate firn elevation or air content to coordinates
        FIRN = np.zeros((count,N))
        T = np.zeros((N))
        #-- create interpolated time series for calculating regression model
        for k in range(N):
            kk = nt - N + k
            #-- time at k
            T[k] = fd['time'][kk]
            #-- spatially extrapolate firn elevation or air content
            firn1 = gs[VARIABLE][kk,ii,jj]
            FIRN[:,k] = np.sum(w*firn1[indices],axis=1)

        #-- calculate regression model
        for n,v in enumerate(ind):
            extrap_firn[v] = regress_model(T, FIRN[n,:], tdec[v], ORDER=2,
                CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[-1])

    #-- replace fill value if specified
    if FILL_VALUE:
        ind, = np.nonzero(extrap_firn == fv)
        extrap_firn[ind] = FILL_VALUE
        fv = FILL_VALUE

    #-- return the interpolated values
    return (extrap_firn,extrap_type,fv)
