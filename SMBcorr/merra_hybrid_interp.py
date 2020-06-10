#!/usr/bin/env python
u"""
merra_hybrid_interp.py
Written by Tyler Sutterley (06/2020)
Interpolates and extrapolates MERRA-2 hybrid variables to times and coordinates
    MERRA-2 Hybrid firn model outputs provided by Brooke Medley at GSFC

CALLING SEQUENCE:
    interp_data = interpolate_merra_hybrid(base_dir, EPSG, REGION, tdec, X, Y,
        VERSION='v1', VARIABLE='FAC', SIGMA=1.5)

INPUTS:
    base_dir: working data directory
    EPSG: projection of input spatial coordinates
    REGION: region to interpolate (gris, ais)
    tdec: dates to interpolate in year-decimal
    X: x-coordinates to interpolate in projection EPSG
    Y: y-coordinates to interpolate in projection EPSG

OPTIONS:
    VERSION: MERRA-2 hybrid model version (v0, v1)
    VARIABLE: MERRA-2 hybrid product to interpolate
        FAC: firn air content
        p_minus_e: precipitation minus evaporation
        melt: snowmelt
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

UPDATE HISTORY:
    Updated 06/2020: updated for version 1 of MERRA-2 Hybrid
    Updated 05/2020: reduced to interpolation function.  output masked array
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
import scipy.spatial
import scipy.ndimage
import scipy.interpolate
from SMBcorr.regress_model import regress_model

#-- PURPOSE: set the projection parameters based on the region name
def set_projection(REGION):
    if (REGION == 'ais'):
        projection_flag = 'EPSG:3031'
    elif (REGION == 'gris'):
        projection_flag = 'EPSG:3413'
    return projection_flag

#-- PURPOSE: read and interpolate MERRA-2 hybrid firn corrections
def interpolate_merra_hybrid(base_dir, EPSG, REGION, tdec, X, Y,
    VERSION='v1', VARIABLE='FAC', SIGMA=1.5, FILL_VALUE=None,
    EXTRAPOLATE=False):

    #-- set the input netCDF4 file for the variable of interest
    if VARIABLE in ('FAC','cum_smb_anomaly','height'):
        hybrid_file='gsfc_fdm_{0}_{1}.nc'.format(VERSION,REGION.lower())
    if VARIABLE in ('FAC') and (VERSION == 'v0'):
        hybrid_file='gsfc_{0}_{1}.nc'.format('FAC',REGION.lower())
    elif VARIABLE in ('p_minus_e','melt') and (VERSION == 'v0'):
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
    gs[VARIABLE].mask = np.zeros((nt,nx,ny), dtype=np.bool)
    for t in range(nt):
        #-- replace fill values before smoothing data
        temp1 = np.zeros((nx,ny))
        #-- reference to first firn field
        temp1[i,j] = fd[VARIABLE][t,i,j] - fd[VARIABLE][0,i,j]
        #-- smooth firn field
        temp2 = scipy.ndimage.gaussian_filter(temp1, SIGMA,
            mode='constant', cval=0)
        #-- scale output smoothed firn field
        gs[VARIABLE].data[t,ii,jj] = temp2[ii,jj]/gs['mask'][ii,jj]
        #-- replace valid firn values with original
        gs[VARIABLE].data[t,i,j] = temp1[i,j]
        #-- set mask variables for time
        gs[VARIABLE].mask[t,:,:] = (gs['mask'] == 0.0)

    #-- convert projection from input coordinates (EPSG) to model coordinates
    #-- MERRA-2 Hybrid models are rotated pole latitude and longitude
    MODEL_EPSG = set_projection(REGION)
    proj1 = pyproj.Proj("+init={0}".format(EPSG))
    proj2 = pyproj.Proj("+init={0}".format(MODEL_EPSG))
    ix,iy = pyproj.transform(proj1, proj2, X, Y)

    #-- check that input points are within convex hull of smoothed model points
    points = np.concatenate((xg[ii,jj,None],yg[ii,jj,None]),axis=1)
    triangle = scipy.spatial.Delaunay(points.data, qhull_options='Qt Qbb Qc Qz')
    interp_points = np.concatenate((ix[:,None],iy[:,None]),axis=1)
    valid = (triangle.find_simplex(interp_points) >= 0)

    #-- output interpolated arrays of variable
    npts = len(tdec)
    interp_data = np.ma.zeros((npts),fill_value=fv)
    #-- interpolation mask of invalid values
    interp_data.mask = np.ones((npts),dtype=np.bool)
    #-- type designating algorithm used (1: interpolate, 2: backward, 3:forward)
    interp_data.interpolation = np.zeros_like(tdec,dtype=np.uint8)

    #-- find days that can be interpolated
    if np.any((tdec >= fd['time'].min()) & (tdec <= fd['time'].max()) & valid):
        #-- indices of dates for interpolated days
        ind, = np.nonzero((tdec >= fd['time'].min()) &
            (tdec <= fd['time'].max()) & valid)
        #-- create an interpolator for firn height or air content
        RGI = scipy.interpolate.RegularGridInterpolator(
            (fd['time'],fd['x'],fd['y']), gs[VARIABLE].data)
        #-- create an interpolator for input mask
        MI = scipy.interpolate.RegularGridInterpolator(
            (fd['time'],fd['x'],fd['y']), gs[VARIABLE].mask)
        #-- interpolate to points
        interp_data.data[ind] = RGI.__call__(np.c_[tdec[ind],ix[ind],iy[ind]])
        interp_data.mask[ind] = MI.__call__(np.c_[tdec[ind],ix[ind],iy[ind]])
        #-- set interpolation type (1: interpolated)
        interp_data.interpolation[ind] = 1

    #-- check if needing to extrapolate backwards in time
    count = np.count_nonzero((tdec < fd['time'].min()) & valid)
    if (count > 0) and EXTRAPOLATE:
        #-- indices of dates before firn model
        ind, = np.nonzero((tdec < fd['time'].min()) & valid)
        #-- calculate a regression model for calculating values
        #-- read first 10 years of data to create regression model
        N = np.int(10.0/time_step)
        #-- spatially interpolate variable to coordinates
        T = np.zeros((N))
        DATA = np.zeros((count,N))
        MASK = np.zeros((count,N))
        #-- create interpolated time series for calculating regression model
        for k in range(N):
            #-- time at k
            T[k] = fd['time'][k]
            #-- spatially interpolate variable and mask
            f1 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE].data[k,:,:], kx=1, ky=1)
            f2 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE].mask[k,:,:], kx=1, ky=1)
            #-- create numpy masked array of interpolated values
            DATA[:,k] = f1.ev(ix[ind],iy[ind])
            MASK[:,k] = f2.ev(ix[ind],iy[ind])
        #-- calculate regression model
        for n,v in enumerate(ind):
            interp_data.data[v] = regress_model(T, DATA[n,:], tdec[v], ORDER=2,
                CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[0])
        #-- mask any invalid points
        interp_data.mask[ind] = np.any(MASK, axis=1)
        #-- set interpolation type (2: extrapolated backward)
        interp_data.interpolation[ind] = 2

    #-- check if needing to extrapolate forward in time
    count = np.count_nonzero((tdec > fd['time'].max()) & valid)
    if (count > 0) and EXTRAPOLATE:
        #-- indices of dates after firn model
        ind, = np.nonzero((tdec > fd['time'].max()) & valid)
        #-- calculate a regression model for calculating values
        #-- read last 10 years of data to create regression model
        N = np.int(10.0/time_step)
        #-- spatially interpolate variable to coordinates
        T = np.zeros((N))
        DATA = np.zeros((count,N))
        MASK = np.zeros((count,N))
        #-- create interpolated time series for calculating regression model
        for k in range(N):
            kk = nt - N + k
            #-- time at kk
            T[k] = fd['time'][kk]
            #-- spatially interpolate firn elevation or air content
            fspl = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE][kk,:,:], kx=1, ky=1)
            #-- spatially interpolate variable and mask
            f1 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE].data[kk,:,:], kx=1, ky=1)
            f2 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE].mask[kk,:,:], kx=1, ky=1)
            #-- create numpy masked array of interpolated values
            DATA[:,k] = f1.ev(ix[ind],iy[ind])
            MASK[:,k] = f2.ev(ix[ind],iy[ind])
        #-- calculate regression model
        for n,v in enumerate(ind):
            interp_data.data[v] = regress_model(T, FIRN[n,:], tdec[v], ORDER=2,
                CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[-1])
        #-- mask any invalid points
        interp_data.mask[ind] = np.any(MASK, axis=1)
        #-- set interpolation type (3: extrapolated forward)
        interp_data.interpolation[ind] = 3

    #-- complete mask if any invalid in data
    invalid, = np.nonzero(interp_data.data == interp_data.fill_value)
    interp_data.mask[invalid] = True
    #-- replace fill value if specified
    if FILL_VALUE:
        interp_data.fill_value = FILL_VALUE
        interp_data.data[interp_data.mask] = interp_data.fill_value

    #-- return the interpolated values
    return interp_data
