#!/usr/bin/env python
u"""
merra_hybrid_extrap.py
Written by Tyler Sutterley (09/2024)
Interpolates and extrapolates MERRA-2 hybrid variables to times and coordinates

MERRA-2 Hybrid firn model outputs provided by Brooke Medley at GSFC

CALLING SEQUENCE:
    interp_data = extrapolate_merra_hybrid(base_dir, EPSG, REGION, tdec, X, Y,
        VERSION='v1', VARIABLE='FAC', SIGMA=1.5, SEARCH='BallTree')

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
    SEARCH: nearest-neighbor search algorithm (BallTree or KDTree)
    NN: number of nearest-neighbor points to use
    POWER: inverse distance weighting power
    FILL_VALUE: output fill_value for invalid points
    EXTRAPOLATE: create a regression model to extrapolate out in time
    GZIP: netCDF4 file is locally gzip compressed

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

PROGRAM DEPENDENCIES:
    regress_model.py: models a time series using least-squares regression

UPDATE HISTORY:
    Updated 09/2024: use wrapper to importlib for optional dependencies
        return masked array if date is outside of model range
    Updated 02/2023: don't recompute min and max time cutoffs for cases
    Updated 08/2022: updated docstrings to numpy documentation format
    Updated 05/2021: set bounds error to false when reducing temporal range
    Updated 04/2021: can reduce input dataset to a temporal subset
    Updated 02/2021: added new MERRA2-hybrid v1.1 variables
        added gzip compression option
    Updated 01/2021: using conversion protocols following pyproj-2 updates
        https://pyproj4.github.io/pyproj/stable/gotchas.html
    Updated 06/2020: updated for version 1 of MERRA-2 Hybrid
    Updated 05/2020: reduced to interpolation function.  output masked array
    Written 10/2019
"""
from __future__ import print_function

import sys
import os
import re
import gzip
import uuid
import numpy as np
import scipy.ndimage
import scipy.interpolate
from SMBcorr.regress_model import regress_model
import SMBcorr.spatial
import SMBcorr.utilities

# attempt imports
netCDF4 = SMBcorr.utilities.import_dependency('netCDF4')
pyproj = SMBcorr.utilities.import_dependency('pyproj')

# PURPOSE: set the projection parameters based on the region name
def set_projection(region):
    """
    Set the coordinate reference system string based on the
    MERRA-2 Hybrid region name

    Parameters
    ----------
    region: str
        Region string

            - ``ais``: Antarctica
            - ``gris``: Greenland
    """
    if (region == 'ais'):
        projection_flag = 'EPSG:3031'
    elif (region == 'gris'):
        projection_flag = 'EPSG:3413'
    return projection_flag

# PURPOSE: read and interpolate MERRA-2 hybrid firn corrections
def extrapolate_merra_hybrid(base_dir, EPSG, REGION, tdec, X, Y,
    VERSION='v1', VARIABLE='FAC', SEARCH='BallTree', N=10, POWER=2.0,
    SIGMA=1.5, FILL_VALUE=None, EXTRAPOLATE=False, GZIP=False):
    """
    Spatially extrapolates MERRA-2 hybrid variables

    Parameters
    ----------
    base_dir: str
        Working data directory
    EPSG: str or int
        input coordinate reference system
    REGION: str
        MERRA-2 region to interpolate

            - ``ais``: Antarctica
            - ``gris``: Greenland
    tdec: float
        time coordinates to interpolate in year-decimal
    X: float
        x-coordinates to interpolate
    Y: float
        y-coordinates to interpolate
    VERSION: str, default 'v1'
        MERRA-2 hybrid model version
    VARIABLE: str, default 'FAC'
        MERRA-2 hybrid product to interpolate

        - ``FAC``: firn air content
        - ``p_minus_e``: precipitation minus evaporation
        - ``melt``: snowmelt
    SEARCH: str, default 'BallTree'
        nearest-neighbor search algorithm
    NN: int, default 10
        number of nearest-neighbor points to use
    POWER: int or float, default 2.0
        Inverse distance weighting power
    SIGMA: float, default 1.5
        Standard deviation for Gaussian kernel
    FILL_VALUE: float or NoneType, default None
        Output fill_value for invalid points

        Default will use fill values from data file
    EXTRAPOLATE: bool, default False
        Create a regression model to extrapolate in time
    GZIP: bool, default False
        netCDF4 file is gzip compressed
    """

    # suffix if compressed
    suffix = '.gz' if GZIP else ''
    # set the input netCDF4 file for the variable of interest
    if VARIABLE in ('FAC') and (VERSION == 'v0'):
        args = ('FAC',REGION.lower(),suffix)
        hybrid_file = 'gsfc_{0}_{1}.nc{2}'.format(*args)
    if VARIABLE in ('p_minus_e','melt') and (VERSION == 'v0'):
        args = (VARIABLE,REGION.lower(),suffix)
        hybrid_file = 'm2_hybrid_{0}_cumul_{1}.nc{2}'.format(*args)
    elif VARIABLE in ('FAC','cum_smb_anomaly','SMB_a','height','h_a'):
        args = (VERSION,REGION.lower(),suffix)
        hybrid_file = 'gsfc_fdm_{0}_{1}.nc{2}'.format(*args)
    elif VARIABLE in ('smb','SMB','Me','Ra','Ru','Sn-Ev'):
        args = (VERSION,REGION.lower(),suffix)
        hybrid_file = 'gsfc_fdm_smb_{0}_{1}.nc{2}'.format(*args)
    elif VARIABLE in ('Me_a','Ra_a','Ru_a','Sn-Ev_a'):
        args = (VERSION,REGION.lower(),suffix)
        hybrid_file = 'gsfc_fdm_smb_cumul_{0}_{1}.nc{2}'.format(*args)

    # Open the MERRA-2 Hybrid NetCDF file for reading
    if GZIP:
        # read as in-memory (diskless) netCDF4 dataset
        with gzip.open(os.path.join(base_dir,hybrid_file),'r') as f:
            fileID = netCDF4.Dataset(uuid.uuid4().hex, memory=f.read())
    else:
        # read netCDF4 dataset
        fileID = netCDF4.Dataset(os.path.join(base_dir,hybrid_file), 'r')

    # invalid data value
    fv = np.float64(fileID.variables[VARIABLE]._FillValue)
    # output interpolated arrays of variable
    npts = len(tdec)
    extrap_data = np.ma.zeros((npts),fill_value=fv,dtype=np.float64)
    extrap_data.mask = np.ones((npts),dtype=bool)
    # type designating algorithm used (1:interpolate, 2:backward, 3:forward)
    extrap_data.interpolation = np.zeros((npts),dtype=np.uint8)

    # Get data from each netCDF variable and remove singleton dimensions
    fd = {}
    # time is year decimal at time step 5 days
    time_step = 5.0/365.25
    # data at first time step for calculating anomalies
    z0 = fileID.variables[VARIABLE][0,:,:].copy()
    # temporary variable for reading time
    tmod = fileID.variables['time'][:].copy()
    # if extrapolating data: read the full dataset
    # if simply interpolating with fill values: reduce to a subset
    if EXTRAPOLATE:
        # copy time variables
        fd['time'] = tmod.copy()
        # read full dataset and remove singleton dimensions
        fd[VARIABLE] = np.squeeze(fileID.variables[VARIABLE][:].copy())
    elif ((np.max(tdec) + 2.0*time_step) < np.max(tmod) and
        ((np.min(tdec) - 2.0*time_step) > np.min(tmod))):
        # reduce grids to time period of input buffered by time steps
        tmin = np.min(tdec) - 2.0*time_step
        tmax = np.max(tdec) + 2.0*time_step
        # find indices to times
        nt, = fileID.variables['time'].shape
        f = scipy.interpolate.interp1d(tmod,
            np.arange(nt), kind='nearest', bounds_error=False,
            fill_value=(0,nt))
        imin,imax = f((tmin,tmax)).astype(np.int64)
        # reduce time variables
        fd['time'] = tmod[imin:imax+1].copy()
        # read reduced dataset and remove singleton dimensions
        fd[VARIABLE] = np.squeeze(fileID.variables[VARIABLE][imin:imax+1,:,:])
    else:
        # return as invalid
        extrap_data.data[extrap_data.mask] = extrap_data.fill_value
        return extrap_data
    # input shape of MERRA-2 Hybrid firn data
    nt,nx,ny = np.shape(fd[VARIABLE])
    # extract x and y coordinate arrays from grids if applicable
    # else create meshgrids of coordinate arrays
    if (np.ndim(fileID.variables['x'][:]) == 2):
        xg = fileID.variables['x'][:].copy()
        yg = fileID.variables['y'][:].copy()
        fd['x'],fd['y'] = (xg[:,0],yg[0,:])
    else:
        fd['x'] = fileID.variables['x'][:].copy()
        fd['y'] = fileID.variables['y'][:].copy()
        xg,yg = np.meshgrid(fd['x'],fd['y'],indexing='ij')
    # close the NetCDF files
    fileID.close()

    # indices of specified ice mask
    i,j = np.nonzero(fd[VARIABLE][0,:,:] != fv)
    # create mask object for interpolating data
    fd['mask'] = np.zeros((nx,ny))
    fd['mask'][i,j] = 1.0

    # use a gaussian filter to smooth mask
    gs = {}
    gs['mask'] = scipy.ndimage.gaussian_filter(fd['mask'], SIGMA,
        mode='constant', cval=0)
    # indices of smoothed ice mask
    ii,jj = np.nonzero(np.ceil(gs['mask']) == 1.0)
    # use a gaussian filter to smooth each firn field
    gs[VARIABLE] = np.ma.zeros((nt,nx,ny), fill_value=fv)
    gs[VARIABLE].mask = np.zeros((nt,nx,ny), dtype=bool)
    for t in range(nt):
        # replace fill values before smoothing data
        temp1 = np.zeros((nx,ny))
        # reference to first firn field (z0)
        temp1[i,j] = fd[VARIABLE][t,i,j] - z0[i,j]
        # smooth firn field
        temp2 = scipy.ndimage.gaussian_filter(temp1, SIGMA,
            mode='constant', cval=0)
        # scale output smoothed firn field
        gs[VARIABLE].data[t,ii,jj] = temp2[ii,jj]/gs['mask'][ii,jj]
        # replace valid firn values with original
        gs[VARIABLE].data[t,i,j] = temp1[i,j]
        # set mask variables for time
        gs[VARIABLE].mask[t,:,:] = (gs['mask'] == 0.0)

    # pyproj transformer for converting to input coordinates (EPSG)
    MODEL_EPSG = set_projection(REGION)
    crs1 = pyproj.CRS.from_string(EPSG)
    crs2 = pyproj.CRS.from_string(MODEL_EPSG)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    direction = pyproj.enums.TransformDirection.INVERSE
    # convert projection from model coordinates
    xg,yg = transformer.transform(fd['x'], fd['y'], direction=direction)

    # construct search tree from original points
    # can use either BallTree or KDTree algorithms
    xy1 = np.concatenate((xg[ii,jj,None],yg[ii,jj,None]),axis=1)
    tree = SMBcorr.spatial.build_tree(xy1, SEARCH=SEARCH)

    # time cutoff without close time interpolation
    time_cutoff = (fd['time'].min(), fd['time'].max())
    # find days that can be interpolated
    if np.any((tdec >= time_cutoff[0]) & (tdec < time_cutoff[1])):
        # indices of dates for interpolated days
        ind,=np.nonzero((tdec >= time_cutoff[0]) & (tdec < time_cutoff[1]))
        # reduce x, y and t coordinates
        xind,yind,tind = (X[ind],Y[ind],tdec[ind])
        # find indices for linearly interpolating in time
        f = scipy.interpolate.interp1d(fd['time'], np.arange(nt), kind='linear')
        date_indice = f(tind).astype(np.int64)
        # for each unique firn date
        # linearly interpolate in time between two firn maps
        # then then inverse distance weighting to extrapolate in space
        for k in np.unique(date_indice):
            kk, = np.nonzero(date_indice==k)
            count = np.count_nonzero(date_indice==k)
            # query the search tree to find the N closest points
            xy2 = np.concatenate((xind[kk,None],yind[kk,None]),axis=1)
            dist,indices = tree.query(xy2, k=N, return_distance=True)
            # normalized weights if POWER > 0 (typically between 1 and 3)
            # in the inverse distance weighting
            power_inverse_distance = dist**(-POWER)
            s = np.sum(power_inverse_distance, axis=1)
            w = power_inverse_distance/np.broadcast_to(s[:,None],(count,N))
            # firn height or air content for times before and after tdec
            firn1 = gs[VARIABLE][k,ii,jj]
            firn2 = gs[VARIABLE][k+1,ii,jj]
            # linearly interpolate to date
            dt = (tind[kk] - fd['time'][k])/(fd['time'][k+1] - fd['time'][k])
            # spatially extrapolate using inverse distance weighting
            extrap_data[kk] = (1.0-dt)*np.sum(w*firn1[indices],axis=1) + \
                dt*np.sum(w*firn2[indices], axis=1)
        # set interpolation type (1: interpolated in time)
        extrap_data.interpolation[ind] = 1

    # check if needing to extrapolate backwards in time
    count = np.count_nonzero(tdec < time_cutoff[0])
    if (count > 0) and EXTRAPOLATE:
        # indices of dates before firn model
        ind, = np.nonzero(tdec < time_cutoff[0])
        # query the search tree to find the N closest points
        xy2 = np.concatenate((X[ind,None],Y[ind,None]),axis=1)
        dist,indices = tree.query(xy2, k=N, return_distance=True)
        # normalized weights if POWER > 0 (typically between 1 and 3)
        # in the inverse distance weighting
        power_inverse_distance = dist**(-POWER)
        s = np.sum(power_inverse_distance, axis=1)
        w = power_inverse_distance/np.broadcast_to(s[:,None],(count,N))
        # calculate a regression model for calculating values
        # read first 10 years of data to create regression model
        N = np.int64(10.0/time_step)
        # spatially interpolate firn elevation or air content to coordinates
        FIRN = np.zeros((count,N))
        T = np.zeros((N))
        # create interpolated time series for calculating regression model
        for k in range(N):
            # time at k
            T[k] = fd['time'][k]
            # spatially extrapolate firn elevation or air content
            firn1 = gs[VARIABLE][k,ii,jj]
            FIRN[:,k] = np.sum(w*firn1[indices],axis=1)
        # calculate regression model
        for n,v in enumerate(ind):
            extrap_data[v] = regress_model(T, FIRN[n,:], tdec[v], ORDER=2,
                CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[0])
        # set interpolation type (2: extrapolated backwards in time)
        extrap_data.interpolation[ind] = 2

    # check if needing to extrapolate forward in time
    count = np.count_nonzero(tdec >= time_cutoff[1])
    if (count > 0) and EXTRAPOLATE:
        # indices of dates after firn model
        ind, = np.nonzero(tdec >= time_cutoff[1])
        # query the search tree to find the N closest points
        xy2 = np.concatenate((X[ind,None],Y[ind,None]),axis=1)
        dist,indices = tree.query(xy2, k=N, return_distance=True)
        # normalized weights if POWER > 0 (typically between 1 and 3)
        # in the inverse distance weighting
        power_inverse_distance = dist**(-POWER)
        s = np.sum(power_inverse_distance, axis=1)
        w = power_inverse_distance/np.broadcast_to(s[:,None],(count,N))
        # calculate a regression model for calculating values
        # read last 10 years of data to create regression model
        N = np.int64(10.0/time_step)
        # spatially interpolate firn elevation or air content to coordinates
        FIRN = np.zeros((count,N))
        T = np.zeros((N))
        # create interpolated time series for calculating regression model
        for k in range(N):
            kk = nt - N + k
            # time at k
            T[k] = fd['time'][kk]
            # spatially extrapolate firn elevation or air content
            firn1 = gs[VARIABLE][kk,ii,jj]
            FIRN[:,k] = np.sum(w*firn1[indices],axis=1)
        # calculate regression model
        for n,v in enumerate(ind):
            extrap_data[v] = regress_model(T, FIRN[n,:], tdec[v], ORDER=2,
                CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[-1])
        # set interpolation type (3: extrapolated forwards in time)
        extrap_data.interpolation[ind] = 3

    # complete mask if any invalid in data
    invalid, = np.nonzero((extrap_data.data == extrap_data.fill_value) |
        np.isnan(extrap_data.data))
    extrap_data.mask[invalid] = True
    # replace fill value if specified
    if FILL_VALUE:
        extrap_data.fill_value = FILL_VALUE
        extrap_data.data[extrap_data.mask] = extrap_data.fill_value

    # return the interpolated values
    return extrap_data
