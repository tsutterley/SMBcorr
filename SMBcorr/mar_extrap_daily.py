#!/usr/bin/env python
u"""
mar_extrap_daily.py
Written by Tyler Sutterley (02/2023)
Interpolates and extrapolates daily MAR products to times and coordinates

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

PROGRAM DEPENDENCIES:
    regress_model.py: models a time series using least-squares regression
    time.py: utilities for calculating time operations

UPDATE HISTORY:
    Updated 02/2023: don't recompute min and max time cutoffs for cases
    Updated 08/2022: updated docstrings to numpy documentation format
    Updated 01/2021: using conversion protocols following pyproj-2 updates
        https://pyproj4.github.io/pyproj/stable/gotchas.html
        using utilities from time module for conversions
    Updated 06/2020: set all values initially to fill_value
    Updated 05/2020: Gaussian average fields before interpolation
        accumulate variable over all available dates. add coordinate options
    Written 04/2020
"""
from __future__ import print_function

import sys
import os
import re
import warnings
import numpy as np
import scipy.spatial
import scipy.ndimage
import scipy.interpolate
from SMBcorr.regress_model import regress_model
import SMBcorr.time

# attempt imports
try:
    import netCDF4
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("netCDF4 not available", ImportWarning)
try:
    import pyproj
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("pyproj not available", ImportWarning)
try:
    from sklearn.neighbors import KDTree, BallTree
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("scikit-learn not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: read and interpolate daily MAR outputs
def extrapolate_mar_daily(DIRECTORY, EPSG, VERSION, tdec, X, Y,
    VARIABLE='SMB', XNAME=None, YNAME=None, TIMENAME='TIME',
    SEARCH='BallTree', NN=10, POWER=2.0, SIGMA=1.5,
    FILL_VALUE=None, EXTRAPOLATE=False):
    """
    Spatially extrapolates daily MAR surface mass balance products

    Parameters
    ----------
    DIRECTORY: str
        Working data directory
    EPSG: str or int
        input coordinate reference system
    VERSION: str
        MAR Version

            - ``v3.5.2``
            - ``v3.9``
            - ``v3.10``
            - ``v3.11``
    tdec: float
        time coordinates to interpolate in year-decimal
    X: float
        x-coordinates to interpolate
    Y: float
        y-coordinates to interpolate
    VARIABLE: str, default 'SMB'
        MAR product to interpolate

            - ``SMB``: Surface Mass Balance
            - ``PRECIP``: Precipitation
            - ``SNOWFALL``: Snowfall
            - ``RAINFALL``: Rainfall
            - ``RUNOFF``: Melt Water Runoff
            - ``SNOWMELT``: Snowmelt
            - ``REFREEZE``: Melt Water Refreeze
            - ``SUBLIM``: Sublimation

    XNAME: str or NoneType, default None
        Name of the x-coordinate variable
    YNAME: str or NoneType, default None
        Name of the y-coordinate variable
    TIMENAME: str or NoneType, default 'TIME'
        Name of the time variable
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
    """

    # start and end years to read
    SY = np.nanmin(np.floor(tdec)).astype(np.int64)
    EY = np.nanmax(np.floor(tdec)).astype(np.int64)
    YRS = '|'.join(['{0:4d}'.format(Y) for Y in range(SY,EY+1)])
    # regular expression pattern for MAR dataset
    rx = re.compile(r'{0}-(.*?)-(\d+)(_subset)?.nc$'.format(VERSION,YRS))

    # create list of files to read
    input_files=sorted([f for f in os.listdir(DIRECTORY) if rx.match(f)])

    # calculate number of time steps to read
    nt = 0
    for f,FILE in enumerate(input_files):
        # Open the MAR NetCDF file for reading
        with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
            nx = len(fileID.variables[XNAME][:])
            ny = len(fileID.variables[YNAME][:])
            TIME = fileID.variables[TIMENAME][:]
            try:
                nt += np.count_nonzero(TIME.data != TIME.fill_value)
            except AttributeError:
                nt += len(TIME)

    # python dictionary with file variables
    fd = {}
    fd['TIME'] = np.zeros((nt))
    # python dictionary with gaussian filtered variables
    gs = {}
    # calculate cumulative sum of gaussian filtered values
    cumulative = np.zeros((ny,nx))
    gs['CUMULATIVE'] = np.ma.zeros((nt,ny,nx), fill_value=FILL_VALUE)
    gs['CUMULATIVE'].mask = np.ones((nt,ny,nx), dtype=bool)
    # create a counter variable for filling variables
    c = 0
    # for each file in the list
    for f,FILE in enumerate(input_files):
        # Open the MAR NetCDF file for reading
        with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
            # number of time variables within file
            TIME = fileID.variables['TIME'][:]
            try:
                t = np.count_nonzero(TIME.data != TIME.fill_value)
            except AttributeError:
                t = len(TIME)
            # create a masked array with all data
            fd[VARIABLE] = np.ma.zeros((t,ny,nx),fill_value=FILL_VALUE)
            fd[VARIABLE].mask = np.zeros((t,ny,nx),dtype=bool)
            # surface type
            SRF=fileID.variables['SRF'][:]
            # indices of specified ice mask
            i,j=np.nonzero(SRF == 4)
            # ice fraction
            FRA=fileID.variables['FRA'][:]/100.0
            # Get data from netCDF variable and remove singleton dimensions
            tmp=np.squeeze(fileID.variables[VARIABLE][:])
            # combine sectors for multi-layered data
            if (np.ndim(tmp) == 4):
                # create mask for combining data
                MASK=np.zeros((t,ny,nx))
                MASK[:,i,j]=FRA[:t,0,i,j]
                # combine data
                fd[VARIABLE][:]=MASK*tmp[:t,0,:,:] + (1.0-MASK)*tmp[:t,1,:,:]
            else:
                # copy data
                fd[VARIABLE][:]=tmp[:t,:,:].copy()
            # verify mask object for interpolating data
            surf_mask = np.broadcast_to(SRF, (t,ny,nx))
            fd[VARIABLE].mask[:,:,:] |= (surf_mask != 4)
            # combine mask object through time to create a single mask
            fd[VARIABLE].mask = fd[VARIABLE].data == fd[VARIABLE].fill_value
            fd['MASK']=1.0-np.any(fd[VARIABLE].mask,axis=0).astype(np.float64)
            # MAR coordinates
            fd['LON']=fileID.variables['LON'][:,:].copy()
            fd['LAT']=fileID.variables['LAT'][:,:].copy()
            # convert x and y coordinates to meters
            fd['x']=1000.0*fileID.variables[XNAME][:].copy()
            fd['y']=1000.0*fileID.variables[YNAME][:].copy()
            # extract delta time and epoch of time
            delta_time=fileID.variables[TIMENAME][:t].astype(np.float64)
            date_string=fileID.variables[TIMENAME].units
        # extract epoch and units
        epoch,to_secs = SMBcorr.time.parse_date_string(date_string)
        # calculate time array in Julian days
        JD = SMBcorr.time.convert_delta_time(delta_time*to_secs, epoch1=epoch,
            epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0) + 2400000.5
        # convert from Julian days to calendar dates
        YY,MM,DD,hh,mm,ss = SMBcorr.time.convert_julian(JD)
        # calculate time in year-decimal
        fd['TIME'][c:c+t] = SMBcorr.time.convert_calendar_decimal(YY,MM,
            day=DD,hour=hh,minute=mm,second=ss)
        # use a gaussian filter to smooth mask
        gs['MASK'] = scipy.ndimage.gaussian_filter(fd['MASK'],SIGMA,
            mode='constant',cval=0)
        # indices of smoothed ice mask
        ii,jj = np.nonzero(np.ceil(gs['MASK']) == 1.0)
        # use a gaussian filter to smooth each model field
        gs[VARIABLE] = np.ma.zeros((t,ny,nx), fill_value=FILL_VALUE)
        gs[VARIABLE].mask = np.ones((t,ny,nx), dtype=bool)
        # for each time
        for tt in range(t):
            # replace fill values before smoothing data
            temp1 = np.zeros((ny,nx))
            i,j = np.nonzero(~fd[VARIABLE].mask[tt,:,:])
            temp1[i,j] = fd[VARIABLE][tt,i,j].copy()
            # smooth spatial field
            temp2 = scipy.ndimage.gaussian_filter(temp1, SIGMA,
                mode='constant', cval=0)
            # scale output smoothed field
            gs[VARIABLE].data[tt,ii,jj] = temp2[ii,jj]/gs['MASK'][ii,jj]
            # replace valid values with original
            gs[VARIABLE].data[tt,i,j] = temp1[i,j]
            # set mask variables for time
            gs[VARIABLE].mask[tt,ii,jj] = False
            # calculate cumulative
            cumulative[ii,jj] += gs[VARIABLE][tt,ii,jj]
            gs['CUMULATIVE'].data[c+tt,ii,jj] = np.copy(cumulative[ii,jj])
            gs['CUMULATIVE'].mask[c+tt,ii,jj] = False
        # add to counter
        c += t

    # convert MAR latitude and longitude to input coordinates (EPSG)
    crs1 = pyproj.CRS.from_string(EPSG)
    crs2 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    direction = pyproj.enums.TransformDirection.INVERSE
    # convert projection from model coordinates
    xg,yg = transformer.transform(fd['LON'], fd['LAT'], direction=direction)

    # construct search tree from original points
    # can use either BallTree or KDTree algorithms
    xy1 = np.concatenate((xg[i,j,None],yg[i,j,None]),axis=1)
    tree = BallTree(xy1) if (SEARCH == 'BallTree') else KDTree(xy1)

    # output interpolated arrays of output variable
    npts = len(tdec)
    extrap = np.ma.zeros((npts),fill_value=FILL_VALUE,dtype=np.float64)
    extrap.mask = np.ones((npts),dtype=bool)
    # initially set all values to fill value
    extrap.data[:] = extrap.fill_value
    # type designating algorithm used (1:interpolate, 2:backward, 3:forward)
    extrap.interpolation = np.zeros((npts),dtype=np.uint8)

    # time cutoff without close time interpolation
    time_cutoff = (fd['TIME'].min(), fd['TIME'].max())
    # find days that can be interpolated
    if np.any((tdec >= time_cutoff[0]) & (tdec < time_cutoff[1])):
        # indices of dates for interpolated days
        ind,=np.nonzero((tdec >= time_cutoff[0]) & (tdec < time_cutoff[1]))
        # reduce x, y and t coordinates
        xind,yind,tind = (X[ind],Y[ind],tdec[ind])
        # find indices for linearly interpolating in time
        f = scipy.interpolate.interp1d(fd['TIME'], np.arange(nt), kind='linear')
        date_indice = f(tind).astype(np.int64)
        # for each unique model date
        # linearly interpolate in time between two model maps
        # then then inverse distance weighting to extrapolate in space
        for k in np.unique(date_indice):
            kk, = np.nonzero(date_indice==k)
            count = np.count_nonzero(date_indice==k)
            # query the search tree to find the NN closest points
            xy2 = np.concatenate((xind[kk,None],yind[kk,None]),axis=1)
            dist,indices = tree.query(xy2, k=NN, return_distance=True)
            # normalized weights if POWER > 0 (typically between 1 and 3)
            # in the inverse distance weighting
            power_inverse_distance = dist**(-POWER)
            s = np.sum(power_inverse_distance, axis=1)
            w = power_inverse_distance/np.broadcast_to(s[:,None],(count,NN))
            # variable for times before and after tdec
            var1 = gs['CUMULATIVE'][k,i,j]
            var2 = gs['CUMULATIVE'][k+1,i,j]
            # linearly interpolate to date
            dt = (tind[kk] - fd['TIME'][k])/(fd['TIME'][k+1] - fd['TIME'][k])
            # spatially extrapolate using inverse distance weighting
            extrap.data[kk] = (1.0-dt)*np.sum(w*var1[indices],axis=1) + \
                dt*np.sum(w*var2[indices], axis=1)
        # set interpolation type (1: interpolated in time)
        extrap.interpolation[ind] = 1

    # check if needing to extrapolate backwards in time
    count = np.count_nonzero(tdec < time_cutoff[0])
    if (count > 0) and EXTRAPOLATE:
        # indices of dates before model
        ind, = np.nonzero(tdec < time_cutoff[0])
        # query the search tree to find the NN closest points
        xy2 = np.concatenate((X[ind,None],Y[ind,None]),axis=1)
        dist,indices = tree.query(xy2, k=NN, return_distance=True)
        # normalized weights if POWER > 0 (typically between 1 and 3)
        # in the inverse distance weighting
        power_inverse_distance = dist**(-POWER)
        s = np.sum(power_inverse_distance, axis=1)
        w = power_inverse_distance/np.broadcast_to(s[:,None],(count,NN))
        # read the first year of data to create regression model
        N = 365
        # calculate a regression model for calculating values
        # spatially interpolate variable to coordinates
        DATA = np.zeros((count,N))
        TIME = np.zeros((N))
        # create interpolated time series for calculating regression model
        for k in range(N):
            # time at k
            TIME[k] = fd['TIME'][k]
            # spatially extrapolate variable
            tmp = gs['CUMULATIVE'][k,i,j]
            DATA[:,k] = np.sum(w*tmp[indices],axis=1)
        # calculate regression model
        for n,v in enumerate(ind):
            extrap.data[v] = regress_model(TIME, DATA[n,:], tdec[v],
                ORDER=2, CYCLES=[0.25,0.5,1.0], RELATIVE=TIME[0])
        # set interpolation type (2: extrapolated backwards in time)
        extrap.interpolation[ind] = 2

    # check if needing to extrapolate forward in time
    count = np.count_nonzero(tdec >= time_cutoff[1])
    if (count > 0) and EXTRAPOLATE:
        # indices of dates after model
        ind, = np.nonzero(tdec >= time_cutoff[1])
        # query the search tree to find the NN closest points
        xy2 = np.concatenate((X[ind,None],Y[ind,None]),axis=1)
        dist,indices = tree.query(xy2, k=NN, return_distance=True)
        # normalized weights if POWER > 0 (typically between 1 and 3)
        # in the inverse distance weighting
        power_inverse_distance = dist**(-POWER)
        s = np.sum(power_inverse_distance, axis=1)
        w = power_inverse_distance/np.broadcast_to(s[:,None],(count,NN))
        # read the last year of data to create regression model
        N = 365
        # calculate a regression model for calculating values
        # spatially interpolate variable to coordinates
        DATA = np.zeros((count,N))
        TIME = np.zeros((N))
        # create interpolated time series for calculating regression model
        for k in range(N):
            kk = nt - N + k
            # time at kk
            TIME[k] = fd['TIME'][kk]
            # spatially extrapolate variable
            tmp = gs['CUMULATIVE'][kk,i,j]
            DATA[:,k] = np.sum(w*tmp[indices],axis=1)
        # calculate regression model
        for n,v in enumerate(ind):
            extrap.data[v] = regress_model(TIME, DATA[n,:], tdec[v],
                ORDER=2, CYCLES=[0.25,0.5,1.0], RELATIVE=TIME[-1])
        # set interpolation type (3: extrapolated forward in time)
        extrap.interpolation[ind] = 3

    # complete mask if any invalid in data
    invalid, = np.nonzero((extrap.data == extrap.fill_value) |
        np.isnan(extrap.data))
    extrap.mask[invalid] = True

    # return the interpolated values
    return extrap
