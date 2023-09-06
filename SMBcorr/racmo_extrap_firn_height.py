#!/usr/bin/env python
u"""
racmo_extrap_firn_height.py
Written by Tyler Sutterley (02/2023)
Spatially extrapolates RACMO firn heights

Uses fast nearest-neighbor search algorithms
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
and inverse distance weighted interpolation to extrapolate spatially

INPUTS:
    base_dir: working data directory
    EPSG: projection of input spatial coordinates
    MODEL: model outputs to interpolate
        FGRN055: 5.5km Greenland RACMO2.3p2
        FGRN11: 11km Greenland RACMO2.3p2
        XANT27: 27km Antarctic RACMO2.3p2
        ASE055: 5.5km Amundsen Sea Embayment RACMO2.3p2
        XPEN055: 5.5km Antarctic Peninsula RACMO2.3p2
    tdec: dates to interpolate in year-decimal
    X: x-coordinates to interpolate in projection EPSG
    Y: y-coordinates to interpolate in projection EPSG

OPTIONS:
    SEARCH: nearest-neighbor search algorithm (BallTree or KDTree)
    NN: number of nearest-neighbor points to use
    POWER: inverse distance weighting power
    VARIABLE: RACMO product to interpolate
        zs: firn height
        FirnAir: firn air content
    SIGMA: Standard deviation for Gaussian kernel
    FILL_VALUE: output fill_value for invalid points
    REFERENCE: calculate firn variables in reference to first field

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
    Updated 02/2023: don't recompute min and max time cutoffs for cases
    Updated 08/2022: updated docstrings to numpy documentation format
    Updated 01/2021: using conversion protocols following pyproj-2 updates
        https://pyproj4.github.io/pyproj/stable/gotchas.html
    Updated 04/2020: reduced to interpolation function.  output masked array
    Updated 10/2019: Gaussian average firn fields before interpolation
    Updated 09/2019: use scipy interpolate to find date indices
    Forked 08/2019 from racmo_interp_firn_height.py
    Updated 08/2019: convert to model coordinates (rotated pole lat/lon)
        and interpolate using N-dimensional functions
        added rotation parameters for Antarctic models (XANT27,ASE055,XPEN055)
        added option to change the fill value for invalid points
    Written 07/2019
"""
from __future__ import print_function

import sys
import os
import re
import warnings
import numpy as np
import scipy.ndimage
import scipy.interpolate
from SMBcorr.regress_model import regress_model

# attempt imports
try:
    import netCDF4
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("netCDF4 not available", ImportWarning)
try:
    import pyproj
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("pyproj not available", ImportWarning)
try:
    from sklearn.neighbors import KDTree, BallTree
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("scikit-learn not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: read and interpolate RACMO2.3 firn corrections
def extrapolate_racmo_firn(base_dir, EPSG, MODEL, tdec, X, Y,
    VARIABLE='zs', SEARCH='BallTree', NN=10, POWER=2.0, SIGMA=1.5,
    FILL_VALUE=None, REFERENCE=False):
    """
    Spatially extrapolates RACMO firn heights

    Parameters
    ----------
    base_dir: str
        Working data directory
    EPSG: str or int
        input coordinate reference system
    MODEL: str
        RACMO firn model

            - ``FGRN055``: 5.5km Greenland RACMO2.3p2
            - ``FGRN11``: 11km Greenland RACMO2.3p2
            - ``XANT27``: 27km Antarctic RACMO2.3p2
            - ``ASE055``: 5.5km Amundsen Sea Embayment RACMO2.3p2
            - ``XPEN055``: 5.5km Antarctic Peninsula RACMO2.3p2
    tdec: float
        time coordinates to interpolate in year-decimal
    X: float
        x-coordinates to interpolate
    Y: float
        y-coordinates to interpolate
    VARIABLE: str, default 'zs'
        RACMO product to interpolate

            - ``zs``: Firn height
            - ``FirnAir``: Firn air content
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
    REFERENCE: bool, default False
        Calculate firn variables in reference to first field
    """

    # set parameters based on input model
    FIRN_FILE = {}
    if (MODEL == 'FGRN11'):
        # filename and directory for input FGRN11 file
        FIRN_FILE['zs'] = 'FDM_zs_FGRN11_1960-2016.nc'
        FIRN_FILE['FirnAir'] = 'FDM_FirnAir_FGRN11_1960-2016.nc'
        FIRN_DIRECTORY = ['RACMO','FGRN11_1960-2016']
    elif (MODEL == 'FGRN055'):
        # filename and directory for input FGRN055 file
        FIRN_FILE['zs'] = 'FDM_zs_FGRN055_1960-2017_interpol.nc'
        FIRN_FILE['FirnAir'] = 'FDM_FirnAir_FGRN055_1960-2017_interpol.nc'
        FIRN_DIRECTORY = ['RACMO','FGRN055_1960-2017']
    elif (MODEL == 'XANT27'):
        # filename and directory for input XANT27 file
        FIRN_FILE['zs'] = 'FDM_zs_ANT27_1979-2016.nc'
        FIRN_FILE['FirnAir'] = 'FDM_FirnAir_ANT27_1979-2016.nc'
        FIRN_DIRECTORY = ['RACMO','XANT27_1979-2016']
    elif (MODEL == 'ASE055'):
        # filename and directory for input ASE055 file
        FIRN_FILE['zs'] = 'FDM_zs_ASE055_1979-2015.nc'
        FIRN_FILE['FirnAir'] = 'FDM_FirnAir_ASE055_1979-2015.nc'
        FIRN_DIRECTORY = ['RACMO','ASE055_1979-2015']
    elif (MODEL == 'XPEN055'):
        # filename and directory for input XPEN055 file
        FIRN_FILE['zs'] = 'FDM_zs_XPEN055_1979-2016.nc'
        FIRN_FILE['FirnAir'] = 'FDM_FirnAir_XPEN055_1979-2016.nc'
        FIRN_DIRECTORY = ['RACMO','XPEN055_1979-2016']

    # Open the RACMO NetCDF file for reading
    ddir = os.path.join(base_dir,*FIRN_DIRECTORY)
    fileID = netCDF4.Dataset(os.path.join(ddir,FIRN_FILE[VARIABLE]), 'r')
    # Get data from each netCDF variable and remove singleton dimensions
    fd = {}
    fd[VARIABLE] = np.squeeze(fileID.variables[VARIABLE][:].copy())
    fd['lon'] = fileID.variables['lon'][:,:].copy()
    fd['lat'] = fileID.variables['lat'][:,:].copy()
    fd['time'] = fileID.variables['time'][:].copy()
    # invalid data value
    fv = np.float64(fileID.variables[VARIABLE]._FillValue)
    # input shape of RACMO firn data
    nt,ny,nx = np.shape(fd[VARIABLE])
    # close the NetCDF files
    fileID.close()

    # indices of specified ice mask
    i,j = np.nonzero(fd[VARIABLE][0,:,:] != fv)

    # use a gaussian filter to smooth mask
    gs = {}
    gs['mask'] = scipy.ndimage.gaussian_filter(fd['mask'], SIGMA,
        mode='constant', cval=0)
    # indices of smoothed ice mask
    ii,jj = np.nonzero(np.ceil(gs['mask']) == 1.0)
    # use a gaussian filter to smooth each firn field
    gs[VARIABLE] = np.ma.zeros((nt,ny,nx), fill_value=fv)
    gs[VARIABLE].mask = np.ma.zeros((nt,ny,nx), dtype=bool)
    for t in range(nt):
        # replace fill values before smoothing data
        temp1 = np.zeros((ny,nx))
        # reference to first firn field
        if REFERENCE:
            temp1[i,j] = fd[VARIABLE][t,i,j] - fd[VARIABLE][0,i,j]
        else:
            temp1[i,j] = fd[VARIABLE][t,i,j].copy()
        # smooth firn field
        temp2 = scipy.ndimage.gaussian_filter(temp1, SIGMA,
            mode='constant', cval=0)
        # scale output smoothed firn field
        gs[VARIABLE][t,ii,jj] = temp2[ii,jj]/gs['mask'][ii,jj]
        # replace valid firn values with original
        gs[VARIABLE][t,i,j] = temp1[i,j]
        # set mask variables for time
        gs[VARIABLE].mask[t,:,:] = (gs['mask'] == 0.0)

    # convert RACMO latitude and longitude to input coordinates (EPSG)
    crs1 = pyproj.CRS.from_string(EPSG)
    crs2 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    direction = pyproj.enums.TransformDirection.INVERSE
    # convert projection from model coordinates
    xg,yg = transformer.transform(fd['lon'], fd['lat'], direction=direction)

    # construct search tree from original points
    # can use either BallTree or KDTree algorithms
    xy1 = np.concatenate((xg[ii,jj,None],yg[ii,jj,None]),axis=1)
    tree = BallTree(xy1) if (SEARCH == 'BallTree') else KDTree(xy1)

    # output interpolated arrays of firn variable (height or firn air content)
    npts = len(tdec)
    extrap_data = np.ma.zeros((npts),fill_value=fv,dtype=np.float64)
    extrap_data.data[:] = extrap_data.fill_value
    extrap_data.mask = np.zeros((npts),dtype=bool)
    # type designating algorithm used (1:interpolate, 2:backward, 3:forward)
    extrap_data.interpolation = np.zeros((npts),dtype=np.uint8)

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
            # query the search tree to find the NN closest points
            xy2 = np.concatenate((xind[kk,None],yind[kk,None]),axis=1)
            dist,indices = tree.query(xy2, k=NN, return_distance=True)
            # normalized weights if POWER > 0 (typically between 1 and 3)
            # in the inverse distance weighting
            power_inverse_distance = dist**(-POWER)
            s = np.sum(power_inverse_distance, axis=1)
            w = power_inverse_distance/np.broadcast_to(s[:,None],(count,NN))
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
    if (count > 0):
        # indices of dates before firn model
        ind, = np.nonzero(tdec < time_cutoff[0])
        # query the search tree to find the NN closest points
        xy2 = np.concatenate((X[ind,None],Y[ind,None]),axis=1)
        dist,indices = tree.query(xy2, k=NN, return_distance=True)
        # normalized weights if POWER > 0 (typically between 1 and 3)
        # in the inverse distance weighting
        power_inverse_distance = dist**(-POWER)
        s = np.sum(power_inverse_distance, axis=1)
        w = power_inverse_distance/np.broadcast_to(s[:,None],(count,NN))
        # calculate a regression model for calculating values
        # read first 10 years of data to create regression model
        N = 365
        # spatially interpolate firn elevation or air content to coordinates
        FIRN = np.zeros((count,N))
        T = np.zeros((N))
        # create interpolated time series for calculating regression model
        for k in range(N):
            # time at k
            T[k] = gs['time'][k]
            # spatially extrapolate firn elevation or air content
            firn1 = fd[VARIABLE][k,ii,jj]
            FIRN[:,k] = np.sum(w*firn1[indices],axis=1)
        # calculate regression model
        for n,v in enumerate(ind):
            extrap_data[v] = regress_model(T, FIRN[n,:], tdec[v], ORDER=2,
                CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[0])
        # set interpolation type (2: extrapolated backwards in time)
        extrap_data.interpolation[ind] = 2

    # check if needing to extrapolate forward in time
    count = np.count_nonzero(tdec >= time_cutoff[1])
    if (count > 0):
        # indices of dates after firn model
        ind, = np.nonzero(tdec >= time_cutoff[1])
        # query the search tree to find the NN closest points
        xy2 = np.concatenate((X[ind,None],Y[ind,None]),axis=1)
        dist,indices = tree.query(xy2, k=NN, return_distance=True)
        # normalized weights if POWER > 0 (typically between 1 and 3)
        # in the inverse distance weighting
        power_inverse_distance = dist**(-POWER)
        s = np.sum(power_inverse_distance, axis=1)
        w = power_inverse_distance/np.broadcast_to(s[:,None],(count,NN))
        # calculate a regression model for calculating values
        # read last 10 years of data to create regression model
        N = 365
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
        # set interpolation type (3: extrapolated forward in time)
        extrap_data.interpolation[ind] = 3

    # complete mask if any invalid in data
    invalid, = np.nonzero(extrap_data.data == extrap_data.fill_value)
    extrap_data.mask[invalid] = True
    # replace fill value if specified
    if FILL_VALUE:
        extrap_data.fill_value = FILL_VALUE
        extrap_data.data[extrap_data.mask] = extrap_data.fill_value

    # return the interpolated values
    return extrap_data
