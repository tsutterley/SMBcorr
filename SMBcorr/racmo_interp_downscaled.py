#!/usr/bin/env python
u"""
racmo_interp_downscaled.py
Written by Tyler Sutterley (09/2024)
Interpolates and extrapolates downscaled RACMO products to times and coordinates

INPUTS:
    base_dir: working data directory
    EPSG: projection of input spatial coordinates
    VERSION: Downscaled RACMO Version
        1.0: RACMO2.3/XGRN11
        2.0: RACMO2.3p2/XGRN11
        3.0: RACMO2.3p2/FGRN055
        4.0: RACMO2.3p2/FGRN055
    tdec: dates to interpolate in year-decimal
    X: x-coordinates to interpolate in projection EPSG
    Y: y-coordinates to interpolate in projection EPSG

OPTIONS:
    VARIABLE: RACMO product to interpolate
        SMB: Surface Mass Balance
        PRECIP: Precipitation
        RUNOFF: Melt Water Runoff
        SNOWMELT: Snowmelt
        REFREEZE: Melt Water Refreeze
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

PROGRAM DEPENDENCIES:
    regress_model.py: models a time series using least-squares regression

UPDATE HISTORY:
    Updated 09/2024: use wrapper to importlib for optional dependencies
    Updated 02/2023: close in time extrapolations with regular grid interpolator
    Updated 10/2022: added version 4.0 (RACMO2.3p2 for 1958-2022 from FGRN055)
    Updated 08/2022: updated docstrings to numpy documentation format
    Updated 11/2021: don't attempt triangulation if large number of points
    Updated 01/2021: using conversion protocols following pyproj-2 updates
        https://pyproj4.github.io/pyproj/stable/gotchas.html
    Updated 08/2020: attempt delaunay triangulation using different options
    Updated 04/2020: reduced to interpolation function.  output masked array
    Updated 09/2019: read subsets of DS1km netCDF4 file to save memory
    Written 09/2019
"""
from __future__ import print_function

import sys
import os
import re
import warnings
import numpy as np
import scipy.spatial
import scipy.interpolate
import SMBcorr.spatial
import SMBcorr.utilities
from SMBcorr.regress_model import regress_model

# attempt imports
netCDF4 = SMBcorr.utilities.import_dependency('netCDF4')
pyproj = SMBcorr.utilities.import_dependency('pyproj')

# PURPOSE: read and interpolate downscaled RACMO products
def interpolate_racmo_downscaled(base_dir, EPSG, VERSION, tdec, X, Y,
    VARIABLE='SMB', FILL_VALUE=None):
    """
    Reads and interpolates downscaled RACMO surface mass balance
    products

    Parameters
    ----------
    base_dir: str
        Working data directory
    EPSG: str or int
        input coordinate reference system
    VERSION: str
        Downscaled RACMO Version

            - ``1.0``: RACMO2.3/XGRN11
            - ``2.0``: RACMO2.3p2/XGRN11
            - ``3.0``: RACMO2.3p2/FGRN055
    tdec: float
        time coordinates to interpolate in year-decimal
    X: float
        x-coordinates to interpolate
    Y: float
        y-coordinates to interpolate
    VARIABLE: str, default 'SMB'
        RACMO product to interpolate

            - ``SMB``: Surface Mass Balance
            - ``PRECIP``: Precipitation
            - ``RUNOFF``: Melt Water Runoff
            - ``SNOWMELT``: Snowmelt
            - ``REFREEZE``: Melt Water Refreeze
    FILL_VALUE: float or NoneType, default None
        Output fill_value for invalid points

        Default will use fill values from data file
    """

    # Full Directory Setup
    DIRECTORY = 'SMB1km_v{0}'.format(VERSION)

    # netcdf variable names
    input_products = {}
    input_products['SMB'] = 'SMB_rec'
    input_products['PRECIP'] = 'precip'
    input_products['RUNOFF'] = 'runoff'
    input_products['SNOWMELT'] = 'snowmelt'
    input_products['REFREEZE'] = 'refreeze'
    # versions 1 and 4 are in separate files for each year
    if (VERSION == '1.0'):
        RACMO_MODEL = ['XGRN11','2.3']
        VARNAME = input_products[VARIABLE]
        SUBDIRECTORY = '{0}_v{1}'.format(VARNAME,VERSION)
        input_dir = os.path.join(base_dir, 'RACMO', DIRECTORY, SUBDIRECTORY)
    elif (VERSION == '2.0'):
        RACMO_MODEL = ['XGRN11','2.3p2']
        var = input_products[VARIABLE]
        VARNAME = var if VARIABLE in ('SMB','PRECIP') else '{0}corr'.format(var)
        input_dir = os.path.join(base_dir, 'RACMO', DIRECTORY)
    elif (VERSION == '3.0'):
        RACMO_MODEL = ['FGRN055','2.3p2']
        var = input_products[VARIABLE]
        VARNAME = var if (VARIABLE == 'SMB') else '{0}corr'.format(var)
        input_dir = os.path.join(base_dir, 'RACMO', DIRECTORY)
    elif (VERSION == '4.0'):
        RACMO_MODEL = ['FGRN055','2.3p2']
        var = input_products[VARIABLE]
        VARNAME = var if (VARIABLE == 'SMB') else '{0}corr'.format(var)
        input_dir = os.path.join(base_dir, 'RACMO', DIRECTORY)
    # input cumulative netCDF4 file
    args = (RACMO_MODEL[0],RACMO_MODEL[1],VERSION,VARIABLE)
    input_file = '{0}_RACMO{1}_DS1km_v{2}_{3}_cumul.nc'.format(*args)

    # pyproj transformer for converting from input coordinates (EPSG)
    # into model coordinates
    try:
        # EPSG projection code string or int
        crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(int(EPSG)))
    except (ValueError,pyproj.exceptions.CRSError):
        # Projection SRS string
        crs1 = pyproj.CRS.from_string(EPSG)
    # coordinate reference system for RACMO model
    crs2 = pyproj.CRS.from_string("epsg:{0:d}".format(3413))
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # calculate projected coordinates of input coordinates
    ix,iy = transformer.transform(X, Y)

    # Open the RACMO NetCDF file for reading
    fileID = netCDF4.Dataset(os.path.join(input_dir,input_file), 'r')
    # input shape of RACMO data
    nt = fileID[VARNAME].shape[0]
    # Get data from each netCDF variable and remove singleton dimensions
    d = {}
    # cell origins on the bottom right
    dx = np.abs(fileID.variables['x'][1]-fileID.variables['x'][0])
    dy = np.abs(fileID.variables['y'][1]-fileID.variables['y'][0])
    # x and y arrays at center of each cell
    d['x'] = fileID.variables['x'][:].copy() - dx/2.0
    d['y'] = fileID.variables['y'][:].copy() - dy/2.0
    # extract time (decimal years)
    d['TIME'] = fileID.variables['TIME'][:].copy()

    # choose a subset of model variables that span the input data
    xr = [ix.min()-dx, ix.max()+dx]
    yr = [iy.min()-dy, iy.max()+dy]
    cols = np.flatnonzero( (d['x'] >= xr[0]) & (d['x'] <= xr[1]) )
    rows = np.flatnonzero( (d['y'] >= yr[0]) & (d['y'] <= yr[1]) )
    ny = rows.size
    nx = cols.size
    # mask object for interpolating data
    d['MASK'] = np.array(fileID.variables['MASK'][rows, cols], dtype=bool)
    d['x'] = d['x'][cols]
    d['y'] = d['y'][rows]
    i,j = np.nonzero(d['MASK'])

    # check that input points are within convex hull of valid model points
    xg,yg = np.meshgrid(d['x'],d['y'])
    v,triangle = SMBcorr.spatial.find_valid_triangulation(xg[i,j],yg[i,j])
    # check where points are within the complex hull of the triangulation
    if v:
        interp_points = np.concatenate((ix[:,None],iy[:,None]),axis=1)
        valid = (triangle.find_simplex(interp_points) >= 0)
    else:
        # Check ix and iy against the bounds of x and y
        valid = (ix >= d['x'].min()) & (ix <= d['x'].max()) & \
            (iy >= d['y'].min()) & (iy <= d['y'].max())

    MI = scipy.interpolate.RegularGridInterpolator(
            (d['y'],d['x']), d['MASK'])
    # check valid points against the mask:
    valid[valid] = MI.__call__(np.c_[iy[valid],ix[valid]])

    # output interpolated arrays of variable
    npts = len(tdec)
    interp_data = np.ma.zeros((npts),fill_value=FILL_VALUE,dtype=np.float64)
    # interpolation mask of invalid values
    interp_data.mask = np.ones((npts),dtype=bool)
    # type designating algorithm used (1:interpolate, 2:backward, 3:forward)
    interp_data.interpolation = np.zeros((npts),dtype=np.uint8)

    # time cutoff allowing for close time interpolation
    dt = np.abs(d['TIME'][1] - d['TIME'][0])
    time_cutoff = (d['TIME'].min() - dt, d['TIME'].max() + dt)
    # find days that can be interpolated
    if np.any((tdec >= time_cutoff[0]) & (tdec <= time_cutoff[1]) & valid):
        # indices of dates for interpolated days
        ind, = np.nonzero((tdec >= time_cutoff[0]) &
            (tdec <= time_cutoff[1]) & valid)
        # determine which subset of time to read from the netCDF4 file
        f = scipy.interpolate.interp1d(d['TIME'], np.arange(nt), kind='linear',
            fill_value=(0,nt-1), bounds_error=False)
        date_indice = f(tdec[ind]).astype(np.int64)
        # months to read
        months = np.arange(date_indice.min(),np.minimum(date_indice.max()+2, d['TIME'].size))
        nm = len(months)
        # extract variable for months of interest
        d[VARNAME] = np.zeros((nm,ny,nx))
        for i,m in enumerate(months):
            d[VARNAME][i,:,:] = fileID.variables[VARNAME][m,rows,cols].copy()
        # create an interpolator for variable
        RGI = scipy.interpolate.RegularGridInterpolator(
            (d['TIME'][months],d['y'],d['x']), d[VARNAME],
            bounds_error=False, fill_value=None)
        # interpolate to points
        interp_data.data[ind] = RGI.__call__(np.c_[tdec[ind],iy[ind],ix[ind]])
        interp_data.mask[ind] = MI.__call__(np.c_[iy[ind],ix[ind]])
        # set interpolation type (1: interpolated)
        interp_data.interpolation[ind] = 1

    # time cutoff without close time interpolation
    time_cutoff = (d['TIME'].min(), d['TIME'].max())
    # check if needing to extrapolate backwards in time
    count = np.count_nonzero((tdec < time_cutoff[0]) & valid)
    if (count > 0):
        # indices of dates before RACMO model
        ind, = np.nonzero((tdec < time_cutoff[0]) & valid)
        # calculate a regression model for calculating values
        # read first 10 years of data to create regression model
        N = 120
        # spatially interpolate variable to coordinates
        VAR = np.zeros((count,N))
        T = np.zeros((N))
        # spatially interpolate mask to coordinates
        mspl = scipy.interpolate.RectBivariateSpline(d['x'], d['y'],
            d['MASK'].T, kx=1, ky=1)
        interp_data.mask[ind] = mspl.ev(ix[ind],iy[ind])
        # create interpolated time series for calculating regression model
        for k in range(N):
            # time at k
            T[k] = d['TIME'][k]
            # spatially interpolate variable
            spl = scipy.interpolate.RectBivariateSpline(d['x'], d['y'],
                fileID.variables[VARNAME][k,rows,cols].T, kx=1, ky=1)
            # create numpy masked array of interpolated values
            VAR[:,k] = spl.ev(ix[ind],iy[ind])
        # calculate regression model
        for n,v in enumerate(ind):
            interp_data.data[v] = regress_model(T, VAR[n,:], tdec[v], ORDER=2,
                CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[0])
        # set interpolation type (2: extrapolated backward)
        interp_data.interpolation[ind] = 2

    # check if needing to extrapolate forward in time
    count = np.count_nonzero((tdec > time_cutoff[1]) & valid)
    if (count > 0):
        # indices of dates after RACMO model
        ind, = np.nonzero((tdec > time_cutoff[1]) & valid)
        # calculate a regression model for calculating values
        # read last 10 years of data to create regression model
        N = 120
        # spatially interpolate variable to coordinates
        VAR = np.zeros((count,N))
        T = np.zeros((N))
        # spatially interpolate mask to coordinates
        mspl = scipy.interpolate.RectBivariateSpline(d['x'], d['y'],
            d['MASK'].T, kx=1, ky=1)
        interp_data.mask[ind] = mspl.ev(ix[ind],iy[ind])
        # create interpolated time series for calculating regression model
        for k in range(N):
            kk = nt - N + k
            # time at k
            T[k] = d['TIME'][kk]
            # spatially interpolate variable
            spl = scipy.interpolate.RectBivariateSpline(d['x'], d['y'],
                fileID.variables[VARNAME][kk,rows, cols].T, kx=1, ky=1)
            # create numpy masked array of interpolated values
            VAR[:,k] = spl.ev(ix[ind],iy[ind])
        # calculate regression model
        for n,v in enumerate(ind):
            interp_data.data[v] = regress_model(T, VAR[n,:], tdec[v], ORDER=2,
                CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[-1])
        # set interpolation type (3: extrapolated forward)
        interp_data.interpolation[ind] = 3

    # complete mask if any invalid in data
    invalid, = np.nonzero(interp_data.data == interp_data.fill_value)
    interp_data.mask[invalid] = True
    # replace fill value
    interp_data.data[interp_data.mask] = interp_data.fill_value

    # close the NetCDF files
    fileID.close()

    # return the interpolated values
    return interp_data
