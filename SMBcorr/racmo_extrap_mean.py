#!/usr/bin/env python
u"""
racmo_extrap_mean.py
Written by Tyler Sutterley (10/2022)
Spatially extrapolates the mean of downscaled RACMO products

Uses fast nearest-neighbor search algorithms
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
and inverse distance weighted interpolation to extrapolate spatially

INPUTS:
    base_dir: Working data directory
    EPSG: input coordinate reference system
    VERSION: Downscaled RACMO Version
        1.0: RACMO2.3/XGRN11
        2.0: RACMO2.3p2/XGRN11
        3.0: RACMO2.3p2/FGRN055
        4.0: RACMO2.3p2/FGRN055
    tdec: time coordinates in year-decimal
    X: x-coordinates
    Y: y-coordinates

OPTIONS:
    VARIABLE: RACMO product to calculate
        SMB: Surface Mass Balance
        PRECIP: Precipitation
        RUNOFF: Melt Water Runoff
        SNOWMELT: Snowmelt
        REFREEZE: Melt Water Refreeze
    RANGE: Start and end year of mean
    SEARCH: nearest-neighbor search algorithm
    NN: number of nearest-neighbor points to use
    POWER: inverse distance weighting power
    FILL_VALUE: Replace invalid values with fill value
        default will use fill values from data file

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
    Updated 10/2022: added version 4.0 (RACMO2.3p2 for 1958-2022 from FGRN055)
    Updated 08/2022: updated docstrings to numpy documentation format
    Updated 01/2021: using conversion protocols following pyproj-2 updates
        https://pyproj4.github.io/pyproj/stable/gotchas.html
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
import scipy.interpolate

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

# PURPOSE: read and interpolate downscaled RACMO products
def extrapolate_racmo_mean(base_dir, EPSG, VERSION, tdec, X, Y,
    VARIABLE='SMB', RANGE=[], SEARCH='BallTree', NN=10, POWER=2.0,
    FILL_VALUE=None):
    """
    Spatially extrapolates the temporal mean of downscaled RACMO products

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
    RANGE: list
        Start and end year of mean
    SEARCH: str, default 'BallTree'
        nearest-neighbor search algorithm
    NN: int, default 10
        number of nearest-neighbor points to use
    POWER: int or float, default 2.0
        Inverse distance weighting power
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

    # read mean from netCDF4 file
    arg = (RACMO_MODEL[0],RACMO_MODEL[1],VERSION,VARIABLE,RANGE[0],RANGE[1])
    mean_file = '{0}_RACMO{1}_DS1km_v{2}_{3}_Mean_{4:4d}-{5:4d}.nc'.format(*arg)
    with netCDF4.Dataset(os.path.join(input_dir,mean_file),'r') as fileID:
        MEAN = fileID[VARNAME][:,:].copy()

    # input cumulative netCDF4 file
    args = (RACMO_MODEL[0],RACMO_MODEL[1],VERSION,VARIABLE)
    input_file = '{0}_RACMO{1}_DS1km_v{2}_{3}_cumul.nc'.format(*args)

    # Open the RACMO NetCDF file for reading
    fileID = netCDF4.Dataset(os.path.join(input_dir,input_file), 'r')
    # input shape of RACMO data
    nt,ny,nx = fileID[VARNAME].shape
    # Get data from each netCDF variable
    d = {}
    # cell origins on the bottom right
    dx = np.abs(fileID.variables['x'][1]-fileID.variables['x'][0])
    dy = np.abs(fileID.variables['y'][1]-fileID.variables['y'][0])
    # latitude and longitude arrays at center of each cell
    d['LON'] = fileID.variables['LON'][:,:].copy()
    d['LAT'] = fileID.variables['LAT'][:,:].copy()
    # extract time (decimal years)
    d['TIME'] = fileID.variables['TIME'][:].copy()
    # mask object for interpolating data
    d['MASK'] = np.array(fileID.variables['MASK'][:],dtype=bool)
    i,j = np.nonzero(d['MASK'])
    # reduce mean to valid points
    var1 = MEAN[i,j]

    # convert RACMO latitude and longitude to input coordinates (EPSG)
    crs1 = pyproj.CRS.from_string(EPSG)
    crs2 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    direction = pyproj.enums.TransformDirection.INVERSE
    # convert projection from model coordinates
    xg,yg = transformer.transform(d['LON'], d['LAT'], direction=direction)

    # construct search tree from original points
    # can use either BallTree or KDTree algorithms
    xy1 = np.concatenate((xg[i,j,None],yg[i,j,None]),axis=1)
    tree = BallTree(xy1) if (SEARCH == 'BallTree') else KDTree(xy1)

    # output extrapolated arrays of variable
    extrap_var = np.zeros_like(tdec,dtype=np.float64)
    # type designating algorithm used (1: interpolate, 2: backward, 3:forward)
    extrap_type = np.ones_like(tdec,dtype=np.uint8)

    # inverse distance weighting to extrapolate in space
    # query the search tree to find the NN closest points
    xy2 = np.concatenate((X[:,None],Y[:,None]),axis=1)
    dist,indices = tree.query(xy2, k=NN, return_distance=True)
    count = len(tdec)
    # normalized weights if POWER > 0 (typically between 1 and 3)
    # in the inverse distance weighting
    power_inverse_distance = dist**(-POWER)
    s = np.sum(power_inverse_distance, axis=1)
    w = power_inverse_distance/np.broadcast_to(s[:,None],(count,NN))
    # spatially extrapolate using inverse distance weighting
    dt = (tdec - d['TIME'][0])/(d['TIME'][1] - d['TIME'][0])
    extrap_var[:] = dt*np.sum(w*var1[indices],axis=1)

    # replace fill value if specified
    if FILL_VALUE:
        ind, = np.nonzero(extrap_type == 0)
        extrap_var[ind] = FILL_VALUE
        fv = FILL_VALUE
    else:
        fv = 0.0

    # close the NetCDF files
    fileID.close()

    # return the extrapolated values
    return (extrap_var,extrap_type,fv)
