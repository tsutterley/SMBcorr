#!/usr/bin/env python
u"""
mar_interp_mean.py
Written by Tyler Sutterley (08/2022)
Interpolates mean MAR products to times and coordinates

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
    RANGE: start year and end year of mean file
    SIGMA: Standard deviation for Gaussian kernel
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

UPDATE HISTORY:
    Updated 08/2022: updated docstrings to numpy documentation format
    Updated 11/2021: don't attempt triangulation if large number of points
    Updated 01/2021: using conversion protocols following pyproj-2 updates
        https://pyproj4.github.io/pyproj/stable/gotchas.html
    Written 08/2020
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
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: find a valid Delaunay triangulation for coordinates x0 and y0
# http://www.qhull.org/html/qhull.htm#options
# Attempt 1: standard qhull options Qt Qbb Qc Qz
# Attempt 2: rescale and center the inputs with option QbB
# Attempt 3: joggle the inputs to find a triangulation with option QJ
# if no passing triangulations: exit with empty list
def find_valid_triangulation(x0, y0, max_points=1e6):
    """
    Attempt to find a valid Delaunay triangulation for coordinates

    - Attempt 1: ``Qt Qbb Qc Qz``
    - Attempt 2: ``Qt Qc QbB``
    - Attempt 3: ``QJ QbB``

    Parameters
    ----------
    x0: float
        x-coordinates
    y0: float
        y-coordinates
    max_points: int or float, default 1e6
        Maximum number of coordinates to attempt to triangulate
    """
    # don't attempt triangulation if there are a large number of points
    if (len(x0) > max_points):
        # if too many points: set triangle as an empty list
        triangle = []
        return (None,triangle)

    # Attempt 1: try with standard options Qt Qbb Qc Qz
    # Qt: triangulated output, all facets will be simplicial
    # Qbb: scale last coordinate to [0,m] for Delaunay triangulations
    # Qc: keep coplanar points with nearest facet
    # Qz: add point-at-infinity to Delaunay triangulation

    # Attempt 2 in case of qhull error from Attempt 1 try Qt Qc QbB
    # Qt: triangulated output, all facets will be simplicial
    # Qc: keep coplanar points with nearest facet
    # QbB: scale input to unit cube centered at the origin

    # Attempt 3 in case of qhull error from Attempt 2 try QJ QbB
    # QJ: joggle input instead of merging facets
    # QbB: scale input to unit cube centered at the origin

    # try each set of qhull_options
    points = np.concatenate((x0[:,None],y0[:,None]),axis=1)
    for i,opt in enumerate(['Qt Qbb Qc Qz','Qt Qc QbB','QJ QbB']):
        try:
            triangle = scipy.spatial.Delaunay(points.data, qhull_options=opt)
        except scipy.spatial.qhull.QhullError:
            pass
        else:
            return (i+1,triangle)

    # if still errors: set triangle as an empty list
    triangle = []
    return (None,triangle)

# PURPOSE: read and interpolate a mean field of MAR outputs
def interpolate_mar_mean(DIRECTORY, EPSG, VERSION, tdec, X, Y,
    VARIABLE='SMB', XNAME=None, YNAME=None, TIMENAME='TIME',
    RANGE=[2000,2019], SIGMA=1.5, FILL_VALUE=None):
    """
    Read and interpolates the temporal mean of MAR products

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
    RANGE: list, default [2000,2019]
        Start and end year of mean
    SIGMA: float, default 1.5
        Standard deviation for Gaussian kernel
    FILL_VALUE: float or NoneType, default None
        Output fill_value for invalid points

        Default will use fill values from data file
    """

    # MAR model projection: Polar Stereographic (Oblique)
    # Earth Radius: 6371229 m
    # True Latitude: 0
    # Center Longitude: -40
    # Center Latitude: 70.5
    proj4_params = ("+proj=sterea +lat_0=+70.5 +lat_ts=0 +lon_0=-40.0 "
        "+a=6371229 +no_defs")

    # regular expression pattern for MAR dataset
    rx = re.compile('MAR_SMBavg(.*?){0}-{1}.nc$'.format(*RANGE))
    # find mar mean file for RANGE
    #print(f"looking for files matching MAR_SMBavg {str(RANGE)} in {DIRECTORY}")
    FILE, = [f for f in os.listdir(DIRECTORY) if rx.match(f)]
    # Open the MAR NetCDF file for reading
    with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
        nx = len(fileID.variables[XNAME][:])
        ny = len(fileID.variables[YNAME][:])

    # python dictionary with file variables
    fd = {}
    # create a masked array with all data
    fd[VARIABLE] = np.ma.zeros((ny,nx),fill_value=FILL_VALUE)
    fd[VARIABLE].mask = np.zeros((ny,nx),dtype=bool)
    # python dictionary with gaussian filtered variables
    gs = {}
    # use a gaussian filter to smooth each model field
    gs[VARIABLE] = np.ma.zeros((ny,nx), fill_value=FILL_VALUE)
    gs[VARIABLE].mask = np.ones((ny,nx), dtype=bool)
    # Open the MAR NetCDF file for reading
    with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
        # surface type
        SRF=fileID.variables['SRF'][:]
        # indices of specified ice mask
        i,j=np.nonzero(SRF == 4)
        # Get data from netCDF variable and remove singleton dimensions
        tmp=np.squeeze(fileID.variables[VARIABLE][:])
        # combine sectors for multi-layered data
        if (np.ndim(tmp) == 3):
            # ice fraction
            FRA=fileID.variables['FRA'][:]/100.0
            # create mask for combining data
            MASK = np.zeros((ny,nx))
            MASK[i,j] = FRA[i,j]
            # combine data
            fd[VARIABLE][:,:] = MASK*tmp[0,:,:] + \
                (1.0-MASK)*tmp[1,:,:]
        else:
            # copy data
            fd[VARIABLE][:,:] = tmp.copy()
        # verify mask object for interpolating data
        fd[VARIABLE].mask[:,:] |= (SRF != 4)
        # combine mask object through time to create a single mask
        fd['MASK']=1.0 - np.array(fd[VARIABLE].mask,dtype=np.float64)
        # MAR coordinates
        fd['LON']=fileID.variables['LON'][:,:].copy()
        fd['LAT']=fileID.variables['LAT'][:,:].copy()
        # convert x and y coordinates to meters
        fd['x']=1000.0*fileID.variables[XNAME][:].copy()
        fd['y']=1000.0*fileID.variables[YNAME][:].copy()
        # use a gaussian filter to smooth mask
        gs['MASK']=scipy.ndimage.gaussian_filter(fd['MASK'],SIGMA,
            mode='constant',cval=0)
        # indices of smoothed ice mask
        ii,jj = np.nonzero(np.ceil(gs['MASK']) == 1.0)
        # replace fill values before smoothing data
        temp1 = np.zeros((ny,nx))
        i,j = np.nonzero(~fd[VARIABLE].mask)
        temp1[i,j] = fd[VARIABLE][i,j].copy()
        # smooth spatial field
        temp2 = scipy.ndimage.gaussian_filter(temp1, SIGMA,
            mode='constant', cval=0)
        # scale output smoothed field
        gs[VARIABLE].data[ii,jj] = temp2[ii,jj]/gs['MASK'][ii,jj]
        # replace valid values with original
        gs[VARIABLE].data[i,j] = temp1[i,j]
        # set mask variables for time
        gs[VARIABLE].mask[ii,jj] = False

    # convert projection from input coordinates (EPSG) to model coordinates
    crs1 = pyproj.CRS.from_string(EPSG)
    crs2 = pyproj.CRS.from_string(proj4_params)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # calculate projected coordinates of input coordinates
    ix,iy = transformer.transform(X, Y)

    # check that input points are within convex hull of valid model points
    gs['x'],gs['y'] = np.meshgrid(fd['x'],fd['y'])
    # attempt to find a valid delaunay triangulation
    v,triangle = find_valid_triangulation(gs['x'][ii,jj],gs['y'][ii,jj])
    # check if there is a valid triangulation
    if v:
        # check where points are within the complex hull of the triangulation
        interp_points = np.concatenate((ix[:,None],iy[:,None]),axis=1)
        valid = (triangle.find_simplex(interp_points) >= 0)
    else:
        # Check ix and iy against the bounds of x and y
        valid = (ix >= fd['x'].min()) & (ix <= fd['x'].max()) & \
            (iy >= fd['y'].min()) & (iy <= fd['y'].max())

    # number of output data points
    npts = len(tdec)
    # output interpolated arrays of model variable
    interp = np.ma.zeros((npts),fill_value=FILL_VALUE,dtype=np.float64)
    interp.mask = np.ones((npts),dtype=bool)
    # initially set all values to fill value
    interp.data[:] = interp.fill_value

    # if there are valid points
    if np.any(valid):
        # indices of valid spatial points
        ind, = np.nonzero(valid)
        # create an interpolator for model variable
        RGI = scipy.interpolate.RegularGridInterpolator(
            (fd['y'],fd['x']), gs[VARIABLE].data)
        # create an interpolator for input mask
        MI = scipy.interpolate.RegularGridInterpolator(
            (fd['y'],fd['x']), gs[VARIABLE].mask)
        # interpolate to points
        interp.data[ind] = RGI.__call__(np.c_[iy[ind],ix[ind]])
        interp.mask[ind] = MI.__call__(np.c_[iy[ind],ix[ind]])

    # complete mask if any invalid in data
    invalid, = np.nonzero((interp.data == interp.fill_value) |
        np.isnan(interp.data))
    interp.mask[invalid] = True

    # return the interpolated values
    return interp
