#!/usr/bin/env python
u"""
racmo_interp_mean.py
Written by Tyler Sutterley (10/2022)
Interpolates the mean of downscaled RACMO products to spatial coordinates

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

UPDATE HISTORY:
    Updated 10/2022: added version 4.0 (RACMO2.3p2 for 1958-2022 from FGRN055)
    Updated 08/2022: updated docstrings to numpy documentation format
    Updated 11/2021: don't attempt triangulation if large number of points
    Updated 01/2021: using conversion protocols following pyproj-2 updates
        https://pyproj4.github.io/pyproj/stable/gotchas.html
        using utilities from time module for conversions
    Updated 08/2020: attempt delaunay triangulation using different options
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
        return (None, triangle)

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
            return (i+1, triangle)

    # if still errors: set triangle as an empty list
    triangle = []
    return (None, triangle)

# PURPOSE: read and interpolate downscaled RACMO products
def interpolate_racmo_mean(base_dir, EPSG, VERSION, tdec, X, Y,
    VARIABLE='SMB', RANGE=[], FILL_VALUE=None):
    """
    Reads and interpolates the temporal mean of downscaled
    RACMO surface mass balance products

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
    # mask object for interpolating data
    d['MASK'] = np.array(fileID.variables['MASK'][:],dtype=bool)
    i,j = np.nonzero(d['MASK'])

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
    # convert projection from input coordinates to projected
    ix,iy = transformer.transform(X, Y)
    # check that input points are within convex hull of valid model points
    xg,yg = np.meshgrid(d['x'],d['y'])
    v,triangle = find_valid_triangulation(xg[i,j],yg[i,j])
    # check where points are within the complex hull of the triangulation
    if v:
        interp_points = np.concatenate((ix[:,None],iy[:,None]),axis=1)
        valid = (triangle.find_simplex(interp_points) >= 0)
    else:
        # Check ix and iy against the bounds of x and y
        valid = (ix >= d['x'].min()) & (ix <= d['x'].max()) & \
            (iy >= d['y'].min()) & (iy <= d['y'].max())

    # output interpolated arrays of variable
    interp_var = np.zeros_like(tdec,dtype=np.float64)
    # type designating algorithm used (1: interpolate, 2: backward, 3:forward)
    interp_type = np.zeros_like(tdec,dtype=np.uint8)
    # interpolation mask of invalid values
    interp_mask = np.zeros_like(tdec,dtype=bool)

    # find days that can be interpolated
    if np.any(valid):
        # indices of dates for interpolated days
        ind, = np.nonzero(valid)
        # create an interpolator for variable
        RGI=scipy.interpolate.RegularGridInterpolator((d['y'],d['x']),MEAN)
        # create an interpolator for input mask
        MI=scipy.interpolate.RegularGridInterpolator((d['y'],d['x']),d['MASK'])
        # interpolate to points
        dt = (tdec[ind] - d['TIME'][0])/(d['TIME'][1] - d['TIME'][0])
        interp_var[ind] = dt*RGI.__call__(np.c_[iy[ind],ix[ind]])
        interp_mask[ind] = MI.__call__(np.c_[iy[ind],ix[ind]])
        # set interpolation type (1: interpolated)
        interp_type[ind] = 1

    # replace fill value if specified
    if FILL_VALUE:
        ind, = np.nonzero(~interp_mask)
        interp_var[ind] = FILL_VALUE
        fv = FILL_VALUE
    else:
        fv = 0.0

    # close the NetCDF files
    fileID.close()

    # return the interpolated values
    return (interp_var, interp_type, fv)
