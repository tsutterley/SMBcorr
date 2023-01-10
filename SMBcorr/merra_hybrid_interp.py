#!/usr/bin/env python
u"""
merra_hybrid_interp.py
Written by Tyler Sutterley (08/2022)
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

PROGRAM DEPENDENCIES:
    regress_model.py: models a time series using least-squares regression

UPDATE HISTORY:
    Updated 08/2022: updated docstrings to numpy documentation format
    Updated 11/2021: don't attempt triangulation if large number of points
    Updated 05/2021: set bounds error to false when reducing temporal range
    Updated 04/2021: can reduce input dataset to a temporal subset
    Updated 02/2021: added new MERRA2-hybrid v1.1 variables
        added gzip compression option
    Updated 01/2021: using conversion protocols following pyproj-2 updates
        https://pyproj4.github.io/pyproj/stable/gotchas.html
    Updated 08/2020: attempt delaunay triangulation using different options
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
import warnings
import numpy as np
import scipy.spatial
import scipy.ndimage
import scipy.interpolate
from SMBcorr.regress_model import regress_model

# attempt imports
try:
    import netCDF4
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("module")
    warnings.warn("netCDF4 not available", ImportWarning)
try:
    import pyproj
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("module")
    warnings.warn("pyproj not available", ImportWarning)
try:
    from sklearn.neighbors import KDTree, BallTree
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("module")
    warnings.warn("scikit-learn not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

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

# PURPOSE: read and interpolate MERRA-2 hybrid firn corrections
def interpolate_merra_hybrid(base_dir, EPSG, REGION, tdec, X, Y,
    VERSION='v1', VARIABLE='FAC', SIGMA=1.5, FILL_VALUE=None,
    EXTRAPOLATE=False, GZIP=False):
    """
    Reads and interpolates MERRA-2 hybrid variables

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
    if VARIABLE in ('FAC','cum_smb_anomaly','SMB_a','height','h_a'):
        args = (VERSION,REGION.lower(),suffix)
        hybrid_file = 'gsfc_fdm_{0}_{1}.nc{2}'.format(*args)
    elif VARIABLE in ('smb','SMB','Me','Ra','Ru','Sn-Ev'):
        args = (VERSION,REGION.lower(),suffix)
        hybrid_file = 'gsfc_fdm_smb_{0}_{1}.nc{2}'.format(*args)
    elif VARIABLE in ('Me_a','Ra_a','Ru_a','Sn-Ev_a'):
        args = (VERSION,REGION.lower(),suffix)
        hybrid_file = 'gsfc_fdm_smb_cumul_{0}_{1}.nc{2}'.format(*args)
    elif VARIABLE in ('FAC') and (VERSION == 'v0'):
        args = ('FAC',REGION.lower(),suffix)
        hybrid_file = 'gsfc_{0}_{1}.nc{2}'.format(*args)
    elif VARIABLE in ('p_minus_e','melt') and (VERSION == 'v0'):
        args = (VARIABLE,REGION.lower(),suffix)
        hybrid_file = 'm2_hybrid_{0}_cumul_{1}.nc{2}'.format(*args)

    # Open the MERRA-2 Hybrid NetCDF file for reading
    if GZIP:
        # read as in-memory (diskless) netCDF4 dataset
        with gzip.open(os.path.join(base_dir,hybrid_file),'r') as f:
            fileID = netCDF4.Dataset(uuid.uuid4().hex, memory=f.read())
    else:
        # read netCDF4 dataset
        fileID = netCDF4.Dataset(os.path.join(base_dir,hybrid_file), 'r')

    # Get data from each netCDF variable and remove singleton dimensions
    fd = {}
    # time is year decimal at time step 5 days
    time_step = 5.0/365.25
    # if extrapolating data: read the full dataset
    # if simply interpolating with fill values: reduce to a subset
    if EXTRAPOLATE:
        # read time variables
        fd['time'] = fileID.variables['time'][:].copy()
        # read full dataset and remove singleton dimensions
        fd[VARIABLE] = np.squeeze(fileID.variables[VARIABLE][:].copy())
    else:
        # reduce grids to time period of input buffered by time steps
        tmin = np.min(tdec) - 2.0*time_step
        tmax = np.max(tdec) + 2.0*time_step
        # find indices to times
        nt, = fileID.variables['time'].shape
        f = scipy.interpolate.interp1d(fileID.variables['time'][:],
            np.arange(nt), kind='nearest', bounds_error=False,
            fill_value=(0,nt))
        imin,imax = f((tmin,tmax)).astype(np.int64)
        # read reduced time variables
        fd['time'] = fileID.variables['time'][imin:imax+1].copy()
        # read reduced dataset and remove singleton dimensions
        fd[VARIABLE] = np.squeeze(fileID.variables[VARIABLE][imin:imax+1,:,:])
    # invalid data value
    fv = np.float64(fileID.variables[VARIABLE]._FillValue)
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
        # reference to first firn field
        temp1[i,j] = fd[VARIABLE][t,i,j] - fd[VARIABLE][0,i,j]
        # smooth firn field
        temp2 = scipy.ndimage.gaussian_filter(temp1, SIGMA,
            mode='constant', cval=0)
        # scale output smoothed firn field
        gs[VARIABLE].data[t,ii,jj] = temp2[ii,jj]/gs['mask'][ii,jj]
        # replace valid firn values with original
        gs[VARIABLE].data[t,i,j] = temp1[i,j]
        # set mask variables for time
        gs[VARIABLE].mask[t,:,:] = (gs['mask'] == 0.0)

    # convert projection from input coordinates (EPSG) to model coordinates
    MODEL_EPSG = set_projection(REGION)
    crs1 = pyproj.CRS.from_string(EPSG)
    crs2 = pyproj.CRS.from_string(MODEL_EPSG)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # calculate projected coordinates of input coordinates
    ix,iy = transformer.transform(X, Y)

    # check that input points are within convex hull of smoothed model points
    v,triangle = find_valid_triangulation(xg[ii,jj],yg[ii,jj])
    # check if there is a valid triangulation
    if v:
        # check where points are within the complex hull of the triangulation
        interp_points = np.concatenate((ix[:,None],iy[:,None]),axis=1)
        valid = (triangle.find_simplex(interp_points) >= 0)
    else:
        # Check ix and iy against the bounds of x and y
        valid = (ix >= fd['x'].min()) & (ix <= fd['x'].max()) & \
            (iy >= fd['y'].min()) & (iy <= fd['y'].max())

    # output interpolated arrays of variable
    npts = len(tdec)
    interp_data = np.ma.zeros((npts),fill_value=fv)
    # interpolation mask of invalid values
    interp_data.mask = np.ones((npts),dtype=bool)
    # type designating algorithm used (1: interpolate, 2: backward, 3:forward)
    interp_data.interpolation = np.zeros_like(tdec,dtype=np.uint8)

    # find days that can be interpolated
    if np.any((tdec >= fd['time'].min()) & (tdec <= fd['time'].max()) & valid):
        # indices of dates for interpolated days
        ind, = np.nonzero((tdec >= fd['time'].min()) &
            (tdec <= fd['time'].max()) & valid)
        # create an interpolator for firn height or air content
        RGI = scipy.interpolate.RegularGridInterpolator(
            (fd['time'],fd['x'],fd['y']), gs[VARIABLE].data)
        # create an interpolator for input mask
        MI = scipy.interpolate.RegularGridInterpolator(
            (fd['time'],fd['x'],fd['y']), gs[VARIABLE].mask)
        # interpolate to points
        interp_data.data[ind] = RGI.__call__(np.c_[tdec[ind],ix[ind],iy[ind]])
        interp_data.mask[ind] = MI.__call__(np.c_[tdec[ind],ix[ind],iy[ind]])
        # set interpolation type (1: interpolated)
        interp_data.interpolation[ind] = 1

    # check if needing to extrapolate backwards in time
    count = np.count_nonzero((tdec < fd['time'].min()) & valid)
    if (count > 0) and EXTRAPOLATE:
        # indices of dates before firn model
        ind, = np.nonzero((tdec < fd['time'].min()) & valid)
        # calculate a regression model for calculating values
        # read first 10 years of data to create regression model
        N = np.int64(10.0/time_step)
        # spatially interpolate variable to coordinates
        T = np.zeros((N))
        DATA = np.zeros((count,N))
        MASK = np.zeros((count,N))
        # create interpolated time series for calculating regression model
        for k in range(N):
            # time at k
            T[k] = fd['time'][k]
            # spatially interpolate variable and mask
            f1 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE].data[k,:,:], kx=1, ky=1)
            f2 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE].mask[k,:,:], kx=1, ky=1)
            # create numpy masked array of interpolated values
            DATA[:,k] = f1.ev(ix[ind],iy[ind])
            MASK[:,k] = f2.ev(ix[ind],iy[ind])
        # calculate regression model
        for n,v in enumerate(ind):
            interp_data.data[v] = regress_model(T, DATA[n,:], tdec[v], ORDER=2,
                CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[0])
        # mask any invalid points
        interp_data.mask[ind] = np.any(MASK, axis=1)
        # set interpolation type (2: extrapolated backward)
        interp_data.interpolation[ind] = 2

    # check if needing to extrapolate forward in time
    count = np.count_nonzero((tdec > fd['time'].max()) & valid)
    if (count > 0) and EXTRAPOLATE:
        # indices of dates after firn model
        ind, = np.nonzero((tdec > fd['time'].max()) & valid)
        # calculate a regression model for calculating values
        # read last 10 years of data to create regression model
        N = np.int64(10.0/time_step)
        # spatially interpolate variable to coordinates
        T = np.zeros((N))
        DATA = np.zeros((count,N))
        MASK = np.zeros((count,N))
        # create interpolated time series for calculating regression model
        for k in range(N):
            kk = nt - N + k
            # time at kk
            T[k] = fd['time'][kk]
            # spatially interpolate firn elevation or air content
            fspl = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE][kk,:,:], kx=1, ky=1)
            # spatially interpolate variable and mask
            f1 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE].data[kk,:,:], kx=1, ky=1)
            f2 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE].mask[kk,:,:], kx=1, ky=1)
            # create numpy masked array of interpolated values
            DATA[:,k] = f1.ev(ix[ind],iy[ind])
            MASK[:,k] = f2.ev(ix[ind],iy[ind])
        # calculate regression model
        for n,v in enumerate(ind):
            interp_data.data[v] = regress_model(T, DATA[n,:], tdec[v], ORDER=2,
                CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[-1])
        # mask any invalid points
        interp_data.mask[ind] = np.any(MASK, axis=1)
        # set interpolation type (3: extrapolated forward)
        interp_data.interpolation[ind] = 3

    # complete mask if any invalid in data
    invalid, = np.nonzero(interp_data.data == interp_data.fill_value)
    interp_data.mask[invalid] = True
    # replace fill value if specified
    if FILL_VALUE:
        interp_data.fill_value = FILL_VALUE
        interp_data.data[interp_data.mask] = interp_data.fill_value

    # return the interpolated values
    return interp_data
