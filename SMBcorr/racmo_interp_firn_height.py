#!/usr/bin/env python
u"""
racmo_interp_firn_height.py
Written by Tyler Sutterley (02/2023)
Interpolates and extrapolates firn heights to times and coordinates

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

PROGRAM DEPENDENCIES:
    regress_model.py: models a time series using least-squares regression

UPDATE HISTORY:
    Updated 02/2023: close in time extrapolations with regular grid interpolator
    Updated 08/2022: updated docstrings to numpy documentation format
    Updated 11/2021: don't attempt triangulation if large number of points
    Updated 01/2021: using conversion protocols following pyproj-2 updates
        https://pyproj4.github.io/pyproj/stable/gotchas.html
    Updated 08/2020: attempt delaunay triangulation using different options
    Updated 04/2020: reduced to interpolation function.  output masked array
    Updated 10/2019: Gaussian average firn fields before interpolation
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
import scipy.spatial
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

# PURPOSE: read and interpolate RACMO2.3 firn corrections
def interpolate_racmo_firn(base_dir, EPSG, MODEL, tdec, X, Y, VARIABLE='zs',
    SIGMA=1.5, FILL_VALUE=None, REFERENCE=False):
    """
    Reads and interpolates downscaled RACMO firn height products

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
        # time is year decimal from 1960-01-01 at time_step 10 days
        time_step = 10.0/365.25
        # rotation parameters
        rot_lat = -18.0
        rot_lon = -37.5
    elif (MODEL == 'FGRN055'):
        # filename and directory for input FGRN055 file
        FIRN_FILE['zs'] = 'FDM_zs_FGRN055_1960-2017_interpol.nc'
        FIRN_FILE['FirnAir'] = 'FDM_FirnAir_FGRN055_1960-2017_interpol.nc'
        FIRN_DIRECTORY = ['RACMO','FGRN055_1960-2017']
        # time is year decimal from 1960-01-01 at time_step 10 days
        time_step = 10.0/365.25
        # rotation parameters
        rot_lat = -18.0
        rot_lon = -37.5
    elif (MODEL == 'XANT27'):
        # filename and directory for input XANT27 file
        FIRN_FILE['zs'] = 'FDM_zs_ANT27_1979-2016.nc'
        FIRN_FILE['FirnAir'] = 'FDM_FirnAir_ANT27_1979-2016.nc'
        FIRN_DIRECTORY = ['RACMO','XANT27_1979-2016']
        # time is year decimal from 1979-01-01 at time_step 10 days
        time_step = 10.0/365.25
        # rotation parameters
        rot_lat = -180.0
        rot_lon = 10.0
    elif (MODEL == 'ASE055'):
        # filename and directory for input ASE055 file
        FIRN_FILE['zs'] = 'FDM_zs_ASE055_1979-2015.nc'
        FIRN_FILE['FirnAir'] = 'FDM_FirnAir_ASE055_1979-2015.nc'
        FIRN_DIRECTORY = ['RACMO','ASE055_1979-2015']
        # time is year decimal from 1979-01-01 at time_step 10 days
        time_step = 10.0/365.25
        # rotation parameters
        rot_lat = 167.0
        rot_lon = 53.0
    elif (MODEL == 'XPEN055'):
        # filename and directory for input XPEN055 file
        FIRN_FILE['zs'] = 'FDM_zs_XPEN055_1979-2016.nc'
        FIRN_FILE['FirnAir'] = 'FDM_FirnAir_XPEN055_1979-2016.nc'
        FIRN_DIRECTORY = ['RACMO','XPEN055_1979-2016']
        # time is year decimal from 1979-01-01 at time_step 10 days
        time_step = 10.0/365.25
        # rotation parameters
        rot_lat = -180.0
        rot_lon = 30.0

    # Open the RACMO NetCDF file for reading
    ddir = os.path.join(base_dir,*FIRN_DIRECTORY)
    fileID = netCDF4.Dataset(os.path.join(ddir,FIRN_FILE[VARIABLE]), 'r')
    fd = {}
    # invalid data value
    fv = np.float64(fileID.variables[VARIABLE]._FillValue)
    # Get data from each netCDF variable and remove singleton dimensions
    fd[VARIABLE] = np.squeeze(fileID.variables[VARIABLE][:].copy())
    # verify mask object for interpolating data
    fd[VARIABLE].mask = (fd[VARIABLE].data[:,:,:] == fv)
    fd['lon'] = fileID.variables['lon'][:,:].copy()
    fd['lat'] = fileID.variables['lat'][:,:].copy()
    fd['time'] = fileID.variables['time'][:].copy()
    # input shape of RACMO firn data
    nt,ny,nx = np.shape(fd[VARIABLE])
    # close the NetCDF files
    fileID.close()

    # indices of specified ice mask
    i,j = np.nonzero(fd[VARIABLE][0,:,:] != fv)
    # create mask object for interpolating data
    fd['mask'] = np.zeros((ny,nx))
    fd['mask'][i,j] = 1.0

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

    # rotated pole longitude and latitude of input model (model coordinates)
    xg,yg = rotate_coordinates(fd['lon'], fd['lat'], rot_lon, rot_lat)
    # recreate arrays to fix small floating point errors
    # (ensure that arrays are monotonically increasing)
    fd['x'] = np.linspace(np.mean(xg[:,0]),np.mean(xg[:,-1]),nx)
    fd['y'] = np.linspace(np.mean(yg[0,:]),np.mean(yg[-1,:]),ny)

    # convert projection from input coordinates (EPSG) to model coordinates
    # RACMO models are rotated pole latitude and longitude
    try:
        # EPSG projection code string or int
        crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(int(EPSG)))
    except (ValueError,pyproj.exceptions.CRSError):
        # Projection SRS string
        crs1 = pyproj.CRS.from_string(EPSG)
    # coordinate reference system for RACMO model
    crs2 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # convert projection from input coordinates to projected
    ilon,ilat = transformer.transform(X, Y)
    # calculate rotated pole coordinates of input coordinates
    ix,iy = rotate_coordinates(ilon, ilat, rot_lon, rot_lat)

    # check that input points are within convex hull of smoothed model points
    v,triangle = find_valid_triangulation(xg[ii,jj],yg[i,j])
    # check where points are within the complex hull of the triangulation
    if v:
        interp_points = np.concatenate((ix[:,None],iy[:,None]),axis=1)
        valid = (triangle.find_simplex(interp_points) >= 0)
    else:
        # Check ix and iy against the bounds of x and y
        valid = (ix >= fd['x'].min()) & (ix <= fd['x'].max()) & \
            (iy >= fd['y'].min()) & (iy <= fd['y'].max())

    # output interpolated arrays of firn variable (height or firn air content)
    npts = len(tdec)
    interp_data = np.ma.zeros((npts),fill_value=fv,dtype=np.float64)
    interp_data.mask = np.ones((npts),dtype=bool)
    # type designating algorithm used (1:interpolate, 2:backward, 3:forward)
    interp_data.interpolation = np.zeros((npts),dtype=np.uint8)

    # time cutoff allowing for close time interpolation
    dt = np.abs(fd['time'][1] - fd['time'][0])
    time_cutoff = (fd['time'].min() - dt, fd['time'].max() + dt)
    # find days that can be interpolated
    if np.any((tdec >= time_cutoff[0]) & (tdec <= time_cutoff[1]) & valid):
        # indices of dates for interpolated days
        ind, = np.nonzero((tdec >= time_cutoff[0]) &
            (tdec <= time_cutoff[1]) & valid)
        # create an interpolator for model variable
        RGI = scipy.interpolate.RegularGridInterpolator(
            (fd['time'],fd['y'],fd['x']), gs[VARIABLE].data,
            bounds_error=False, fill_value=None)
        # create an interpolator for input mask
        MI = scipy.interpolate.RegularGridInterpolator(
            (fd['time'],fd['y'],fd['x']), gs[VARIABLE].mask,
            bounds_error=False, fill_value=None)
        # interpolate to points
        interp_data.data[ind] = RGI.__call__(np.c_[tdec[ind],iy[ind],ix[ind]])
        interp_data.mask[ind] = MI.__call__(np.c_[tdec[ind],iy[ind],ix[ind]])
        # set interpolation type (1: interpolated)
        interp_data.interpolation[ind] = 1

    # time cutoff without close time interpolation
    time_cutoff = (fd['time'].min(), fd['time'].max())
    # check if needing to extrapolate backwards in time
    count = np.count_nonzero((tdec < time_cutoff[0]) & valid)
    if (count > 0):
        # indices of dates before firn model
        ind, = np.nonzero((tdec < time_cutoff[0]) & valid)
        # calculate a regression model for calculating values
        # read first 10 years of data to create regression model
        N = 365
        # spatially interpolate firn elevation or air content to coordinates
        FIRN = np.zeros((count,N))
        MASK = np.zeros((count,N),dtype=bool)
        T = np.zeros((N))
        # create interpolated time series for calculating regression model
        for k in range(N):
            # time at k
            T[k] = fd['time'][k]
            # spatially interpolate firn elevation or air content
            S1 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE].data[k,:,:].T, kx=1, ky=1)
            S2 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE].mask[k,:,:].T, kx=1, ky=1)
            # create numpy masked array of interpolated values
            FIRN[:,k] = S1.ev(ix[ind],iy[ind])
            MASK[:,k] = S2.ev(ix[ind],iy[ind])
        # calculate regression model
        for n,v in enumerate(ind):
            interp_data.data[v] = regress_model(T, FIRN[n,:], tdec[v], ORDER=2,
                CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[0])
        # mask any invalid points
        interp_data.mask[ind] = np.any(MASK, axis=1)
        # set interpolation type (2: extrapolated backward)
        interp_data.interpolation[ind] = 2

    # check if needing to extrapolate forward in time
    count = np.count_nonzero((tdec > time_cutoff[1]) & valid)
    if (count > 0):
        # indices of dates after firn model
        ind, = np.nonzero((tdec > time_cutoff[1]) & valid)
        # calculate a regression model for calculating values
        # read last 10 years of data to create regression model
        N = 365
        # spatially interpolate firn elevation or air content to coordinates
        FIRN = np.zeros((count,N))
        MASK = np.zeros((count,N),dtype=bool)
        T = np.zeros((N))
        # spatially interpolate mask to coordinates
        mspl = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
            fd['mask'].T, kx=1, ky=1)
        interp_data.mask[ind] = mspl.ev(ix[ind],iy[ind]).astype(bool)
        # create interpolated time series for calculating regression model
        for k in range(N):
            kk = nt - N + k
            # time at k
            T[k] = fd['time'][kk]
            # spatially interpolate firn elevation or air content
            S1 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE].data[kk,:,:].T, kx=1, ky=1)
            S2 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs[VARIABLE].mask[kk,:,:].T, kx=1, ky=1)
            # create numpy masked array of interpolated values
            FIRN[:,k] = S1.ev(ix[ind],iy[ind])
            MASK[:,k] = S2.ev(ix[ind],iy[ind])
        # calculate regression model
        for n,v in enumerate(ind):
            interp_data.data[v] = regress_model(T, FIRN[n,:], tdec[v], ORDER=2,
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

# PURPOSE: calculate rotated pole coordinates
def rotate_coordinates(lon, lat, rot_lon, rot_lat):
    # convert from degrees to radians
    phi = np.pi*lon/180.0
    phi_r = np.pi*rot_lon/180.0
    th = np.pi*lat/180.0
    th_r = np.pi*rot_lat/180.0
    # calculate rotation parameters
    R1 = np.sin(phi - phi_r)*np.cos(th)
    R2 = np.cos(th_r)*np.sin(th) - np.sin(th_r)*np.cos(th)*np.cos(phi - phi_r)
    R3 = -np.sin(th_r)*np.sin(th) - np.cos(th_r)*np.cos(th)*np.cos(phi - phi_r)
    # rotated pole longitude and latitude of input model
    # convert back into degrees
    Xr = np.arctan2(R1,R2)*180.0/np.pi
    Yr = np.arcsin(R3)*180.0/np.pi
    # return the rotated coordinates
    return (Xr,Yr)
