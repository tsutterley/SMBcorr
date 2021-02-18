#!/usr/bin/env python
u"""
racmo_interp_daily.py
Written by Tyler Sutterley (02/2023)
Interpolates and extrapolates daily RACMO products to times and coordinates

INPUTS:
    base_dir: working data directory
    EPSG: projection of input spatial coordinates
    MODEL: daily model outputs to interpolate
        FGRN055: 5.5km Greenland RACMO2.3p2
        FGRN11: 11km Greenland RACMO2.3p2
        XANT27: 27km Antarctic RACMO2.3p2
        ASE055: 5.5km Amundsen Sea Embayment RACMO2.3p2
        XPEN055: 5.5km Antarctic Peninsula RACMO2.3p2
    tdec: dates to interpolate in year-decimal
    X: x-coordinates to interpolate
    Y: y-coordinates to interpolate

OPTIONS:
    VARIABLE: RACMO product to interpolate
        smb: Surface Mass Balance
        hgtsrf: Change of Surface Height
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
    time.py: utilities for calculating time operations

UPDATE HISTORY:
    Updated 02/2023: close in time extrapolations with regular grid interpolator
    Updated 08/2022: updated docstrings to numpy documentation format
    Updated 11/2021: don't attempt triangulation if large number of points
    Updated 08/2020: attempt delaunay triangulation using different options
    Updated 06/2020: set all values initially to fill_value
    Updated 05/2020: Gaussian average model fields before interpolation
        accumulate variable over all available dates
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

# PURPOSE: read and interpolate daily RACMO2.3 outputs
def interpolate_racmo_daily(base_dir, EPSG, MODEL, tdec, X, Y, VARIABLE='smb',
    SIGMA=1.5, FILL_VALUE=None, EXTRAPOLATE=False):
    """
    Reads and interpolates daily RACMO surface mass balance products

    Parameters
    ----------
    base_dir: str
        Working data directory
    EPSG: str or int
        input coordinate reference system
    MODEL: str
        Daily model outputs to interpolate

            - ``FGRN055``: 5.5km Greenland RACMO2.3p2
    tdec: float
        time coordinates to interpolate in year-decimal
    X: float
        x-coordinates to interpolate
    Y: float
        y-coordinates to interpolate
    VARIABLE: str, default 'smb'
        RACMO product to interpolate

            - ``smb``: Surface Mass Balance
            - ``hgtsrf``: Change of Surface Height
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
    # input list of files
    if (MODEL == 'FGRN055'):
        # filename and directory for input FGRN055 files
        file_pattern = 'RACMO2.3p2_FGRN055_{0}_daily_(\d+).nc'
        DIRECTORY = os.path.join(base_dir,'RACMO','GL','RACMO2.3p2_FGRN055')

    # create list of files to read
    rx = re.compile(file_pattern.format(VARIABLE,YRS),re.VERBOSE)
    input_files=sorted([f for f in os.listdir(DIRECTORY) if rx.match(f)])

    # calculate number of time steps to read
    nt = 0
    for f,FILE in enumerate(input_files):
        # Open the RACMO NetCDF file for reading
        with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
            nx = len(fileID.variables['rlon'][:])
            ny = len(fileID.variables['rlat'][:])
            nt += len(fileID.variables['time'][:])
            # invalid data value
            fv = np.float64(fileID.variables[VARIABLE]._FillValue)

    # scaling factor for converting units
    if (VARIABLE == 'hgtsrf'):
        scale_factor = 86400.0
    elif (VARIABLE == 'smb'):
        scale_factor = 1.0

    # python dictionary with file variables
    fd = {}
    fd['time'] = np.zeros((nt))
    # python dictionary with gaussian filtered variables
    gs = {}
    # calculate cumulative sum of gaussian filtered values
    cumulative = np.zeros((ny,nx))
    gs['cumulative'] = np.ma.zeros((nt,ny,nx), fill_value=fv)
    gs['cumulative'].mask = np.zeros((nt,ny,nx), dtype=bool)
    # create a counter variable for filling variables
    c = 0
    # for each file in the list
    for f,FILE in enumerate(input_files):
        # Open the RACMO NetCDF file for reading
        with netCDF4.Dataset(os.path.join(DIRECTORY,FILE), 'r') as fileID:
            # number of time variables within file
            t=len(fileID.variables['time'][:])
            fd[VARIABLE] = np.ma.zeros((t,ny,nx),fill_value=fv)
            fd[VARIABLE].mask = np.ones((t,ny,nx),dtype=bool)
            # Get data from netCDF variable and remove singleton dimensions
            tmp=np.squeeze(fileID.variables[VARIABLE][:])
            fd[VARIABLE][:] = scale_factor*tmp
            # indices of specified ice mask
            i,j = np.nonzero(tmp[0,:,:] != fv)
            fd[VARIABLE].mask[:,i,j] = False
            # combine mask object through time to create a single mask
            fd['mask']=1.0-np.any(fd[VARIABLE].mask,axis=0).astype(np.float64)
            # racmo coordinates
            fd['lon']=fileID.variables['lon'][:,:].copy()
            fd['lat']=fileID.variables['lat'][:,:].copy()
            fd['x']=fileID.variables['rlon'][:].copy()
            fd['y']=fileID.variables['rlat'][:].copy()
            # rotated pole parameters
            proj4_params=fileID.variables['rotated_pole'].proj4_params
            # extract delta time and epoch of time
            delta_time=fileID.variables['time'][:].astype(np.float64)
            date_string=fileID.variables['time'].units
        # extract epoch and units
        epoch,to_secs = SMBcorr.time.parse_date_string(date_string)
        # calculate time array in Julian days
        JD = SMBcorr.time.convert_delta_time(delta_time*to_secs, epoch1=epoch,
            epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0) + 2400000.5
        # convert from Julian days to calendar dates
        YY,MM,DD,hh,mm,ss = SMBcorr.time.convert_julian(JD)
        # calculate time in year-decimal
        fd['time'][c:c+t] = SMBcorr.time.convert_calendar_decimal(YY,MM,
            day=DD,hour=hh,minute=mm,second=ss)
        # use a gaussian filter to smooth mask
        gs['mask'] = scipy.ndimage.gaussian_filter(fd['mask'],SIGMA,
            mode='constant',cval=0)
        # indices of smoothed ice mask
        ii,jj = np.nonzero(np.ceil(gs['mask']) == 1.0)
        # use a gaussian filter to smooth each model field
        gs[VARIABLE] = np.ma.zeros((t,ny,nx), fill_value=fv)
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
            gs[VARIABLE][tt,ii,jj] = temp2[ii,jj]/gs['mask'][ii,jj]
            # replace valid values with original
            gs[VARIABLE][tt,i,j] = temp1[i,j]
            # set mask variables for time
            gs[VARIABLE].mask[tt,ii,jj] = False
            # calculate cumulative
            cumulative[ii,jj] += gs[VARIABLE][tt,ii,jj]
            gs['cumulative'].data[c+tt,ii,jj] = np.copy(cumulative[ii,jj])
            gs['cumulative'].mask[c+tt,ii,jj] = False
        # add to counter
        c += t

    # pyproj transformer for converting from input coordinates (EPSG)
    # RACMO models are rotated pole latitude and longitude
    try:
        # EPSG projection code string or int
        crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(int(EPSG)))
    except (ValueError,pyproj.exceptions.CRSError):
        # Projection SRS string
        crs1 = pyproj.CRS.from_string(EPSG)
    # coordinate reference system for RACMO model
    crs2 = pyproj.CRS.from_string(proj4_params)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # calculate projected coordinates of input coordinates
    ix,iy = transformer.transform(X, Y)

    # check that input points are within convex hull of valid model points
    gs['x'],gs['y'] = np.meshgrid(fd['x'],fd['y'])
    v,triangle = find_valid_triangulation(gs['x'][ii,jj],gs['y'][ii,jj])
    # check where points are within the complex hull of the triangulation
    if v:
        interp_points = np.concatenate((ix[:,None],iy[:,None]),axis=1)
        valid = (triangle.find_simplex(interp_points) >= 0)
    else:
        # Check ix and iy against the bounds of x and y
        valid = (ix >= fd['x'].min()) & (ix <= fd['x'].max()) & \
            (iy >= fd['y'].min()) & (iy <= fd['y'].max())

    # output interpolated arrays of model variable
    npts = len(tdec)
    interp = np.ma.zeros((npts),fill_value=fv,dtype=np.float64)
    interp.mask = np.ones((npts),dtype=bool)
    # initially set all values to fill value
    interp.data[:] = interp.fill_value
    # type designating algorithm used (1:interpolate, 2:backward, 3:forward)
    interp.interpolation = np.zeros((npts),dtype=np.uint8)

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
            (fd['time'],fd['y'],fd['x']), gs['cumulative'].data,
            bounds_error=False, fill_value=None)
        # create an interpolator for input mask
        MI = scipy.interpolate.RegularGridInterpolator(
            (fd['time'],fd['y'],fd['x']), gs['cumulative'].mask,
            bounds_error=False, fill_value=None)
        # interpolate to points
        interp.data[ind] = RGI.__call__(np.c_[tdec[ind],iy[ind],ix[ind]])
        interp.mask[ind] = MI.__call__(np.c_[tdec[ind],iy[ind],ix[ind]])
        # set interpolation type (1: interpolated)
        interp.interpolation[ind] = 1

    # time cutoff without close time interpolation
    time_cutoff = (fd['time'].min(), fd['time'].max())
    # check if needing to extrapolate backwards in time
    count = np.count_nonzero((tdec < time_cutoff[0]) & valid)
    if (count > 0) and EXTRAPOLATE:
        # indices of dates before model
        ind, = np.nonzero((tdec < time_cutoff[0]) & valid)
        # read the first year of data to create regression model
        N = 365
        # calculate a regression model for calculating values
        # spatially interpolate model variable to coordinates
        DATA = np.zeros((count,N))
        MASK = np.zeros((count,N),dtype=bool)
        TIME = np.zeros((N))
        # create interpolated time series for calculating regression model
        for k in range(N):
            # time at k
            TIME[k] = fd['time'][k]
            # spatially interpolate model variable
            S1 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs['cumulative'].data[k,:,:].T, kx=1, ky=1)
            S2 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs['cumulative'].mask[k,:,:].T, kx=1, ky=1)
            # create numpy masked array of interpolated values
            DATA[:,k] = S1.ev(ix[ind],iy[ind])
            MASK[:,k] = S2.ev(ix[ind],iy[ind])
        # calculate regression model
        for n,v in enumerate(ind):
            interp.data[v] = regress_model(TIME, DATA[n,:], tdec[v],
                ORDER=2, CYCLES=[0.25,0.5,1.0], RELATIVE=TIME[0])
        # mask any invalid points
        interp.mask[ind] = np.any(MASK, axis=1)
        # set interpolation type (2: extrapolated backward)
        interp.interpolation[ind] = 2

    # check if needing to extrapolate forward in time
    count = np.count_nonzero((tdec > time_cutoff[1]) & valid)
    if (count > 0) and EXTRAPOLATE:
        # indices of dates after model
        ind, = np.nonzero((tdec > time_cutoff[1]) & valid)
        # read the last year of data to create regression model
        N = 365
        # calculate a regression model for calculating values
        # spatially interpolate model variable to coordinates
        DATA = np.zeros((count,N))
        MASK = np.zeros((count,N),dtype=bool)
        TIME = np.zeros((N))
        # create interpolated time series for calculating regression model
        for k in range(N):
            kk = nt - N + k
            # time at kk
            TIME[k] = fd['time'][kk]
            # spatially interpolate model variable
            S1 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs['cumulative'].data[kk,:,:].T, kx=1, ky=1)
            S2 = scipy.interpolate.RectBivariateSpline(fd['x'], fd['y'],
                gs['cumulative'].mask[kk,:,:].T, kx=1, ky=1)
            # create numpy masked array of interpolated values
            DATA[:,k] = S1.ev(ix[ind],iy[ind])
            MASK[:,k] = S2.ev(ix[ind],iy[ind])
        # calculate regression model
        for n,v in enumerate(ind):
            interp.data[v] = regress_model(TIME, DATA[n,:], tdec[v],
                ORDER=2, CYCLES=[0.25,0.5,1.0], RELATIVE=TIME[-1])
        # mask any invalid points
        interp.mask[ind] = np.any(MASK, axis=1)
        # set interpolation type (3: extrapolated forward)
        interp.interpolation[ind] = 3

    # complete mask if any invalid in data
    invalid, = np.nonzero((interp.data == interp.fill_value) |
        np.isnan(interp.data))
    interp.mask[invalid] = True
    # replace fill value if specified
    if FILL_VALUE:
        interp.fill_value = FILL_VALUE
        interp.data[interp.mask] = interp.fill_value

    # return the interpolated values
    return interp
