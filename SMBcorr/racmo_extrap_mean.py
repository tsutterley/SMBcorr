#!/usr/bin/env python
u"""
racmo_extrap_mean.py
Written by Tyler Sutterley (09/2019)
Interpolates and extrapolates downscaled RACMO products to times and coordinates

Uses fast nearest-neighbor search algorithms
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
and inverse distance weighted interpolation to extrapolate spatially

CALLING SEQUENCE:
    python racmo_extrap_mean.py --directory=<path> --version=3.0 \
        --product=SMB,PRECIP,RUNOFF --coordinate=[-39e4,-133e4],[-39e4,-133e4] \
        --date=2016.1,2018.1

COMMAND LINE OPTIONS:
    -D X, --directory=X: Working data directory
    --version=X: Downscaled RACMO Version
        1.0: RACMO2.3/XGRN11
        2.0: RACMO2.3p2/XGRN11
        3.0: RACMO2.3p2/FGRN055
    --product: RACMO product to calculate
        SMB: Surface Mass Balance
        PRECIP: Precipitation
        RUNOFF: Melt Water Runoff
        SNOWMELT: Snowmelt
        REFREEZE: Melt Water Refreeze
    --mean: Start and end year of mean (separated by commas)
    --coordinate=X: Polar Stereographic X and Y of point
    --date=X: Date to interpolate in year-decimal format
    --csv=X: Read dates and coordinates from a csv file
    --fill-value: Replace invalid values with fill value
        (default uses original fill values from data file)

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
    Updated 04/2020: reduced to interpolation function.  output masked array
    Updated 09/2019: read subsets of DS1km netCDF4 file to save memory
    Written 09/2019
"""
from __future__ import print_function

import sys
import os
import re
import pyproj
import getopt
import netCDF4
import numpy as np
import scipy.interpolate
from sklearn.neighbors import KDTree, BallTree

#-- PURPOSE: read and interpolate downscaled RACMO products
def extrapolate_racmo_mean(base_dir, EPSG, VERSION, PRODUCT, tdec, X, Y,
    RANGE=[], SEARCH='BallTree', NN=10, POWER=2.0, FILL_VALUE=None):

    #-- Full Directory Setup
    DIRECTORY = 'SMB1km_v{0}'.format(VERSION)

    #-- netcdf variable names
    input_products = {}
    input_products['SMB'] = 'SMB_rec'
    input_products['PRECIP'] = 'precip'
    input_products['RUNOFF'] = 'runoff'
    input_products['SNOWMELT'] = 'snowmelt'
    input_products['REFREEZE'] = 'refreeze'
    #-- version 1 was in separate files for each year
    if (VERSION == '1.0'):
        RACMO_MODEL = ['XGRN11','2.3']
        VARNAME = input_products[PRODUCT]
        SUBDIRECTORY = '{0}_v{1}'.format(VARNAME,VERSION)
        input_dir = os.path.join(base_dir, 'RACMO', DIRECTORY, SUBDIRECTORY)
    elif (VERSION == '2.0'):
        RACMO_MODEL = ['XGRN11','2.3p2']
        var = input_products[PRODUCT]
        VARNAME = var if PRODUCT in ('SMB','PRECIP') else '{0}corr'.format(var)
        input_dir = os.path.join(base_dir, 'RACMO', DIRECTORY)
    elif (VERSION == '3.0'):
        RACMO_MODEL = ['FGRN055','2.3p2']
        var = input_products[PRODUCT]
        VARNAME = var if (PRODUCT == 'SMB') else '{0}corr'.format(var)
        input_dir = os.path.join(base_dir, 'RACMO', DIRECTORY)

    #-- read mean from netCDF4 file
    arg = (RACMO_MODEL[0],RACMO_MODEL[1],VERSION,PRODUCT,RANGE[0],RANGE[1])
    mean_file = '{0}_RACMO{1}_DS1km_v{2}_{3}_Mean_{4:4d}-{5:4d}.nc'.format(*arg)
    with netCDF4.Dataset(os.path.join(input_dir,mean_file),'r') as fileID:
        MEAN = fileID[VARNAME][:,:].copy()

    #-- input cumulative netCDF4 file
    args = (RACMO_MODEL[0],RACMO_MODEL[1],VERSION,PRODUCT)
    input_file = '{0}_RACMO{1}_DS1km_v{2}_{3}_cumul.nc'.format(*args)

    #-- Open the RACMO NetCDF file for reading
    fileID = netCDF4.Dataset(os.path.join(input_dir,input_file), 'r')
    #-- input shape of RACMO data
    nt,ny,nx = fileID[VARNAME].shape
    #-- Get data from each netCDF variable
    d = {}
    #-- cell origins on the bottom right
    dx = np.abs(fileID.variables['x'][1]-fileID.variables['x'][0])
    dy = np.abs(fileID.variables['y'][1]-fileID.variables['y'][0])
    #-- latitude and longitude arrays at center of each cell
    d['LON'] = fileID.variables['LON'][:,:].copy()
    d['LAT'] = fileID.variables['LAT'][:,:].copy()
    #-- extract time (decimal years)
    d['TIME'] = fileID.variables['TIME'][:].copy()
    #-- mask object for interpolating data
    d['MASK'] = np.array(fileID.variables['MASK'][:],dtype=np.bool)
    i,j = np.nonzero(d['MASK'])
    #-- reduce mean to valid points
    var1 = MEAN[i,j]

    #-- convert RACMO latitude and longitude to input coordinates (EPSG)
    proj1 = pyproj.Proj("+init={0}".format(EPSG))
    proj2 = pyproj.Proj("+init=EPSG:{0:d}".format(4326))
    xg,yg = pyproj.transform(proj2, proj1, d['LON'], d['LAT'])

    #-- construct search tree from original points
    #-- can use either BallTree or KDTree algorithms
    xy1 = np.concatenate((xg[i,j,None],yg[i,j,None]),axis=1)
    tree = BallTree(xy1) if (SEARCH == 'BallTree') else KDTree(xy1)

    #-- output extrapolated arrays of variable
    extrap_var = np.zeros_like(tdec,dtype=np.float)
    #-- type designating algorithm used (1: interpolate, 2: backward, 3:forward)
    extrap_type = np.ones_like(tdec,dtype=np.uint8)

    #-- inverse distance weighting to extrapolate in space
    #-- query the search tree to find the NN closest points
    xy2 = np.concatenate((X[:,None],Y[:,None]),axis=1)
    dist,indices = tree.query(xy2, k=NN, return_distance=True)
    count = len(tdec)
    #-- normalized weights if POWER > 0 (typically between 1 and 3)
    #-- in the inverse distance weighting
    power_inverse_distance = dist**(-POWER)
    s = np.sum(power_inverse_distance, axis=1)
    w = power_inverse_distance/np.broadcast_to(s[:,None],(count,NN))
    #-- spatially extrapolate using inverse distance weighting
    dt = (tdec - d['TIME'][0])/(d['TIME'][1] - d['TIME'][0])
    extrap_var[:] = dt*np.sum(w*var1[indices],axis=1)

    #-- replace fill value if specified
    if FILL_VALUE:
        ind, = np.nonzero(extrap_type == 0)
        extrap_var[ind] = FILL_VALUE
        fv = FILL_VALUE
    else:
        fv = 0.0

    #-- close the NetCDF files
    fileID.close()

    #-- return the extrapolated values
    return (extrap_var,extrap_type,fv)

#-- PURPOSE: interpolate RACMO products to a set of coordinates and times
#-- wrapper function to extract EPSG and print to terminal
def racmo_extrap_mean(base_dir, VERSION, PRODUCT, RANGE=[],
    COORDINATES=None, DATES=None, CSV=None, FILL_VALUE=None):

    #-- this is the projection of the coordinates being extrapolated into
    EPSG = "EPSG:{0:d}".format(3413)

    #-- read coordinates and dates from a csv file (X,Y,year decimal)
    if CSV:
        X,Y,tdec = np.loadtxt(CSV,delimiter=',').T
    else:
        #-- regular expression pattern for extracting x and y coordinates
        numerical_regex = '([-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?)'
        regex = re.compile('\[{0},{0}\]'.format(numerical_regex))
        #-- number of coordinates
        npts = len(regex.findall(COORDINATES))
        #-- x and y coordinates of interpolation points
        X = np.zeros((npts))
        Y = np.zeros((npts))
        for i,XY in enumerate(regex.findall(COORDINATES)):
            X[i],Y[i] = np.array(XY, dtype=np.float)
        #-- convert dates to ordinal days (count of days of the Common Era)
        tdec = np.array(DATES, dtype=np.float)

    #-- read and interpolate/extrapolate RACMO2.3 products
    vi,itype,fv = extrapolate_racmo_mean(base_dir, EPSG, VERSION, PRODUCT,
        tdec, X, Y, RANGE=RANGE, FILL_VALUE=FILL_VALUE)
    interpolate_types = ['invalid','extrapolated','backward','forward']
    for v,t in zip(vi,itype):
        print(v,interpolate_types[t])

#-- PURPOSE: help module to describe the optional input parameters
def usage():
    print('\nHelp: {}'.format(os.path.basename(sys.argv[0])))
    print(' -D X, --directory=X\tWorking data directory')
    print(' --version=X\t\tDownscaled RACMO Version')
    print('\t1.0: RACMO2.3/XGRN11')
    print('\t2.0: RACMO2.3p2/XGRN11')
    print('\t3.0: RACMO2.3p2/FGRN055')
    print(' --product:\t\tRACMO product to calculate')
    print('\tSMB: Surface Mass Balance')
    print('\tPRECIP: Precipitation')
    print('\tRUNOFF: Melt Water Runoff')
    print('\tSNOWMELT: Snowmelt')
    print('\tREFREEZE: Melt Water Refreeze')
    print(' --mean:\t\tStart and end year of mean (separated by commas)')
    print(' --coordinate=X\t\tPolar Stereographic X and Y of point')
    print(' --date=X\t\tDates to interpolate in year-decimal format')
    print(' --csv=X\t\tRead dates and coordinates from a csv file')
    print(' --fill-value\t\tReplace invalid values with fill value\n')

#-- Main program that calls racmo_extrap_mean()
def main():
    #-- Read the system arguments listed after the program
    long_options = ['help','directory=','version=','product=','mean=',
        'coordinate=','date=','csv=','fill-value=']
    optlist,arglist = getopt.getopt(sys.argv[1:], 'hD:', long_options)

    #-- data directory
    base_dir = os.getcwd()
    #-- Downscaled version
    VERSION = '3.0'
    #-- Products to calculate cumulative
    PRODUCTS = ['SMB']
    #-- mean range
    RANGE = [1961,1990]
    #-- coordinates and times to run
    COORDINATES = None
    DATES = None
    #-- read coordinates and dates from csv file
    CSV = None
    #-- invalid value (default is nan)
    FILL_VALUE = np.nan
    #-- extract parameters
    for opt, arg in optlist:
        if opt in ('-h','--help'):
            usage()
            sys.exit()
        elif opt in ("-D","--directory"):
            base_dir = os.path.expanduser(arg)
        elif opt in ("--version"):
            VERSION = arg
        elif opt in ("--product"):
            PRODUCTS = arg.split(',')
        elif opt in ("--mean"):
            RANGE = np.array(arg.split(','),dtype=np.int)
        elif opt in ("--coordinate"):
            COORDINATES = arg
        elif opt in ("--date"):
            DATES = arg.split(',')
        elif opt in ("--csv"):
            CSV = os.path.expanduser(arg)
        elif opt in ("--fill-value"):
            FILL_VALUE = eval(arg)

    #-- data product longnames
    longname = {}
    longname['SMB'] = 'Cumulative Surface Mass Balance Anomalies'
    longname['PRECIP'] = 'Cumulative Precipitation Anomalies'
    longname['RUNOFF'] = 'Cumulative Runoff Anomalies'
    longname['SNOWMELT'] = 'Cumulative Snowmelt Anomalies'
    longname['REFREEZE'] = 'Cumulative Melt Water Refreeze Anomalies'

    #-- for each product
    for p in PRODUCTS:
        #-- check that product was entered correctly
        if p not in longname.keys():
            raise IOError('{0} not in valid RACMO products'.format(p))
        #-- run program with parameters
        racmo_extrap_mean(base_dir, VERSION, p, RANGE=RANGE,
            COORDINATES=COORDINATES, DATES=DATES, CSV=CSV,
            FILL_VALUE=FILL_VALUE)

#-- run main program
if __name__ == '__main__':
    main()
