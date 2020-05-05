#!/usr/bin/env python
u"""
merra_hybrid_cumulative.py
Written by Tyler Sutterley (10/2019)
Calculates cumulative anomalies of MERRA-2 hybrid surface mass balance products
    MERRA-2 Hybrid model outputs provided by Brooke Medley at GSFC

CALLING SEQUENCE:
    python merra_hybrid_cumulative.py --directory=<path> --region=gris \
        --mean=1980,1995 --product=p_minus_e

COMMAND LINE OPTIONS:
    -D X, --directory=X: Working data directory
    -R X, --region=X: Region to interpolate (gris, ais)
    --mean: Start and end year of mean (separated by commas)
    --product: MERRA-2 hybrid product to calculate
        p_minus_e: Precipitation minus Evaporation
        melt: Snowmelt
    -M X, --mode=X: Local permissions mode of the directories and files

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        http://www.numpy.org
        http://www.scipy.org/NumPy_for_Matlab_Users
    netCDF4: Python interface to the netCDF C library
         https://unidata.github.io/netcdf4-python/netCDF4/index.html

UPDATE HISTORY:
    Written 10/2019
"""
from __future__ import print_function

import sys
import os
import re
import time
import getopt
import netCDF4
import numpy as np

#-- PURPOSE: read and interpolate MERRA-2 hybrid surface mass balance variables
def merra_hybrid_cumulative(base_dir, REGION, DIRECTORY=None,
    VARIABLE='p_minus_e', RANGE=None, MODE=0o775):

    #-- set the input netCDF4 file for the variable of interest
    if VARIABLE in ('p_minus_e','melt'):
        hybrid_file = 'm2_hybrid_p_minus_e_melt_{0}.nc'.format(REGION.lower())
    #-- Open the MERRA-2 Hybrid NetCDF file for reading
    fileID = netCDF4.Dataset(os.path.join(base_dir,hybrid_file), 'r')
    #-- Get data from each netCDF variable and remove singleton dimensions
    fd = {}
    DATA = np.squeeze(fileID.variables[VARIABLE][:].copy())
    fd['x'] = fileID.variables['x'][:,:].copy()
    fd['y'] = fileID.variables['y'][:,:].copy()
    fd['time'] = fileID.variables['time'][:].copy()
    #-- invalid data value
    fill_value = np.float(fileID.variables[VARIABLE]._FillValue)
    #-- input shape of MERRA-2 Hybrid firn data
    nt,nx,ny = np.shape(DATA)
    #-- close the NetCDF files
    fileID.close()
    #-- time is year decimal at time step 5 days
    time_step = 5.0/365.25

    #-- indices of specified ice mask
    i,j = np.nonzero(DATA[0,:,:] != fill_value)
    valid_count = len(DATA[0,i,j])

    #-- calculate mean period for MERRA-2
    tt, = np.nonzero((fd['time'] >= RANGE[0]) & (fd['time'] < (RANGE[1]+1)))
    MEAN = np.mean(DATA[tt,:,:], axis=0)

    #-- cumulative mass anomalies calculated by removing mean balance flux
    fd[VARIABLE] = np.full((nt,nx,ny),fill_value)
    CUMULATIVE = np.zeros((valid_count))

    #-- Writing output cumulative anomalies to netcdf file
    for t in range(nt):
        #-- calculating cumulative anomalies for time t
        CUMULATIVE += (DATA[t,i,j] - MEAN[i,j])
        fd[VARIABLE][t,i,j] = CUMULATIVE.copy()

    #-- set directory to base directory if None
    if DIRECTORY is None:
        DIRECTORY = os.path.expanduser(base_dir)
    #-- create output directory if non-existent
    if not os.access(DIRECTORY, os.F_OK):
        os.makedirs(DIRECTORY,MODE)
    #-- output MERRA-2 data file for cumulative data
    FILE = 'm2_hybrid_{0}_cumul_{1}.nc'.format(VARIABLE,REGION.lower())
    #-- opening NetCDF file for writing
    fileID = netCDF4.Dataset(os.path.join(DIRECTORY,FILE),'w',format="NETCDF4")

    #-- Defining the NetCDF dimensions
    fileID.createDimension('x', nx)
    fileID.createDimension('y', ny)
    fileID.createDimension('time', nt)

    #-- python dictionary with netCDF4 variables
    nc = {}

    #-- defining the NetCDF variables
    nc['x'] = fileID.createVariable('x', fd['x'].dtype, ('x','y',))
    nc['y'] = fileID.createVariable('y', fd['y'].dtype, ('x','y',))
    nc['time'] = fileID.createVariable('time', fd['time'].dtype, ('time',))
    nc[VARIABLE] = fileID.createVariable(VARIABLE, fd[VARIABLE].dtype,
        ('time','x','y',), fill_value=fill_value, zlib=True)

    #-- filling NetCDF variables
    for key,val in fd.items():
        nc[key][:] = val.copy()

    #-- Defining attributes for x and y coordinates
    nc['x'].long_name = 'polar stereographic x coordinate, 12.5km resolution'
    nc['x'].units = 'meters'
    nc['y'].long_name = 'polar stereographic y coordinate, 12.5km resolution'
    nc['y'].units = 'meters'
    #-- Defining attributes for dataset
    if (VARIABLE == 'p_minus_e'):
        nc[VARIABLE].long_name = ('MERRA-2 hybrid '
            'precipitation-minus-evaporation (net accumulation)')
        nc[VARIABLE].units = 'meters of ice equivalent per year'
        nc[VARIABLE].comment = ('developed using a degree-day model from our '
            'MERRA-2 hybrid skin temperature product and MARv3.5.2 meltwater '
            'for 1980-2019')
    elif (VARIABLE == 'melt'):
        nc[VARIABLE].long_name = ('MERRA-2 meltwater, calibrated to '
            'MARv3.5.2 melt')
        nc[VARIABLE].units = 'meters of ice equivalent per year'
    #-- Defining attributes for date
    nc['time'].long_name = 'time, 5-daily resolution'
    nc['time'].units = 'decimal years, 5-daily resolution'
    #-- global variable of NetCDF file
    fileID.TITLE = ('Cumulative anomalies in MERRA-2 Hybrid variables relative '
        'to {0:4d}-{1:4d}').format(*RANGE)
    fileID.date_created = time.strftime('%Y-%m-%d',time.localtime())
    #-- Closing the NetCDF file
    fileID.close()

    os.chmod(os.path.join(DIRECTORY,FILE), MODE)

#-- PURPOSE: help module to describe the optional input parameters
def usage():
    print('\nHelp: {}'.format(os.path.basename(sys.argv[0])))
    print(' -D X, --directory=X\tWorking data directory')
    print(' -O X, --output=X\tOutput working data directory')
    print(' -R X, --region=X\tRegion of firn model to interpolate')
    print(' --mean\t\t\tStart and end year of mean (separated by commas)')
    print(' --product\t\tMERRA-2 hybrid product to calculate')
    print('\tp_minus_e: Precipitation minus Evaporation\n\tmelt: Snowmelt')
    print(' -M X, --mode=X\t\tPermission mode of directories and files\n')

#-- Main program that calls merra_hybrid_cumulative()
def main():
    #-- Read the system arguments listed after the program
    lopt = ['help','directory=','output=','region=','mean=','product=','mode=']
    optlist,arglist = getopt.getopt(sys.argv[1:], 'hD:O:R:M:', lopt)

    #-- data directory
    base_dir = os.getcwd()
    DIRECTORY = None
    #-- region of firn model
    REGION = 'gris'
    #-- surface mass balance product
    PRODUCTS = ['p_minus_e','melt']
    #-- start and end year of mean
    RANGE = [1980,1995]
    #-- permissions mode
    MODE = 0o775
    #-- extract parameters
    for opt, arg in optlist:
        if opt in ('-h','--help'):
            usage()
            sys.exit()
        elif opt in ("-D","--directory"):
            base_dir = os.path.expanduser(arg)
        elif opt in ("-O","--output"):
            DIRECTORY = os.path.expanduser(arg)
        elif opt in ("-R","--region"):
            REGION = arg.lower()
        elif opt in ("--product"):
            PRODUCTS = arg.split(',')
        elif opt in ("--mean"):
            RANGE = np.array(arg.split(','),dtype=np.int)
        elif opt in ("-M","--mode"):
            MODE = int(arg,8)

    #-- run program with parameters
    for p in PRODUCTS:
        merra_hybrid_cumulative(base_dir, REGION, DIRECTORY=DIRECTORY,
            VARIABLE=p, RANGE=RANGE, MODE=MODE)

#-- run main program
if __name__ == '__main__':
    main()
