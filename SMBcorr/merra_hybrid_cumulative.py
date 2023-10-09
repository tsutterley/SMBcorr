#!/usr/bin/env python
u"""
merra_hybrid_cumulative.py
Written by Tyler Sutterley (02/2021)
Calculates cumulative anomalies of MERRA-2 hybrid surface mass balance products
    MERRA-2 Hybrid model outputs provided by Brooke Medley at GSFC

CALLING SEQUENCE:
    python merra_hybrid_cumulative.py --directory <path> --region gris \
        --mean 1980 1995 --product p_minus_e

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -R X, --region X: Region to calculate (gris, ais)
    --mean: Start and end year of mean
    --product: MERRA-2 hybrid product to calculate
        p_minus_e: Precipitation minus Evaporation
        melt: Snowmelt
    -G, --gzip: netCDF4 file is locally gzip compressed
    -M X, --mode X: Local permissions mode of the directories and files

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    netCDF4: Python interface to the netCDF C library
         https://unidata.github.io/netcdf4-python/netCDF4/index.html

UPDATE HISTORY:
    Updated 02/2021: using argparse to set parameters
        added gzip compression option
    Written 10/2019
"""
from __future__ import print_function

import sys
import os
import re
import gzip
import time
import uuid
import netCDF4
import argparse
import numpy as np

#-- PURPOSE: read and interpolate MERRA-2 hybrid surface mass balance variables
def merra_hybrid_cumulative(base_dir, REGION, VERSION, VARIABLE='p_minus_e',
    RANGE=None, GZIP=False, MODE=0o775):

    #-- set the input netCDF4 file for the variable of interest
    suffix = '.gz' if GZIP else ''
    if VARIABLE in ('p_minus_e','melt') and (VERSION == 'v0'):
        args = (REGION.lower(),suffix)
        hybrid_file = 'm2_hybrid_p_minus_e_melt_{0}.nc{1}'.format(*args)

    #-- Open the MERRA-2 Hybrid NetCDF file for reading
    if GZIP:
        #-- read as in-memory (diskless) netCDF4 dataset
        with gzip.open(os.path.join(base_dir,hybrid_file),'r') as f:
            fileID = netCDF4.Dataset(uuid.uuid4().hex, memory=f.read())
    else:
        #-- read netCDF4 dataset
        fileID = netCDF4.Dataset(os.path.join(base_dir,hybrid_file), 'r')

    #-- Get data from each netCDF variable and remove singleton dimensions
    fd = {}
    DATA = np.squeeze(fileID.variables[VARIABLE][:].copy())
    fd['x'] = fileID.variables['x'][:,:].copy()
    fd['y'] = fileID.variables['y'][:,:].copy()
    fd['time'] = fileID.variables['time'][:].copy()
    #-- invalid data value
    fill_value = float(fileID.variables[VARIABLE]._FillValue)
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

    #-- output MERRA-2 data file for cumulative data
    FILE = 'm2_hybrid_{0}_cumul_{1}.nc'.format(VARIABLE,REGION.lower())
    #-- opening NetCDF file for writing
    fileID = netCDF4.Dataset(os.path.join(base_dir,FILE),'w',format="NETCDF4")

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

    os.chmod(os.path.join(base_dir,FILE), MODE)

#-- Main program that calls merra_hybrid_cumulative()
def main():
    #-- Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Reads MERRA-2 Hybrid datafiles to
            calculate monthly cumulative anomalies in surface
            mass balance products
            """
    )
    #-- command line parameters
    #-- working data directory
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
    #-- region of firn model
    parser.add_argument('--region','-R',
        type=str, default='gris', choices=['gris','ais'],
        help='Region of firn model to calculate')
    #-- version of firn model
    parser.add_argument('--version','-v',
        type=str, default='v1.1', choices=['v0','v1','v1.1'],
        help='Version of firn model to calculate')
    #-- firn model product
    parser.add_argument('--product','-p',
        type=str, nargs='+', default=['p_minus_e','melt'],
        choices=['p_minus_e','melt'],
        help='MERRA-2 Hybrid product')
    #-- start and end years to run for mean
    parser.add_argument('--mean','-m',
        metavar=('START','END'), type=int, nargs=2,
        default=[1980,1995],
        help='Start and end year range for mean')
    #-- netCDF4 files are gzip compressed
    parser.add_argument('--gzip','-G',
        default=False, action='store_true',
        help='netCDF4 file is locally gzip compressed')
    #-- permissions mode of the local directories and files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files')
    args = parser.parse_args()

    #-- run program for each input product
    for PRODUCT in args.product:
        merra_hybrid_cumulative(args.directory, args.region, args.version,
            VARIABLE=PRODUCT, RANGE=args.mean, GZIP=args.gzip, MODE=args.mode)

#-- run main program
if __name__ == '__main__':
    main()

