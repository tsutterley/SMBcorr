#!/usr/bin/env python
u"""
racmo_downscaled_mean.py
Written by Tyler Sutterley (07/2019)
Calculates the temporal mean of downscaled RACMO surface mass balance products

COMMAND LINE OPTIONS:
    --help: list the command line options
    --directory=X: set the base data directory
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
    -G, --gzip: Input netCDF data files are compressed (*.gz)
    -M X, --mode=X: Permission mode of directories and files created
    -V, --verbose: Verbose output of netCDF4 variables

PROGRAM DEPENDENCIES:
    convert_calendar_decimal.py: converts from calendar dates to decimal years

UPDATE HISTORY:
    Forked 09/2019 from downscaled_mean_netcdf.py
    Updated 07/2019: added version 3.0 (RACMO2.3p2 for 1958-2018 from FGRN055)
    Updated 06/2018: using python3 compatible octal and input
    Updated 11/2017: added version 2.0 (RACMO2.3p2 for 1958-2016)
    Updated 02/2017: using getopt to set base directory
    Written 11/2016
"""
from __future__ import print_function

import sys
import os
import re
import uuid
import gzip
import getopt
import netCDF4
import numpy as np
from datetime import date
from SMBcorr.convert_calendar_decimal import convert_calendar_decimal

#-- data product longnames
longname = {}
longname['SMB'] = 'Surface Mass Balance'
longname['PRECIP'] = 'Precipitation'
longname['RUNOFF'] = 'Runoff'
longname['SNOWMELT'] = 'Snowmelt'
longname['REFREEZE'] = 'Melt Water Refreeze'
#-- netcdf variable names
input_products = {}
input_products['SMB'] = 'SMB_rec'
input_products['PRECIP'] = 'precip'
input_products['RUNOFF'] = 'runoff'
input_products['SNOWMELT'] = 'snowmelt'
input_products['REFREEZE'] = 'refreeze'

#-- PURPOSE: get the dimensions for the input data matrices
def get_dimensions(input_dir,VERSION,PRODUCT,GZIP=False):
    #-- names within netCDF4 files
    VARIABLE = input_products[PRODUCT]
    #-- variable of interest
    if PRODUCT in ('SMB','PRECIP') and (VERSION == '2.0'):
        VARNAME = VARIABLE
    else:
        VARNAME = '{0}corr'.format(VARIABLE)
    #-- if reading yearly files or compressed files
    if (VERSION == '1.0'):
        #-- find input files
        pattern = '{0}.(\d+).BN_\d+_\d+_1km.MM.nc'.format(VARIABLE)
        rx = re.compile(pattern, re.VERBOSE)
        infiles = sorted([f for f in os.listdir(input_dir) if rx.match(f)])
        nt = 12*len(infiles)
        #-- read netCDF file for dataset (could also set memory=None)
        fileID = netCDF4.Dataset(os.path.join(input_dir,infiles[0]), mode='r')
        #-- shape of the input data matrix
        nm,ny,nx = fileID.variables[VARIABLE].shape
        fileID.close()
    elif VERSION in ('2.0','3.0'):
        #-- if reading bytes from compressed file or netcdf file directly
        gz = '.gz' if GZIP else ''
        #-- input dataset for variable
        file_format = {}
        file_format['2.0'] = '{0}.1958-2016.BN_RACMO2.3p2_FGRN11_GrIS.MM.nc{1}'
        file_format['3.0'] = '{0}.1958-2016.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc{1}'
        f = file_format[VERSION].format(VARIABLE.lower(),gz)
        if GZIP:
            #-- read bytes from compressed file
            fd = gzip.open(os.path.join(input_dir,f),'rb')
            #-- read netCDF file for dataset from bytes
            fileID = netCDF4.Dataset(uuid.uuid4().hex,mode='r',memory=fd.read())
        else:
            #-- read netCDF file for dataset (could also set memory=None)
            fileID = netCDF4.Dataset(os.path.join(input_dir,f), mode='r')
        #-- shape of the input data matrix
        nt,ny,nx = fileID.variables[VARNAME].shape
        fd.close() if GZIP else fileID.close()
    #-- return the data dimensions
    return (nt,ny,nx)

#-- PURPOSE: read individual yearly netcdf files and calculate mean over period
def yearly_file_mean(input_dir,VERSION,RACMO_MODEL,PRODUCT,START,END,GZIP):
    #-- names within netCDF4 files
    VARIABLE = input_products[PRODUCT]
    #-- find input files for years of interest
    regex_years = '|'.join('{0:4d}'.format(Y) for Y in range(START,END+1))
    pattern = '{0}.({1}).BN_\d+_\d+_1km.MM.nc'.format(VARIABLE,regex_years)
    rx = re.compile(pattern, re.VERBOSE)
    input_files = sorted([fi for fi in os.listdir(input_dir) if rx.match(fi)])
    #-- number of input files
    n_files = len(input_files)
    #-- input dimensions and counter variable
    #-- get dimensions for input VERSION
    nt,ny,nx = get_dimensions(input_dir,VERSION,PRODUCT,GZIP=GZIP)
    #-- create counter variable
    c = 0
    #-- allocate for all data
    dinput = {}
    dinput['LON'] = np.zeros((ny,nx))
    dinput['LAT'] = np.zeros((ny,nx))
    dinput['x'] = np.zeros((nx))
    dinput['y'] = np.zeros((ny))
    dinput['MASK'] = np.zeros((ny,nx),dtype=np.int8)
    #-- calculate total
    dinput[VARIABLE] = np.zeros((ny,nx))
    #-- calendar year and month
    year = np.zeros((nt))
    month = np.zeros((nt))

    #-- for each file of interest
    for t in range(n_files):
        #-- Open the NetCDF file for reading
        fileID = netCDF4.Dataset(os.path.join(input_dir,input_files[t]), 'r')
        #-- Getting the data from each netCDF variable
        dinput['LON'][:,:] = fileID.variables['LON'][:,:].copy()
        dinput['LAT'][:,:] = fileID.variables['LAT'][:,:].copy()
        dinput['x'][:] = fileID.variables['x'][:].copy()
        dinput['y'][:] = fileID.variables['y'][:].copy()
        dinput['MASK'][:,:] = fileID.variables['icemask'][:,:].astype(np.int8)
        #-- get year from file
        year[c], = np.array(rx.findall(input_files[t]),dtype=np.float)
        #-- for each month
        for m in range(12):
            #-- calendar month
            month[c] = np.float(m+1)
            #-- read product of interest and add to total
            dinput[VARIABLE] += fileID.variables[VARIABLE][m,:,:].copy()
            #-- add to counter
            c += 1
        #-- close the NetCDF file
        fileID.close()

    #-- calculate mean time over period
    dinput['TIME'] = np.mean(convert_calendar_decimal(year, month))
    #-- convert from total to mean
    dinput[VARIABLE] /= np.float(c)

    #-- return the mean variables
    return dinput

#-- PURPOSE: read compressed netCDF4 files and calculate mean over period
def compressed_file_mean(input_dir,VERSION,RACMO_MODEL,PRODUCT,START,END,GZIP):
    #-- names within netCDF4 files
    VARIABLE = input_products[PRODUCT]
    #-- variable of interest
    if (PRODUCT == 'SMB') or ((PRODUCT == 'PRECIP') and (VERSION == '2.0')):
        VARNAME = VARIABLE
    else:
        VARNAME = '{0}corr'.format(VARIABLE)

    #-- if reading bytes from compressed file or netcdf file directly
    gz = '.gz' if GZIP else ''
    #-- allocate for all data
    dinput = {}

    #-- input area file with ice mask and model topography
    f1 = 'Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc{0}'.format(gz)
    if GZIP:
        #-- read bytes from compressed file
        fd = gzip.open(os.path.join(input_dir,f1),'rb')
        #-- read netCDF file for topography and ice classes from bytes
        fileID = netCDF4.Dataset(uuid.uuid4().hex, mode='r', memory=fd.read())
    else:
        #-- read netCDF file for topography and ice classes
        fileID = netCDF4.Dataset(os.path.join(input_dir,f1), mode='r')
    #-- Getting the data from each netCDF variable
    dinput['LON'] = np.array(fileID.variables['LON'][:,:])
    dinput['LAT'] = np.array(fileID.variables['LAT'][:,:])
    dinput['x'] = np.array(fileID.variables['x'][:])
    dinput['y'] = np.array(fileID.variables['y'][:])
    promicemask = np.array(fileID.variables['Promicemask'][:,:])
    topography = np.array(fileID.variables['Topography'][:,:])
    #-- close the compressed file objects
    fd.close() if GZIP else fileID.close()

    #-- file format for each version
    file_format = {}
    file_format['2.0'] = '{0}.1958-2016.BN_RACMO2.3p2_FGRN11_GrIS.MM.nc{1}'
    file_format['3.0'] = '{0}.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc{1}'

    #-- input dataset for variable
    f2 = file_format[VERSION].format(VARIABLE.lower(),gz)
    if GZIP:
        #-- read bytes from compressed file
        fd = gzip.open(os.path.join(input_dir,f2),'rb')
        #-- read netCDF file for dataset from bytes
        fileID = netCDF4.Dataset(uuid.uuid4().hex, mode='r', memory=fd.read())
    else:
        #-- read netCDF file for dataset (could also set memory=None)
        fileID = netCDF4.Dataset(os.path.join(input_dir,f2), mode='r')
    #-- shape of the input data matrix
    nt,ny,nx = fileID.variables[VARNAME].shape

    #-- find ice sheet points from promicemask that valid
    ii,jj = np.nonzero((promicemask >= 1) & (promicemask <= 3))
    dinput['MASK'] = np.zeros((ny,nx),dtype=np.int8)
    dinput['MASK'][ii,jj] = 1

    #-- calculate dates
    #-- Months since 1958-01-15 at 00:00:00
    itime = np.array(fileID.variables['time'][:])
    year = np.zeros((nt))
    month = np.zeros((nt))
    for t in range(nt):
        #-- divide t by 12 to get the year
        year[t] = 1958 + np.floor(t/12.0)
        #-- use the modulus operator to get the month
        month[t] = (t % 12) + 1
    #-- convert to decimal format (using mid-month values)
    tdec = convert_calendar_decimal(year, month)

    #-- calculate total
    dinput[VARNAME] = np.zeros((ny,nx))
    #-- find indices for dates of interest
    indices, = np.nonzero((tdec >= START) & (tdec < END+1))
    c = np.count_nonzero((tdec >= START) & (tdec < END+1))
    for t in indices:
        #-- read product of interest and add to total
        dinput[VARNAME] += fileID.variables[VARNAME][t,:,:].copy()
    #-- convert from total to mean
    dinput[VARNAME] /= np.float(c),
    #-- calculate mean time over period
    dinput['TIME'] = np.mean(tdec)

    #-- close the compressed file objects
    fd.close() if GZIP else fileID.close()
    #-- return the mean variables
    return dinput

#-- PURPOSE: write RACMO downscaled data to netCDF4
def ncdf_racmo(dinput, FILENAME=None, UNITS=None, LONGNAME=None, VARNAME=None,
    LONNAME=None, LATNAME=None, XNAME=None, YNAME=None, TIMENAME=None,
    MASKNAME=None, TIME_UNITS='years', TIME_LONGNAME='Date_in_Decimal_Years',
    TITLE = None, CLOBBER = False, VERBOSE=False):

    #-- setting NetCDF clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'a'

    #-- opening NetCDF file for writing
    #-- Create the NetCDF file
    fileID = netCDF4.Dataset(FILENAME, clobber, format="NETCDF4")

    #-- Dimensions of parameters
    n_time = 1 if (np.ndim(dinput[TIMENAME]) == 0) else len(dinput[TIMENAME])
    #-- Defining the NetCDF dimensions
    fileID.createDimension(XNAME, len(dinput[XNAME]))
    fileID.createDimension(YNAME, len(dinput[YNAME]))
    fileID.createDimension(TIMENAME, n_time)

    #-- python dictionary with netCDF4 variables
    nc = {}

    #-- defining the NetCDF variables
    nc[XNAME] = fileID.createVariable(XNAME, dinput[XNAME].dtype, (XNAME,))
    nc[YNAME] = fileID.createVariable(YNAME, dinput[YNAME].dtype, (YNAME,))
    nc[TIMENAME] = fileID.createVariable(TIMENAME, dinput[TIMENAME].dtype,
        (TIMENAME,))
    nc[LONNAME] = fileID.createVariable(LONNAME, dinput[LONNAME].dtype,
        (YNAME,XNAME,))
    nc[LATNAME] = fileID.createVariable(LATNAME, dinput[LATNAME].dtype,
        (YNAME,XNAME,))
    nc[MASKNAME] = fileID.createVariable(MASKNAME, dinput[MASKNAME].dtype,
        (YNAME,XNAME,), fill_value=0, zlib=True)
    if (n_time > 1):
        nc[VARNAME] = fileID.createVariable(VARNAME, dinput[VARNAME].dtype,
            (TIMENAME,YNAME,XNAME,), zlib=True)
    else:
        nc[VARNAME] = fileID.createVariable(VARNAME, dinput[VARNAME].dtype,
            (YNAME,XNAME,), zlib=True)

    #-- filling NetCDF variables
    for key,val in dinput.items():
        nc[key][:] = val.copy()

    #-- Defining attributes for longitude and latitude
    nc[LONNAME].long_name = 'longitude'
    nc[LONNAME].units = 'degrees_east'
    nc[LATNAME].long_name = 'latitude'
    nc[LATNAME].units = 'degrees_north'
    #-- Defining attributes for x and y coordinates
    nc[XNAME].long_name = 'easting'
    nc[XNAME].units = 'meters'
    nc[YNAME].long_name = 'northing'
    nc[YNAME].units = 'meters'
    #-- Defining attributes for dataset
    nc[VARNAME].long_name = LONGNAME
    nc[VARNAME].units = UNITS
    nc[MASKNAME].long_name = 'mask'
    #-- Defining attributes for date
    nc[TIMENAME].long_name = TIME_LONGNAME
    nc[TIMENAME].units = TIME_UNITS
    #-- global variable of NetCDF file
    fileID.TITLE = TITLE
    fileID.date_created = date.isoformat(date.today())

    #-- Output NetCDF structure information
    if VERBOSE:
        print(FILENAME)
        print(list(fileID.variables.keys()))

    #-- Closing the NetCDF file
    fileID.close()

#-- PURPOSE: calculate RACMO mean data over a polar stereographic grid
def racmo_downscaled_mean(base_dir, VERSION, PRODUCT, RANGE=[1961,1990],
    GZIP=False, VERBOSE=False, MODE=0o775):

    #-- Full Directory Setup
    DIRECTORY = 'SMB1km_v{0}'.format(VERSION)

    #-- version 1 was in separate files for each year
    if (VERSION == '1.0'):
        RACMO_MODEL = ['XGRN11','2.3']
        VARNAME = input_products[PRODUCT]
        SUBDIRECTORY = '{0}_v{1}'.format(VARNAME,VERSION)
        input_dir = os.path.join(base_dir, 'RACMO', DIRECTORY, SUBDIRECTORY)
        dinput = yearly_file_mean(input_dir, VERSION, RACMO_MODEL, PRODUCT,
            RANGE[0], RANGE[1], GZIP)
    elif (VERSION == '2.0'):
        RACMO_MODEL = ['XGRN11','2.3p2']
        var = input_products[PRODUCT]
        VARNAME = var if PRODUCT in ('SMB','PRECIP') else '{0}corr'.format(var)
        input_dir = os.path.join(base_dir, 'RACMO', DIRECTORY)
        dinput = compressed_file_mean(input_dir, VERSION, RACMO_MODEL, PRODUCT,
            RANGE[0], RANGE[1], GZIP)
    elif (VERSION == '3.0'):
        RACMO_MODEL = ['FGRN055','2.3p2']
        var = input_products[PRODUCT]
        VARNAME = var if (PRODUCT == 'SMB') else '{0}corr'.format(var)
        input_dir = os.path.join(base_dir, 'RACMO', DIRECTORY)
        dinput = compressed_file_mean(input_dir, VERSION, RACMO_MODEL, PRODUCT,
            RANGE[0], RANGE[1], GZIP)

    #-- output mean as netCDF4 file
    arg = (RACMO_MODEL[0],RACMO_MODEL[1],VERSION,PRODUCT,RANGE[0],RANGE[1])
    mean_file = '{0}_RACMO{1}_DS1km_v{2}_{3}_Mean_{4:4d}-{5:4d}.nc'.format(*arg)
    ncdf_racmo(dinput, FILENAME=os.path.join(input_dir,mean_file), UNITS='mmWE',
        LONGNAME=longname[PRODUCT], VARNAME=VARNAME, LONNAME='LON',
        LATNAME='LAT', XNAME='x', YNAME='y', TIMENAME='TIME', MASKNAME='MASK',
        TITLE='Mean_downscaled_field', CLOBBER=True, VERBOSE=VERBOSE)
    #-- change the permission mode
    os.chmod(os.path.join(input_dir,mean_file), MODE)

#-- PURPOSE: help module to describe the optional input parameters
def usage():
    print('\nHelp: {0}'.format(os.path.basename(sys.argv[0])))
    print(' -D X, --directory=X\tSet the base data directory')
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
    print(' -G, --gzip\t\tInput netCDF data files are compressed (*.gz)')
    print(' -M X, --mode=X\t\tPermission mode of directories and files created')
    print(' -V, --verbose\t\tVerbose output of netCDF4 variables\n')

#-- This is the main part of the program that calls the individual modules
def main():
    #-- Read the system arguments listed after the program and run the analyses
    #--    with the specific parameters.
    long_options = ['help','directory=','version=','product=','mean=','gzip',
        'verbose','mode=']
    optlist, input_files=getopt.getopt(sys.argv[1:],'hD:GVM:',long_options)

    #-- command line parameters
    base_dir = os.getcwd()
    #-- Downscaled version
    VERSION = '3.0'
    #-- Product to calculate cumulative
    PRODUCTS = ['SMB']
    #-- mean range
    RANGE = [1961,1990]
    GZIP = False
    VERBOSE = False
    MODE = 0o775
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
        elif opt in ("-G","--gzip"):
            GZIP = True
        elif opt in ("-V","--verbose"):
            VERBOSE = True
        elif opt in ("-M","--mode"):
            MODE = int(arg,8)

    #-- for each product
    for p in PRODUCTS:
        #-- check that product was entered correctly
        if p not in longname.keys():
            raise IOError('{0} not in valid RACMO products'.format(p))
        #-- run downscaled mean program with parameters
        racmo_downscaled_mean(base_dir, VERSION, p, RANGE=RANGE, GZIP=GZIP,
            VERBOSE=VERBOSE, MODE=MODE)

#-- run main program
if __name__ == '__main__':
    main()
