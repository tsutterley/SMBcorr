#!/usr/bin/env python
u"""
merra_smb_cumulative.py
Written by Tyler Sutterley (01/2017)

Calculates cumulative anomalies of MERRA-2 surface mass balance products

INPUTS:
    SMB: Surface Mass Balance
    PRECIP: Total Precipitation
    RUNOFF: Meltwater Runoff

COMMAND LINE OPTIONS:
    --help: list the command line options
    --directory: working data directory
    --mean: Start and end year of mean (separated by commas)
        Greenland climatology to match RCMs as close as possible: 1980 - 1990
        Antarctic climatology to match RCMs as close as possible: 1980 - 2008
    -M X, --mode=X: Local permissions mode of the directories and files

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python (https://numpy.org)
    netCDF4: Python interface to the netCDF C library
         (https://unidata.github.io/netcdf4-python/netCDF4/index.html)

UPDATE HISTORY:
    Updated 01/2017: can output different data products (SMB, PRECIP, RUNOFF)
    Written 11/2016
"""
from __future__ import print_function

import sys
import os
import re
import time
import getopt
import netCDF4
import numpy as np

#-- PURPOSE: read monthly MERRA-2 datasets to calculate cumulative anomalies
def merra_smb_cumulative(base_dir, PRODUCT, RANGE=None, MODE=0o775):
    #-- directory setup
    DIRECTORY = os.path.join(base_dir,'merra.dir')
    #-- setup subdirectories
    month_dir = '{0}.5.12.4'.format(PRODUCT)
    cumul_dir = '{0}.5.12.4.CUMUL.{1:d}.{2:d}'.format(PRODUCT,*RANGE)
    #-- input yearly subdirectories
    YEARS=sorted([sd for sd in os.listdir(os.path.join(DIRECTORY,month_dir))
        if re.match('\d+',sd)])

    #-- global grid parameters
    nlon = 576
    nlat = 361
    #-- regular expression operator to find input datafiles
    regex_pattern = 'MERRA2_(\d+).{0}.{1:4d}(\d+).nc'

    #-- netcdf titles for each data product
    merra_products = {}
    merra_products['SMB'] = 'MERRA-2 Surface Mass Balance'
    merra_products['PRECIP'] = 'MERRA-2 Precipitation'
    merra_products['RUNOFF'] = 'MERRA-2 Meltwater Runoff'

    #-- calculate total number of files
    nfiles = 0
    for Y in YEARS:
        #-- compile regular expression operator
        args = ('tavgM_2d_{0}_Nx'.format(PRODUCT),int(Y))
        rx = re.compile(regex_pattern.format(*args),re.VERBOSE)
        #-- find input files for PRODUCT and YEAR
        FILES = [f for f in os.listdir(os.path.join(DIRECTORY,month_dir,Y))
            if rx.match(f)]
        nfiles += len(FILES)
        #-- make cumulative subdirectory for year
        if not os.access(os.path.join(DIRECTORY,cumul_dir,Y), os.F_OK):
            os.makedirs(os.path.join(DIRECTORY,cumul_dir,Y), MODE)

    #-- counter variable
    c = 0
    #-- read input data
    DATA = np.zeros((nlon,nlat,nfiles))
    tdec = np.zeros((nfiles))
    glon = np.zeros((nlon))
    glat = np.zeros((nlat))
    N = np.zeros((nfiles),dtype=np.uint16)
    YY = np.zeros((nfiles),dtype=np.uint16)
    MM = np.zeros((nfiles),dtype=np.uint8)
    for Y in YEARS:
        #-- compile regular expression operator
        args = ('tavgM_2d_{0}_Nx'.format(PRODUCT),int(Y))
        rx = re.compile(regex_pattern.format(*args),re.VERBOSE)
        #-- find input files for PRODUCT and YEAR
        FILES = [f for f in os.listdir(os.path.join(DIRECTORY,month_dir,Y))
            if rx.match(f)]
        #-- for each input file
        for fi in sorted(FILES):
            #-- extract month from input file
            YY[c] = int(Y)
            N[c],MM[c] = np.array(rx.findall(fi).pop(),dtype=np.int)
            #-- read input data file
            dinput = ncdf_read(os.path.join(DIRECTORY,month_dir,Y,fi),
                DATE='Y', VARNAME=PRODUCT, MISSING='Y')
            DATA[:,:,c] = dinput['data']
            glon[:] = dinput['lon']
            glat[:] = dinput['lat']
            tdec[c] = dinput['time']
            fill_value = dinput['attributes']['_FillValue']
            #-- add 1 to counter
            c += 1

    #-- check if MERRA-2 mean balance flux file exists
    args = (PRODUCT, RANGE[0], RANGE[1])
    FILE = 'MERRA2.tavgM_2d_{0}_mean_Nx.{1:4d}-{2:4d}.nc'.format(*args)
    merra_mean_file = os.path.join(DIRECTORY,month_dir,FILE)
    if not os.access(merra_mean_file, os.F_OK):
        #-- calculate mean period for MERRA-2 and save mean balance flux to file
        tt, = np.nonzero((tdec >= RANGE[0]) & (tdec < (RANGE[1]+1)))
        print('Mean Period: {0:6.2f}-{1:6.2f}'.format(tdec[tt][0],tdec[tt][-1]))
        MEAN = np.mean(DATA[:,:,tt], axis=2)
        ncdf_write(np.transpose(MEAN), glon, glat, len(tt),
            FILL_VALUE=fill_value, FILENAME=merra_mean_file, VARNAME=PRODUCT,
            LONGNAME='Equivalent Water Thickness', UNITS='mm w.e.',
            TITLE=merra_products[PRODUCT], CLOBBER='Y')
        os.chmod(merra_mean_file, MODE)
    else:
        #-- read mean balance flux
        MEAN = ncdf_read(merra_mean_file, VARNAME=PRODUCT)['data']

    #-- cumulative mass anomalies calculated by removing mean balance flux
    CUMULATIVE = np.full((nlon,nlat),fill_value)
    #-- set valid indices to 0
    ii,jj = np.nonzero(DATA[:,:,0] != fill_value)
    CUMULATIVE[ii,jj] = 0.0

    #-- Writing output cumulative anomalies to netcdf file
    for t in range(nfiles):
        #-- calculating cumulative anomalies for time t
        CUMULATIVE[ii,jj] += (DATA[ii,jj,t] - MEAN[ii,jj])
        #-- output MERRA-2 data file for cumulative data
        args = (N[t],PRODUCT,YY[t],MM[t])
        FILE='MERRA2_{0:d}.tavgM_2d_{1}_cumul_Nx.{2:4d}{3:02d}.nc'.format(*args)
        merra_data_file = os.path.join(DIRECTORY,cumul_dir,str(YY[t]),FILE)
        ncdf_write(np.transpose(CUMULATIVE), glon, glat, tdec[t],
            FILL_VALUE=fill_value, FILENAME=merra_data_file, VARNAME=PRODUCT,
            LONGNAME='Equivalent Water Thickness', UNITS='mm w.e.',
            TITLE=merra_products[PRODUCT], CLOBBER='Y')
        os.chmod(merra_data_file, MODE)

#-- PURPOSE: reads COARDS-compliant NetCDF4 files
def ncdf_read(filename, DATE='N', MISSING='N', VERBOSE='N', VARNAME='z',
    LONNAME='lon', LATNAME='lat', TIMENAME='time', ATTRIBUTES='Y', TITLE='Y'):
    #-- Open the NetCDF file for reading
    fileID = netCDF4.Dataset(filename, 'r')
    #-- create python dictionary for output variables
    dinput = {}

    #-- Output NetCDF file information
    if VERBOSE in ('Y','y'):
        print(fileID.filepath())
        print(list(fileID.variables.keys()))

    #-- netcdf variable names
    NAMES = {}
    NAMES['lon'] = LONNAME
    NAMES['lat'] = LATNAME
    NAMES['data'] = VARNAME
    if DATE in ('Y','y'):
        NAMES['time'] = TIMENAME
    #-- for each variable
    for key in NAMES.keys():
        #-- Getting the data from each NetCDF variable
        #-- filling numpy arrays with NetCDF objects
        nc_variable = fileID.variables[NAMES[key]][:].copy()
        dinput[key] = np.asarray(nc_variable).squeeze()

    #-- switching data array to lon/lat if lat/lon
    sz = dinput['data'].shape
    if (np.ndim(dinput['data']) == 2) and (len(dinput['lat']) == sz[0]):
        dinput['data'] = np.transpose(dinput['data'])

    #-- getting attributes of included variables
    dinput['attributes'] = {}
    if ATTRIBUTES in ('Y','y'):
        #-- create python dictionary for variable attributes
        attributes = {}
        #-- for each variable
        #-- get attributes for the included variables
        for key in NAMES.keys():
            attributes[key] = [fileID.variables[NAMES[key]].units, \
                fileID.variables[NAMES[key]].long_name]
        #-- put attributes in output python dictionary
        dinput['attributes'] = attributes
    #-- missing data fill value
    if MISSING in ('Y','y'):
        dinput['attributes']['_FillValue']=fileID.variables[VARNAME]._FillValue
    #-- Global attribute (title of dataset)
    if TITLE in ('Y','y'):
        rx = re.compile('TITLE',re.IGNORECASE)
        title, = [st for st in dir(fileID) if rx.match(st)]
        dinput['attributes']['title'] = getattr(fileID, title)

    #-- Closing the NetCDF file
    fileID.close()
    #-- return the output variable
    return dinput

#-- PURPOSE: writes COARDS-compliant NetCDF4 files
def ncdf_write(data, lon, lat, tim, FILENAME='sigma.H5',
    UNITS='cmH2O', LONGNAME='Equivalent_Water_Thickness',
    LONNAME='lon', LATNAME='lat', VARNAME='z', TIMENAME='time',
    TIME_UNITS='years', TIME_LONGNAME='Date_in_Decimal_Years',
    FILL_VALUE=None, TITLE = 'Spatial_Data', CLOBBER='Y', VERBOSE='N'):

    #-- setting NetCDF clobber attribute
    if CLOBBER in ('Y','y'):
        clobber = 'w'
    else:
        clobber = 'a'

    #-- opening NetCDF file for writing
    #-- Create the NetCDF file
    fileID = netCDF4.Dataset(FILENAME, clobber, format="NETCDF4")

    #-- Dimensions of parameters
    n_time = 1 if (np.ndim(tim) == 0) else len(tim)
    #-- Defining the NetCDF dimensions
    fileID.createDimension(LONNAME, len(lon))
    fileID.createDimension(LATNAME, len(lat))
    fileID.createDimension(TIMENAME, n_time)

    #-- defining the NetCDF variables
    nc = {}
    #-- lat and lon
    nc[LONNAME] = fileID.createVariable(LONNAME, lon.dtype, (LONNAME,))
    nc[LATNAME] = fileID.createVariable(LATNAME, lat.dtype, (LATNAME,))
    #-- spatial data
    if (n_time > 1):
        nc[VARNAME] = fileID.createVariable(VARNAME, data.dtype,
            (LATNAME,LONNAME,TIMENAME,), fill_value=FILL_VALUE, zlib=True)
    else:
        nc[VARNAME] = fileID.createVariable(VARNAME, data.dtype,
            (LATNAME,LONNAME,), fill_value=FILL_VALUE, zlib=True)
    #-- time (in decimal form)
    nc[TIMENAME] = fileID.createVariable(TIMENAME, 'f8', (TIMENAME,))

    #-- filling NetCDF variables
    nc[LONNAME][:] = lon
    nc[LATNAME][:] = lat
    if (n_time > 1):
        nc[VARNAME][:,:,:] = data
    else:
        nc[VARNAME][:,:] = data
    nc[TIMENAME][:] = tim

    #-- Defining attributes for longitude and latitude
    nc[LONNAME].long_name = 'longitude'
    nc[LONNAME].units = 'degrees_east'
    nc[LATNAME].long_name = 'latitude'
    nc[LATNAME].units = 'degrees_north'
    #-- Defining attributes for dataset
    nc[VARNAME].long_name = LONGNAME
    nc[VARNAME].units = UNITS
    #-- Defining attributes for date
    nc[TIMENAME].long_name = TIME_LONGNAME
    nc[TIMENAME].units = TIME_UNITS
    #-- global variable of NetCDF file
    fileID.TITLE = TITLE
    fileID.date_created = time.strftime('%Y-%m-%d',time.localtime())

    #-- Output NetCDF structure information
    if VERBOSE in ('Y','y'):
        print(FILENAME)
        print(list(fileID.variables.keys()))

    #-- Closing the NetCDF file
    fileID.close()

#-- PURPOSE: help module to describe the optional input parameters
def usage():
    print('\nHelp: {}'.format(os.path.basename(sys.argv[0])))
    print(' --directory=X\t\tWorking data directory')
    print(' --mean=X\t\tDate Range for climatology')
    print(' -M X, --mode=X\t\tPermission mode of directories and files\n')

#-- Main program that calls merra_smb_cumulative()
def main():
    #-- Read the system arguments listed after the program
    long_options = ['help','directory=','mean=','mode=']
    optlist,arglist = getopt.getopt(sys.argv[1:],'hM:',long_options)

    #-- command line parameters
    base_dir = os.getcwd()
    RANGE = [1980,1990]
    #-- permissions mode of the local directories and files (number in octal)
    MODE = 0o775
    for opt, arg in optlist:
        if opt in ('-h','--help'):
            usage()
            sys.exit()
        elif opt in ("--directory"):
            base_dir = os.path.expanduser(arg)
        elif opt in ("-C","--climatology"):
            RANGE = np.array(arg.split(','), dtype=np.int)
        elif opt in ("-M","--mode"):
            MODE = int(arg, 8)

    #-- enter MERRA-2 Product as system argument
    if not arglist:
        raise Exception('No System Arguments Listed')

    #-- run program with parameters
    for PRODUCT in arglist:
        merra_smb_cumulative(base_dir, PRODUCT, RANGE=RANGE, MODE=MODE)

#-- run main program
if __name__ == '__main__':
    main()
