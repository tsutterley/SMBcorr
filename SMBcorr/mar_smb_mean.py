#!/usr/bin/env python
u"""
mar_smb_mean.py
Written by Tyler Sutterley (08/2022)
Calculates the temporal mean of MAR surface mass balance products

COMMAND LINE OPTIONS:
    --help: list the command line options
    --directory X: set the full path to the MAR data directory
    --version X: MAR version to run
        v3.5.2
        v3.9
        v3.10
        v3.11
    -d, --downscaled: run downscaled MAR
    -p X, --product X: MAR product to calculate
        SMB: Surface Mass Balance
        PRECIP: Precipitation
        SNOWFALL: Snowfall
        RAINFALL: Rainfall
        RUNOFF: Melt Water Runoff
        SNOWMELT: Snowmelt
        REFREEZE: Melt Water Refreeze
        SUBLIM = Sublimation
    -m X, --mean X: Start and end year of mean
    -M X, --mode X: Permission mode of directories and files created
    -V, --verbose: Verbose output of netCDF4 variables

PROGRAM DEPENDENCIES:
    time.py: utilities for calculating time operations

UPDATE HISTORY:
    Updated 08/2022: updated docstrings to numpy documentation format
    Updated 02/2021: using argparse to set parameters
        using utilities from time module for conversions
    Written 11/2019
"""
from __future__ import print_function, division

import sys
import os
import re
import argparse
import warnings
import numpy as np
import SMBcorr.time

# attempt imports
try:
    import netCDF4
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("module")
    warnings.warn("netCDF4 not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

# data product longnames
longname = {}
longname['SMB'] = 'Surface_Mass_Balance'
longname['PRECIP'] = 'Precipitation'
longname['SNOWFALL'] = 'Snowfall'
longname['RAINFALL'] = 'Rainfall'
longname['RUNOFF'] = 'Melt_Water_Runoff'
longname['SNOWMELT'] = 'Snowmelt'
longname['REFREEZE'] = 'Melt_Water_Refreeze'
longname['SUBLIM'] = 'Sublimation'

# PURPOSE: sort input files by year
def sort_files(regex, input_files):
    """
    Sort the list of input files by date

    Parameters
    ----------
    regex: obj
        Regular expression object for matching files
    input_files: list
        Input MAR surface mass balance files
    """
    sort_indices = np.argsort([regex.match(f).group(2) for f in input_files])
    return np.array(input_files)[sort_indices]

# PURPOSE: get the dimensions for the input data matrices
def get_dimensions(directory, input_files, XNAME=None, YNAME=None):
    """
    Get the total dimensions of the input data

    Parameters
    ----------
    directory: str
        Working data directory
    input_files: list
        Input MAR surface mass balance files
    XNAME: str or NoneType, default None
        Name of the x-coordinate variable
    YNAME: str or NoneType, default None
        Name of the y-coordinate variable
    """
    # get grid dimensions from first file and 12*number of files
    # Open the NetCDF file for reading
    fileID = netCDF4.Dataset(os.path.join(directory,input_files[0]), 'r')
    nx, = fileID[XNAME].shape
    ny, = fileID[YNAME].shape
    fileID.close()
    nt = 12*len(input_files)
    return ny,nx,nt

# PURPOSE: create an output netCDF4 file for the output data fields
def create_netCDF4(OUTPUT, FILENAME=None, UNITS=None, LONGNAME=None,
    VARNAME=None, LONNAME=None, LATNAME=None, XNAME=None, YNAME=None,
    TIMENAME=None, MASKNAME=None, TITLE=None, VERBOSE=False, PROJECTION=None):
    # output netCDF4 file
    fileID = netCDF4.Dataset(FILENAME,'w',format="NETCDF4")
    nc = {}
    # Defining the netCDF dimensions
    # create each netCDF4 dimension variable
    for key in (XNAME,YNAME):
        fileID.createDimension(key, len(OUTPUT[key]))
        nc[key] = fileID.createVariable(key, 'f', (key,), zlib=True)
    fileID.createDimension(TIMENAME, 1)
    nc[TIMENAME] = fileID.createVariable(TIMENAME, 'f', (TIMENAME,), zlib=True)
    # create each netCDF4 variable
    for key,type in zip([LONNAME,LATNAME,MASKNAME],['f','f','b']):
        nc[key] = fileID.createVariable(key, type, ('y','x',), zlib=True)
    nc[VARNAME] = fileID.createVariable(VARNAME, 'f', ('y','x',),
        fill_value=OUTPUT[VARNAME].fill_value, zlib=True)
    # fill each output netCDF4 variable
    for key in (XNAME,YNAME,TIMENAME,LONNAME,LATNAME,MASKNAME,VARNAME):
        nc[key][:] = OUTPUT[key]
    # Defining attributes for each netCDF4 variable
    nc[XNAME].units = 'm'
    nc[YNAME].units = 'm'
    nc[TIMENAME].units = 'years'
    nc[TIMENAME].long_name = 'Date_in_Decimal_Years'
    nc[LONNAME].long_name = 'longitude'
    nc[LONNAME].units = 'degrees_east'
    nc[LATNAME].long_name = 'latitude'
    nc[LATNAME].units = 'degrees_north'
    nc[VARNAME].long_name = LONGNAME
    nc[VARNAME].units = UNITS
    # global variables of netCDF file
    fileID.projection = PROJECTION
    fileID.TITLE = TITLE
    # Output NetCDF structure information
    if VERBOSE:
        print(FILENAME)
        print(list(fileID.variables.keys()))
    # Closing the netCDF file
    fileID.close()

# PURPOSE: calculates mean of MAR products
def mar_smb_mean(input_dir, VERSION, PRODUCT, RANGE=[1961,1990],
    DOWNSCALED=False, VERBOSE=False, MODE=0o775):
    """
    Calculates the temporal mean of MAR surface mass balance products

    Parameters
    ----------
    input_dir: str
        Working data directory
    VERSION: str
        MAR Version

            - ``v3.5.2``
            - ``v3.9``
            - ``v3.10``
            - ``v3.11``
    PRODUCT: str
        MAR product to calculate

            - ``SMB``: Surface Mass Balance
            - ``PRECIP``: Precipitation
            - ``SNOWFALL``: Snowfall
            - ``RAINFALL``: Rainfall
            - ``RUNOFF``: Melt Water Runoff
            - ``SNOWMELT``: Snowmelt
            - ``REFREEZE``: Melt Water Refreeze
            - ``SUBLIM``: Sublimation
    RANGE: list, default [1961,1990]
        Start and end year of mean
    DOWNSCALED: bool, default False
        Run downscaled MAR products
    VERBOSE: bool, default False
        Verbose output of netCDF4 variables
    MODE: oct, default 0o775
        Permission mode of directories and files created
    """

    # regular expression pattern for MAR dataset
    regex_year = '|'.join([str(yr) for yr in range(RANGE[0],RANGE[1]+1)])
    rx = re.compile('MAR{0}-monthly-(.*?)-({1}).nc$'.format(VERSION,regex_year))
    # netCDF4 variable names (for both direct and derived products)
    input_products = {}
    # SMB from downscaled product
    if DOWNSCALED:
        # variable coordinates
        XNAME,YNAME,TIMENAME = ('x','y','time')
        # SMBcorr is topography corrected SMB for the ice covered area
        # SMB2 is the SMB for the tundra covered area
        input_products['SMB'] = ['SMBcorr','SMB2']
        # RU from downscaled product
        # RUcorr is topography corrected runoff for the ice covered area
        # RU2corr is topography corrected runoff for the tundra covered area
        input_products['RUNOFF'] = ['RUcorr','RU2corr']
        input_products['PRECIP'] = ['RF','SF']
        input_products['SNOWFALL'] = 'SF'
        # ME from downscaled product
        # MEcorr is topography corrected melt
        input_products['SNOWMELT'] = 'MEcorr'
        input_products['SUBLIM'] = 'SU'
        input_products['REFREEZE'] = ['MEcorr','RUcorr','RU2corr']
        input_products['RAINFALL'] = 'RF'
        input_products['ALBEDO'] = 'AL'
        input_products['CLOUD_COVER'] = 'CC'
        input_products['LONGWAVE'] = 'LWD'
        input_products['SHORTWAVE'] = 'SWD'
        input_products['LATENT_HEAT'] = 'LHF'
        input_products['SENSIBLE_HEAT'] = 'SHF'
        # downscaled projection: WGS84/NSIDC Sea Ice Polar Stereographic North
        proj4_params = "+init=EPSG:{0:d}".format(3413)
    else:
        # variable coordinates
        XNAME,YNAME,TIMENAME = ('X10_105','Y21_199','TIME')
        # SMB is SMB for the ice covered area
        input_products['SMB'] = 'SMB'
        # RU is runoff for the ice covered area
        # RU2 is runoff for the tundra covered area
        input_products['RUNOFF'] = ['RU','RU2']
        input_products['PRECIP'] = ['RF','SF']
        input_products['SNOWFALL'] = 'SF'
        input_products['SNOWMELT'] = 'ME'
        input_products['SUBLIM'] = 'SU'
        input_products['REFREEZE'] = 'RZ'
        input_products['RAINFALL'] = 'RF'
        input_products['ALBEDO'] = 'AL'
        input_products['CLOUD_COVER'] = 'CC'
        input_products['LONGWAVE'] = 'LWD'
        input_products['SHORTWAVE'] = 'SWD'
        input_products['LATENT_HEAT'] = 'LHF'
        input_products['SENSIBLE_HEAT'] = 'SHF'
        # MAR model projection: Polar Stereographic (Oblique)
        # Earth Radius: 6371229 m
        # True Latitude: 0
        # Center Longitude: -40
        # Center Latitude: 70.5
        proj4_params = ("+proj=sterea +lat_0=+70.5 +lat_ts=0 +lon_0=-40.0 "
            "+a=6371229 +no_defs")

    # create flag to differentiate between direct and directed products
    if (np.ndim(input_products[PRODUCT]) == 0):
        # direct products
        derived_product = False
    else:
        # derived products
        derived_product = True

    # find input files
    input_files=sort_files(rx,[f for f in os.listdir(input_dir) if rx.match(f)])
    # input dimensions and counter variable
    # get dimensions for input dataset
    ny,nx,nt = get_dimensions(input_dir,input_files,XNAME,YNAME)
    # allocate for all data
    MEAN = {}
    MEAN['LON'] = np.zeros((ny,nx))
    MEAN['LAT'] = np.zeros((ny,nx))
    MEAN['VALID'] = np.zeros((ny,nx),dtype=bool)
    MEAN['x'] = np.zeros((nx))
    MEAN['y'] = np.zeros((ny))
    # calculate mean data
    MEAN[PRODUCT] = np.ma.zeros((ny,nx),fill_value=-9999.0)
    MEAN[PRODUCT].mask = np.ones((ny,nx),dtype=bool)
    # input monthly data
    MONTH = {}
    MONTH['TIME'] = np.zeros((nt))
    MONTH['MASK'] = np.zeros((ny,nx))
    # counter for mean
    c = 0

    # for each file
    for t,input_file in enumerate(input_files):
        # Open the NetCDF file for reading
        fileID = netCDF4.Dataset(os.path.join(input_dir,input_file), 'r')
        # Getting the data from each netCDF variable
        # latitude and longitude
        MEAN['LON'][:,:] = fileID.variables['LON'][:,:].copy()
        MEAN['LAT'][:,:] = fileID.variables['LAT'][:,:].copy()

        # extract model x and y
        MEAN['x'][:] = fileID.variables[XNAME][:].copy()
        MEAN['y'][:] = fileID.variables[YNAME][:].copy()
        # extract delta time and epoch of time
        delta_time = fileID.variables[TIMENAME][:].astype(np.float64)
        date_string = fileID.variables[TIMENAME].units
        # extract epoch and units
        epoch,to_secs = SMBcorr.time.parse_date_string(date_string)
        # calculate time array in Julian days
        JD = SMBcorr.time.convert_delta_time(delta_time*to_secs, epoch1=epoch,
            epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0) + 2400000.5
        # read land/ice mask
        LAND_MASK = fileID.variables['MSK'][:,:].copy()
        # finding valid points only from land mask
        iy,ix = np.nonzero(LAND_MASK > 1)
        MEAN['VALID'][iy,ix] = True
        MEAN[PRODUCT].mask[iy,ix] = False
        # read downscaled masks
        if DOWNSCALED:
            # read glacier and ice sheet mask (tundra=1, permanent ice=2)
            MASK_MAR = fileID.variables['MSK_MAR'][:,:].copy()
            SURF_MAR = fileID.variables['SRF_MAR'][:,:].copy()
            iy,ix = np.nonzero((SURF_MAR >= 0.0) & (LAND_MASK > 1))
            MONTH['MASK'][iy,ix] = MASK_MAR[iy,ix]
        else:
            MONTH['MASK'][iy,ix] = 2.0

        # invalid value from MAR product
        FILL_VALUE = fileID.variables['SMB']._FillValue

        # for each Julian Day
        for m,julian in enumerate(JD):
            # convert from Julian days to calendar dates
            YY,MM,DD,hh,mm,ss = SMBcorr.time.convert_julian(julian)
            # calculate time in year-decimal
            MONTH['TIME'][c] = SMBcorr.time.convert_calendar_decimal(YY,MM,
                day=DD,hour=hh,minute=mm,second=ss)
            # read each product of interest contained within the dataset
            # read variables for both direct and derived products
            if derived_product:
                for p in input_products[PRODUCT]:
                    MONTH[p] = fileID.variables[p][m,:,:].copy()
            else:
                p = input_products[PRODUCT]
                MONTH[PRODUCT] = fileID.variables[p][m,:,:].copy()

            # calculate derived products
            if (PRODUCT == 'PRECIP'):
                # PRECIP = SNOWFALL + RAINFALL
                MONTH['PRECIP'] = MONTH['SF'] + MONTH['RF']
            elif (PRODUCT == 'REFREEZE') and DOWNSCALED:
                # runoff from permanent ice covered regions and tundra regions
                RU1,RU2 = input_products['RUNOFF']
                ME = input_products['SNOWMELT']
                MONTH['RUNOFF'] = (MONTH['MASK'] - 1.0)*MONTH[RU1] + \
                    (2.0 - MONTH['MASK'])*MONTH[RU2]
                # REFREEZE = (total) SNOWMELT - RUNOFF
                MONTH['REFREEZE'] = MONTH[ME] - MONTH['RUNOFF']
            elif (PRODUCT == 'RUNOFF'):
                # runoff from permanent ice covered regions and tundra regions
                RU1,RU2 = input_products['RUNOFF']
                MONTH['RUNOFF'] = (MONTH['MASK'] - 1.0)*MONTH[RU1] + \
                    (2.0 - MONTH['MASK'])*MONTH[RU2]
            elif (PRODUCT == 'SMB'):
                # SMB from permanent ice covered regions and tundra regions
                SMB1,SMB2 = input_products['SMB']
                MONTH['SMB'] = (MONTH['MASK'] - 1.0)*MONTH[SMB1] + \
                    (2.0 - MONTH['MASK'])*MONTH[SMB2]

            # add to mean at each time step
            MEAN[PRODUCT][iy,ix] += MONTH[PRODUCT][iy,ix]
            # add to counter
            c += 1

        # close the netcdf file
        fileID.close()

    # convert from total to mean
    MEAN[PRODUCT].data[iy,ix] /= np.float64(c)
    # replace masked values with fill value
    MEAN[PRODUCT].data[MEAN[PRODUCT].mask] = MEAN[PRODUCT].fill_value
    # calculate mean time over period
    MEAN['TIME'] = np.mean(MONTH['TIME'])

    # output netCDF4 filename
    args = (VERSION, PRODUCT, RANGE[0], RANGE[1])
    mean_file = 'MAR_{0}_{1}_mean_{2:4.0f}-{3:4.0f}.nc'.format(*args)
    create_netCDF4(MEAN, FILENAME=os.path.join(input_dir,mean_file),
        UNITS='mmWE', LONGNAME=longname[PRODUCT], VARNAME=PRODUCT,
        LONNAME='LON', LATNAME='LAT', XNAME='x', YNAME='y', TIMENAME='TIME',
        MASKNAME='VALID', VERBOSE=VERBOSE, PROJECTION=proj4_params,
        TITLE='{0:4d}-{1:4d}_mean_field'.format(RANGE[0],RANGE[1]))
    # change the permissions mode
    os.chmod(os.path.join(input_dir,mean_file),MODE)

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates the temporal mean of MAR
            surface mass balance products
            """
    )
    # command line parameters
    # working data directory
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
    # MAR model version
    parser.add_argument('--version','-v',
        metavar='VERSION', type=str,
        default='v3.11', choices=['v3.5.2','v3.9','v3.10','v3.11'],
        help='MAR version to run')
    # Products to calculate cumulative
    parser.add_argument('--product','-p',
        metavar='PRODUCT', type=str, nargs='+',
        default=['SMB'], choices=longname.keys(),
        help='MAR product to calculate')
    # mean range
    parser.add_argument('--mean','-m',
        metavar=('START','END'), type=int, nargs=2,
        default=[1961,1990],
        help='Start and end year range for mean')
    # run Downscaled version of MAR
    parser.add_argument('--downscaled','-d',
        default=False, action='store_true',
        help='Run downscaled MAR')
    # verbose output of processing run
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of netCDF4 variables')
    # permissions mode of the local directories and files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files')
    # return the parser
    return parser

# This is the main part of the program that calls the individual modules
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args = parser.parse_args()

    # run program for each input product
    for PRODUCT in args.product:
        mar_smb_mean(args.directory, args.version, PRODUCT,
            RANGE=args.mean, DOWNSCALED=args.downscaled,
            VERBOSE=args.verbose, MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
