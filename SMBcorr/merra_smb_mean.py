#!/usr/bin/env python
u"""
merra_smb_mean.py
Written by Tyler Sutterley (09/2023)
Reads monthly MERRA-2 datafiles to calculate multi-annual means
    of derived surface mass balance products

From tavgM_2d_int (Vertically Integrated Diagnostics) collection:
    PRECCU (convective rain)
    PRECLS (large-scale rain)
    PRECSN (snow)
    and EVAP (evaporation)
From tavgM_2d_glc (Land Ice Surface Diagnostics) collection:
    RUNOFF (runoff over glaciated land)

INPUTS:
    SMB: Surface Mass Balance
    ACCUM: Snowfall accumulation
    PRECIP: Total Precipitation
    RAIN: Total Rainfall
    SUBLIM: Evaporation and Sublimation
    RUNOFF: Meltwater Runoff

COMMAND LINE OPTIONS:
    -D X, --directory X: working data directory
    -m X, --mean X: Year range for mean
    -M X, --mode X: Permission mode of directories and files
    -V, --verbose: Output information for each output file

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    dateutil: powerful extensions to datetime
        https://dateutil.readthedocs.io/en/stable/
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/netCDF4/index.html
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://h5py.org

PROGRAM DEPENDENCIES:
    spatial.py: utilities for working with geospatial data
    time.py: utilities for calculating time operations

UPDATE HISTORY:
    Updated 09/2023: using updated spatial functions
    Updated 08/2022: updated docstrings to numpy documentation format
    Updated 10/2021: using python logging for handling verbose output
        add more derived products and include sublimation and condensation
    Updated 02/2021: replaced numpy bool to prevent deprecation warning
        sort files by month as September 2020 was reprocessed
        https://daac.gsfc.nasa.gov/information/alerts
    Updated 01/2021: use argparse to set command line parameters
        using spatial module for read/write operations
        using utilities from time module
    Updated 10/2019: changing Y/N flags to True/False
    Updated 01/2017: can output different data products
    Written 11/2016
"""
from __future__ import print_function

import sys
import os
import re
import logging
import argparse
import warnings
import numpy as np
import SMBcorr.time
import SMBcorr.spatial

# attempt imports
try:
    import netCDF4
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("netCDF4 not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: read variables from MERRA-2 tavgM_2d_int and tavgM_2d_glc files
def read_merra_variables(merra_flux_file, merra_ice_surface_file):
    """Read a subset of MERRA-2 variables for deriving
        surface mass balance products

    From ``tavgM_2d_int`` (Vertically Integrated Diagnostics) collection:

        - ``PRECCU``: convective rain
        - ``PRECLS``: large-scale rain
        - ``PRECSN``: snow
        - ``EVAP``: evaporation

    From ``tavgM_2d_glc`` (Land Ice Surface Diagnostics) collection:

        - ``RUNOFF``: runoff over glaciated land

    Parameters
    ----------
    merra_flux_file: str
        Path to ``tavgM_2d_int`` file
    merra_ice_surface_file: str
        Path to ``tavgM_2d_glc`` file
    """
    # python dictionary of output variables
    dinput = {}
    # read each variable of interest in MERRA-2 flux file
    with netCDF4.Dataset(merra_flux_file, 'r') as fid1:
        # extract geolocation variables
        dinput['lon'] = fid1.variables['lon'][:].copy()
        dinput['lat'] = fid1.variables['lat'][:].copy()
        # convert time from netCDF4 units to Julian Days
        date_string = fid1.variables['time'].units
        epoch,to_secs = SMBcorr.time.parse_date_string(date_string)
        dinput['time'] = SMBcorr.time.convert_delta_time(
            to_secs*fid1.variables['time'][:],epoch1=epoch,
            epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0) + 2400000.5
        # read each variable of interest in MERRA-2 flux file
        for key in ['PRECCU','PRECLS','PRECSN','EVAP']:
            # Getting the data from each NetCDF variable of interest
            dinput[key] = np.ma.array(fid1.variables[key][:].squeeze(),
                fill_value=fid1.variables[key]._FillValue)
            dinput[key].mask = (dinput[key].data == dinput[key].fill_value)
    # read each variable of interest in MERRA-2 ice surface file
    with netCDF4.Dataset(merra_ice_surface_file, 'r') as fid2:
        for key in ['RUNOFF','WESNSC']:
            # Getting the data from each NetCDF variable of interest
            dinput[key] = np.ma.array(fid2.variables[key][:].squeeze(),
                fill_value=fid2.variables[key]._FillValue)
            dinput[key].mask = (dinput[key].data == dinput[key].fill_value)
    return dinput

# PURPOSE: read monthly MERRA-2 datasets to calculate multi-annual means
def merra_smb_mean(DIRECTORY, PRODUCT, RANGE=None,
    VERBOSE=False, MODE=0o775):
    """
    Reads monthly MERRA-2 datafiles to calculate multi-annual means
    of derived surface mass balance products

    Parameters
    ----------
    base_dir: str
        Working data directory
    PRODUCT: str
        MERRA-2 product to calculate

            - ``SMB``: Surface Mass Balance
            - ``ACCUM``: Snowfall accumulation
            - ``PRECIP``: Precipitation
            - ``RAIN``: Total Rainfall
            - ``SUBLIM``: Evaporation and Sublimation
            - ``REFREEZE``: Melt Water Refreeze
    RANGE: list, default [1961,1990]
        Start and end year of mean
    VERBOSE: bool, default False
        Verbose output of netCDF4 variables
    MODE: oct, default 0o775
        Permission mode of directories and files created
    """

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # MERRA-2 product subdirectories
    P1 = 'M2TMNXINT.5.12.4'
    P2 = 'M2TMNXGLC.5.12.4'
    # regular expression operator to find datafiles (and not the xml files)
    regex_pattern = 'MERRA2_(\d+).{0}.(\d{{4}})(\d{{2}}).nc4(?!.xml)'
    # sign for each product to calculate total SMB
    smb_sign = {'PRECCU':1.0,'PRECLS':1.0,'PRECSN':1.0,'EVAP':-1.0,
        'RUNOFF':-1.0,'WESNSC':1.0}
    # titles for each output data product
    merra_products = {}
    merra_products['SMB'] = 'MERRA-2 Surface Mass Balance'
    merra_products['ACCUM'] = 'MERRA-2 Snowfall accumulation'
    merra_products['PRECIP'] = 'MERRA-2 Precipitation'
    merra_products['RAINFALL'] = 'MERRA-2 Rainfall'
    merra_products['SUBLIM'] = 'MERRA-2 Evaporation and Sublimation'
    merra_products['RUNOFF'] = 'MERRA-2 Meltwater Runoff'
    # source of each output data product
    merra_sources = {}
    merra_sources['SMB'] = ['PRECCU','PRECLS','PRECSN','EVAP','RUNOFF','WESNSC']
    merra_sources['ACCUM'] = ['PRECSN','EVAP']
    merra_sources['PRECIP'] = ['PRECCU','PRECLS','PRECSN']
    merra_sources['RAINFALL'] = ['PRECCU','PRECLS']
    merra_sources['SUBLIM'] = ['EVAP','WESNSC']
    merra_sources['RUNOFF'] = ['RUNOFF']
    merra_reference = ', '.join(merra_sources[PRODUCT])
    # output bad value
    fill_value = -9999.0
    # output dimensions and extents
    nlat,nlon = (361,576)

    # years of available data between RANGE
    YEARS = sorted(map(str,range(int(RANGE[0]),int(RANGE[-1])+1)))
    # check that are years for RANGE are available
    CHECK = [Y in os.listdir(os.path.join(DIRECTORY,P1)) for Y in YEARS]
    if not np.all(CHECK):
        raise Exception('Not all years available on file system')
    # compile regular expression operator for flux product
    rx = re.compile(regex_pattern.format('tavgM_2d_int_Nx'), re.VERBOSE)

    # output data and attributes
    output = {}
    attrib = {}
    # latitude
    attrib['lat'] = {}
    attrib['lat']['long_name'] = 'Latitude'
    attrib['lat']['units'] = 'Degrees_North'
    # longitude
    attrib['lon'] = {}
    attrib['lon']['long_name'] = 'Longitude'
    attrib['lon']['units'] = 'Degrees_East'
    # output mean of product
    attrib[PRODUCT] = {}
    attrib[PRODUCT]['description'] = merra_products[PRODUCT]
    attrib[PRODUCT]['reference'] = merra_reference
    attrib[PRODUCT]['model'] = 'MERRA-2'
    attrib[PRODUCT]['units'] = 'mm w.e.'
    attrib[PRODUCT]['long_name'] = 'Equivalent_Water_Thickness'
    attrib[PRODUCT]['_FillValue'] = fill_value
    # time
    attrib['time'] = {}
    attrib['time']['long_name'] = 'Time'
    attrib['time']['units'] = 'years'

    # output data and mask
    output[PRODUCT] = np.ma.zeros((nlat,nlon,1), fill_value=fill_value)
    output[PRODUCT].mask = np.zeros((nlat,nlon,1),dtype=bool)
    output['time'] = 0.0
    count = 0.0
    # for each input file
    for Y in YEARS:
        # find input files for PRODUCT
        f=[f for f in os.listdir(os.path.join(DIRECTORY,P1,Y)) if rx.match(f)]
        # sort files by month
        indices = np.argsort([rx.match(f1).group(3) for f1 in f])
        f = [f[indice] for indice in indices]
        # days per month in year
        dpm = SMBcorr.time.calendar_days(int(Y))
        # for each monthly file
        for M,f1 in enumerate(f):
            # extract parameters from input flux file
            MOD,Y1,M1 = rx.findall(f1).pop()
            # corresponding ice surface product file
            args = (MOD,'tavgM_2d_glc_Nx',Y1,M1)
            f2 = 'MERRA2_{0}.{1}.{2}{3}.nc4'.format(*args)
            # full path for flux and ice surface files
            merra_flux_file = os.path.join(DIRECTORY,P1,Y,f1)
            merra_ice_surface_file = os.path.join(DIRECTORY,P2,Y,f2)
            if not os.access(merra_ice_surface_file,os.F_OK):
                raise Exception('File {0} not in file system'.format(f2))
            # read netCDF4 files for variables of interest
            var = read_merra_variables(merra_flux_file,merra_ice_surface_file)
            # convert from Julian days to calendar dates
            YY,MM,DD,hh,mm,ss = SMBcorr.time.convert_julian(var['time'],
                FORMAT='tuple')
            # calculate the total seconds in month
            seconds = dpm[M]*24.0*60.0*60.0
            # calculate time in year decimal
            tdec = SMBcorr.time.convert_calendar_decimal(YY,
                MM,day=DD,hour=hh,minute=mm,second=ss)
            # spatial object for monthly variable
            dinput = np.ma.zeros((nlat,nlon), fill_value=fill_value)
            dinput.mask = np.zeros((nlat,nlon),dtype=bool)
            for p in ['PRECCU','PRECLS','PRECSN','EVAP','RUNOFF','WESNSC']:
                dinput.mask |= var[p].mask
            # valid indices for all variables
            indy,indx = np.nonzero(np.logical_not(dinput.mask))
            if (PRODUCT == 'SMB'):
                # calculate SMB and convert from flux to monthly
                for p in ['PRECCU','PRECLS','PRECSN','EVAP','RUNOFF','WESNSC']:
                    tmp = var[p][indy,indx]*seconds*smb_sign[p]
                    dinput.data[indy,indx] += tmp
            elif (PRODUCT == 'ACCUM'):
                # calculate accumulation and convert from flux to monthly
                for p in ['PRECSN','EVAP','WESNSC']:
                    tmp = var[p][indy,indx]*seconds*smb_sign[p]
                    dinput.data[indy,indx] += tmp
            elif (PRODUCT == 'PRECIP'):
                # calculate precipitation and convert from flux to monthly
                for p in ['PRECCU','PRECLS','PRECSN']:
                    tmp = var[p][indy,indx]*seconds
                    dinput.data[indy,indx] += tmp
            elif (PRODUCT == 'RAINFALL'):
                # calculate rainfall and convert from flux to monthly
                for p in ['PRECCU','PRECLS']:
                    tmp = var[p][indy,indx]*seconds
                    dinput.data[indy,indx] += tmp
            elif (PRODUCT == 'SUBLIM'):
                # calculate sublimation and convert from flux to monthly
                for p in ['EVAP','WESNSC']:
                    tmp = var[p][indy,indx]*seconds
                    dinput.data[indy,indx] += tmp
            elif (PRODUCT == 'RUNOFF'):
                # convert runoff from flux to monthly
                for p in ['RUNOFF']:
                    tmp = var[p][indy,indx]*seconds
                    dinput.data[indy,indx] += tmp
            # add monthly fluxes to total
            output[PRODUCT].data[:,:] += np.atleast_3d(dinput.data)
            output[PRODUCT].mask[:,:] |= np.atleast_3d(dinput.mask)
            output['time'] += tdec
            output['lat'] = np.copy(var['lat'])
            output['lon'] = np.copy(var['lon'])
            # add to count
            count += 1.0

    # convert from totals to means
    indy,indx,indt = np.nonzero(np.logical_not(output[PRODUCT].mask))
    output[PRODUCT][indy,indx,indt] /= count
    output[PRODUCT].data[output[PRODUCT].mask] = output[PRODUCT].fill_value
    output['time'] /= count

    # output MERRA-2 mean data file
    FILE = f'MERRA2.tavgM_2d_{PRODUCT}_mean_Nx.{RANGE[0]:4d}-{RANGE[1]:4d}.nc'
    output_file = os.path.join(DIRECTORY, FILE)
    # netcdf (.nc)
    SMBcorr.spatial.to_netCDF4(output, attrib, output_file,
        data_type='grid')
    # change the permissions mode
    os.chmod(output_file, MODE)

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Reads monthly MERRA-2 datafiles to calculate
            calculate multi-annual means of derived surface mass
            balance products
            """
    )
    # command line parameters
    choices = ['SMB','ACCUM','PRECIP','RAINFALL','SUBLIM','RUNOFF']
    parser.add_argument('product',
        type=str, nargs='+', choices=choices,
        help='MERRA-2 derived product')
    # working data directory
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
    # start and end years to run for mean
    parser.add_argument('--mean','-m',
        metavar=('START','END'), type=int, nargs=2,
        default=[1980,1995],
        help='Start and end year range for mean')
    # print information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
    # permissions mode of the local directories and files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files')
    # return the argument parser
    return parser

# Main program that calls merra_smb_mean()
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # run program for each input product
    for PRODUCT in args.product:
        merra_smb_mean(args.directory, PRODUCT, RANGE=args.mean,
            VERBOSE=args.verbose, MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()

