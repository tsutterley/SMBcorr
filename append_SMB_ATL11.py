#!/usr/bin/env python
u"""
append_SMB_ATL11.py
Written by Tyler Sutterley (05/2020)
Interpolates daily model firn estimates to the coordinates of an ATL11 file

CALLING SEQUENCE:
    python append_SMB_ATL11.py --directory=<path> --region=GL <path_to_file>

COMMAND LINE OPTIONS:
    -D X, --directory=X: Working data directory
    -R X, --region=X: Region to interpolate (GL, AA)
    -M X, --model=X: Regional climate models to run

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
    netCDF4: Python interface to the netCDF C library
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
        https://unidata.github.io/netcdf4-python/netCDF4/index.html
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
    pointCollection: Utilities for organizing and manipulating point data
        https://github.com/SmithB/pointCollection

UPDATE HISTORY:
    Updated 05/2020: reduce variables imported from HDF5
        add crossover reading and interpolation
    Written 05/2020
"""
import sys
import os
import re
import getopt
import SMBcorr
import h5py
import numpy as np
import pointCollection as pc

# PURPOSE: convert time from delta seconds into Julian and year-decimal
def convert_delta_time(delta_time, gps_epoch=1198800018.0):
    # calculate gps time from delta_time
    gps_seconds = gps_epoch + delta_time
    time_leaps = SMBcorr.count_leap_seconds(gps_seconds)
    # calculate julian time
    time_julian = 2444244.5 + (gps_seconds - time_leaps)/86400.0
    # convert to calendar date with convert_julian.py
    Y,M,D,h,m,s = SMBcorr.convert_julian(time_julian,FORMAT='tuple')
    # calculate year-decimal time
    time_decimal = SMBcorr.convert_calendar_decimal(Y,M,DAY=D,HOUR=h,MINUTE=m,SECOND=s)
    # return both the Julian and year-decimal formatted dates
    return dict(julian=time_julian, decimal=time_decimal)

# PURPOSE: set the projection parameters based on the region name
def set_projection(REGION):
    if (REGION == 'AA'):
        projection_flag = 'EPSG:3031'
    elif (REGION == 'GL'):
        projection_flag = 'EPSG:3413'
    return projection_flag

def append_SMB_ATL11(input_file, base_dir, REGION, MODEL):
    # read input file
    field_dict = {None:('delta_time','h_corr','x','y')}
    D11 = pc.data().from_h5(input_file, field_dict=field_dict)
    # check if running crossover or along-track ATL11
    if (D11.h_corr.ndim == 3):
        nseg,ncycle,ncross = D11.shape
    else:
        nseg,ncycle = D11.shape

    # get projection of input coordinates
    EPSG = set_projection(REGION)

    # available models
    models = dict(AA={}, GL={})
    models['GL']['MAR'] = []
    # models['GL']['MAR'].append('MARv3.9-ERA')
    # models['GL']['MAR'].append('MARv3.10-ERA')
    # models['GL']['MAR'].append('MARv3.11-NCEP')
    models['GL']['MAR'].append('MARv3.11-ERA')
    models['GL']['RACMO'] = []
    # models['GL']['RACMO'].append('RACMO2.3-XGRN11')
    # models['GL']['RACMO'].append('RACMO2.3p2-XGRN11')
    models['GL']['RACMO'].append('RACMO2.3p2-FGRN055')

    for model_version in models[REGION][MODEL]:
        if (MODEL == 'MAR'):
            MAR_VERSION,=re.findall('MARv\d+\.\d+',model_version)
            MAR_REGION=dict(GL='Greenland',AA='Antarctic')[REGION]
            SUBDIRECTORY={}
            SUBDIRECTORY['MARv3.9-ERA']=['ERA_1958-2018_10km','daily_10km']
            SUBDIRECTORY['MARv3.10-ERA']=['ERA_1958-2019-15km','daily_15km']
            SUBDIRECTORY['MARv3.11-NCEP']=['NCEP1_1948-2020_20km','daily_20km']
            SUBDIRECTORY['MARv3.11-ERA']=['ERA_1958-2019-15km','daily_15km']
            MAR_MODEL=SUBDIRECTORY[model_version]
            DIRECTORY=os.path.join(base_dir,'MAR',MAR_VERSION,MAR_REGION,*MAR_MODEL)
        elif (MODEL == 'RACMO'):
            RACMO_VERSION,RACMO_MODEL=model_version.split('-')

        # check if running crossover or along track
        if (D11.h_corr.ndim == 3):
            # allocate for output FIRN height for crossover data
            FIRN = np.ma.zeros((nseg,ncycle,ncross),fill_value=np.nan)
            FIRN.annual = np.zeros((nseg,ncycle,ncross))
            FIRN.interpolation = np.zeros((nseg,ncycle,ncross),dtype=np.uint8)
            # for each cycle of ICESat-2 ATL11 data
            for c in range(ncycle):
                # check that there are valid crossovers
                cross = [xo for xo in range(ncross) if np.any(np.isfinite(D11.delta_time[:,c,xo]))]
                # for each valid crossing
                for xo in cross:
                    # find valid crossovers
                    i, = np.nonzero(np.isfinite(D11.delta_time[:,c,xo]))
                    # convert from delta time to decimal-years
                    tdec = convert_delta_time(D11.delta_time[i,c,xo])['decimal']
                    if (MODEL == 'MAR'):
                        # read and interpolate daily MAR outputs
                        firn_out = SMBcorr.interpolate_mar_daily(DIRECTORY, EPSG,
                            MAR_VERSION, tdec, D11.x[i,c,xo], D11.y[i,c,xo],
                            VARIABLE='ZN6', SIGMA=1.5, FILL_VALUE=np.nan)
                    elif (MODEL == 'RACMO'):
                        # read and interpolate daily RACMO outputs
                        firn_out = SMBcorr.interpolate_racmo_daily(base_dir, EPSG,
                            RACMO_MODEL, tdec, D11.x[i,c,xo], D11.y[i,c,xo],
                            VARIABLE='hgtsrf', SIGMA=1.5, FILL_VALUE=np.nan)
                    # set attributes to output for iteration
                    FIRN[i,c,xo] = np.copy(firn_out)
                    FIRN.annual[i,c,xo] = np.copy(firn_out.annual)
                    FIRN.interpolation[i,c,xo] = np.copy(firn_out.interpolation)
        else:
            # allocate for output FIRN height for along track data
            FIRN = np.ma.zeros((nseg,ncycle),fill_value=np.nan)
            FIRN.annual = np.zeros((nseg,ncycle))
            FIRN.interpolation = np.zeros((nseg,ncycle),dtype=np.uint8)
            # check that there are valid elevations
            cycle = [c for c in range(ncycle) if np.any(np.isfinite(D11.delta_time[:,c]))]
            # for each valid cycle of ICESat-2 ATL11 data
            for c in cycle:
                # find valid elevations
                i, = np.nonzero(np.isfinite(D11.delta_time[:,c]))
                # convert from delta time to decimal-years
                tdec = convert_delta_time(D11.delta_time[i,c])['decimal']
                if (MODEL == 'MAR'):
                    # read and interpolate daily MAR outputs
                    firn_out = SMBcorr.interpolate_mar_daily(DIRECTORY, EPSG,
                        MAR_VERSION, tdec, D11.x[i,c], D11.y[i,c],
                        VARIABLE='ZN6', SIGMA=1.5, FILL_VALUE=np.nan)
                elif (MODEL == 'RACMO'):
                    # read and interpolate daily RACMO outputs
                    firn_out = SMBcorr.interpolate_racmo_daily(base_dir, EPSG,
                        RACMO_MODEL, tdec, D11.x[i,c], D11.y[i,c],
                        VARIABLE='hgtsrf', SIGMA=1.5, FILL_VALUE=np.nan)
                # set attributes to output for iteration
                FIRN[i,c] = np.copy(firn_out)
                FIRN.annual[i,c] = np.copy(firn_out.annual)
                FIRN.interpolation[i,c] = np.copy(firn_out.interpolation)
                    
        # replace mask values
        FIRN.mask = (FIRN.data == FIRN.fill_value)

        # append input HDF5 file with new firn model outputs
        fileID = h5py.File(os.path.expanduser(input_file),'a')
        fileID.create_group(model_version)
        h5 = {}
        val = '{0}/{1}'.format(model_version,'zsurf')
        h5['zsurf'] = fileID.create_dataset(val, FIRN.shape, data=FIRN,
            dtype=FIRN.dtype, compression='gzip', fillvalue=FIRN.fill_value)
        h5['zsurf'].attrs['units'] = "m"
        h5['zsurf'].attrs['long_name'] = "Snow Height Change"
        h5['zsurf'].attrs['coordinates'] = "../delta_time ../latitude ../longitude"
        h5['zsurf'].attrs['model'] = model_version
        val = '{0}/{1}'.format(model_version,'zannual')
        h5['zannual'] = fileID.create_dataset(val, FIRN.shape, data=FIRN.annual,
            dtype=FIRN.dtype, compression='gzip', fillvalue=FIRN.fill_value)
        h5['zannual'].attrs['units'] = "m"
        h5['zannual'].attrs['long_name'] = "Annual Snow Height Change"
        h5['zannual'].attrs['coordinates'] = "../delta_time ../latitude ../longitude"
        h5['zannual'].attrs['model'] = model_version
        fileID.close()

#-- PURPOSE: help module to describe the optional input parameters
def usage():
    print('\nHelp: {}'.format(os.path.basename(sys.argv[0])))
    print(' -D X, --directory=X\tWorking data directory')
    print(' -R X, --region=X\tRegion of model to interpolate')
    print(' -M X, --model=X\tRegional climate model to run\n')

#-- Main program that calls merra_hybrid_cumulative()
def main():
    #-- Read the system arguments listed after the program
    long_options = ['help','directory=','region=','model=']
    optlist,arglist = getopt.getopt(sys.argv[1:], 'hD:R:M:', long_options)

    #-- data directory
    base_dir = None
    #-- region of firn model
    REGION = 'GL'
    #-- surface mass balance product
    MODELS = ['RACMO','MAR']
    #-- extract parameters
    for opt, arg in optlist:
        if opt in ('-h','--help'):
            usage()
            sys.exit()
        elif opt in ("-D","--directory"):
            base_dir = os.path.expanduser(arg)
        elif opt in ("-R","--region"):
            REGION = arg
        elif opt in ("-M","--model"):
            MODELS = arg.split(',')

    #-- run program with parameters
    for f in arglist:
        for m in MODELS:
            append_SMB_ATL11(os.path.expanduser(f),base_dir,REGION,m)

#-- run main program
if __name__ == '__main__':
    main()
