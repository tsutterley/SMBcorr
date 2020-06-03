#!/usr/bin/env python
u"""
append_SMB_ATL11.py
Written by Tyler Sutterley (06/2020)
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
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/netCDF4/index.html
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
    pointCollection: Utilities for organizing and manipulating point data
        https://github.com/SmithB/pointCollection

UPDATE HISTORY:
    Updated 06/2020: verify masked values are set to fill_value
    Updated 05/2020: reduce variables imported from HDF5
        add crossover reading and interpolation.  add more models
        copy mask variable from interpolation programs
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
    models['GL']['MAR'].append('MARv3.11.2-ERA-7.5km')
    models['GL']['MAR'].append('MARv3.11.2-ERA-10km')
    models['GL']['MAR'].append('MARv3.11.2-ERA-15km')
    models['GL']['MAR'].append('MARv3.11.2-ERA-20km')
    models['GL']['MAR'].append('MARv3.11.2-NCEP-20km')
    models['GL']['RACMO'] = []
    # models['GL']['RACMO'].append('RACMO2.3-XGRN11')
    # models['GL']['RACMO'].append('RACMO2.3p2-XGRN11')
    models['GL']['RACMO'].append('RACMO2.3p2-FGRN055')

    for model_version in models[REGION][MODEL]:
        if (MODEL == 'MAR'):
            match_object=re.match('(MARv\d+\.\d+(.\d+)?)',model_version)
            MAR_VERSION=match_object.group(0)
            MAR_REGION=dict(GL='Greenland',AA='Antarctic')[REGION]
            # model subdirectories
            SUBDIRECTORY={}
            SUBDIRECTORY['MARv3.9-ERA']=['ERA_1958-2018_10km','daily_10km']
            SUBDIRECTORY['MARv3.10-ERA']=['ERA_1958-2019-15km','daily_15km']
            SUBDIRECTORY['MARv3.11-NCEP']=['NCEP1_1948-2020_20km','daily_20km']
            SUBDIRECTORY['MARv3.11-ERA']=['ERA_1958-2019-15km','daily_15km']
            SUBDIRECTORY['MARv3.11.2-ERA-7.5km']=['7.5km_ERA5']
            SUBDIRECTORY['MARv3.11.2-ERA-10km']=['10km_ERA5']
            SUBDIRECTORY['MARv3.11.2-ERA-15km']=['15km_ERA5']
            SUBDIRECTORY['MARv3.11.2-ERA-20km']=['20km_ERA5']
            SUBDIRECTORY['MARv3.11.2-NCEP-20km']=['20km_NCEP1']
            MAR_MODEL=SUBDIRECTORY[model_version]
            DIRECTORY=os.path.join(base_dir,'MAR',MAR_VERSION,MAR_REGION,*MAR_MODEL)
            # variable coordinates
            KWARGS = {}
            KWARGS['MARv3.9-ERA']=dict(XNAME='X10_153',YNAME='Y21_288')
            KWARGS['MARv3.10-ERA']=dict(XNAME='X10_105',YNAME='Y21_199')
            KWARGS['MARv3.11-NCEP']=dict(XNAME='X12_84',YNAME='Y21_155')
            KWARGS['MARv3.11-ERA']=dict(XNAME='X10_105',YNAME='Y21_199')
            KWARGS['MARv3.11.2-ERA-7.5km']=dict(XNAME='X12_203',YNAME='Y20_377')
            KWARGS['MARv3.11.2-ERA-10km']=dict(XNAME='X10_153',YNAME='Y21_288')
            KWARGS['MARv3.11.2-ERA-15km']=dict(XNAME='X10_105',YNAME='Y21_199')
            KWARGS['MARv3.11.2-ERA-20km']=dict(XNAME='X12_84',YNAME='Y21_155')
            KWARGS['MARv3.11.2-NCEP-20km']=dict(XNAME='X12_84',YNAME='Y21_155')            
            MAR_KWARGS=KWARGS[model_version]
            # output variable keys for both direct and derived fields
            KEYS = ['zsurf','zfirn','zmelt','zsmb','zaccum']
            # HDF5 longname attributes for each variable
            LONGNAME = {}
            LONGNAME['zsurf'] = "Snow Height Change"
            LONGNAME['zfirn'] = "Snow Height Change due to Compaction"
            LONGNAME['zmelt'] = "Snow Height Change due to Surface Melt"
            LONGNAME['zsmb'] = "Snow Height Change due to Surface Mass Balance"
            LONGNAME['zaccum'] = "Snow Height Change due to Surface Accumulation"
        elif (MODEL == 'RACMO'):
            RACMO_VERSION,RACMO_MODEL=model_version.split('-')
            # output variable keys
            KEYS = ['zsurf']
            # HDF5 longname attributes for each variable
            LONGNAME = {}
            LONGNAME['zsurf'] = "Snow Height Change"

        # check if running crossover or along track
        if (D11.h_corr.ndim == 3):
            # allocate for output height for crossover data
            OUTPUT = {}
            for key in KEYS:
                OUTPUT[key] = np.ma.zeros((nseg,ncycle,ncross),fill_value=np.nan)
                OUTPUT[key].mask = np.ones((nseg,ncycle,ncross),dtype=np.bool)
                OUTPUT[key].interpolation = np.zeros((nseg,ncycle,ncross),dtype=np.uint8)
            # for each cycle of ICESat-2 ATL11 data
            for c in range(ncycle):
                # check that there are valid crossovers
                cross = [xo for xo in range(ncross) if 
                    np.any(np.isfinite(D11.delta_time[:,c,xo]))]
                # for each valid crossing
                for xo in cross:
                    # find valid crossovers
                    i, = np.nonzero(np.isfinite(D11.delta_time[:,c,xo]))
                    # convert from delta time to decimal-years
                    tdec = convert_delta_time(D11.delta_time[i,c,xo])['decimal']
                    if (MODEL == 'MAR'):
                        # read and interpolate daily MAR outputs
                        ZN4 = SMBcorr.interpolate_mar_daily(DIRECTORY, EPSG,
                            MAR_VERSION, tdec, D11.x[i,c,xo], D11.y[i,c,xo],
                            VARIABLE='ZN4', SIGMA=1.5, FILL_VALUE=np.nan, **MAR_KWARGS)
                        ZN5 = SMBcorr.interpolate_mar_daily(DIRECTORY, EPSG,
                            MAR_VERSION, tdec, D11.x[i,c,xo], D11.y[i,c,xo],
                            VARIABLE='ZN5', SIGMA=1.5, FILL_VALUE=np.nan, **MAR_KWARGS)
                        ZN6 = SMBcorr.interpolate_mar_daily(DIRECTORY, EPSG,
                            MAR_VERSION, tdec, D11.x[i,c,xo], D11.y[i,c,xo],
                            VARIABLE='ZN6', SIGMA=1.5, FILL_VALUE=np.nan, **MAR_KWARGS)
                        # set attributes to output for iteration
                        OUTPUT['zfirn'].data[i,c,xo] = np.copy(ZN4.data)
                        OUTPUT['zfirn'].mask[i,c,xo] = np.copy(ZN4.mask)
                        OUTPUT['zfirn'].interpolation[i,c,xo] = np.copy(ZN4.interpolation)
                        OUTPUT['zsurf'].data[i,c,xo] = np.copy(ZN6.data)
                        OUTPUT['zsurf'].mask[i,c,xo] = np.copy(ZN6.mask)
                        OUTPUT['zsurf'].interpolation[i,c,xo] = np.copy(ZN6.interpolation)
                        OUTPUT['zmelt'].data[i,c,xo] = np.copy(ZN5.data)
                        OUTPUT['zmelt'].mask[i,c,xo] = np.copy(ZN5.mask)
                        OUTPUT['zmelt'].interpolation[i,c,xo] = np.copy(ZN5.interpolation)
                        # calculate derived fields
                        OUTPUT['zsmb'].data[i,c,xo] = ZN6.data - ZN4.data
                        OUTPUT['zsmb'].mask[i,c,xo] = ZN4.mask | ZN6.mask
                        OUTPUT['zaccum'].data[i,c,xo] = ZN6.data - ZN4.data - ZN5.data
                        OUTPUT['zaccum'].mask[i,c,xo] = ZN4.mask | ZN5.mask | ZN6.mask
                    elif (MODEL == 'RACMO'):
                        # read and interpolate daily RACMO outputs
                        hgtsrf = SMBcorr.interpolate_racmo_daily(base_dir, EPSG,
                            RACMO_MODEL, tdec, D11.x[i,c,xo], D11.y[i,c,xo],
                            VARIABLE='hgtsrf', SIGMA=1.5, FILL_VALUE=np.nan)
                        # set attributes to output for iteration
                        OUTPUT['zsurf'].data[i,c,xo] = np.copy(hgtsrf.data)
                        OUTPUT['zsurf'].mask[i,c,xo] = np.copy(hgtsrf.mask)
                        OUTPUT['zsurf'].interpolation[i,c,xo] = np.copy(hgtsrf.interpolation)
        else:
            # allocate for output height for along-track data
            OUTPUT = {}
            for key in KEYS:
                OUTPUT[key] = np.ma.zeros((nseg,ncycle),fill_value=np.nan)
                OUTPUT[key].mask = np.ones((nseg,ncycle),dtype=np.bool)
                OUTPUT[key].interpolation = np.zeros((nseg,ncycle),dtype=np.uint8)
            # check that there are valid elevations
            cycle = [c for c in range(ncycle) if 
                np.any(np.isfinite(D11.delta_time[:,c]))]
            # for each valid cycle of ICESat-2 ATL11 data
            for c in cycle:
                # find valid elevations
                i, = np.nonzero(np.isfinite(D11.delta_time[:,c]))
                # convert from delta time to decimal-years
                tdec = convert_delta_time(D11.delta_time[i,c])['decimal']
                if (MODEL == 'MAR'):
                    # read and interpolate daily MAR outputs
                    ZN4 = SMBcorr.interpolate_mar_daily(DIRECTORY, EPSG,
                        MAR_VERSION, tdec, D11.x[i,c], D11.y[i,c],
                        VARIABLE='ZN4', SIGMA=1.5, FILL_VALUE=np.nan, **MAR_KWARGS)
                    ZN5 = SMBcorr.interpolate_mar_daily(DIRECTORY, EPSG,
                        MAR_VERSION, tdec, D11.x[i,c], D11.y[i,c],
                        VARIABLE='ZN5', SIGMA=1.5, FILL_VALUE=np.nan, **MAR_KWARGS)
                    ZN6 = SMBcorr.interpolate_mar_daily(DIRECTORY, EPSG,
                        MAR_VERSION, tdec, D11.x[i,c], D11.y[i,c],
                        VARIABLE='ZN6', SIGMA=1.5, FILL_VALUE=np.nan, **MAR_KWARGS)
                    # set attributes to output for iteration
                    OUTPUT['zfirn'].data[i,c] = np.copy(ZN4.data)
                    OUTPUT['zfirn'].mask[i,c] = np.copy(ZN4.mask)
                    OUTPUT['zfirn'].interpolation[i,c] = np.copy(ZN4.interpolation)
                    OUTPUT['zsurf'].data[i,c] = np.copy(ZN6.data)
                    OUTPUT['zsurf'].mask[i,c] = np.copy(ZN6.mask)
                    OUTPUT['zsurf'].interpolation[i,c] = np.copy(ZN6.interpolation)
                    OUTPUT['zmelt'].data[i,c] = np.copy(ZN5.data)
                    OUTPUT['zmelt'].mask[i,c] = np.copy(ZN5.mask)
                    OUTPUT['zmelt'].interpolation[i,c] = np.copy(ZN5.interpolation)
                    # calculate derived fields
                    OUTPUT['zsmb'].data[i,c] = ZN6.data - ZN4.data
                    OUTPUT['zsmb'].mask[i,c] = ZN4.mask | ZN6.mask
                    OUTPUT['zaccum'].data[i,c] = ZN6.data - ZN4.data - ZN5.data
                    OUTPUT['zaccum'].mask[i,c] = ZN4.mask | ZN5.mask | ZN6.mask
                elif (MODEL == 'RACMO'):
                    # read and interpolate daily RACMO outputs
                    hgtsrf = SMBcorr.interpolate_racmo_daily(base_dir, EPSG,
                        RACMO_MODEL, tdec, D11.x[i,c], D11.y[i,c],
                        VARIABLE='hgtsrf', SIGMA=1.5, FILL_VALUE=np.nan)
                    # set attributes to output for iteration
                    OUTPUT['zsurf'].data[i,c] = np.copy(hgtsrf.data)
                    OUTPUT['zsurf'].mask[i,c] = np.copy(hgtsrf.mask)
                    OUTPUT['zsurf'].interpolation[i,c] = np.copy(hgtsrf.interpolation)

        # append input HDF5 file with new firn model outputs
        fileID = h5py.File(os.path.expanduser(input_file),'a')
        fileID.create_group(model_version)
        h5 = {}
        for key in KEYS:
            # verify mask values
            OUTPUT[key].mask |= (OUTPUT[key].data == OUTPUT[key].fill_value) | \
                    np.isnan(OUTPUT[key].data)
            OUTPUT[key].data[OUTPUT[key].mask] = OUTPUT[key].fill_value
            # output variable to HDF5
            val = '{0}/{1}'.format(model_version,key)
            h5[key] = fileID.create_dataset(val, OUTPUT[key].shape,
                data=OUTPUT[key], dtype=OUTPUT[key].dtype,
                compression='gzip', fillvalue=OUTPUT[key].fill_value)
            h5[key].attrs['units'] = "m"
            h5[key].attrs['long_name'] = LONGNAME[key]
            h5[key].attrs['coordinates'] = "../delta_time ../latitude ../longitude"
            h5[key].attrs['model'] = model_version
        # close the output HDF5 file
        fileID.close()

# PURPOSE: help module to describe the optional input parameters
def usage():
    print('\nHelp: {}'.format(os.path.basename(sys.argv[0])))
    print(' -D X, --directory=X\tWorking data directory')
    print(' -R X, --region=X\tRegion of model to interpolate')
    print(' -M X, --model=X\tRegional climate model to run\n')

# Main program that calls append_SMB_ATL11()
def main():
    # Read the system arguments listed after the program
    long_options = ['help','directory=','region=','model=']
    optlist,arglist = getopt.getopt(sys.argv[1:], 'hD:R:M:', long_options)

    # data directory
    base_dir = None
    # region of firn model
    REGION = 'GL'
    # surface mass balance product
    MODELS = ['RACMO','MAR']
    # extract parameters
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

    # run program with parameters
    for f in arglist:
        for m in MODELS:
            append_SMB_ATL11(os.path.expanduser(f),base_dir,REGION,m)

# run main program
if __name__ == '__main__':
    main()
