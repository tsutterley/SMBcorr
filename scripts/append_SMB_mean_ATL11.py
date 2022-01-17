#!/usr/bin/env python
u"""
append_SMB_mean_ATL11.py
Written by Tyler Sutterley (01/2021)
Interpolates mean estimates of model firn variable to the coordinates
    of an ATL11 file

CALLING SEQUENCE:
    python append_SMB_mean_ATL11.py --directory=<path> --region=GL <path_to_file>

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
    Updated 01/2021: using utilities from time module for conversions
    Updated 09/2020: added MARv3.11.2 6km outputs and MERRA2-hybrid subversions
    Written 06/2020
"""
import os
import re
import h5py
import SMBcorr
import argparse
import numpy as np
import pointCollection as pc

# PURPOSE: convert time from delta seconds into Julian and year-decimal
def convert_delta_time(delta_time, gps_epoch=1198800018.0):
    # calculate gps time from delta_time
    gps_seconds = gps_epoch + delta_time
    time_leaps = SMBcorr.time.count_leap_seconds(gps_seconds)
    # calculate julian time
    julian = 2400000.5 + SMBcorr.time.convert_delta_time(gps_seconds - time_leaps,
        epoch1=(1980,1,6,0,0,0), epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0)
    # convert to calendar date
    Y,M,D,h,m,s = SMBcorr.time.convert_julian(julian,FORMAT='tuple')
    # calculate year-decimal time
    decimal = SMBcorr.time.convert_calendar_decimal(Y,M,day=D,
        hour=h,minute=m,second=s)
    # return both the Julian and year-decimal formatted dates
    return dict(julian=julian, decimal=decimal)

# PURPOSE: set the projection parameters based on the region name
def set_projection(REGION):
    if (REGION == 'AA'):
        projection_flag = 'EPSG:3031'
    elif (REGION == 'GL'):
        projection_flag = 'EPSG:3413'
    return projection_flag

def append_SMB_mean_ATL11(input_file,base_dir,REGION,MODEL,RANGE=[2000,2019]):

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
    # MAR
    models['GL']['MAR'] = []
    # models['GL']['MAR'].append('MARv3.9-ERA')
    # models['GL']['MAR'].append('MARv3.10-ERA')
    # models['GL']['MAR'].append('MARv3.11-NCEP')
    # models['GL']['MAR'].append('MARv3.11-ERA')
    models['GL']['MAR'].append('MARv3.11.2-ERA-6km')
    #models['GL']['MAR'].append('MARv3.11.2-ERA-7.5km')
    models['GL']['MAR'].append('MARv3.11.2-ERA-10km')
    #models['GL']['MAR'].append('MARv3.11.2-ERA-15km')
    models['GL']['MAR'].append('MARv3.11.2-ERA-20km')
    models['GL']['MAR'].append('MARv3.11.2-NCEP-20km')
    # RACMO
    models['GL']['RACMO'] = []
    # models['GL']['RACMO'].append('RACMO2.3-XGRN11')
    # models['GL']['RACMO'].append('RACMO2.3p2-XGRN11')
    models['GL']['RACMO'].append('RACMO2.3p2-FGRN055')
    # MERRA2-hybrid
    models['GL']['MERRA2-hybrid'] = []
    # models['GL']['MERRA2-hybrid'].append('GSFC-fdm-v0')
    models['GL']['MERRA2-hybrid'].append('GSFC-fdm-v1')
    models['AA']['MERRA2-hybrid'] = []
    # models['AA']['MERRA2-hybrid'].append('GSFC-fdm-v0')
    models['AA']['MERRA2-hybrid'].append('GSFC-fdm-v1')

    for model_version in models[REGION][MODEL]:
        if (MODEL == 'MAR'):
            match_object=re.match(r'(MARv\d+\.\d+(.\d+)?)',model_version)
            MAR_VERSION=match_object.group(0)
            MAR_REGION=dict(GL='Greenland',AA='Antarctic')[REGION]
            # model subdirectories
            SUBDIRECTORY=dict(AA={}, GL={})
            SUBDIRECTORY['GL']['MARv3.9-ERA']=['ERA_1958-2018_10km','daily_10km']
            SUBDIRECTORY['GL']['MARv3.10-ERA']=['ERA_1958-2019-15km','daily_15km']
            SUBDIRECTORY['GL']['MARv3.11-NCEP']=['NCEP1_1948-2020_20km','daily_20km']
            SUBDIRECTORY['GL']['MARv3.11-ERA']=['ERA_1958-2019-15km','daily_15km']
            SUBDIRECTORY['GL']['MARv3.11.2-ERA-6km']=['6km_ERA5']
            SUBDIRECTORY['GL']['MARv3.11.2-ERA-7.5km']=['7.5km_ERA5']
            SUBDIRECTORY['GL']['MARv3.11.2-ERA-10km']=['10km_ERA5']
            SUBDIRECTORY['GL']['MARv3.11.2-ERA-15km']=['15km_ERA5']
            SUBDIRECTORY['GL']['MARv3.11.2-ERA-20km']=['20km_ERA5']
            SUBDIRECTORY['GL']['MARv3.11.2-NCEP-20km']=['20km_NCEP1']
            MAR_MODEL=SUBDIRECTORY[REGION][model_version]
            DIRECTORY=os.path.join(base_dir,'MAR',MAR_VERSION,MAR_REGION,*MAR_MODEL)
            # variable coordinates
            KWARGS=dict(AA={}, GL={})
            KWARGS['GL']['MARv3.9-ERA'] = dict(XNAME='X10_153',YNAME='Y21_288')
            KWARGS['GL']['MARv3.10-ERA'] = dict(XNAME='X10_105',YNAME='Y21_199')
            KWARGS['GL']['MARv3.11-NCEP'] = dict(XNAME='X12_84',YNAME='Y21_155')
            KWARGS['GL']['MARv3.11-ERA'] = dict(XNAME='X10_105',YNAME='Y21_199')
            KWARGS['GL']['MARv3.11.2-ERA-6km'] = dict(XNAME='X12_251',YNAME='Y20_465')
            KWARGS['GL']['MARv3.11.2-ERA-7.5km'] = dict(XNAME='X12_203',YNAME='Y20_377')
            KWARGS['GL']['MARv3.11.2-ERA-10km'] = dict(XNAME='X10_153',YNAME='Y21_288')
            KWARGS['GL']['MARv3.11.2-ERA-15km'] = dict(XNAME='X10_105',YNAME='Y21_199')
            KWARGS['GL']['MARv3.11.2-ERA-20km'] = dict(XNAME='X12_84',YNAME='Y21_155')
            KWARGS['GL']['MARv3.11.2-NCEP-20km'] = dict(XNAME='X12_84',YNAME='Y21_155')
            MAR_KWARGS=KWARGS[REGION][model_version]
            MAR_KWARGS['RANGE']=RANGE
            # output variable keys for both direct and derived fields
            KEYS = ['smb_mean','zsurf_mean']
            # HDF5 longname attributes for each variable
            LONGNAME = {}
            LONGNAME['smb_mean'] = "Ice Sheet Surface Mass Balance"
            LONGNAME['zsurf_mean'] = "Mean change in surface height"
        elif (MODEL == 'RACMO'):
            RACMO_VERSION,RACMO_MODEL=model_version.split('-')
            # output variable keys
            KEYS = ['smb_mean']
            # HDF5 longname attributes for each variable
            LONGNAME = {}
            LONGNAME['smb_mean'] = "Ice Sheet Surface Mass Balance"
        elif (MODEL == 'MERRA2-hybrid'):
            merra2_regex = re.compile(r'GSFC-fdm-((v\d+)(\.\d+)?)$')
            # get MERRA-2 version and major version
            MERRA2_VERSION = merra2_regex.match(model_version).group(1)
            MERRA2_MAJOR_VERSION = merra2_regex.match(model_version).group(2)
            # MERRA-2 hybrid directory
            DIRECTORY=os.path.join(base_dir,'MERRA2_hybrid',MERRA2_VERSION)
            MERRA2_REGION = dict(AA='ais',GL='gris')[REGION]
            # output variable keys for both direct and derived fields
            KEYS = ['smb_mean']
            # HDF5 longname attributes for each variable
            LONGNAME = {}
            LONGNAME['smb_mean'] = "Ice Sheet Surface Mass Balance"

        # check if running crossover or along track
        if (D11.h_corr.ndim == 3):
            # allocate for output height for crossover data
            OUTPUT = {}
            for key in KEYS:
                OUTPUT[key] = np.ma.zeros((nseg,ncycle,ncross),fill_value=np.nan)
                OUTPUT[key].mask = np.ones((nseg,ncycle,ncross),dtype=np.bool)
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
                        SMB = SMBcorr.interpolate_mar_mean(DIRECTORY, EPSG,
                            MAR_VERSION, tdec, D11.x[i,c,xo], D11.y[i,c,xo],
                            VARIABLE='SMB', SIGMA=1.5, FILL_VALUE=np.nan, **MAR_KWARGS)
                        # set attributes to output for iteration
                        OUTPUT['smb_mean'].data[i,c,xo] = np.copy(SMB.data)
                        OUTPUT['smb_mean'].mask[i,c,xo] = np.copy(SMB.mask)
                        zsurf = SMBcorr.interpolate_mar_mean(DIRECTORY, EPSG,
                             MAR_VERSION, tdec, D11.x[i,c,xo], D11.y[i,c,xo],
                             VARIABLE='ZN6', SIGMA=1.5, FILL_VALUE=np.nan, **MAR_KWARGS)
                        OUTPUT['zsurf_mean'].data[i,c,xo] = np.copy(zsurf.data)
                        OUTPUT['zsurf_mean'].mask[i,c,xo] = np.copy(zsurf.mask)
                    # elif (MODEL == 'RACMO'):
                    #     # read and interpolate daily RACMO outputs
                    #     SMB = SMBcorr.interpolate_racmo_mean(base_dir, EPSG,
                    #         RACMO_MODEL, tdec, D11.x[i,c,xo], D11.y[i,c,xo],
                    #         VARIABLE='SMB', SIGMA=1.5, FILL_VALUE=np.nan)
                    #     # set attributes to output for iteration
                    #     OUTPUT['smb_mean'].data[i,c,xo] = np.copy(SMB.data)
                    #     OUTPUT['smb_mean'].mask[i,c,xo] = np.copy(SMB.mask)
                    # elif (MODEL == 'MERRA2-hybrid'):
                    #     # read and interpolate 5-day MERRA2-Hybrid outputs
                    #     smb = SMBcorr.interpolate_merra_hybrid_mean(DIRECTORY, EPSG,
                    #         MERRA2_REGION, tdec, D11.x[i,c,xo], D11.y[i,c,xo],
                    #         VERSION=MERRA2_MAJOR_VERSION, VARIABLE='cum_smb_anomaly',
                    #         SIGMA=1.5, FILL_VALUE=np.nan)
                    #     # set attributes to output for iteration
                    #     OUTPUT['smb_mean'].data[i,c,xo] = np.copy(smb.data)
                    #     OUTPUT['smb_mean'].mask[i,c,xo] = np.copy(smb.mask)
        else:
            # allocate for output height for along-track data
            OUTPUT = {}
            for key in KEYS:
                OUTPUT[key] = np.ma.zeros((nseg,ncycle),fill_value=np.nan)
                OUTPUT[key].mask = np.ones((nseg,ncycle),dtype=np.bool)
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
                    SMB = SMBcorr.interpolate_mar_mean(DIRECTORY, EPSG,
                        MAR_VERSION, tdec, D11.x[i,c], D11.y[i,c],
                        VARIABLE='SMB', SIGMA=1.5, FILL_VALUE=np.nan, **MAR_KWARGS)
                    # set attributes to output for iteration
                    OUTPUT['smb_mean'].data[i,c] = np.copy(SMB.data)
                    OUTPUT['smb_mean'].mask[i,c] = np.copy(SMB.mask)
                    zsurf = SMBcorr.interpolate_mar_mean(DIRECTORY, EPSG,
                        MAR_VERSION, tdec, D11.x[i,c], D11.y[i,c],
                        VARIABLE='ZN6', SIGMA=1.5, FILL_VALUE=np.nan, **MAR_KWARGS)
                    # set attributes to output for iteration
                    OUTPUT['zsurf_mean'].data[i,c] = np.copy(zsurf.data)
                    OUTPUT['zsurf_mean'].mask[i,c] = np.copy(zsurf.mask)
                # elif (MODEL == 'RACMO'):
                #     # read and interpolate daily RACMO outputs
                #     SMB = SMBcorr.interpolate_racmo_mean(base_dir, EPSG,
                #         RACMO_MODEL, tdec, D11.x[i,c], D11.y[i,c],
                #         VARIABLE='SMB', SIGMA=1.5, FILL_VALUE=np.nan)
                #     # set attributes to output for iteration
                #     OUTPUT['smb_mean'].data[i,c] = np.copy(SMB.data)
                #     OUTPUT['smb_mean'].mask[i,c] = np.copy(SMB.mask)
                # elif (MODEL == 'MERRA2-hybrid'):
                #     # read and interpolate 5-day MERRA2-Hybrid outputs
                #     smb = SMBcorr.interpolate_merra_hybrid_mean(DIRECTORY, EPSG,
                #         MERRA2_REGION, tdec, D11.x[i,c], D11.y[i,c],
                #         VERSION=MERRA2_MAJOR_VERSION, VARIABLE='cum_smb_anomaly',
                #         SIGMA=1.5, FILL_VALUE=np.nan)
                #     # set attributes to output for iteration
                #     OUTPUT['smb_mean'].data[i,c] = np.copy(smb.data)
                #     OUTPUT['smb_mean'].mask[i,c] = np.copy(smb.mask)

        # append input HDF5 file with new firn model outputs
        fileID = h5py.File(os.path.expanduser(input_file),'a')
        # fileID.create_group(model_version) if model_version not in fileID.keys() else None
        h5 = {}
        for key in KEYS:
            # verify mask values
            OUTPUT[key].mask |= (OUTPUT[key].data == OUTPUT[key].fill_value) | \
                    np.isnan(OUTPUT[key].data)
            OUTPUT[key].data[OUTPUT[key].mask] = OUTPUT[key].fill_value
            # output variable to HDF5
            val = '{0}/{1}'.format(model_version,key)
            if val not in fileID:
                h5[key] = fileID.create_dataset(val, OUTPUT[key].shape,
                                                data=OUTPUT[key], dtype=OUTPUT[key].dtype,
                                                compression='gzip', fillvalue=OUTPUT[key].fill_value)
            else:
                h5[key]=fileID[val]
                fileID[val][...]=OUTPUT[key]
            h5[key].attrs['units'] = "m"
            h5[key].attrs['long_name'] = LONGNAME[key]
            h5[key].attrs['coordinates'] = "../delta_time ../latitude ../longitude"
            h5[key].attrs['model'] = model_version
        # close the output HDF5 file
        fileID.close()

# Main program that calls append_SMB_mean_ATL11()
def main():
    # Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Interpolates mean estimates of model firn
            variable to the coordinates of an ATL11 file
            """
    )
    # command line parameters
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat-2 ATL11 file to run')
    # data directory
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
    # region of firn model
    parser.add_argument('--region','-R',
        metavar='REGION', type=str, nargs='+',
        default=['GL'], choices=('AA','GL'),
        help='Region of model to interpolate')
    # surface mass balance product
    parser.add_argument('--model','-M',
        metavar='MODEL', type=str, nargs='+',
        default=['MAR'], choices=('MAR','RACMO','MERRA2-hybrid'),
        help='Regional climate model to run')
    # range of years to use for climatology
    parser.add_argument('--year','-Y',
        metavar=('START','END'), type=int, nargs=2,
        default=[2000,2019],
        help='Range of years to use in climatology')
    args = parser.parse_args()

    # run program with parameters
    for f in args.infile:
        for m in args.model:
            append_SMB_mean_ATL11(f,args.directory,
                args.region,m,RANGE=args.year)

# run main program
if __name__ == '__main__':
    main()