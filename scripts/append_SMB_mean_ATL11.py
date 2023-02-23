#!/usr/bin/env python
u"""
append_SMB_mean_ATL11.py
Written by Tyler Sutterley (08/2022)
Interpolates mean estimates of model firn variable to the coordinates
    of an ATL11 file

CALLING SEQUENCE:
    python append_SMB_mean_ATL11.py --directory=<path> --region=GL <path_to_file>

INPUTS:
    Merged ATL11 file

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -R X, --region X: Region to interpolate (GL, AA)
    -M X, --model X: Regional climate models to run
    -V, --verbose: Output information about each created file

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
    Updated 08/2022: use argparse descriptions within documentation
    Updated 12/2021: added GSFC MERRA-2 Hybrid Greenland v1.2
    Updated 10/2021: using python logging for handling verbose output
    Updated 04/2021: added GSFC MERRA-2 Hybrid Antarctica v1.1
    Updated 02/2021: added new MERRA2-hybrid v1.1 variables
        added new MARv3.11.5 Greenland outputs
        set a keyword argument dict with standard and optional parameters
    Updated 01/2021: using utilities from time module for conversions
    Updated 09/2020: added MARv3.11.2 6km outputs and MERRA2-hybrid subversions
    Written 06/2020
"""
import sys
import os
import re
import logging
import SMBcorr
import argparse
import warnings
import numpy as np
# attempt imports
try:
    import h5py
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("h5py not available", ImportWarning)
try:
    import pointCollection as pc
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("pointCollection not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

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

def append_SMB_mean_ATL11(input_file, base_dir, REGION, MODEL,
    RANGE=[2000,2019], VERBOSE=False):

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

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
    # models['GL']['MAR'].append('MARv3.11.2-ERA-6km')
    # models['GL']['MAR'].append('MARv3.11.2-ERA-7.5km')
    # models['GL']['MAR'].append('MARv3.11.2-ERA-10km')
    # models['GL']['MAR'].append('MARv3.11.2-ERA-15km')
    # models['GL']['MAR'].append('MARv3.11.2-ERA-20km')
    # models['GL']['MAR'].append('MARv3.11.2-NCEP-20km')
    # models['GL']['MAR'].append('MARv3.11.5-ERA-6km')
    models['GL']['MAR'].append('MARv3.11.5-ERA-10km')
    models['GL']['MAR'].append('MARv3.11.5-ERA-15km')
    models['GL']['MAR'].append('MARv3.11.5-ERA-20km')

    # RACMO
    models['GL']['RACMO'] = []
    # models['GL']['RACMO'].append('RACMO2.3-XGRN11')
    # models['GL']['RACMO'].append('RACMO2.3p2-XGRN11')
    models['GL']['RACMO'].append('RACMO2.3p2-FGRN055')

    # MERRA2-hybrid
    models['GL']['MERRA2-hybrid'] = []
    # models['GL']['MERRA2-hybrid'].append('GSFC-fdm-v0')
    # models['GL']['MERRA2-hybrid'].append('GSFC-fdm-v1')
    # models['GL']['MERRA2-hybrid'].append('GSFC-fdm-v1.0')
    # models['GL']['MERRA2-hybrid'].append('GSFC-fdm-v1.1')
    # models['GL']['MERRA2-hybrid'].append('GSFC-fdm-v1.2')
    models['GL']['MERRA2-hybrid'].append('GSFC-fdm-v1.2.1')
    models['AA']['MERRA2-hybrid'] = []
    # models['AA']['MERRA2-hybrid'].append('GSFC-fdm-v0')
    # models['AA']['MERRA2-hybrid'].append('GSFC-fdm-v1')
    # models['AA']['MERRA2-hybrid'].append('GSFC-fdm-v1.1')
    models['AA']['MERRA2-hybrid'].append('GSFC-fdm-v1.2.1')

    # for each model to append to ATL11
    for model_version in models[REGION][MODEL]:
        # keyword arguments for all models
        # add range to input keyword arguments
        KWARGS = dict(SIGMA=1.5, FILL_VALUE=np.nan, RANGE=RANGE)
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
            SUBDIRECTORY['GL']['MARv3.11.5-ERA-6km']=['6km_ERA5']
            SUBDIRECTORY['GL']['MARv3.11.5-ERA-10km']=['10km_ERA5']
            SUBDIRECTORY['GL']['MARv3.11.5-ERA-15km']=['15km_ERA5']
            SUBDIRECTORY['GL']['MARv3.11.5-ERA-20km']=['20km_ERA5']
            MAR_MODEL=SUBDIRECTORY[REGION][model_version]
            DIRECTORY=os.path.join(base_dir,'MAR',MAR_VERSION,MAR_REGION,*MAR_MODEL)
            # keyword arguments for variable coordinates
            MAR_KWARGS=dict(AA={}, GL={})
            MAR_KWARGS['GL']['MARv3.9-ERA'] = dict(XNAME='X10_153',YNAME='Y21_288')
            MAR_KWARGS['GL']['MARv3.10-ERA'] = dict(XNAME='X10_105',YNAME='Y21_199')
            MAR_KWARGS['GL']['MARv3.11-NCEP'] = dict(XNAME='X12_84',YNAME='Y21_155')
            MAR_KWARGS['GL']['MARv3.11-ERA'] = dict(XNAME='X10_105',YNAME='Y21_199')
            MAR_KWARGS['GL']['MARv3.11.2-ERA-6km'] = dict(XNAME='X12_251',YNAME='Y20_465')
            MAR_KWARGS['GL']['MARv3.11.2-ERA-7.5km'] = dict(XNAME='X12_203',YNAME='Y20_377')
            MAR_KWARGS['GL']['MARv3.11.2-ERA-10km'] = dict(XNAME='X10_153',YNAME='Y21_288')
            MAR_KWARGS['GL']['MARv3.11.2-ERA-15km'] = dict(XNAME='X10_105',YNAME='Y21_199')
            MAR_KWARGS['GL']['MARv3.11.2-ERA-20km'] = dict(XNAME='X12_84',YNAME='Y21_155')
            MAR_KWARGS['GL']['MARv3.11.2-NCEP-20km'] = dict(XNAME='X12_84',YNAME='Y21_155')
            MAR_KWARGS['GL']['MARv3.11.5-ERA-6km'] = dict(XNAME='X12_251',YNAME='Y20_465')
            MAR_KWARGS['GL']['MARv3.11.5-ERA-10km'] = dict(XNAME='X10_153',YNAME='Y21_288')
            MAR_KWARGS['GL']['MARv3.11.5-ERA-15km'] = dict(XNAME='X10_105',YNAME='Y21_199')
            MAR_KWARGS['GL']['MARv3.11.5-ERA-20km'] = dict(XNAME='X12_84',YNAME='Y21_155')
            KWARGS.update(MAR_KWARGS[REGION][model_version])
            # netCDF4 variable names for direct fields
            VARIABLES = ['SMB','ZN6']
            # output variable keys for both direct and derived fields
            KEYS = ['smb_mean','zsurf_mean']
            # HDF5 longname attributes for each variable
            LONGNAME = {}
            LONGNAME['smb_mean'] = "Ice Sheet Surface Mass Balance"
            LONGNAME['zsurf_mean'] = "Mean change in surface height"
        elif (MODEL == 'RACMO'):
            RACMO_VERSION,RACMO_MODEL=model_version.split('-')
            # netCDF4 variable names
            VARIABLES = ['hgtsrf']
            # output variable keys
            KEYS = ['smb_mean']
            # HDF5 longname attributes for each variable
            LONGNAME = {}
            LONGNAME['smb_mean'] = "Ice Sheet Surface Mass Balance"
        elif (MODEL == 'MERRA2-hybrid'):
            # regular expression pattern for extracting version
            merra2_regex = re.compile(r'GSFC-fdm-((v\d+)(\.\d+)?(\.\d+)?)$')
            # get MERRA-2 version and major version
            MERRA2_VERSION = merra2_regex.match(model_version).group(1)
            # MERRA-2 hybrid directory
            DIRECTORY=os.path.join(base_dir,'MERRA2_hybrid',MERRA2_VERSION)
            # MERRA-2 region name from ATL11 region
            MERRA2_REGION = dict(AA='ais',GL='gris')[REGION]
            # keyword arguments for MERRA-2 interpolation programs
            if MERRA2_VERSION in ('v0','v1','v1.0'):
                KWARGS['VERSION'] = merra2_regex.match(model_version).group(2)
                VARIABLES = ['smb']
            else:
                KWARGS['VERSION'] = MERRA2_VERSION.replace('.','_')
                VARIABLES = ['SMB']
            # output variable keys for both direct and derived fields
            KEYS = ['smb_mean']
            # HDF5 longname attributes for each variable
            LONGNAME = {}
            LONGNAME['smb_mean'] = "Ice Sheet Surface Mass Balance"
        else:
            raise ValueError(f'Unknown model {MODEL}')

        # check if running crossover or along track
        if (D11.h_corr.ndim == 3):
            # allocate for output height for crossover data
            OUTPUT = {}
            for key in KEYS:
                OUTPUT[key] = np.ma.zeros((nseg,ncycle,ncross),fill_value=np.nan)
                OUTPUT[key].mask = np.ones((nseg,ncycle,ncross),dtype=bool)
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
                        for key,var in zip(KEYS,VARIABLES):
                            # read and interpolate daily MAR outputs
                            OUT =  SMBcorr.interpolate_mar_daily(DIRECTORY, EPSG,
                                MAR_VERSION, tdec, D11.x[i,c,xo], D11.y[i,c,xo],
                                VARIABLE=var, **KWARGS)
                            # set attributes to output for iteration
                            OUTPUT[key].data[i,c,xo] = np.copy(OUT.data)
                            OUTPUT[key].mask[i,c,xo] = np.copy(OUT.mask)
                            OUTPUT[key].interpolation[i,c,xo] = np.copy(OUT.interpolation)
                        # calculate derived fields
                        OUTPUT['zsmb_mean'].data[i,c,xo] = OUTPUT['zsurf_mean'].data[i,c,xo] - \
                            OUTPUT['zfirn_mean'].data[i,c,xo]
                        OUTPUT['zsmb_mean'].mask[i,c,xo] = OUTPUT['zsurf_mean'].mask[i,c,xo] | \
                            OUTPUT['zfirn_mean'].mask[i,c,xo]
                        OUTPUT['zaccum_mean'].data[i,c,xo] = OUTPUT['zsurf_mean'].data[i,c,xo] - \
                            OUTPUT['zfirn_mean'].data[i,c,xo] - OUTPUT['zmelt_mean'].data[i,c,xo]
                        OUTPUT['zaccum_mean'].mask[i,c,xo] = OUTPUT['zsurf_mean'].mask[i,c,xo] | \
                            OUTPUT['zfirn_mean'].mask[i,c,xo] | OUTPUT['zmelt_mean'].mask[i,c,xo]
                    elif (MODEL == 'RACMO'):
                        # read and interpolate daily RACMO outputs
                        for key,var in zip(KEYS,VARIABLES):
                            OUT = SMBcorr.interpolate_racmo_daily(base_dir, EPSG,
                                RACMO_MODEL, tdec, D11.x[i,c,xo], D11.y[i,c,xo],
                                VARIABLE=var, **KWARGS)
                            # set attributes to output for iteration
                            OUTPUT[key].data[i,c,xo] = np.copy(OUT.data)
                            OUTPUT[key].mask[i,c,xo] = np.copy(OUT.mask)
                            OUTPUT[key].interpolation[i,c,xo] = np.copy(OUT.interpolation)
                    elif (MODEL == 'MERRA2-hybrid'):
                        # read and interpolate 5-day MERRA2-Hybrid outputs
                        for key,var in zip(KEYS,VARIABLES):
                            OUT = SMBcorr.interpolate_merra_hybrid(DIRECTORY, EPSG,
                                MERRA2_REGION, tdec, D11.x[i,c,xo], D11.y[i,c,xo],
                                VARIABLE=var, **KWARGS)
                            # set attributes to output for iteration
                            OUTPUT[key].data[i,c,xo] = np.copy(OUT.data)
                            OUTPUT[key].mask[i,c,xo] = np.copy(OUT.mask)
                            OUTPUT[key].interpolation[i,c,xo] = np.copy(OUT.interpolation)
        else:
            # allocate for output height for along-track data
            OUTPUT = {}
            for key in KEYS:
                OUTPUT[key] = np.ma.zeros((nseg,ncycle),fill_value=np.nan)
                OUTPUT[key].mask = np.ones((nseg,ncycle),dtype=bool)
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
                    for key,var in zip(KEYS,VARIABLES):
                        # read and interpolate daily MAR outputs
                        OUT =  SMBcorr.interpolate_mar_daily(DIRECTORY, EPSG,
                            MAR_VERSION, tdec, D11.x[i,c], D11.y[i,c],
                            VARIABLE=var, **KWARGS)
                        # set attributes to output for iteration
                        OUTPUT[key].data[i,c] = np.copy(OUT.data)
                        OUTPUT[key].mask[i,c] = np.copy(OUT.mask)
                        OUTPUT[key].interpolation[i,c] = np.copy(OUT.interpolation)
                    # calculate derived fields
                    OUTPUT['zsmb_mean'].data[i,c] = OUTPUT['zsurf_mean'].data[i,c] - \
                        OUTPUT['zfirn_mean'].data[i,c]
                    OUTPUT['zsmb_mean'].mask[i,c] = OUTPUT['zsurf_mean'].mask[i,c] | \
                        OUTPUT['zfirn_mean'].mask[i,c]
                    OUTPUT['zaccum'].data[i,c] = OUTPUT['zsurf_mean'].data[i,c] - \
                        OUTPUT['zfirn_mean'].data[i,c] - OUTPUT['zmelt_mean'].data[i,c]
                    OUTPUT['zaccum'].mask[i,c] = OUTPUT['zsurf_mean'].mask[i,c] | \
                        OUTPUT['zfirn_mean'].mask[i,c] | OUTPUT['zmelt_mean'].mask[i,c]
                elif (MODEL == 'RACMO'):
                    # read and interpolate daily RACMO outputs
                    for key,var in zip(KEYS,VARIABLES):
                        OUT = SMBcorr.interpolate_racmo_daily(base_dir, EPSG,
                            RACMO_MODEL, tdec, D11.x[i,c], D11.y[i,c],
                            VARIABLE=var, **KWARGS)
                        # set attributes to output for iteration
                        OUTPUT[key].data[i,c] = np.copy(OUT.data)
                        OUTPUT[key].mask[i,c] = np.copy(OUT.mask)
                        OUTPUT[key].interpolation[i,c] = np.copy(OUT.interpolation)
                elif (MODEL == 'MERRA2-hybrid'):
                    # read and interpolate 5-day MERRA2-Hybrid outputs
                    for key,var in zip(KEYS,VARIABLES):
                        OUT = SMBcorr.interpolate_merra_hybrid(DIRECTORY, EPSG,
                            MERRA2_REGION, tdec, D11.x[i,c], D11.y[i,c],
                            VARIABLE=var, **KWARGS)
                        # set attributes to output for iteration
                        OUTPUT[key].data[i,c] = np.copy(OUT.data)
                        OUTPUT[key].mask[i,c] = np.copy(OUT.mask)
                        OUTPUT[key].interpolation[i,c] = np.copy(OUT.interpolation)

        # append input HDF5 file with new firn model outputs
        fileID = h5py.File(os.path.expanduser(input_file),'a')
        logging.info(input_file)
        # fileID.create_group(model_version) if model_version not in fileID.keys() else None
        h5 = {}
        for key in KEYS:
            logging.info(f'{sys.argv[0]}: writing{key}')
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

# PURPOSE: create arguments parser
def arguments():
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
        metavar='REGION', type=str,
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
    # verbosity settings
    # verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Output information about each created file')
    # return the parser
    return parser

# Main program that calls append_SMB_mean_ATL11()
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args = parser.parse_args()

    # run program with parameters
    for f in args.infile:
        for m in args.model:
            append_SMB_mean_ATL11(f, args.directory, args.region, m,
                RANGE=args.year, VERBOSE=args.verbose)

# run main program
if __name__ == '__main__':
    main()