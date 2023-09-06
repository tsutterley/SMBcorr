#!/usr/bin/env python
u"""
interp_SMB_correction.py
Written by Tyler Sutterley (05/2023)
Interpolates SMB and firn model estimates for correcting surface elevations

INPUTS:
    csv file with columns for spatial and temporal coordinates
    HDF5 file with variables for spatial and temporal coordinates
    netCDF4 file with variables for spatial and temporal coordinates
    geotiff file with bands in spatial coordinates
    parquet file with variables for spatial and temporal coordinates

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -m X, --model X: Regional firn model to run
    -r X, --region X: Region for firn model
    -F X, --format X: input and output data format
        csv (default)
        netCDF4
        HDF5
        geotiff
        parquet
    -v X, --variables X: variable names of data in csv, HDF5 or netCDF4 file
        for csv files: the order of the columns within the file
        for HDF5, netCDF4 and parquet files: time, y, x and data variable names
    -H X, --header X: number of header lines for csv files
    --delimiter X: Delimiter for csv or ascii files
    -t X, --type X: input data type
        drift: drift buoys or satellite/airborne altimetry (time per data point)
        grid: spatial grids or images (single time for all data points)
        time series: time series at a single point
    -e X, --epoch X: Reference epoch of input time (default Modified Julian Day)
        days since 1858-11-17T00:00:00
    -d X, --deltatime X: Input delta time for files without date information
        can be set to 0 to use exact calendar date from epoch
    -s X, --standard X: Input time standard for delta times or input time type
        UTC: Coordinate Universal Time
        GPS: GPS Time
        LORAN: Long Range Navigator Time
        TAI: International Atomic Time
        datetime: formatted datetime string in UTC
    -P X, --projection X: spatial projection as EPSG code or PROJ4 string
        4326: latitude and longitude coordinates on WGS84 reference ellipsoid
    -G, --gzip: Model files are gzip compressed
    -V, --verbose: Verbose output of processing run
    -M X, --mode X: Permission mode of output file

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
    netCDF4: Python interface to the netCDF C library
         https://unidata.github.io/netcdf4-python/netCDF4/index.html
    gdal: Pythonic interface to the Geospatial Data Abstraction Library (GDAL)
        https://pypi.python.org/pypi/GDAL
    dateutil: powerful extensions to datetime
        https://dateutil.readthedocs.io/en/stable/
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/

PROGRAM DEPENDENCIES:
    time.py: utilities for calculating time operations
    spatial.py: utilities for reading and writing spatial data

UPDATE HISTORY:
    Written 09/2023
"""
from __future__ import print_function

import sys
import re
import logging
import pathlib
import argparse
import numpy as np
import SMBcorr

# attempt imports
try:
    import pandas as pd
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    logging.critical("geopandas not available")
try:
    import pyproj
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    logging.critical("pyproj not available")
try:
    import timescale
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    logging.critical("timescale not available")

# available models
models = dict(AA={}, GL={})
# MAR
models['GL']['MAR'] = []
models['GL']['MAR'].append('MARv3.9-ERA')
models['GL']['MAR'].append('MARv3.10-ERA')
models['GL']['MAR'].append('MARv3.11-NCEP')
models['GL']['MAR'].append('MARv3.11-ERA')
models['GL']['MAR'].append('MARv3.11.2-ERA-6km')
models['GL']['MAR'].append('MARv3.11.2-ERA-7.5km')
models['GL']['MAR'].append('MARv3.11.2-ERA-10km')
models['GL']['MAR'].append('MARv3.11.2-ERA-15km')
models['GL']['MAR'].append('MARv3.11.2-ERA-20km')
models['GL']['MAR'].append('MARv3.11.2-NCEP-20km')
models['GL']['MAR'].append('MARv3.11.5-ERA-6km')
models['GL']['MAR'].append('MARv3.11.5-ERA-10km')
models['GL']['MAR'].append('MARv3.11.5-ERA-15km')
models['GL']['MAR'].append('MARv3.11.5-ERA-20km')
# RACMO
models['GL']['RACMO'] = []
models['GL']['RACMO'].append('RACMO2.3-XGRN11')
models['GL']['RACMO'].append('RACMO2.3p2-XGRN11')
models['GL']['RACMO'].append('RACMO2.3p2-FGRN055')
# MERRA2-hybrid
models['GL']['MERRA2-hybrid'] = []
models['GL']['MERRA2-hybrid'].append('GSFC-fdm-v0')
models['GL']['MERRA2-hybrid'].append('GSFC-fdm-v1')
models['GL']['MERRA2-hybrid'].append('GSFC-fdm-v1.0')
models['GL']['MERRA2-hybrid'].append('GSFC-fdm-v1.1')
models['GL']['MERRA2-hybrid'].append('GSFC-fdm-v1.2.1')
models['AA']['MERRA2-hybrid'] = []
models['AA']['MERRA2-hybrid'].append('GSFC-fdm-v0')
models['AA']['MERRA2-hybrid'].append('GSFC-fdm-v1')
models['AA']['MERRA2-hybrid'].append('GSFC-fdm-v1.1')
models['AA']['MERRA2-hybrid'].append('GSFC-fdm-v1.2.1')

# PURPOSE: try to get the projection information for the input file
def get_projection(attributes, PROJECTION):
    # coordinate reference system string from file
    try:
        crs = pyproj.CRS.from_string(attributes['projection'])
    except (ValueError,KeyError,pyproj.exceptions.CRSError):
        pass
    else:
        return crs
    # EPSG projection code
    try:
        crs = pyproj.CRS.from_epsg(int(PROJECTION))
    except (ValueError,pyproj.exceptions.CRSError):
        pass
    else:
        return crs
    # coordinate reference system string
    try:
        crs = pyproj.CRS.from_string(PROJECTION)
    except (ValueError,pyproj.exceptions.CRSError):
        pass
    else:
        return crs
    # no projection can be made
    raise pyproj.exceptions.CRSError

# PURPOSE: Interpolates SMB and firn model estimates for
# correcting surface elevations
def interp_SMB_correction(base_dir, input_file, output_file, model_version,
    REGION='AA',
    FORMAT='csv',
    VARIABLES=['time','lat','lon','data'],
    HEADER=0,
    DELIMITER=',',
    TYPE='drift',
    TIME_UNITS='days since 1858-11-17T00:00:00',
    TIME=None,
    TIME_STANDARD='UTC',
    PROJECTION='4326',
    GZIP=False,
    VERBOSE=False,
    MODE=0o775):

    # create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)


    # verify data directory
    base_dir = pathlib.Path(base_dir).expanduser().absolute()
    # determine main model group from region and model_version
    MODEL, = [key for key,val in models[REGION].items() if model_version in val]
    # keyword arguments for all models
    KWARGS = dict(SIGMA=1.5, FILL_VALUE=np.nan)
    # set model specific parameters
    if (MODEL == 'MAR'):
        match_object = re.match(r'(MARv\d+\.\d+(.\d+)?)',model_version)
        MAR_VERSION = match_object.group(0)
        MAR_REGION = dict(GL='Greenland', AA='Antarctic')[REGION]
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
        MAR_MODEL = SUBDIRECTORY[REGION][model_version]
        DIRECTORY = base_dir.joinpath('MAR', MAR_VERSION, MAR_REGION, *MAR_MODEL)
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
        VARIABLES = ['SMB','ZN6','ZN4','ZN5']
        # output variable keys for both direct and derived fields
        KEYS = ['SMB','zsurf','zfirn','zmelt','zsmb','zaccum']
        # HDF5 longname and description attributes for each variable
        LONGNAME = {}
        LONGNAME['SMB'] = "Cumulative SMB"
        LONGNAME['zsurf'] = "Height"
        LONGNAME['zfirn'] = "Compaction"
        LONGNAME['zmelt'] = "Surface Melt"
        LONGNAME['zsmb'] = "Surface Mass Balance"
        LONGNAME['zaccum'] = "Surface Accumulation"
        DESCRIPTION = {}
        DESCRIPTION['SMB'] = "Cumulative Surface Mass Balance"
        DESCRIPTION['zsurf'] = "Snow Height Change"
        DESCRIPTION['zfirn'] = "Snow Height Change due to Compaction"
        DESCRIPTION['zmelt'] = "Snow Height Change due to Surface Melt"
        DESCRIPTION['zsmb'] = "Snow Height Change due to Surface Mass Balance"
        DESCRIPTION['zaccum'] = "Snow Height Change due to Surface Accumulation"
    elif (MODEL == 'RACMO'):
        RACMO_VERSION, RACMO_MODEL = model_version.split('-')
        # netCDF4 variable names
        VARIABLES = ['hgtsrf']
        # output variable keys
        KEYS = ['zsurf']
        # HDF5 longname attributes for each variable
        LONGNAME = {}
        LONGNAME['zsurf'] = "Height"
        DESCRIPTION = {}
        DESCRIPTION['zsurf'] = "Snow Height Change"
    elif (MODEL == 'MERRA2-hybrid'):
        # regular expression pattern for extracting version
        merra2_regex = re.compile(r'GSFC-fdm-((v\d+)(\.\d+)?(\.\d+)?)$')
        # get MERRA-2 version and major version
        MERRA2_VERSION = merra2_regex.match(model_version).group(1)
        # MERRA-2 hybrid directory
        DIRECTORY = base_dir.joinpath('MERRA2_hybrid', MERRA2_VERSION)
        # MERRA-2 region name from ATL11 region
        MERRA2_REGION = dict(AA='ais',GL='gris')[REGION]
        # keyword arguments for MERRA-2 interpolation programs
        if MERRA2_VERSION in ('v0','v1','v1.0'):
            KWARGS['VERSION'] = merra2_regex.match(model_version).group(2)
            # netCDF4 variable names
            VARIABLES = ['FAC','cum_smb_anomaly','height']
            # add additional Greenland variables
            if (MERRA2_REGION == 'gris'):
                VARIABLES.append('runoff_anomaly')
        else:
            KWARGS['VERSION'] = MERRA2_VERSION.replace('.','_')
            # netCDF4 variable names
            VARIABLES = ['FAC','SMB_a','h_a']
            # add additional Greenland variables
            if (MERRA2_REGION == 'gris'):
                VARIABLES.append('Me_a')
        # use compressed files
        KWARGS['GZIP'] = GZIP
        # output variable keys
        KEYS = ['zfirn','zsmb','zsurf','zmelt']
        # HDF5 longname and description attributes for each variable
        LONGNAME = {}
        LONGNAME['zsurf'] = "Height"
        LONGNAME['zfirn'] = "Compaction"
        LONGNAME['zsmb'] = "Surface Mass Balance"
        LONGNAME['zmelt'] = "Surface Melt"
        DESCRIPTION = {}
        DESCRIPTION['zsurf'] = "Snow Height Change"
        DESCRIPTION['zfirn'] = "Snow Height Change due to Compaction"
        DESCRIPTION['zsmb'] = "Snow Height Change due to Surface Mass Balance"
        DESCRIPTION['zmelt'] = "Snow Height Change due to Surface Melt"

    # invalid value
    fill_value = -9999.0
    # output netCDF4 and HDF5 file attributes
    # will be added to YAML header in csv files
    attrib = {}
    # latitude
    attrib['lat'] = {}
    attrib['lat']['long_name'] = 'Latitude'
    attrib['lat']['units'] = 'Degrees_North'
    # longitude
    attrib['lon'] = {}
    attrib['lon']['long_name'] = 'Longitude'
    attrib['lon']['units'] = 'Degrees_East'
    # time
    attrib['time'] = {}
    attrib['time']['long_name'] = 'Time'
    attrib['time']['units'] = TIME_UNITS
    attrib['time']['calendar'] = 'standard'

    # read input file to extract time, spatial coordinates and data
    if (FORMAT == 'csv'):
        parse_dates = (TIME_STANDARD.lower() == 'datetime')
        dinput = SMBcorr.spatial.from_ascii(input_file, columns=VARIABLES,
            delimiter=DELIMITER, header=HEADER, parse_dates=parse_dates)
        attributes = dinput['attributes']
    elif (FORMAT == 'netCDF4'):
        field_mapping = SMBcorr.spatial.default_field_mapping(VARIABLES)
        dinput = SMBcorr.spatial.from_netCDF4(input_file,
            field_mapping=field_mapping)
        attributes = dinput['attributes']
    elif (FORMAT == 'HDF5'):
        field_mapping = SMBcorr.spatial.default_field_mapping(VARIABLES)
        dinput = SMBcorr.spatial.from_HDF5(input_file,
            field_mapping=field_mapping)
        attributes = dinput['attributes']
    elif (FORMAT == 'geotiff'):
        dinput = SMBcorr.spatial.from_geotiff(input_file)
        attributes = dinput['attributes']
        # copy global geotiff attributes for projection and grid parameters
        for att_name in ['projection','wkt','spacing','extent']:
            attrib[att_name] = dinput['attributes'][att_name]
    elif (FORMAT == 'parquet'):
        field_mapping = SMBcorr.spatial.default_field_mapping(VARIABLES)
        remap = SMBcorr.spatial.inverse_mapping(field_mapping)
        dinput.rename(columns=remap, inplace=True)
        attributes = None

    # update time variable if entered as argument
    if TIME is not None:
        dinput['time'] = np.copy(TIME)

    # get coordinate reference system of input data
    proj4_params = get_projection(attributes, PROJECTION).to_proj4()

    # extract time units from netCDF4 and HDF5 attributes or from TIME_UNITS
    try:
        time_string = dinput['attributes']['time']['units']
        epoch1, to_secs = timescale.time.parse_date_string(time_string)
    except (TypeError, KeyError, ValueError):
        epoch1, to_secs = timescale.time.parse_date_string(TIME_UNITS)

    # copy variables to output
    output = {}
    output['time'] = np.ravel(dinput['time'])
    if (TYPE == 'drift'):
        X = np.ravel(dinput['x'])
        Y = np.ravel(dinput['y'])
    else:
        raise ValueError(f'Unsupported data type {TYPE}')

    # convert delta times or datetimes objects to timescale
    if (TIME_STANDARD.lower() == 'datetime'):
        ts = timescale.time.Timescale().from_datetime(output['time'])
    else:
        # convert time to seconds
        ts = timescale.time.Timescale().from_deltatime(to_secs*output['time'],
            epoch=epoch1, standard=TIME_STANDARD)
    # number of time points
    nt = len(ts)

    # allocate for output height for each variable
    for key,var in zip(KEYS,VARIABLES):
        output[key] = np.ma.empty((nt), fill_value=fill_value)
        output[key].mask = np.ones((nt), dtype=bool)

    if (MODEL == 'MAR'):
        # read and interpolate daily MAR outputs
        for key,var in zip(KEYS,VARIABLES):
            OUT = SMBcorr.interpolate_mar_daily(DIRECTORY, proj4_params,
                MAR_VERSION, ts.year, X, Y,
                VARIABLE=var, **KWARGS)
            # set attributes to output for iteration
            output[key].data[:] = np.copy(OUT.data)
            output[key].mask[:] = np.copy(OUT.mask)
        # calculate derived fields
        output['zsmb'].data[:] = output['zsurf'].data[:] - \
            output['zfirn'].data[:]
        output['zsmb'].mask[:] = output['zsurf'].mask[:] | \
            output['zfirn'].mask[:]
        output['zaccum'].data[:] = output['zsurf'].data[:] - \
            output['zfirn'].data[:] - output['zmelt'].data
        output['zaccum'].mask[:] = output['zsurf'].mask[:] | \
            output['zfirn'].mask[:] | output['zmelt'].mask[:]
    elif (MODEL == 'RACMO'):
        # read and interpolate daily RACMO outputs
        for key,var in zip(KEYS,VARIABLES):
            OUT = SMBcorr.interpolate_racmo_daily(base_dir, proj4_params,
                RACMO_MODEL, ts.year, X, Y,
                VARIABLE=var, **KWARGS)
            # set attributes to output for iteration
            output[key].data[:] = np.copy(OUT.data)
            output[key].mask[:] = np.copy(OUT.mask)
    elif (MODEL == 'MERRA2-hybrid'):
        # read and interpolate 5-day MERRA2-Hybrid outputs
        for key,var in zip(KEYS,VARIABLES):
            OUT = SMBcorr.interpolate_merra_hybrid(DIRECTORY, proj4_params,
                MERRA2_REGION, ts.year, X, Y,
                VARIABLE=var, **KWARGS)
            # set attributes to output for iteration
            output[key].data[:] = np.copy(OUT.data)
            output[key].mask[:] = np.copy(OUT.mask)

    # output to file
    if (FORMAT == 'csv'):
        SMBcorr.spatial.to_ascii(output, attrib, output_file,
            delimiter=DELIMITER, header=False,
            columns=['time','y','x',*KEYS])
        # change the permissions level to MODE
        outfile.chmod(mode=MODE)
    elif (FORMAT == 'netCDF4'):
        SMBcorr.spatial.to_netCDF4(output, attrib, output_file,
            data_type=TYPE)
        # change the permissions level to MODE
        outfile.chmod(mode=MODE)
    elif (FORMAT == 'HDF5'):
        SMBcorr.spatial.to_HDF5(output, attrib, output_file)
        # change the permissions level to MODE
        outfile.chmod(mode=MODE)
    elif (FORMAT == 'geotiff'):
        for key in KEYS:
            # individual output files for each variable
            vars = (output_file.stem, key, output_file.suffix)
            outfile = output_file.with_name('{0}_{1}{2}'.format(*vars))
            SMBcorr.spatial.to_geotiff(output, attrib, outfile,
                varname=key, dtype=np.float32, fill_value=fill_value)
            # change the permissions level to MODE
            outfile.chmod(mode=MODE)
    elif (FORMAT == 'parquet'):
        # write to parquet file
        pd.DataFrame(output).to_parquet(output_file)
        # change the permissions level to MODE
        outfile.chmod(mode=MODE)

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Calculates SMB and firn model estimates for
            correcting surface elevations
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = SMBcorr.utilities.convert_arg_line_to_args
    # command line options
    # input and output file
    parser.add_argument('infile',
        type=pathlib.Path, nargs='?',
        help='Input file to run')
    parser.add_argument('outfile',
        type=pathlib.Path, nargs='?',
        help='Computed output file')
    # directory with model data
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=pathlib.Path.cwd(),
        help='Working data directory')
    # firn model
    choices = [v for mdl in models.values() for val in mdl.values() for v in val]
    parser.add_argument('--model','-m',
        metavar='FIRN', type=str, default='GSFC-fdm-v1.2',
        choices=sorted(set(choices)),
        help='Regional firn model to run')
    # model region
    parser.add_argument('--region','-r',
        type=str, default='AA',
        choices=('AA', 'GL'),
        help='Region for firn model')
    # input and output data format
    parser.add_argument('--format','-F',
        type=str, default='parquet',
        choices=('csv','netCDF4','HDF5','geotiff','parquet'),
        help='Input and output data format')
    # variable names (for csv names of columns)
    parser.add_argument('--variables','-v',
        type=str, nargs='+', default=['time','lat','lon','data'],
        help='Variable names of data in input file')
    # number of header lines for csv files
    parser.add_argument('--header','-H',
        type=int, default=0,
        help='Number of header lines for csv files')
    # delimiter for csv or ascii files
    parser.add_argument('--delimiter',
        type=str, default=',',
        help='Delimiter for csv or ascii files')
    # input data type
    # drift: drift buoys or satellite/airborne altimetry (time per data point)
    # grid: spatial grids or images (single time for all data points)
    # time series: station locations with multiple time values
    parser.add_argument('--type','-t',
        type=str, default='drift',
        choices=('drift','grid','time series'),
        help='Input data type')
    # time epoch (default Modified Julian Days)
    # in form "time-units since yyyy-mm-dd hh:mm:ss"
    parser.add_argument('--epoch','-e',
        type=str, default='days since 1858-11-17T00:00:00',
        help='Reference epoch of input time')
    # input delta time for files without date information
    parser.add_argument('--deltatime','-d',
        type=float, nargs='+',
        help='Input delta time for files without date variables')
    # input time standard definition
    parser.add_argument('--standard','-s',
        type=str, choices=('UTC','GPS','TAI','LORAN','datetime'), default='UTC',
        help='Input time standard for delta times')
    # spatial projection (EPSG code or PROJ4 string)
    parser.add_argument('--projection','-P',
        type=str, default='4326',
        help='Spatial projection as EPSG code or PROJ4 string')
    # use compressed model files
    parser.add_argument('--gzip','-G',
        default=False, action='store_true',
        help='Model files are gzip compressed')
    # verbose output of processing run
    # print information about each input and output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
    # permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of output file')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # set output file from input filename if not entered
    if not args.outfile:
        vars = (args.infile.stem,args.model,args.infile.suffix)
        args.outfile = args.infile.with_name('{0}_{1}{2}'.format(*vars))

    # run SMB interpolation program for input file
    interp_SMB_correction(args.directory, args.infile, args.outfile, args.model,
        REGION=args.region,
        FORMAT=args.format,
        VARIABLES=args.variables,
        HEADER=args.header,
        DELIMITER=args.delimiter,
        TYPE=args.type,
        TIME_UNITS=args.epoch,
        TIME=args.deltatime,
        TIME_STANDARD=args.standard,
        PROJECTION=args.projection,
        GZIP=args.gzip,
        VERBOSE=args.verbose,
        MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
