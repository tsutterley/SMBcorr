#!/usr/bin/env python
u"""
interp_SMB_ICESat2_ATL15.py
Written by Tyler Sutterley (09/2024)
Interpolates daily firn model estimates to the times and locations of
    ICESat-2 ATL15 gridded land ice height change data

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -m X, --model X: Regional firn model to run
    -G, --gzip: Model files are gzip compressed
    -V, --verbose: Output information about each created file
    -M X, --mode X: Permission mode of directories and files created

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
    netCDF4: Python interface to the netCDF C library
         https://unidata.github.io/netcdf4-python/netCDF4/index.html

PROGRAM DEPENDENCIES:
    time.py: utilities for calculating time operations
    regress_model.py: models a time series using least-squares regression
    mar_interp_daily.py: interpolates daily MAR products
    racmo_interp_daily.py: interpolates daily RACMO products
    merra_hybrid_interp.py: interpolates GSFC MERRA-2 hybrid products

UPDATE HISTORY:
    Updated 09/2024: use hemisphere flag to set model options
        fixes model parameters to use the hemisphere flags
    Written 02/2023
"""
from __future__ import print_function

import os
import re
import sys
import time
import logging
import argparse
import warnings
import traceback
import numpy as np
import SMBcorr

# attempt imports
try:
    import netCDF4
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("netCDF4 not available", ImportWarning)
try:
    import pyproj
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("pyproj not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

# available models
models = dict(N={}, S={})
# MAR
models['N']['MAR'] = []
models['N']['MAR'].append('MARv3.9-ERA')
models['N']['MAR'].append('MARv3.10-ERA')
models['N']['MAR'].append('MARv3.11-NCEP')
models['N']['MAR'].append('MARv3.11-ERA')
models['N']['MAR'].append('MARv3.11.2-ERA-6km')
models['N']['MAR'].append('MARv3.11.2-ERA-7.5km')
models['N']['MAR'].append('MARv3.11.2-ERA-10km')
models['N']['MAR'].append('MARv3.11.2-ERA-15km')
models['N']['MAR'].append('MARv3.11.2-ERA-20km')
models['N']['MAR'].append('MARv3.11.2-NCEP-20km')
models['N']['MAR'].append('MARv3.11.5-ERA-6km')
models['N']['MAR'].append('MARv3.11.5-ERA-10km')
models['N']['MAR'].append('MARv3.11.5-ERA-15km')
models['N']['MAR'].append('MARv3.11.5-ERA-20km')
# RACMO
models['N']['RACMO'] = []
models['N']['RACMO'].append('RACMO2.3-XGRN11')
models['N']['RACMO'].append('RACMO2.3p2-XGRN11')
models['N']['RACMO'].append('RACMO2.3p2-FGRN055')
# MERRA2-hybrid
models['N']['MERRA2-hybrid'] = []
models['N']['MERRA2-hybrid'].append('GSFC-fdm-v0')
models['N']['MERRA2-hybrid'].append('GSFC-fdm-v1')
models['N']['MERRA2-hybrid'].append('GSFC-fdm-v1.0')
models['N']['MERRA2-hybrid'].append('GSFC-fdm-v1.1')
models['N']['MERRA2-hybrid'].append('GSFC-fdm-v1.2')
models['N']['MERRA2-hybrid'].append('GSFC-fdm-v1.2.1')
models['S']['MERRA2-hybrid'] = []
models['S']['MERRA2-hybrid'].append('GSFC-fdm-v0')
models['S']['MERRA2-hybrid'].append('GSFC-fdm-v1')
models['S']['MERRA2-hybrid'].append('GSFC-fdm-v1.1')
models['S']['MERRA2-hybrid'].append('GSFC-fdm-v1.2.1')

# PURPOSE: keep track of threads
def info(args):
    logging.info(os.path.basename(sys.argv[0]))
    logging.info(args)
    logging.info(f'module name: {__name__}')
    if hasattr(os, 'getppid'):
        logging.info(f'parent process: {os.getppid():d}')
    logging.info(f'process id: {os.getpid():d}')

# PURPOSE: set the hemisphere of interest based on the ATL14/15 region
def set_hemisphere(REGION):
    if REGION in ('AA','A1','A2','A3','A4'):
        projection_flag = 'S'
    else:
        projection_flag = 'N'
    # return the hemisphere flag
    return projection_flag

# PURPOSE: read a variable group from ICESat-2 ATL15
def read_ATL15(infile, group='delta_h'):
    # dictionary with ATL15 variables
    ATL15 = {}
    attributes = {}
    with netCDF4.Dataset(os.path.expanduser(infile),'r') as fileID:
        # check if reading from root group or sub-group
        ncf = fileID.groups[group] if group else fileID
        # netCDF4 structure information
        logging.debug(os.path.expanduser(infile))
        logging.debug(list(ncf.variables.keys()))
        for key,val in ncf.variables.items():
            ATL15[key] = val[:].copy()
            attributes[key] = {}
            for att_name in val.ncattrs():
                attributes[key][att_name] = val.getncattr(att_name)
    # return the data and attributes
    return (ATL15, attributes)

# PURPOSE: read ICESat-2 grided land ice height change data (ATL15) from NSIDC
# calculate and interpolate daily model firn outputs
def interp_SMB_ICESat2(base_dir, input_file, model_version,
    GZIP=False, MODE=0o775):

    # parse ATL15 file
    pattern = r'(ATL\d{2})_(.*?)_(\d{2})(\d{2})_(.*?)_(\d{3})_(\d{2}).nc$'
    rx = re.compile(pattern, re.VERBOSE)
    PRD, RGN, SCYC, ECYC, RES, RL, VERS = rx.findall(input_file).pop()
    # directory with ATL15 data
    ddir = os.path.dirname(input_file)
    # read ATL15 data for group
    ATL15, attrib = read_ATL15(input_file, group='delta_h')
    nt, ny, nx = ATL15['delta_h'].shape
    grid_mapping_name = attrib['delta_h']['grid_mapping']
    crs_wkt = attrib[grid_mapping_name]['crs_wkt']
    fill_value = attrib['delta_h']['_FillValue']
    # coordinate reference systems for converting from projection
    crs1 = pyproj.CRS.from_wkt(crs_wkt)
    crs2 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # dictionary of coordinate reference system variables
    cs_to_cf = crs1.cs_to_cf()
    crs_to_dict = crs1.to_dict()
    # input coordinate reference system as proj4 string
    proj4_params = crs1.to_proj4()
    # epoch of delta heights
    EPOCH = 2020.0
    # set the hemisphere of interest
    HEM = set_hemisphere(RGN)

    # determine main model group from region and model_version
    MODEL, = [key for key,val in models[HEM].items() if model_version in val]

    # keyword arguments for all models
    KWARGS = dict(SIGMA=1.5, FILL_VALUE=fill_value)
    # set model specific parameters
    if (MODEL == 'MAR'):
        match_object=re.match(r'(MARv\d+\.\d+(.\d+)?)',model_version)
        MAR_VERSION=match_object.group(0)
        MAR_REGION=dict(N='Greenland',S='Antarctic')[HEM]
        # model subdirectories
        SUBDIRECTORY=dict(S={}, N={})
        SUBDIRECTORY['N']['MARv3.9-ERA']=['ERA_1958-2018_10km','daily_10km']
        SUBDIRECTORY['N']['MARv3.10-ERA']=['ERA_1958-2019-15km','daily_15km']
        SUBDIRECTORY['N']['MARv3.11-NCEP']=['NCEP1_1948-2020_20km','daily_20km']
        SUBDIRECTORY['N']['MARv3.11-ERA']=['ERA_1958-2019-15km','daily_15km']
        SUBDIRECTORY['N']['MARv3.11.2-ERA-6km']=['6km_ERA5']
        SUBDIRECTORY['N']['MARv3.11.2-ERA-7.5km']=['7.5km_ERA5']
        SUBDIRECTORY['N']['MARv3.11.2-ERA-10km']=['10km_ERA5']
        SUBDIRECTORY['N']['MARv3.11.2-ERA-15km']=['15km_ERA5']
        SUBDIRECTORY['N']['MARv3.11.2-ERA-20km']=['20km_ERA5']
        SUBDIRECTORY['N']['MARv3.11.2-NCEP-20km']=['20km_NCEP1']
        SUBDIRECTORY['N']['MARv3.11.5-ERA-6km']=['6km_ERA5']
        SUBDIRECTORY['N']['MARv3.11.5-ERA-10km']=['10km_ERA5']
        SUBDIRECTORY['N']['MARv3.11.5-ERA-15km']=['15km_ERA5']
        SUBDIRECTORY['N']['MARv3.11.5-ERA-20km']=['20km_ERA5']
        MAR_MODEL=SUBDIRECTORY[HEM][model_version]
        DIRECTORY=os.path.join(base_dir,'MAR',MAR_VERSION,MAR_REGION,*MAR_MODEL)
        # keyword arguments for variable coordinates
        MAR_KWARGS=dict(S={}, N={})
        MAR_KWARGS['N']['MARv3.9-ERA'] = dict(XNAME='X10_153',YNAME='Y21_288')
        MAR_KWARGS['N']['MARv3.10-ERA'] = dict(XNAME='X10_105',YNAME='Y21_199')
        MAR_KWARGS['N']['MARv3.11-NCEP'] = dict(XNAME='X12_84',YNAME='Y21_155')
        MAR_KWARGS['N']['MARv3.11-ERA'] = dict(XNAME='X10_105',YNAME='Y21_199')
        MAR_KWARGS['N']['MARv3.11.2-ERA-6km'] = dict(XNAME='X12_251',YNAME='Y20_465')
        MAR_KWARGS['N']['MARv3.11.2-ERA-7.5km'] = dict(XNAME='X12_203',YNAME='Y20_377')
        MAR_KWARGS['N']['MARv3.11.2-ERA-10km'] = dict(XNAME='X10_153',YNAME='Y21_288')
        MAR_KWARGS['N']['MARv3.11.2-ERA-15km'] = dict(XNAME='X10_105',YNAME='Y21_199')
        MAR_KWARGS['N']['MARv3.11.2-ERA-20km'] = dict(XNAME='X12_84',YNAME='Y21_155')
        MAR_KWARGS['N']['MARv3.11.2-NCEP-20km'] = dict(XNAME='X12_84',YNAME='Y21_155')
        MAR_KWARGS['N']['MARv3.11.5-ERA-6km'] = dict(XNAME='X12_251',YNAME='Y20_465')
        MAR_KWARGS['N']['MARv3.11.5-ERA-10km'] = dict(XNAME='X10_153',YNAME='Y21_288')
        MAR_KWARGS['N']['MARv3.11.5-ERA-15km'] = dict(XNAME='X10_105',YNAME='Y21_199')
        MAR_KWARGS['N']['MARv3.11.5-ERA-20km'] = dict(XNAME='X12_84',YNAME='Y21_155')
        KWARGS.update(MAR_KWARGS[HEM][model_version])
        # netCDF4 variable names for direct fields
        VARIABLES = ['SMB','ZN6','ZN4','ZN5']
        # output variable keys for both direct and derived fields
        KEYS = ['SMB', 'delta_h_surf', 'delta_h_firn', 'delta_h_melt',
            'delta_h_smb', 'delta_h_accum']
        # output netCDF4 long_name attributes for each variable
        LONGNAME = {}
        LONGNAME['SMB'] = "Cumulative SMB"
        LONGNAME['delta_h_surf'] = "Height"
        LONGNAME['delta_h_firn'] = "Compaction"
        LONGNAME['delta_h_melt'] = "Surface Melt"
        LONGNAME['delta_h_smb'] = "Surface Mass Balance"
        LONGNAME['delta_h_accum'] = "Surface Accumulation"
        # output netCDF4 description attributes for each variable
        DESCRIPTION = {}
        DESCRIPTION['SMB'] = \
            "Cumulative Surface Mass Balance"
        DESCRIPTION['delta_h_surf'] = \
            "Relative Snow Height Change"
        DESCRIPTION['delta_h_firn'] = \
            "Relative Snow Height Change due to Compaction"
        DESCRIPTION['delta_h_melt'] = \
            "Relative Snow Height Change due to Surface Melt"
        DESCRIPTION['delta_h_smb'] = \
            "Relative Snow Height Change due to Surface Mass Balance"
        DESCRIPTION['delta_h_accum'] = \
            "Relative Snow Height Change due to Surface Accumulation"
    elif (MODEL == 'RACMO'):
        RACMO_VERSION,RACMO_MODEL=model_version.split('-')
        # netCDF4 variable names
        VARIABLES = ['hgtsrf']
        # output variable keys
        KEYS = ['delta_h_surf']
        # output netCDF4 long_name attributes for each variable
        LONGNAME = {}
        LONGNAME['delta_h_surf'] = "Height"
        # output netCDF4 description attributes for each variable
        DESCRIPTION = {}
        DESCRIPTION['delta_h_surf'] = \
            "Relative Snow Height Change"
    elif (MODEL == 'MERRA2-hybrid'):
        # regular expression pattern for extracting version
        merra2_regex = re.compile(r'GSFC-fdm-((v\d+)(\.\d+)?(\.\d+)?)$')
        # get MERRA-2 version and major version
        MERRA2_VERSION = merra2_regex.match(model_version).group(1)
        # MERRA-2 hybrid directory
        DIRECTORY=os.path.join(base_dir,'MERRA2_hybrid',MERRA2_VERSION)
        # MERRA-2 region name from ATL15 region
        MERRA2_REGION = dict(S='ais', N='gris')[HEM]
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
        KEYS = ['delta_h_firn','delta_h_smb','delta_h_surf','delta_h_melt']
        # output netCDF4 long_name attributes for each variable
        LONGNAME = {}
        LONGNAME['delta_h_surf'] = "Height"
        LONGNAME['delta_h_firn'] = "Compaction"
        LONGNAME['delta_h_smb'] = "Surface Mass Balance"
        LONGNAME['delta_h_melt'] = "Surface Melt"
        # output netCDF4 description attributes for each variable
        DESCRIPTION = {}
        DESCRIPTION['delta_h_surf'] = \
            "Relative Snow Height Change"
        DESCRIPTION['delta_h_firn'] = \
            "Relative Snow Height Change due to Compaction"
        DESCRIPTION['delta_h_smb'] = \
            "Relative Snow Height Change due to Surface Mass Balance"
        DESCRIPTION['delta_h_melt'] = \
            "Relative Snow Height Change due to Surface Melt"

    # output spatial data and attributes dictionaries
    output_data = {}
    attributes = dict(x={}, y={}, time={})
    # x, y and time
    output_data['x'] = ATL15['x'].copy()
    output_data['y'] = ATL15['y'].copy()
    output_data['time'] = ATL15['time'].copy()
    for att_name in ['long_name','standard_name','units']:
        attributes['x'][att_name] = cs_to_cf[0][att_name]
        attributes['y'][att_name] = cs_to_cf[1][att_name]
    for att_name in ['description','source','long_name','units']:
        attributes['time'][att_name] = attrib['time'][att_name]
    # allocate for output variables
    for key,var in zip(KEYS,VARIABLES):
        output_data[key] = np.ma.empty((nt, ny, nx), fill_value=fill_value)
        output_data[key].mask = np.ones((nt, ny, nx), dtype=bool)
        # attributes for output variable
        attributes[key] = {}
        attributes[key]['units'] = "meters"
        attributes[key]['long_name'] = LONGNAME[key]
        attributes[key]['description'] = DESCRIPTION[key]
        attributes[key]['source'] = MODEL
        attributes[key]['reference'] = model_version
        attributes[key]['grid_mapping'] = 'crs'
        attributes[key]['_FillValue'] = fill_value

    # convert x and y axes to grid
    gridx, gridy = np.meshgrid(ATL15['x'], ATL15['y'])
    gridlon, latitude_geodetic = transformer.transform(gridx, gridy)
    # make sure points are valid
    valid_mask = np.logical_not(ATL15['delta_h'].mask)
    iY, iX = np.nonzero(np.any(valid_mask, axis=0))
    nind = len(iY)

    # for each time
    for iT, delta_time in enumerate(ATL15['time']):
        # convert delta times to year-decimal
        tdec = 2018.0 + delta_time/365.25 + np.zeros((nind))
        if (MODEL == 'MAR') and np.any(valid_mask):
            # read and interpolate daily MAR outputs
            for key,var in zip(KEYS,VARIABLES):
                OUT = SMBcorr.interpolate_mar_daily(DIRECTORY, proj4_params,
                    MAR_VERSION, tdec, gridx[iY,iX], gridy[iY,iX],
                    VARIABLE=var, **KWARGS)
                # set attributes to output for iteration
                output_data[key].data[iT,iY,iX] = np.copy(OUT.data)
                output_data[key].mask[iT,iY,iX] = np.copy(OUT.mask)
            # calculate derived fields
            output_data['delta_h_smb'].data[iT,iY,iX] = \
                output_data['delta_h_surf'].data[iT,iY,iX] - \
                output_data['delta_h_firn'].data[iT,iY,iX]
            output_data['delta_h_smb'].mask[iT,iY,iX] = \
                output_data['delta_h_surf'].mask[iT,iY,iX] | \
                output_data['delta_h_firn'].mask[iT,iY,iX]
            output_data['delta_h_accum'].data[iT,iY,iX] = \
                output_data['delta_h_surf'].data[iT,iY,iX] - \
                output_data['delta_h_firn'].data[iT,iY,iX] - \
                output_data['delta_h_melt'].data
            output_data['delta_h_accum'].mask[iT,iY,iX] = \
                output_data['delta_h_surf'].mask[iT,iY,iX] | \
                output_data['delta_h_firn'].mask[iT,iY,iX] | \
                output_data['delta_h_melt'].mask[iT,iY,iX]
        elif (MODEL == 'RACMO') and np.any(valid_mask):
            # read and interpolate daily RACMO outputs
            for key,var in zip(KEYS,VARIABLES):
                OUT = SMBcorr.interpolate_racmo_daily(base_dir, proj4_params,
                    RACMO_MODEL, tdec, gridx[iY,iX], gridy[iY,iX],
                    VARIABLE=var, **KWARGS)
                # set attributes to output for iteration
                output_data[key].data[iT,iY,iX] = np.copy(OUT.data)
                output_data[key].mask[iT,iY,iX] = np.copy(OUT.mask)
        elif (MODEL == 'MERRA2-hybrid') and np.any(valid_mask):
            # read and interpolate 5-day MERRA2-Hybrid outputs
            for key,var in zip(KEYS,VARIABLES):
                OUT = SMBcorr.interpolate_merra_hybrid(DIRECTORY, proj4_params,
                    MERRA2_REGION, tdec, gridx[iY,iX], gridy[iY,iX],
                    VARIABLE=var, **KWARGS)
                # set attributes to output for iteration
                output_data[key].data[iT,iY,iX] = np.copy(OUT.data)
                output_data[key].mask[iT,iY,iX] = np.copy(OUT.mask)

    # remove epoch from all time points
    indt, = np.flatnonzero(ATL15['time'] == (EPOCH - 2018.0)*365.25)
    # calculate as anomalies with respect to epoch
    for key,var in zip(KEYS,VARIABLES):
        # data at epoch for calculating anomalies
        z0 = output_data[key].data[indt,iY,iX].copy()
        for iT, delta_time in enumerate(ATL15['time']):
            output_data[key].data[iT,iY,iX] -= z0

    # output file
    output_file = f'{PRD}_{RGN}_{SCYC}{ECYC}_{RES}_{model_version}.nc'
    fileID = netCDF4.Dataset(os.path.join(ddir, output_file),'w')

    # dictionary with netCDF4 variables
    nc = {}
    # netCDF4 dimension variables
    dimensions = []
    dimensions.append('time')
    dimensions.append('y')
    dimensions.append('x')
    dims = tuple(dimensions)

    # create projection variable
    nc['crs'] = fileID.createVariable('crs', np.byte, ())
    # add projection attributes
    nc['crs'].setncattr('standard_name', 'Polar_Stereographic')
    # nc['crs'].setncattr('spatial_epsg', crs1.to_epsg())
    nc['crs'].setncattr('spatial_ref', crs1.to_wkt())
    nc['crs'].setncattr('proj4_params', proj4_params)
    nc['crs'].setncattr('latitude_of_projection_origin', crs_to_dict['lat_0'])
    for att_name, att_val in crs1.to_cf().items():
        nc['crs'].setncattr(att_name, att_val)

    # netCDF4 dimensions
    for i,key in enumerate(dimensions):
        val = output_data[key]
        fileID.createDimension(key, len(val))
        nc[key] = fileID.createVariable(key, val.dtype, (key,))
        # filling netCDF4 dimension variables
        nc[key][:] = val
        # Defining attributes for variable
        for att_name,att_val in attributes[key].items():
            nc[key].setncattr(att_name,att_val)

    # netCDF4 spatial variables
    variables = set(output_data.keys()) - set(dimensions)
    for key in sorted(variables):
        val = output_data[key]
        if '_FillValue' in attributes[key].keys():
            nc[key] = fileID.createVariable(key, val.dtype, dims,
                fill_value=attributes[key]['_FillValue'], zlib=True)
            attributes[key].pop('_FillValue')
        elif val.shape:
            nc[key] = fileID.createVariable(key, val.dtype, dims,
                zlib=True)
        else:
            nc[key] = fileID.createVariable(key, val.dtype, ())
        # filling netCDF4 variables
        nc[key][:] = val
        # Defining attributes for variable
        for att_name,att_val in attributes[key].items():
            nc[key].setncattr(att_name, att_val)

    # add root level attributes
    fileID.setncattr('title', 'ATL15_SMB_and_Firn_Correction')
    fileID.setncattr('summary', 'Surface_Mass_Balance_and_Firn_Corrections_'
        'for_NASA_ICESat-2_ATL15_Gridded_Land_Ice_Height_Change_data.')
    fileID.setncattr('GDAL_AREA_OR_POINT', 'Area')
    fileID.setncattr('Conventions', 'CF-1.6')
    today = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    fileID.setncattr('date_created', today)
    # add software information
    fileID.setncattr('software_reference', SMBcorr.version.project_name)
    fileID.setncattr('software_version', SMBcorr.version.full_version)
    # add geospatial and temporal attributes
    fileID.setncattr('geospatial_lat_min', latitude_geodetic.min())
    fileID.setncattr('geospatial_lat_max', latitude_geodetic.max())
    fileID.setncattr('geospatial_lon_min', gridlon.min())
    fileID.setncattr('geospatial_lon_max', gridlon.max())
    fileID.setncattr('geospatial_lat_units', "degrees_north")
    fileID.setncattr('geospatial_lon_units', "degrees_east")
    # Output NetCDF structure information
    logging.info(os.path.join(ddir, output_file))
    logging.info(list(fileID.variables.keys()))
    # Closing the netCDF4 file
    fileID.close()
    # change the permissions mode
    os.chmod(os.path.join(ddir, output_file), MODE)

# PURPOSE: create arguments parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Interpolates daily firn model estimates to times and
            locations of ICESat-2 ATL15 gridded land ice height change data
            """
    )
    # command line parameters
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        help='ICESat-2 ATL15 file to run')
    # directory with model data
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
    # firn model
    choices = [v for mdl in models.values() for val in mdl.values() for v in val]
    parser.add_argument('--model','-m',
        metavar='FIRN', type=str, default='GSFC-fdm-v1.2',
        choices=sorted(set(choices)),
        help='Regional firn model to run')
    # use compressed model files
    parser.add_argument('--gzip','-G',
        default=False, action='store_true',
        help='Model files are gzip compressed')
    # print information about each input and output file
    parser.add_argument('--verbose','-V',
        action='count', default=0,
        help='Verbose output of run')
    # permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files created')
    # return the parser
    return parser

# Main program that calls interp_SMB_ICESat2()
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # create logger
    loglevels = [logging.CRITICAL, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=loglevels[args.verbose])

    # try to run the analysis with listed parameters
    try:
        info(args)
        # run algorithm with parameters
        interp_SMB_ICESat2(args.directory, args.infile, args.model,
            GZIP=args.gzip, MODE=args.mode)
    except Exception as exc:
        # if there has been an error exception
        # print the type, value, and stack trace of the
        # current exception being handled
        logging.critical(f'process id {os.getpid():d} failed')
        logging.error(traceback.format_exc())

# run main program
if __name__ == '__main__':
    main()