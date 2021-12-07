#!/usr/bin/env python
u"""
interp_SMB_ICESat2_ATL11.py
Written by Tyler Sutterley (12/2021)
Interpolates daily firn model estimates to the times and locations of
    ICESat-2 ATL11 annual land ice height data

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -m X, --model X: Regional firn model to run
    -C, --crossovers: Run ATL11 Crossovers
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
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://h5py.org
    netCDF4: Python interface to the netCDF C library
         https://unidata.github.io/netcdf4-python/netCDF4/index.html

PROGRAM DEPENDENCIES:
    time.py: utilities for calculating time operations
    regress_model.py: models a time series using least-squares regression
    mar_interp_daily.py: interpolates daily MAR products
    racmo_interp_daily.py: interpolates daily RACMO products
    merra_hybrid_interp.py: interpolates GSFC MERRA-2 hybrid products

UPDATE HISTORY:
    Updated 12/2021: added GSFC MERRA-2 Hybrid Greenland v1.2
    Updated 05/2021: make GSFC MERRA-2 compression keyword an option
    Updated 04/2021: added GSFC MERRA-2 Hybrid Antarctica v1.1
    Written 03/2021
"""
from __future__ import print_function

import os
import re
import h5py
import pyproj
import logging
import argparse
import datetime
import numpy as np
import collections
import SMBcorr

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
models['GL']['MERRA2-hybrid'].append('GSFC-fdm-v1.2')
models['AA']['MERRA2-hybrid'] = []
models['AA']['MERRA2-hybrid'].append('GSFC-fdm-v0')
models['AA']['MERRA2-hybrid'].append('GSFC-fdm-v1')
models['AA']['MERRA2-hybrid'].append('GSFC-fdm-v1.1')

# PURPOSE: set the projection parameters based on the input granule
def set_projection(GRANULE):
    if GRANULE in ('10','11','12'):
        REGION = 'AA'
        projection_flag = 'EPSG:3031'
    elif GRANULE in ('02','03','04','05','06'):
        REGION = 'GL'
        projection_flag = 'EPSG:3413'
    return (REGION,projection_flag)

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

# PURPOSE: read ICESat-2 annual land ice height data (ATL11) from NSIDC
# calculate and interpolate daily model firn outputs
def interp_SMB_ICESat2(base_dir, FILE, model_version, CROSSOVERS=False,
    GZIP=False, VERBOSE=False, MODE=0o775):

    #-- create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logging.basicConfig(level=loglevel)

    # read data from input file
    logging.info('{0} -->'.format(os.path.basename(FILE)))
    # Open the HDF5 file for reading
    fileID = h5py.File(FILE, 'r')
    # output data directory
    ddir = os.path.dirname(FILE)
    # extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})_(\d{2})(\d{2})_'
        r'(\d{3})_(\d{2})(.*?).h5$')
    SUB,PRD,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX = rx.findall(FILE).pop()
    # get projection and region name based on granule
    REGION,proj4_params = set_projection(GRAN)
    # determine main model group from region and model_version
    MODEL, = [key for key,val in models[REGION].items() if model_version in val]

    # keyword arguments for all models
    KWARGS = dict(SIGMA=1.5, FILL_VALUE=np.nan)
    # set model specific parameters
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
        RACMO_VERSION,RACMO_MODEL=model_version.split('-')
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
        merra2_regex = re.compile(r'GSFC-fdm-((v\d+)(\.\d+)?)$')
        # get MERRA-2 version and major version
        MERRA2_VERSION = merra2_regex.match(model_version).group(1)
        # MERRA-2 hybrid directory
        DIRECTORY=os.path.join(base_dir,'MERRA2_hybrid',MERRA2_VERSION)
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
        KEYS = ['zsurf','zfirn','zsmb','zmelt']
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

    # pyproj transformer for converting from latitude/longitude
    # into polar stereographic coordinates
    crs1 = pyproj.CRS.from_string("epsg:{0:d}".format(4326))
    crs2 = pyproj.CRS.from_string(proj4_params)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)

    # read each input beam pair within the file
    IS2_atl11_pairs = []
    for ptx in [k for k in fileID.keys() if bool(re.match(r'pt\d',k))]:
        # check if subsetted beam contains reference points
        try:
            fileID[ptx]['ref_pt']
        except KeyError:
            pass
        else:
            IS2_atl11_pairs.append(ptx)

    # copy variables for outputting to HDF5 file
    IS2_atl11_corr = {}
    IS2_atl11_fill = {}
    IS2_atl11_dims = {}
    IS2_atl11_corr_attrs = {}
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    IS2_atl11_corr['ancillary_data'] = {}
    IS2_atl11_corr_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        # get each HDF5 variable
        IS2_atl11_corr['ancillary_data'][key] = fileID['ancillary_data'][key][:]
        # Getting attributes of group and included variables
        IS2_atl11_corr_attrs['ancillary_data'][key] = {}
        for att_name,att_val in fileID['ancillary_data'][key].attrs.items():
            IS2_atl11_corr_attrs['ancillary_data'][key][att_name] = att_val
    # HDF5 group name for across-track data
    XT = 'crossing_track_data'

    # for each input beam pair within the file
    for ptx in sorted(IS2_atl11_pairs):
        # output data dictionaries for beam
        IS2_atl11_corr[ptx] = dict(cycle_stats=collections.OrderedDict(),
            crossing_track_data=collections.OrderedDict())
        IS2_atl11_fill[ptx] = dict(cycle_stats={},crossing_track_data={})
        IS2_atl11_dims[ptx] = dict(cycle_stats={},crossing_track_data={})
        IS2_atl11_corr_attrs[ptx] = dict(cycle_stats={},crossing_track_data={})

        # extract along-track and across-track variables
        ref_pt = {}
        latitude = {}
        longitude = {}
        delta_time = {}
        groups = ['AT']
        # dictionary with output variables
        OUTPUT = {}
        # number of average segments and number of included cycles
        # fill_value for invalid heights and corrections
        fv = fileID[ptx]['h_corr'].attrs['_FillValue']
        # shape of along-track data
        n_points,n_cycles = fileID[ptx]['delta_time'][:].shape
        # along-track (AT) reference point, latitude, longitude and time
        ref_pt['AT'] = fileID[ptx]['ref_pt'][:].copy()
        latitude['AT'] = np.ma.array(fileID[ptx]['latitude'][:],
            fill_value=fileID[ptx]['latitude'].attrs['_FillValue'])
        latitude['AT'].mask = (latitude['AT'] == latitude['AT'].fill_value)
        longitude['AT'] = np.ma.array(fileID[ptx]['longitude'][:],
            fill_value=fileID[ptx]['longitude'].attrs['_FillValue'])
        longitude['AT'].mask = (longitude['AT'] == longitude['AT'].fill_value)
        delta_time['AT'] = np.ma.array(fileID[ptx]['delta_time'][:],
            fill_value=fileID[ptx]['delta_time'].attrs['_FillValue'])
        delta_time['AT'].mask = (delta_time['AT'] == delta_time['AT'].fill_value)
        # allocate for output height for along-track data
        OUTPUT['AT'] = {}
        for key in KEYS:
            OUTPUT['AT'][key] = np.ma.empty((n_points,n_cycles),fill_value=fv)
            OUTPUT['AT'][key].mask = np.ones((n_points,n_cycles),dtype=bool)
            OUTPUT['AT'][key].interpolation = np.zeros((n_points,n_cycles),dtype=np.uint8)
        # if running ATL11 crossovers
        if CROSSOVERS:
            # add to group
            groups.append('XT')
            # shape of across-track data
            n_cross, = fileID[ptx][XT]['delta_time'].shape
            # across-track (XT) reference point, latitude, longitude and time
            ref_pt['XT'] = fileID[ptx][XT]['ref_pt'][:].copy()
            latitude['XT'] = np.ma.array(fileID[ptx][XT]['latitude'][:],
                fill_value=fileID[ptx][XT]['latitude'].attrs['_FillValue'])
            latitude['XT'].mask = (latitude['XT'] == latitude['XT'].fill_value)
            longitude['XT'] = np.ma.array(fileID[ptx][XT]['longitude'][:],
                fill_value=fileID[ptx][XT]['longitude'].attrs['_FillValue'])
            latitude['XT'].mask = (latitude['XT'] == longitude['XT'].fill_value)
            delta_time['XT'] = np.ma.array(fileID[ptx][XT]['delta_time'][:],
                fill_value=fileID[ptx][XT]['delta_time'].attrs['_FillValue'])
            delta_time['XT'].mask = (delta_time['XT'] == delta_time['XT'].fill_value)
            # allocate for output height for across-track data
            OUTPUT['XT'] = {}
            for key in KEYS:
                OUTPUT['XT'][key] = np.ma.empty((n_cross),fill_value=fv)
                OUTPUT['XT'][key].mask = np.ones((n_cross),dtype=bool)
                OUTPUT['XT'][key].interpolation = np.zeros((n_cross),dtype=np.uint8)

        # extract lat/lon and convert to polar stereographic
        X,Y = transformer.transform(longitude['AT'],longitude['AT'])

        # for each valid cycle of ICESat-2 ATL11 data
        for c in range(n_cycles):
            # find valid elevations for cycle
            valid = np.logical_not(delta_time['AT'].mask[:,c])
            i, = np.nonzero(valid)
            # convert time from ATLAS SDP to date in decimal-years
            tdec = convert_delta_time(delta_time['AT'][i,c])['decimal']
            if (MODEL == 'MAR') and np.any(valid):
                # read and interpolate daily MAR outputs
                for key,var in zip(KEYS,VARIABLES):
                    OUT = SMBcorr.interpolate_mar_daily(DIRECTORY, proj4_params,
                        MAR_VERSION, tdec, X[i], Y[i], VARIABLE=var, **KWARGS)
                    # set attributes to output for iteration
                    OUTPUT['AT'][key].data[i,c] = np.copy(OUT.data)
                    OUTPUT['AT'][key].mask[i,c] = np.copy(OUT.mask)
                    OUTPUT['AT'][key].interpolation[i,c] = np.copy(OUT.interpolation)
                # calculate derived fields
                OUTPUT['AT']['zsmb'].data[i,c] = OUTPUT['AT']['zsurf'].data[i,c] - \
                    OUTPUT['AT']['zfirn'].data[i,c]
                OUTPUT['AT']['zsmb'].mask[i,c] = OUTPUT['AT']['zsurf'].mask[i,c] | \
                    OUTPUT['AT']['zfirn'].mask[i,c]
                OUTPUT['AT']['zaccum'].data[i,c] = OUTPUT['AT']['zsurf'].data[i,c] - \
                    OUTPUT['AT']['zfirn'].data[i,c] - OUTPUT['AT']['zmelt'].data
                OUTPUT['AT']['zaccum'].mask[i,c] = OUTPUT['AT']['zsurf'].mask[i,c] | \
                    OUTPUT['AT']['zfirn'].mask[i,c] | OUTPUT['AT']['zmelt'].mask[i,c]
            elif (MODEL == 'RACMO') and np.any(valid):
                # read and interpolate daily RACMO outputs
                for key,var in zip(KEYS,VARIABLES):
                    OUT = SMBcorr.interpolate_racmo_daily(base_dir, proj4_params,
                        RACMO_MODEL, tdec, X[i], Y[i], VARIABLE=var, **KWARGS)
                    # set attributes to output for iteration
                    OUTPUT['AT'][key].data[i,c] = np.copy(OUT.data)
                    OUTPUT['AT'][key].mask[i,c] = np.copy(OUT.mask)
                    OUTPUT['AT'][key].interpolation[i,c] = np.copy(OUT.interpolation)
            elif (MODEL == 'MERRA2-hybrid') and np.any(valid):
                # read and interpolate 5-day MERRA2-Hybrid outputs
                for key,var in zip(KEYS,VARIABLES):
                    OUT = SMBcorr.interpolate_merra_hybrid(DIRECTORY, proj4_params,
                        MERRA2_REGION, tdec, X[i], Y[i], VARIABLE=var, **KWARGS)
                    # set attributes to output for iteration
                    OUTPUT['AT'][key].data[i,c] = np.copy(OUT.data)
                    OUTPUT['AT'][key].mask[i,c] = np.copy(OUT.mask)
                    OUTPUT['AT'][key].interpolation[i,c] = np.copy(OUT.interpolation)

        #-- if interpolating to ATL11 crossover locations
        if CROSSOVERS:
            # extract lat/lon and convert to polar stereographic
            X,Y = transformer.transform(longitude['XT'],longitude['XT'])
            # find valid elevations for cycle
            valid = np.logical_not(delta_time['XT'].mask[:])
            i, = np.nonzero(valid)
            # convert time from ATLAS SDP to date in decimal-years
            tdec = convert_delta_time(delta_time['XT'][i])['decimal']
            if (MODEL == 'MAR') and np.any(valid):
                # read and interpolate daily MAR outputs
                for key,var in zip(KEYS,VARIABLES):
                    OUT = SMBcorr.interpolate_mar_daily(DIRECTORY, proj4_params,
                        MAR_VERSION, tdec, X[i], Y[i], VARIABLE=var, **KWARGS)
                    # set attributes to output for iteration
                    OUTPUT['XT'][key].data[i] = np.copy(OUT.data)
                    OUTPUT['XT'][key].mask[i] = np.copy(OUT.mask)
                    OUTPUT['XT'][key].interpolation[i] = np.copy(OUT.interpolation)
                # calculate derived fields
                OUTPUT['XT']['zsmb'].data[i] = OUTPUT['XT']['zsurf'].data[i] - \
                    OUTPUT['XT']['zfirn'].data[i]
                OUTPUT['XT']['zsmb'].mask[i] = OUTPUT['XT']['zsurf'].mask[i] | \
                    OUTPUT['XT']['zfirn'].mask[i]
                OUTPUT['XT']['zaccum'].data[i] = OUTPUT['XT']['zsurf'].data[i] - \
                    OUTPUT['XT']['zfirn'].data[i] - OUTPUT['AT']['zmelt'].data[i]
                OUTPUT['XT']['zaccum'].mask[i] = OUTPUT['XT']['zsurf'].mask[i] | \
                    OUTPUT['XT']['zfirn'].mask[i] | OUTPUT['XT']['zmelt'].mask[i]
            elif (MODEL == 'RACMO') and np.any(valid):
                # read and interpolate daily RACMO outputs
                for key,var in zip(KEYS,VARIABLES):
                    OUT = SMBcorr.interpolate_racmo_daily(base_dir, proj4_params,
                        RACMO_MODEL, tdec, X[i], Y[i], VARIABLE=var, **KWARGS)
                    # set attributes to output for iteration
                    OUTPUT['XT'][key].data[i] = np.copy(OUT.data)
                    OUTPUT['XT'][key].mask[i] = np.copy(OUT.mask)
                    OUTPUT['XT'][key].interpolation[i] = np.copy(OUT.interpolation)
            elif (MODEL == 'MERRA2-hybrid') and np.any(valid):
                # read and interpolate 5-day MERRA2-Hybrid outputs
                for key,var in zip(KEYS,VARIABLES):
                    OUT = SMBcorr.interpolate_merra_hybrid(DIRECTORY, proj4_params,
                        MERRA2_REGION, tdec, X[i], Y[i], VARIABLE=var, **KWARGS)
                    # set attributes to output for iteration
                    OUTPUT['XT'][key].data[i] = np.copy(OUT.data)
                    OUTPUT['XT'][key].mask[i] = np.copy(OUT.mask)
                    OUTPUT['XT'][key].interpolation[i] = np.copy(OUT.interpolation)

        # group attributes for beam
        IS2_atl11_corr_attrs[ptx]['description'] = ('Contains the primary science parameters '
            'for this data set')
        IS2_atl11_corr_attrs[ptx]['beam_pair'] = fileID[ptx].attrs['beam_pair']
        IS2_atl11_corr_attrs[ptx]['ReferenceGroundTrack'] = fileID[ptx].attrs['ReferenceGroundTrack']
        IS2_atl11_corr_attrs[ptx]['first_cycle'] = fileID[ptx].attrs['first_cycle']
        IS2_atl11_corr_attrs[ptx]['last_cycle'] = fileID[ptx].attrs['last_cycle']
        IS2_atl11_corr_attrs[ptx]['equatorial_radius'] = fileID[ptx].attrs['equatorial_radius']
        IS2_atl11_corr_attrs[ptx]['polar_radius'] = fileID[ptx].attrs['polar_radius']

        # geolocation, time and reference point
        # reference point
        IS2_atl11_corr[ptx]['ref_pt'] = ref_pt['AT'].copy()
        IS2_atl11_fill[ptx]['ref_pt'] = None
        IS2_atl11_dims[ptx]['ref_pt'] = None
        IS2_atl11_corr_attrs[ptx]['ref_pt'] = collections.OrderedDict()
        IS2_atl11_corr_attrs[ptx]['ref_pt']['units'] = "1"
        IS2_atl11_corr_attrs[ptx]['ref_pt']['contentType'] = "referenceInformation"
        IS2_atl11_corr_attrs[ptx]['ref_pt']['long_name'] = "Reference point number"
        IS2_atl11_corr_attrs[ptx]['ref_pt']['source'] = "ATL06"
        IS2_atl11_corr_attrs[ptx]['ref_pt']['description'] = ("The reference point is the "
            "7 digit segment_id number corresponding to the center of the ATL06 data used "
            "for each ATL11 point.  These are sequential, starting with 1 for the first "
            "segment after an ascending equatorial crossing node.")
        IS2_atl11_corr_attrs[ptx]['ref_pt']['coordinates'] = \
            "delta_time latitude longitude"
        # cycle_number
        IS2_atl11_corr[ptx]['cycle_number'] = fileID[ptx]['cycle_number'][:].copy()
        IS2_atl11_fill[ptx]['cycle_number'] = None
        IS2_atl11_dims[ptx]['cycle_number'] = None
        IS2_atl11_corr_attrs[ptx]['cycle_number'] = collections.OrderedDict()
        IS2_atl11_corr_attrs[ptx]['cycle_number']['units'] = "1"
        IS2_atl11_corr_attrs[ptx]['cycle_number']['long_name'] = "Orbital cycle number"
        IS2_atl11_corr_attrs[ptx]['cycle_number']['source'] = "ATL06"
        IS2_atl11_corr_attrs[ptx]['cycle_number']['description'] = ("Number of 91-day periods "
            "that have elapsed since ICESat-2 entered the science orbit. Each of the 1,387 "
            "reference ground track (RGTs) is targeted in the polar regions once "
            "every 91 days.")
        # delta time
        IS2_atl11_corr[ptx]['delta_time'] = delta_time['AT'].copy()
        IS2_atl11_fill[ptx]['delta_time'] = delta_time['AT'].fill_value
        IS2_atl11_dims[ptx]['delta_time'] = ['ref_pt','cycle_number']
        IS2_atl11_corr_attrs[ptx]['delta_time'] = collections.OrderedDict()
        IS2_atl11_corr_attrs[ptx]['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl11_corr_attrs[ptx]['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl11_corr_attrs[ptx]['delta_time']['standard_name'] = "time"
        IS2_atl11_corr_attrs[ptx]['delta_time']['calendar'] = "standard"
        IS2_atl11_corr_attrs[ptx]['delta_time']['source'] = "ATL06"
        IS2_atl11_corr_attrs[ptx]['delta_time']['description'] = ("Number of GPS "
            "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
            "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
            "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
            "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
            "time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl11_corr_attrs[ptx]['delta_time']['coordinates'] = \
            "ref_pt cycle_number latitude longitude"
        # latitude
        IS2_atl11_corr[ptx]['latitude'] = latitude['AT'].copy()
        IS2_atl11_fill[ptx]['latitude'] = latitude['AT'].fill_value
        IS2_atl11_dims[ptx]['latitude'] = ['ref_pt']
        IS2_atl11_corr_attrs[ptx]['latitude'] = collections.OrderedDict()
        IS2_atl11_corr_attrs[ptx]['latitude']['units'] = "degrees_north"
        IS2_atl11_corr_attrs[ptx]['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_corr_attrs[ptx]['latitude']['long_name'] = "Latitude"
        IS2_atl11_corr_attrs[ptx]['latitude']['standard_name'] = "latitude"
        IS2_atl11_corr_attrs[ptx]['latitude']['source'] = "ATL06"
        IS2_atl11_corr_attrs[ptx]['latitude']['description'] = ("Center latitude of "
            "selected segments")
        IS2_atl11_corr_attrs[ptx]['latitude']['valid_min'] = -90.0
        IS2_atl11_corr_attrs[ptx]['latitude']['valid_max'] = 90.0
        IS2_atl11_corr_attrs[ptx]['latitude']['coordinates'] = \
            "ref_pt delta_time longitude"
        # longitude
        IS2_atl11_corr[ptx]['longitude'] = longitude['AT'].copy()
        IS2_atl11_fill[ptx]['longitude'] = longitude['AT'].fill_value
        IS2_atl11_dims[ptx]['longitude'] = ['ref_pt']
        IS2_atl11_corr_attrs[ptx]['longitude'] = collections.OrderedDict()
        IS2_atl11_corr_attrs[ptx]['longitude']['units'] = "degrees_east"
        IS2_atl11_corr_attrs[ptx]['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl11_corr_attrs[ptx]['longitude']['long_name'] = "Longitude"
        IS2_atl11_corr_attrs[ptx]['longitude']['standard_name'] = "longitude"
        IS2_atl11_corr_attrs[ptx]['longitude']['source'] = "ATL06"
        IS2_atl11_corr_attrs[ptx]['longitude']['description'] = ("Center longitude of "
            "selected segments")
        IS2_atl11_corr_attrs[ptx]['longitude']['valid_min'] = -180.0
        IS2_atl11_corr_attrs[ptx]['longitude']['valid_max'] = 180.0
        IS2_atl11_corr_attrs[ptx]['longitude']['coordinates'] = \
            "ref_pt delta_time latitude"

        # cycle statistics variables
        IS2_atl11_corr_attrs[ptx]['cycle_stats']['Description'] = ("The cycle_stats subgroup "
            "contains summary information about segments for each reference point, including "
            "the uncorrected mean heights for reference surfaces, blowing snow and cloud "
            "indicators, and geolocation and height misfit statistics.")
        IS2_atl11_corr_attrs[ptx]['cycle_stats']['data_rate'] = ("Data within this group "
            "are stored at the average segment rate.")

        # for each along-track dataset
        for key,val in OUTPUT['AT'].items():
            # add to output
            IS2_atl11_corr[ptx]['cycle_stats'][key] = val.copy()
            IS2_atl11_fill[ptx]['cycle_stats'][key] = val.fill_value
            IS2_atl11_dims[ptx]['cycle_stats'][key] = ['ref_pt','cycle_number']
            IS2_atl11_corr_attrs[ptx]['cycle_stats'][key] = collections.OrderedDict()
            IS2_atl11_corr_attrs[ptx]['cycle_stats'][key]['units'] = "meters"
            IS2_atl11_corr_attrs[ptx]['cycle_stats'][key]['contentType'] = "referenceInformation"
            IS2_atl11_corr_attrs[ptx]['cycle_stats'][key]['long_name'] = LONGNAME[key]
            IS2_atl11_corr_attrs[ptx]['cycle_stats'][key]['description'] = DESCRIPTION[key]
            IS2_atl11_corr_attrs[ptx]['cycle_stats'][key]['source'] = MODEL
            IS2_atl11_corr_attrs[ptx]['cycle_stats'][key]['reference'] = model_version
            IS2_atl11_corr_attrs[ptx]['cycle_stats'][key]['coordinates'] = \
                "../ref_pt ../cycle_number ../delta_time ../latitude ../longitude"

        # if crossover measurements were calculated
        if CROSSOVERS:
            # crossing track variables
            IS2_atl11_corr_attrs[ptx][XT]['Description'] = ("The crossing_track_data "
                "subgroup contains elevation data at crossover locations. These are "
                "locations where two ICESat-2 pair tracks cross, so data are available "
                "from both the datum track, for which the granule was generated, and "
                "from the crossing track.")
            IS2_atl11_corr_attrs[ptx][XT]['data_rate'] = ("Data within this group are "
                "stored at the average segment rate.")

            # reference point
            IS2_atl11_corr[ptx][XT]['ref_pt'] = ref_pt['XT'].copy()
            IS2_atl11_fill[ptx][XT]['ref_pt'] = None
            IS2_atl11_dims[ptx][XT]['ref_pt'] = None
            IS2_atl11_corr_attrs[ptx][XT]['ref_pt'] = collections.OrderedDict()
            IS2_atl11_corr_attrs[ptx][XT]['ref_pt']['units'] = "1"
            IS2_atl11_corr_attrs[ptx][XT]['ref_pt']['contentType'] = "referenceInformation"
            IS2_atl11_corr_attrs[ptx][XT]['ref_pt']['long_name'] = ("fit center reference point number, "
                "segment_id")
            IS2_atl11_corr_attrs[ptx][XT]['ref_pt']['source'] = "derived, ATL11 algorithm"
            IS2_atl11_corr_attrs[ptx][XT]['ref_pt']['description'] = ("The reference-point number of the "
                "fit center for the datum track. The reference point is the 7 digit segment_id number "
                "corresponding to the center of the ATL06 data used for each ATL11 point.  These are "
                "sequential, starting with 1 for the first segment after an ascending equatorial "
                "crossing node.")
            IS2_atl11_corr_attrs[ptx][XT]['ref_pt']['coordinates'] = \
                "delta_time latitude longitude"

            # reference ground track of the crossing track
            IS2_atl11_corr[ptx][XT]['rgt'] = fileID[ptx][XT]['rgt'][:].copy()
            IS2_atl11_fill[ptx][XT]['rgt'] = fileID[ptx][XT]['rgt'].attrs['_FillValue']
            IS2_atl11_dims[ptx][XT]['rgt'] = None
            IS2_atl11_corr_attrs[ptx][XT]['rgt'] = collections.OrderedDict()
            IS2_atl11_corr_attrs[ptx][XT]['rgt']['units'] = "1"
            IS2_atl11_corr_attrs[ptx][XT]['rgt']['contentType'] = "referenceInformation"
            IS2_atl11_corr_attrs[ptx][XT]['rgt']['long_name'] = "crossover reference ground track"
            IS2_atl11_corr_attrs[ptx][XT]['rgt']['source'] = "ATL06"
            IS2_atl11_corr_attrs[ptx][XT]['rgt']['description'] = "The RGT number for the crossing data."
            IS2_atl11_corr_attrs[ptx][XT]['rgt']['coordinates'] = \
                "ref_pt delta_time latitude longitude"
            # cycle_number of the crossing track
            IS2_atl11_corr[ptx][XT]['cycle_number'] = fileID[ptx][XT]['cycle_number'][:].copy()
            IS2_atl11_fill[ptx][XT]['cycle_number'] = fileID[ptx][XT]['cycle_number'].attrs['_FillValue']
            IS2_atl11_dims[ptx][XT]['cycle_number'] = None
            IS2_atl11_corr_attrs[ptx][XT]['cycle_number'] = collections.OrderedDict()
            IS2_atl11_corr_attrs[ptx][XT]['cycle_number']['units'] = "1"
            IS2_atl11_corr_attrs[ptx][XT]['cycle_number']['long_name'] = "crossover cycle number"
            IS2_atl11_corr_attrs[ptx][XT]['cycle_number']['source'] = "ATL06"
            IS2_atl11_corr_attrs[ptx][XT]['cycle_number']['description'] = ("Cycle number for the "
                "crossing data. Number of 91-day periods that have elapsed since ICESat-2 entered "
                "the science orbit. Each of the 1,387 reference ground track (RGTs) is targeted "
                "in the polar regions once every 91 days.")
            # delta time of the crossing track
            IS2_atl11_corr[ptx][XT]['delta_time'] = delta_time['XT'].copy()
            IS2_atl11_fill[ptx][XT]['delta_time'] = delta_time['XT'].fill_value
            IS2_atl11_dims[ptx][XT]['delta_time'] = ['ref_pt']
            IS2_atl11_corr_attrs[ptx][XT]['delta_time'] = {}
            IS2_atl11_corr_attrs[ptx][XT]['delta_time']['units'] = "seconds since 2018-01-01"
            IS2_atl11_corr_attrs[ptx][XT]['delta_time']['long_name'] = "Elapsed GPS seconds"
            IS2_atl11_corr_attrs[ptx][XT]['delta_time']['standard_name'] = "time"
            IS2_atl11_corr_attrs[ptx][XT]['delta_time']['calendar'] = "standard"
            IS2_atl11_corr_attrs[ptx][XT]['delta_time']['source'] = "ATL06"
            IS2_atl11_corr_attrs[ptx][XT]['delta_time']['description'] = ("Number of GPS "
                "seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch offset "
                "is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS seconds "
                "between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP epoch. By "
                "adding the offset contained within atlas_sdp_gps_epoch to delta time parameters, the "
                "time in gps_seconds relative to the GPS epoch can be computed.")
            IS2_atl11_corr_attrs[ptx]['delta_time']['coordinates'] = \
                "ref_pt latitude longitude"
            # latitude of the crossover measurement
            IS2_atl11_corr[ptx][XT]['latitude'] = latitude['XT'].copy()
            IS2_atl11_fill[ptx][XT]['latitude'] = latitude['XT'].fill_value
            IS2_atl11_dims[ptx][XT]['latitude'] = ['ref_pt']
            IS2_atl11_corr_attrs[ptx][XT]['latitude'] = collections.OrderedDict()
            IS2_atl11_corr_attrs[ptx][XT]['latitude']['units'] = "degrees_north"
            IS2_atl11_corr_attrs[ptx][XT]['latitude']['contentType'] = "physicalMeasurement"
            IS2_atl11_corr_attrs[ptx][XT]['latitude']['long_name'] = "crossover latitude"
            IS2_atl11_corr_attrs[ptx][XT]['latitude']['standard_name'] = "latitude"
            IS2_atl11_corr_attrs[ptx][XT]['latitude']['source'] = "ATL06"
            IS2_atl11_corr_attrs[ptx][XT]['latitude']['description'] = ("Center latitude of "
                "selected segments")
            IS2_atl11_corr_attrs[ptx][XT]['latitude']['valid_min'] = -90.0
            IS2_atl11_corr_attrs[ptx][XT]['latitude']['valid_max'] = 90.0
            IS2_atl11_corr_attrs[ptx][XT]['latitude']['coordinates'] = \
                "ref_pt delta_time longitude"
            # longitude of the crossover measurement
            IS2_atl11_corr[ptx][XT]['longitude'] = longitude['XT'].copy()
            IS2_atl11_fill[ptx][XT]['longitude'] = longitude['XT'].fill_value
            IS2_atl11_dims[ptx][XT]['longitude'] = ['ref_pt']
            IS2_atl11_corr_attrs[ptx][XT]['longitude'] = collections.OrderedDict()
            IS2_atl11_corr_attrs[ptx][XT]['longitude']['units'] = "degrees_east"
            IS2_atl11_corr_attrs[ptx][XT]['longitude']['contentType'] = "physicalMeasurement"
            IS2_atl11_corr_attrs[ptx][XT]['longitude']['long_name'] = "crossover longitude"
            IS2_atl11_corr_attrs[ptx][XT]['longitude']['standard_name'] = "longitude"
            IS2_atl11_corr_attrs[ptx][XT]['longitude']['source'] = "ATL06"
            IS2_atl11_corr_attrs[ptx][XT]['longitude']['description'] = ("Center longitude of "
                "selected segments")
            IS2_atl11_corr_attrs[ptx][XT]['longitude']['valid_min'] = -180.0
            IS2_atl11_corr_attrs[ptx][XT]['longitude']['valid_max'] = 180.0
            IS2_atl11_corr_attrs[ptx][XT]['longitude']['coordinates'] = \
                "ref_pt delta_time latitude"

            # for each crossover dataset
            for key,val in OUTPUT['XT'].items():
                # add to output
                IS2_atl11_corr[ptx][XT][key] = val.copy()
                IS2_atl11_fill[ptx][XT][key] = val.fill_value
                IS2_atl11_dims[ptx][XT][key] = ['ref_pt']
                IS2_atl11_corr_attrs[ptx][XT][key] = collections.OrderedDict()
                IS2_atl11_corr_attrs[ptx][XT][key]['units'] = "meters"
                IS2_atl11_corr_attrs[ptx][XT][key]['contentType'] = "referenceInformation"
                IS2_atl11_corr_attrs[ptx][XT][key]['long_name'] = LONGNAME[key]
                IS2_atl11_corr_attrs[ptx][XT][key]['description'] = DESCRIPTION[key]
                IS2_atl11_corr_attrs[ptx][XT][key]['source'] = MODEL
                IS2_atl11_corr_attrs[ptx][XT][key]['reference'] = model_version
                IS2_atl11_corr_attrs[ptx][XT][key]['coordinates'] = \
                    "ref_pt delta_time latitude longitude"

    # output HDF5 files with interpolated surface mass balance data
    args = (PRD,model_version,TRK,GRAN,SCYC,ECYC,RL,VERS,AUX)
    file_format = '{0}_{1}_{2}{3}_{4}{5}_{6}_{7}{8}.h5'
    # print file information
    logging.info('\t{0}'.format(file_format.format(*args)))
    HDF5_ATL11_corr_write(IS2_atl11_corr, IS2_atl11_corr_attrs,
        CLOBBER=True, INPUT=os.path.basename(FILE), CROSSOVERS=CROSSOVERS,
        FILL_VALUE=IS2_atl11_fill, DIMENSIONS=IS2_atl11_dims,
        FILENAME=os.path.join(ddir,file_format.format(*args)))
    # change the permissions mode
    os.chmod(os.path.join(ddir,file_format.format(*args)), MODE)

# PURPOSE: outputting the correction values for ICESat-2 data to HDF5
def HDF5_ATL11_corr_write(IS2_atl11_corr, IS2_atl11_attrs, INPUT=None,
    FILENAME='', FILL_VALUE=None, DIMENSIONS=None, CROSSOVERS=False,
    CLOBBER=False):
    # setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    # open output HDF5 file
    fileID = h5py.File(os.path.expanduser(FILENAME), clobber)

    # create HDF5 records
    h5 = {}

    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    h5['ancillary_data'] = {}
    for k,v in IS2_atl11_corr['ancillary_data'].items():
        # Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        # add HDF5 variable attributes
        for att_name,att_val in IS2_atl11_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    # write each output beam pair
    pairs = [k for k in IS2_atl11_corr.keys() if bool(re.match(r'pt\d',k))]
    for ptx in pairs:
        fileID.create_group(ptx)
        h5[ptx] = {}
        # add HDF5 group attributes for beam
        for att_name in ['description','beam_pair','ReferenceGroundTrack',
            'first_cycle','last_cycle','equatorial_radius','polar_radius']:
            fileID[ptx].attrs[att_name] = IS2_atl11_attrs[ptx][att_name]

        # ref_pt, cycle number, geolocation and delta_time variables
        for k in ['ref_pt','cycle_number','delta_time','latitude','longitude']:
            # values and attributes
            v = IS2_atl11_corr[ptx][k]
            attrs = IS2_atl11_attrs[ptx][k]
            fillvalue = FILL_VALUE[ptx][k]
            # Defining the HDF5 dataset variables
            val = '{0}/{1}'.format(ptx,k)
            if fillvalue:
                h5[ptx][k] = fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, fillvalue=fillvalue, compression='gzip')
            else:
                h5[ptx][k] = fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, compression='gzip')
            # create or attach dimensions for HDF5 variable
            if DIMENSIONS[ptx][k]:
                # attach dimensions
                for i,dim in enumerate(DIMENSIONS[ptx][k]):
                    h5[ptx][k].dims[i].attach_scale(h5[ptx][dim])
            else:
                # make dimension
                h5[ptx][k].make_scale(k)
            # add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[ptx][k].attrs[att_name] = att_val

        # add to cycle_stats variables
        groups = ['cycle_stats']
        # if running crossovers: add to crossing_track_data variables
        if CROSSOVERS:
            groups.append('crossing_track_data')
        for key in groups:
            fileID[ptx].create_group(key)
            h5[ptx][key] = {}
            for att_name in ['Description','data_rate']:
                att_val=IS2_atl11_attrs[ptx][key][att_name]
                fileID[ptx][key].attrs[att_name] = att_val
            for k,v in IS2_atl11_corr[ptx][key].items():
                # attributes
                attrs = IS2_atl11_attrs[ptx][key][k]
                fillvalue = FILL_VALUE[ptx][key][k]
                # Defining the HDF5 dataset variables
                val = '{0}/{1}/{2}'.format(ptx,key,k)
                if fillvalue:
                    h5[ptx][key][k] = fileID.create_dataset(val, np.shape(v), data=v,
                        dtype=v.dtype, fillvalue=fillvalue, compression='gzip')
                else:
                    h5[ptx][key][k] = fileID.create_dataset(val, np.shape(v), data=v,
                        dtype=v.dtype, compression='gzip')
                # create or attach dimensions for HDF5 variable
                if DIMENSIONS[ptx][key][k]:
                    # attach dimensions
                    for i,dim in enumerate(DIMENSIONS[ptx][key][k]):
                        if (key == 'cycle_stats'):
                            h5[ptx][key][k].dims[i].attach_scale(h5[ptx][dim])
                        else:
                            h5[ptx][key][k].dims[i].attach_scale(h5[ptx][key][dim])
                else:
                    # make dimension
                    h5[ptx][key][k].make_scale(k)
                # add HDF5 variable attributes
                for att_name,att_val in attrs.items():
                    h5[ptx][key][k].attrs[att_name] = att_val

    # HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 Annual Land Ice Height'
    fileID.attrs['summary'] = ('The purpose of ATL11 is to provide an ICESat-2 '
        'satellite cycle summary of heights and height changes of land-based '
        'ice and will be provided as input to ATL15 and ATL16, gridded '
        'estimates of heights and height-changes.')
    fileID.attrs['description'] = ('Land ice parameters for each beam pair. '
        'All parameters are calculated for the same along-track increments '
        'for each beam pair and repeat.')
    date_created = datetime.datetime.today()
    fileID.attrs['date_created'] = date_created.isoformat()
    project = 'ICESat-2 > Ice, Cloud, and land Elevation Satellite-2'
    fileID.attrs['project'] = project
    platform = 'ICESat-2 > Ice, Cloud, and land Elevation Satellite-2'
    fileID.attrs['project'] = platform
    # add attribute for elevation instrument and designated processing level
    instrument = 'ATLAS > Advanced Topographic Laser Altimeter System'
    fileID.attrs['instrument'] = instrument
    fileID.attrs['source'] = 'Spacecraft'
    fileID.attrs['references'] = 'https://nsidc.org/data/icesat-2'
    fileID.attrs['processing_level'] = '4'
    # add attributes for input ATL11 files
    fileID.attrs['input_files'] = os.path.basename(INPUT)
    # find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for ptx in pairs:
        lon = IS2_atl11_corr[ptx]['longitude']
        lat = IS2_atl11_corr[ptx]['latitude']
        delta_time = IS2_atl11_corr[ptx]['delta_time']
        valid = np.nonzero(delta_time != FILL_VALUE[ptx]['delta_time'])
        # setting the geospatial and temporal ranges
        lnmn = lon.min() if (lon.min() < lnmn) else lnmn
        lnmx = lon.max() if (lon.max() > lnmx) else lnmx
        ltmn = lat.min() if (lat.min() < ltmn) else ltmn
        ltmx = lat.max() if (lat.max() > ltmx) else ltmx
        tmn = delta_time[valid].min() if (delta_time[valid].min() < tmn) else tmn
        tmx = delta_time[valid].max() if (delta_time[valid].max() > tmx) else tmx
    # add geospatial and temporal attributes
    fileID.attrs['geospatial_lat_min'] = ltmn
    fileID.attrs['geospatial_lat_max'] = ltmx
    fileID.attrs['geospatial_lon_min'] = lnmn
    fileID.attrs['geospatial_lon_max'] = lnmx
    fileID.attrs['geospatial_lat_units'] = "degrees_north"
    fileID.attrs['geospatial_lon_units'] = "degrees_east"
    fileID.attrs['geospatial_ellipsoid'] = "WGS84"
    fileID.attrs['date_type'] = 'UTC'
    fileID.attrs['time_type'] = 'CCSDS UTC-A'
    # convert start and end time from ATLAS SDP seconds into Julian days
    JD = convert_delta_time(np.array([tmn,tmx]))['julian']
    # convert to calendar date
    YY,MM,DD,HH,MN,SS = SMBcorr.time.convert_julian(JD,FORMAT='tuple')
    # add attributes with measurement date start, end and duration
    tcs = datetime.datetime(int(YY[0]), int(MM[0]), int(DD[0]),
        int(HH[0]), int(MN[0]), int(SS[0]), int(1e6*(SS[0] % 1)))
    fileID.attrs['time_coverage_start'] = tcs.isoformat()
    tce = datetime.datetime(int(YY[1]), int(MM[1]), int(DD[1]),
        int(HH[1]), int(MN[1]), int(SS[1]), int(1e6*(SS[1] % 1)))
    fileID.attrs['time_coverage_end'] = tce.isoformat()
    fileID.attrs['time_coverage_duration'] = '{0:0.0f}'.format(tmx-tmn)
    # Closing the HDF5 file
    fileID.close()

# Main program that calls interp_SMB_ICESat2()
def main():
    # Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Interpolates daily firn model estimates to times
            and locations of ICESat-2 ATL11 annual land ice height data
            """
    )
    # command line parameters
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat-2 ATL11 file to run')
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
    # run with ATL11 crossovers
    parser.add_argument('--crossovers','-C',
        default=False, action='store_true',
        help='Run ATL11 Crossovers')
    # use compressed model files
    parser.add_argument('--gzip','-G',
        default=False, action='store_true',
        help='Model files are gzip compressed')
    # verbosity settings
    # verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Output information about each created file')
    # permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files created')
    args = parser.parse_args()

    # run for each input ATL11 file
    for FILE in args.infile:
        interp_SMB_ICESat2(args.directory, FILE, args.model,
            CROSSOVERS=args.crossovers, GZIP=args.gzip,
            VERBOSE=args.verbose, MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()