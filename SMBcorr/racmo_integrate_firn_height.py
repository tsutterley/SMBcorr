#!/usr/bin/env python
u"""
racmo_integrate_firn_height.py
Written by Tyler Sutterley (10/2021)
Integrate RACMO firn heights for each Promice ice class

CALLING SEQUENCE:
    python racmo_integrate_firn_height.py --directory <path> FGRN055

INPUTS:
    model: Firn model outputs to interpolate
        FGRN055: 1km interpolated Greenland RACMO2.3p2
        FGRN11: 11km Greenland RACMO2.3p2

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -O, --output: Output integrated results to file

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    netCDF4: Python interface to the netCDF C library
         https://unidata.github.io/netcdf4-python/netCDF4/index.html

PROGRAM DEPENDENCIES:
    regress_model.py: models a time series using least-squares regression

UPDATE HISTORY:
    Updated 10/2021: using argparse to set command line parameters
    Written 10/2019
"""
from __future__ import print_function

import sys
import os
import re
import netCDF4
import argparse
import numpy as np
import scipy.interpolate
from SMBcorr.regress_model import regress_model

#-- PURPOSE: read and integrate RACMO2.3 firn corrections
def racmo_integrate_firn_height(base_dir, MODEL, VARIABLE='zs', OUTPUT=True):

    #-- set parameters based on input model
    FIRN_FILE = {}
    if (MODEL == 'FGRN11'):
        #-- filename and directory for input FGRN11 file
        FIRN_FILE['zs'] = 'FDM_zs_FGRN11_1960-2016.nc'
        FIRN_FILE['FirnAir'] = 'FDM_FirnAir_FGRN11_1960-2016.nc'
        FIRN_DIRECTORY = ['RACMO','FGRN11_1960-2016']
        FIRN_OUTPUT = 'FDM_{0}_FGRN11_1960-2016_Promice.txt'
        #-- time is year decimal from 1960-01-01 at time_step 10 days
        time_step = 10.0/365.25
        #-- rotation parameters
        rot_lat = -18.0
        rot_lon = -37.5
    elif (MODEL == 'FGRN055'):
        #-- filename and directory for input FGRN055 file
        FIRN_FILE['zs'] = 'FDM_zs_FGRN055_1960-2017_interpol.nc'
        FIRN_FILE['FirnAir'] = 'FDM_FirnAir_FGRN055_1960-2017_interpol.nc'
        FIRN_FILE['Mask'] = 'FGRN055_Masks_5.5km.nc'
        FIRN_DIRECTORY = ['RACMO','FGRN055_1960-2017']
        FIRN_OUTPUT = 'FDM_{0}_FGRN055_1960-2017_Promice.txt'
        #-- time is year decimal from 1960-01-01 at time_step 10 days
        time_step = 10.0/365.25
        #-- rotation parameters
        rot_lat = -18.0
        rot_lon = -37.5

    #-- Open the RACMO NetCDF file for reading
    ddir = os.path.join(base_dir,*FIRN_DIRECTORY)
    fileID = netCDF4.Dataset(os.path.join(ddir,FIRN_FILE[VARIABLE]), 'r')
    #-- Get data from each netCDF variable and remove singleton dimensions
    fd = {}
    fd[VARIABLE] = np.squeeze(fileID.variables[VARIABLE][:].copy())
    fd['lon'] = fileID.variables['lon'][:,:].copy()
    fd['lat'] = fileID.variables['lat'][:,:].copy()
    fd['time'] = fileID.variables['time'][:].copy()
    #-- invalid data value
    fv = np.float(fileID.variables[VARIABLE]._FillValue)
    #-- input shape of RACMO firn data
    nt,ny,nx = np.shape(fd[VARIABLE])
    #-- close the NetCDF files
    fileID.close()

    #-- Open the RACMO Mask NetCDF file for reading
    fileID = netCDF4.Dataset(os.path.join(ddir,FIRN_FILE['Mask']), 'r')
    #-- Get data from each netCDF mask variable and remove singleton dimensions
    mask = {}
    for var in ['Area','Icemask_GR','Promicemask','Topography','lon','lat']:
        mask[var] = np.squeeze(fileID.variables[var][:].copy())
    my,mx = np.shape(mask['Area'])
    #-- close the NetCDF files
    fileID.close()

    #-- rotated pole longitude and latitude of input model (model coordinates)
    xg,yg = rotate_coordinates(fd['lon'], fd['lat'], rot_lon, rot_lat)
    xmask,ymask = rotate_coordinates(mask['lon'], mask['lat'], rot_lon, rot_lat)
    #-- recreate arrays to fix small floating point errors
    #-- (ensure that arrays are monotonically increasing)
    mask['x'] = np.linspace(np.mean(xmask[:,0]),np.mean(xmask[:,-1]),mx)
    mask['y'] = np.linspace(np.mean(ymask[0,:]),np.mean(ymask[-1,:]),my)

    #-- create an interpolator for input masks
    #-- masks are on the original RACMO grid and not the firn model grid
    IMI = scipy.interpolate.RegularGridInterpolator((mask['y'],mask['x']),
        mask['Icemask_GR'])
    PMI = scipy.interpolate.RegularGridInterpolator((mask['y'],mask['x']),
        mask['Promicemask'])
    AMI = scipy.interpolate.RegularGridInterpolator((mask['y'],mask['x']),
        mask['Area'])
    #-- interpolate masks to firn model coordinates
    Icemask_GR = IMI.__call__(np.c_[yg.flatten(),xg.flatten()])
    Promicemask = PMI.__call__(np.c_[yg.flatten(),xg.flatten()])
    #-- reshape, round to fix interpolation errors and convert to integers
    fd['Icemask_GR'] = np.round(Icemask_GR.reshape(ny,nx)).astype('i')
    fd['Promicemask'] = np.round(Promicemask.reshape(ny,nx)).astype('i')
    #-- interpolate area to firn model coordinates
    fd['Area'] = AMI.__call__(np.c_[yg.flatten(),xg.flatten()]).reshape(ny,nx)
    #-- clear memory of flattened interpolation masks
    Icemask_GR = None
    Promicemask = None

    #-- output integrated arrays of firn variable (height or firn air content)
    #-- for each land classification mask in km^3
    firn_volume = np.full((nt,3),fv,dtype=np.float)
    #-- extrapolate out in time two years
    tdec = np.arange(fd['time'][-1]+time_step,fd['time'][-1]+2,time_step)
    ntx = len(tdec)
    firn_extrap = np.full((ntx,3),fv,dtype=np.float)
    for m in range(3):
        #-- indices of specified mask (0==ocean, 1==ice caps outside Greenland)
        #-- masks of interest: Greenland ice sheet and peripheral glaciers (2-4)
        i,j = np.nonzero((fd[VARIABLE][0,:,:] != fv) & (fd['Icemask_GR'] == 1) &
            (fd['Promicemask'] == (m+2)))
        #-- for each time
        for t in range(nt):
            #-- convert firn height change to km
            firn_volume[t,m] = np.sum(fd[VARIABLE][t,i,j]*fd['Area'][i,j]/1e3)
        #-- calculate a regression model for calculating values
        #-- read last 10 years of data to create regression model
        N = 365
        T = np.zeros((N))
        FIRN = np.zeros((N))
        #-- reduce time series for calculating regression model
        for k in range(N):
            kk = nt - N + k
            #-- time at k
            T[k] = fd['time'][kk]
            FIRN[k] = firn_volume[kk,m]
        #-- calculate regression model
        firn_extrap[:,m] = regress_model(T, FIRN, tdec, ORDER=2,
            CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=T[-1])

    #-- combine into single arrays
    combined_time = np.concatenate((fd['time'],tdec),axis=0)
    combined_firn = np.concatenate((firn_volume,firn_extrap),axis=0)

    #-- print to file
    if OUTPUT:
        #-- open the file
        fid = open(os.path.join(ddir,FIRN_OUTPUT.format(VARIABLE)),'w')
        #-- print for each time
        for i,t in enumerate(combined_time):
            args = (t, *combined_firn[i,:])
            print('{0:0.4f}{1:12.4f}{2:12.4f}{3:12.4f}'.format(*args),file=fid)
        #-- close the file
        fid.close()

    #-- return the combined integrated values
    return (combined_firn,combined_time)

#-- PURPOSE: calculate rotated pole coordinates
def rotate_coordinates(lon, lat, rot_lon, rot_lat):
    #-- convert from degrees to radians
    phi = np.pi*lon/180.0
    phi_r = np.pi*rot_lon/180.0
    th = np.pi*lat/180.0
    th_r = np.pi*rot_lat/180.0
    #-- calculate rotation parameters
    R1 = np.sin(phi - phi_r)*np.cos(th)
    R2 = np.cos(th_r)*np.sin(th) - np.sin(th_r)*np.cos(th)*np.cos(phi - phi_r)
    R3 = -np.sin(th_r)*np.sin(th) - np.cos(th_r)*np.cos(th)*np.cos(phi - phi_r)
    #-- rotated pole longitude and latitude of input model
    #-- convert back into degrees
    Xr = np.arctan2(R1,R2)*180.0/np.pi
    Yr = np.arcsin(R3)*180.0/np.pi
    #-- return the rotated coordinates
    return (Xr,Yr)

#-- Main program that calls racmo_integrate_firn_height()
def main():
    #-- Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Integrate RACMO firn heights for each Promice
            ice class
            """
    )
    #-- working data directory
    parser.add_argument('model',
        type=str, choices=('FGRN055','FGRN11'),
        help='Firn model outputs to interpolate')
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
    #-- output integrated results to file
    parser.add_argument('--output','-O',
        default=False, action='store_true',
        help='Output integrated results to file')
    args,_ = parser.parse_known_args()

    #-- read and integrate RACMO2.3 firn corrections
    zs,tzs = racmo_integrate_firn_height(args.directory,
        args.model, VARIABLE='zs', OUTPUT=args.output)
    air,tair = racmo_integrate_firn_height(args.directory,
        args.model, VARIABLE='FirnAir', OUTPUT=args.output)

#-- run main program
if __name__ == '__main__':
    main()
