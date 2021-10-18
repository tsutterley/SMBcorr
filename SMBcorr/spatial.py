#!/usr/bin/env python
u"""
spatial.py
Written by Tyler Sutterley (10/2021)

Data class for reading, writing and processing spatial data

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    dateutil: powerful extensions to datetime
        https://dateutil.readthedocs.io/en/stable/
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/netCDF4/index.html
    h5py: Pythonic interface to the HDF5 binary data format.
        https://www.h5py.org/

PROGRAM DEPENDENCIES:
    time.py: utilities for calculating time operations

UPDATE HISTORY:
    Written 10/2021
"""
import os
import re
import io
import copy
import gzip
import h5py
import time
import uuid
import logging
import zipfile
import netCDF4
import numpy as np

class spatial(object):
    """
    Data class for reading, writing and processing spatial data
    """
    np.seterr(invalid='ignore')
    def __init__(self, **kwargs):
        #-- set default keyword arguments
        kwargs.setdefault('spacing',[None,None])
        kwargs.setdefault('nlat',None)
        kwargs.setdefault('nlon',None)
        kwargs.setdefault('extent',[None]*4)
        kwargs.setdefault('fill_value',None)
        #-- set default class attributes
        self.data=None
        self.mask=None
        self.lon=None
        self.lat=None
        self.time=None
        self.fill_value=kwargs['fill_value']
        self.attributes=dict()
        self.extent=kwargs['extent']
        self.spacing=kwargs['spacing']
        self.shape=[kwargs['nlat'],kwargs['nlon'],None]
        self.ndim=None
        self.filename=None

    def case_insensitive_filename(self,filename):
        """
        Searches a directory for a filename without case dependence
        """
        #-- check if filename is open file object
        if isinstance(filename, io.IOBase):
            self.filename = copy.copy(filename)
        else:
            #-- tilde-expand input filename
            self.filename = os.path.expanduser(filename)
            #-- check if file presently exists with input case
            if not os.access(self.filename,os.F_OK):
                #-- search for filename without case dependence
                basename = os.path.basename(filename)
                directory = os.path.dirname(os.path.expanduser(filename))
                f = [f for f in os.listdir(directory) if re.match(basename,f,re.I)]
                if not f:
                    errmsg = '{0} not found in file system'.format(filename)
                    raise FileNotFoundError(errmsg)
                self.filename = os.path.join(directory,f.pop())
        return self

    def from_ascii(self, filename, date=True, **kwargs):
        """
        Read a spatial object from an ascii file
        Inputs: full path of input ascii file
        Options:
            ascii file contains date information
            keyword arguments for ascii input
        """
        #-- set filename
        self.case_insensitive_filename(filename)
        #-- set default parameters
        kwargs.setdefault('verbose',False)
        kwargs.setdefault('compression',None)
        kwargs.setdefault('columns',['lon','lat','data','time'])
        kwargs.setdefault('header',0)
        #-- open the ascii file and extract contents
        logging.info(self.filename)
        if (kwargs['compression'] == 'gzip'):
            #-- read input ascii data from gzip compressed file and split lines
            with gzip.open(self.filename,'r') as f:
                file_contents = f.read().decode('ISO-8859-1').splitlines()
        elif (kwargs['compression'] == 'zip'):
            #-- read input ascii data from zipped file and split lines
            base,_ = os.path.splitext(self.filename)
            with zipfile.ZipFile(self.filename) as z:
                file_contents = z.read(base).decode('ISO-8859-1').splitlines()
        elif (kwargs['compression'] == 'bytes'):
            #-- read input file object and split lines
            file_contents = self.filename.read().splitlines()
        else:
            #-- read input ascii file (.txt, .asc) and split lines
            with open(self.filename,'r') as f:
                file_contents = f.read().splitlines()
        #-- compile regular expression operator for extracting numerical values
        #-- from input ascii files of spatial data
        regex_pattern = r'[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[EeD][+-]?\d+)?'
        rx = re.compile(regex_pattern, re.VERBOSE)
        #-- output spatial dimensions
        if (None not in self.extent):
            self.lat = np.linspace(self.extent[3],self.extent[2],self.shape[0])
            self.lon = np.linspace(self.extent[0],self.extent[1],self.shape[1])
        else:
            self.lat = np.zeros((self.shape[0]))
            self.lon = np.zeros((self.shape[1]))
        #-- output spatial data
        self.data = np.zeros((self.shape[0],self.shape[1]))
        self.mask = np.zeros((self.shape[0],self.shape[1]),dtype=bool)
        #-- remove time from list of column names if not date
        columns = [c for c in kwargs['columns'] if (c != 'time')]
        #-- extract spatial data array and convert to matrix
        #-- for each line in the file
        header = kwargs['header']
        for line in file_contents[header:]:
            #-- extract columns of interest and assign to dict
            #-- convert fortran exponentials if applicable
            d = {c:r.replace('D','E') for c,r in zip(columns,rx.findall(line))}
            #-- convert line coordinates to integers
            ilon = np.int64(np.float64(d['lon'])/self.spacing[0])
            ilat = np.int64((90.0-np.float64(d['lat']))//self.spacing[1])
            self.data[ilat,ilon] = np.float64(d['data'])
            self.mask[ilat,ilon] = False
            self.lon[ilon] = np.float64(d['lon'])
            self.lat[ilat] = np.float64(d['lat'])
            #-- if the ascii file contains date variables
            if date:
                self.time = np.array(d['time'],dtype='f')
        #-- get spacing and dimensions
        self.update_spacing()
        self.update_extents()
        self.update_dimensions()
        self.update_mask()
        return self

    def from_netCDF4(self, filename, **kwargs):
        """
        Read a spatial object from a netCDF4 file
        Inputs: full path of input netCDF4 file
        Options:
            netCDF4 file contains date information
            keyword arguments for netCDF4 reader
        """
        #-- set filename
        self.case_insensitive_filename(filename)
        #-- set default parameters
        kwargs.setdefault('date',True)
        kwargs.setdefault('verbose',False)
        kwargs.setdefault('compression',None)
        kwargs.setdefault('varname','z')
        kwargs.setdefault('lonname','lon')
        kwargs.setdefault('latname','lat')
        kwargs.setdefault('timename','time')
        #-- read data from netCDF5 file
        data = ncdf_read(self.filename, **kwargs)
        #-- copy variables to spatial object
        self.data = data['data'].copy()
        if '_FillValue' in data['attributes']['data'].keys():
            self.fill_value = data['attributes']['data']['_FillValue']
        self.mask = np.zeros(self.data.shape, dtype=bool)
        self.lon = data['lon'].copy()
        self.lat = data['lat'].copy()
        #-- if the netCDF4 file contains date variables
        if kwargs['date']:
            self.time = data['time'].copy()
        #-- update attributes
        self.attributes.update(data['attributes'])
        #-- get spacing and dimensions
        self.update_spacing()
        self.update_extents()
        self.update_dimensions()
        self.update_mask()
        return self

    def from_HDF5(self, filename, **kwargs):
        """
        Read a spatial object from a HDF5 file
        Inputs: full path of input HDF5 file
        Options:
            keyword arguments for HDF5 reader
        """
        #-- set filename
        self.case_insensitive_filename(filename)
        #-- set default parameters
        kwargs.setdefault('date',True)
        kwargs.setdefault('verbose',False)
        kwargs.setdefault('compression',None)
        kwargs.setdefault('varname','z')
        kwargs.setdefault('lonname','lon')
        kwargs.setdefault('latname','lat')
        kwargs.setdefault('timename','time')
        #-- read data from HDF5 file
        data = hdf5_read(self.filename, **kwargs)
        #-- copy variables to spatial object
        self.data = data['data'].copy()
        if '_FillValue' in data['attributes']['data'].keys():
            self.fill_value = data['attributes']['_FillValue']
        self.mask = np.zeros(self.data.shape, dtype=bool)
        self.lon = data['lon'].copy()
        self.lat = data['lat'].copy()
        #-- if the HDF5 file contains date variables
        if kwargs['date']:
            self.time = data['time'].copy()
        #-- update attributes
        self.attributes.update(data['attributes'])
        #-- get spacing and dimensions
        self.update_spacing()
        self.update_extents()
        self.update_dimensions()
        self.update_mask()
        return self

    def from_file(self, filename, format=None, **kwargs):
        """
        Read a spatial object from a specified format
        Inputs: full path of input file
        Options:
        file format (ascii, netCDF4, HDF5)
        file contains date information
        **kwargs: keyword arguments for input readers
        """
        #-- set filename
        self.case_insensitive_filename(filename)
        #-- set default verbosity
        kwargs.setdefault('verbose',False)
        #-- read from file
        if (format == 'ascii'):
            #-- ascii (.txt)
            return spatial().from_ascii(filename, **kwargs)
        elif (format == 'netCDF4'):
            #-- netcdf (.nc)
            return spatial().from_netCDF4(filename, **kwargs)
        elif (format == 'HDF5'):
            #-- HDF5 (.H5)
            return spatial().from_HDF5(filename, **kwargs)

    def from_list(self, object_list, **kwargs):
        """
        Build a sorted spatial object from a list of other spatial objects
        Inputs: list of spatial object to be merged
        Options:
            spatial objects contain date information
            sort spatial objects by date information
            clear the spatial list from memory
        """
        #-- set default keyword arguments
        kwargs.setdefault('date',True)
        kwargs.setdefault('sort',True)
        kwargs.setdefault('clear',False)
        #-- number of spatial objects in list
        n = len(object_list)
        #-- indices to sort data objects if spatial list contain dates
        if kwargs['date'] and kwargs['sort']:
            list_sort = np.argsort([d.time for d in object_list],axis=None)
        else:
            list_sort = np.arange(n)
        #-- extract dimensions and grid spacing
        self.spacing = object_list[0].spacing
        self.extent = object_list[0].extent
        self.shape = object_list[0].shape
        #-- create output spatial grid and mask
        self.data = np.zeros((self.shape[0],self.shape[1],n))
        self.mask = np.zeros((self.shape[0],self.shape[1],n),dtype=bool)
        self.fill_value = object_list[0].fill_value
        self.lon = object_list[0].lon.copy()
        self.lat = object_list[0].lat.copy()
        #-- create list of files and attributes
        self.filename = []
        self.attributes = []
        #-- output dates
        if kwargs['date']:
            self.time = np.zeros((n))
        #-- for each indice
        for t,i in enumerate(list_sort):
            self.data[:,:,t] = object_list[i].data[:,:].copy()
            self.mask[:,:,t] |= object_list[i].mask[:,:]
            if kwargs['date']:
                self.time[t] = np.atleast_1d(object_list[i].time)
            #-- append filename to list
            if getattr(object_list[i], 'filename'):
                self.filename.append(object_list[i].filename)
            #-- append attributes to list
            if getattr(object_list[i], 'attributes'):
                self.attributes.append(object_list[i].attributes)
        #-- update the dimensions
        self.update_dimensions()
        self.update_mask()
        #-- clear the input list to free memory
        if kwargs['clear']:
            object_list = None
        #-- return the single spatial object
        return self

    def from_dict(self, d, **kwargs):
        """
        Convert a dict object to a spatial object
        Inputs: dictionary object to be converted
        """
        #-- assign variables to self
        for key in ['lon','lat','data','error','time']:
            try:
                setattr(self, key, d[key].copy())
            except (AttributeError, KeyError):
                pass
        #-- create output mask for data
        self.mask = np.zeros_like(self.data,dtype=bool)
        #-- get spacing and dimensions
        self.update_spacing()
        self.update_extents()
        self.update_dimensions()
        self.update_mask()
        return self

    def to_ascii(self, filename, date=True, **kwargs):
        """
        Write a spatial object to ascii file
        Inputs: full path of output ascii file
        Options:
            spatial objects contain date information
            keyword arguments for ascii output
        """
        self.filename = os.path.expanduser(filename)
        #-- set default verbosity
        kwargs.setdefault('verbose',False)
        logging.info(self.filename)
        #-- open the output file
        fid = open(self.filename, 'w')
        if date:
            file_format = '{0:10.4f} {1:10.4f} {2:12.4f} {3:10.4f}'
        else:
            file_format = '{0:10.4f} {1:10.4f} {2:12.4f}'
        #-- write to file for each valid latitude and longitude
        ii,jj = np.nonzero((self.data != self.fill_value) & (~self.mask))
        for ln,lt,dt in zip(self.lon[jj],self.lat[ii],self.data[ii,jj]):
            print(file_format.format(ln,lt,dt,self.time), file=fid)
        #-- close the output file
        fid.close()

    def to_netCDF4(self, filename, **kwargs):
        """
        Write a spatial object to netCDF4 file
        Inputs: full path of output netCDF4 file
        Options: spatial objects contain date information
        **kwargs: keyword arguments for netCDF4 writer
        """
        self.filename = os.path.expanduser(filename)
        #-- set default verbosity and parameters
        kwargs.setdefault('date',True)
        kwargs.setdefault('verbose',False)
        kwargs.setdefault('varname','z')
        kwargs.setdefault('lonname','lon')
        kwargs.setdefault('latname','lat')
        kwargs.setdefault('timename','time')
        kwargs.setdefault('time_units','years')
        kwargs.setdefault('time_longname','Date_in_Decimal_Years')
        #-- write to netCDF4
        ncdf_write(self.data, self.lon, self.lat, self.time,
            **kwargs)

    def to_HDF5(self, filename, **kwargs):
        """
        Write a spatial object to HDF5 file
        Inputs: full path of output HDF5 file
        Options: spatial objects contain date information
        **kwargs: keyword arguments for HDF5 writer
        """
        self.filename = os.path.expanduser(filename)
        #-- set default verbosity and parameters
        kwargs.setdefault('date',True)
        kwargs.setdefault('verbose',False)
        kwargs.setdefault('varname','z')
        kwargs.setdefault('lonname','lon')
        kwargs.setdefault('latname','lat')
        kwargs.setdefault('timename','time')
        kwargs.setdefault('time_units','years')
        kwargs.setdefault('time_longname','Date_in_Decimal_Years')
        #-- write to HDF5
        hdf5_write(self.data, self.lon, self.lat, self.time,
            **kwargs)

    def to_file(self, filename, format=None, date=True, **kwargs):
        """
        Write a spatial object to a specified format
        Inputs: full path of output file
        Options:
            file format (ascii, netCDF4 or HDF5)
            spatial object contains date information
            keyword arguments for output writers
        """
        #-- set default verbosity
        kwargs.setdefault('verbose',False)
        #-- write to file
        if (format == 'ascii'):
            #-- ascii (.txt)
            self.to_ascii(filename, date=date, **kwargs)
        elif (format == 'netCDF4'):
            #-- netcdf (.nc)
            self.to_netCDF4(filename, date=date, **kwargs)
        elif (format == 'HDF5'):
            #-- HDF5 (.H5)
            self.to_HDF5(filename, date=date, **kwargs)

    def to_masked_array(self):
        """
        Convert a spatial object to a masked numpy array
        """
        return np.ma.array(self.data, mask=self.mask,
            fill_value=self.fill_value)

    def update_spacing(self):
        """
        Calculate the step size of spatial object
        """
        #-- calculate degree spacing
        dlat = np.abs(self.lat[1] - self.lat[0])
        dlon = np.abs(self.lon[1] - self.lon[0])
        self.spacing = (dlon,dlat)
        return self

    def update_extents(self):
        """
        Calculate the bounds of spatial object
        """
        self.extent[0] = np.min(self.lon)
        self.extent[1] = np.max(self.lon)
        self.extent[2] = np.min(self.lat)
        self.extent[3] = np.max(self.lat)

    def update_dimensions(self):
        """
        Update the dimensions of the spatial object
        """
        self.shape = np.shape(self.data)
        self.ndim = np.ndim(self.data)
        return self

    def update_mask(self):
        """
        Update the mask of the spatial object
        """
        if self.fill_value is not None:
            self.mask |= (self.data == self.fill_value)
            self.mask |= np.isnan(self.data)
            self.data[self.mask] = self.fill_value
        return self

    def copy(self):
        """
        Copy a spatial object to a new spatial object
        """
        temp = spatial(fill_value=self.fill_value)
        #-- copy attributes or update attributes dictionary
        if isinstance(self.attributes,list):
            setattr(temp,'attributes',self.attributes)
        elif isinstance(self.attributes,dict):
            temp.attributes.update(self.attributes)
        #-- assign variables to self
        var = ['lon','lat','data','mask','error','time']
        for key in var:
            try:
                val = getattr(self, key)
                setattr(temp, key, np.copy(val))
            except AttributeError:
                pass
        #-- get spacing and dimensions
        temp.update_spacing()
        temp.update_extents()
        temp.update_dimensions()
        temp.replace_masked()
        return temp

    def zeros_like(self):
        """
        Create a spatial object using the dimensions of another
        """
        temp = spatial(fill_value=self.fill_value)
        #-- assign variables to self
        temp.lon = self.lon.copy()
        temp.lat = self.lat.copy()
        var = ['data','mask','error','time']
        for key in var:
            try:
                val = getattr(self, key)
                setattr(temp, key, np.zeros_like(val))
            except AttributeError:
                pass
        #-- get spacing and dimensions
        temp.update_spacing()
        temp.update_extents()
        temp.update_dimensions()
        temp.replace_masked()
        return temp

    def expand_dims(self):
        """
        Add a singleton dimension to a spatial object if non-existent
        """
        #-- change time dimensions to be iterable
        self.time = np.atleast_1d(self.time)
        #-- output spatial with a third dimension
        if (np.ndim(self.data) == 2):
            self.data = self.data[:,:,None]
            self.mask = self.mask[:,:,None]
        #-- get spacing and dimensions
        self.update_spacing()
        self.update_extents()
        self.update_dimensions()
        self.update_mask()
        return self

    def squeeze(self):
        """
        Remove singleton dimensions from a spatial object
        """
        #-- squeeze singleton dimensions
        self.time = np.squeeze(self.time)
        self.data = np.squeeze(self.data)
        self.mask = np.squeeze(self.mask)
        #-- get spacing and dimensions
        self.update_spacing()
        self.update_extents()
        self.update_dimensions()
        self.update_mask()
        return self

    def index(self, indice, date=True):
        """
        Subset a spatial object to specific index
        Inputs: indice in matrix to subset
        Options: spatial objects contain date information
        """
        #-- output spatial object
        temp = spatial(fill_value=self.fill_value)
        #-- subset output spatial field
        temp.data = self.data[:,:,indice].copy()
        temp.mask = self.mask[:,:,indice].copy()
        #-- subset output spatial error
        try:
            temp.error = self.error[:,:,indice].copy()
        except AttributeError:
            pass
        #-- copy dimensions
        temp.lon = self.lon.copy()
        temp.lat = self.lat.copy()
        #-- subset output dates
        if date:
            temp.time = self.time[indice].copy()
        #-- subset filenames
        if getattr(self, 'filename'):
            temp.filename = self.filename[indice]
        #-- get spacing and dimensions
        temp.update_spacing()
        temp.update_extents()
        temp.update_dimensions()
        return temp

    def offset(self, var):
        """
        Offset a spatial object by a constant
        Inputs: scalar value to which the spatial object will be offset
        """
        temp = self.copy()
        #-- offset by a single constant or a time-variable scalar
        if (np.ndim(var) == 0):
            temp.data = self.data + var
        elif (np.ndim(var) == 1) and (self.ndim == 2):
            n = len(var)
            temp.data = np.zeros((temp.shape[0],temp.shape[1],n))
            temp.mask = np.zeros((temp.shape[0],temp.shape[1],n),dtype=bool)
            for i,v in enumerate(var):
                temp.data[:,:,i] = self.data[:,:] + v
                temp.mask[:,:,i] = np.copy(self.mask[:,:])
        elif (np.ndim(var) == 1) and (self.ndim == 3):
            for i,v in enumerate(var):
                temp.data[:,:,i] = self.data[:,:,i] + v
        elif (np.ndim(var) == 2) and (self.ndim == 2):
            temp.data = self.data + var
        elif (np.ndim(var) == 2) and (self.ndim == 3):
            for i,t in enumerate(self.time):
                temp.data[:,:,i] = self.data[:,:,i] + var
        elif (np.ndim(var) == 3) and (self.ndim == 3):
            for i,t in enumerate(self.time):
                temp.data[:,:,i] = self.data[:,:,i] + var[:,:,i]
        #-- get spacing and dimensions
        temp.update_spacing()
        temp.update_extents()
        temp.update_dimensions()
        #-- update mask
        temp.update_mask()
        return temp

    def scale(self, var):
        """
        Multiply a spatial object by a constant
        Inputs: scalar value to which the spatial object will be multiplied
        """
        temp = self.copy()
        #-- multiply by a single constant or a time-variable scalar
        if (np.ndim(var) == 0):
            temp.data = var*self.data
        elif (np.ndim(var) == 1) and (self.ndim == 2):
            n = len(var)
            temp.data = np.zeros((temp.shape[0],temp.shape[1],n))
            temp.mask = np.zeros((temp.shape[0],temp.shape[1],n),dtype=bool)
            for i,v in enumerate(var):
                temp.data[:,:,i] = v*self.data[:,:]
                temp.mask[:,:,i] = np.copy(self.mask[:,:])
        elif (np.ndim(var) == 1) and (self.ndim == 3):
            for i,v in enumerate(var):
                temp.data[:,:,i] = v*self.data[:,:,i]
        elif (np.ndim(var) == 2) and (self.ndim == 2):
            temp.data = var*self.data
        elif (np.ndim(var) == 2) and (self.ndim == 3):
            for i,t in enumerate(self.time):
                temp.data[:,:,i] = var*self.data[:,:,i]
        elif (np.ndim(var) == 3) and (self.ndim == 3):
            for i,t in enumerate(self.time):
                temp.data[:,:,i] = var[:,:,i]*self.data[:,:,i]
        #-- get spacing and dimensions
        temp.update_spacing()
        temp.update_extents()
        temp.update_dimensions()
        #-- update mask
        temp.update_mask()
        return temp

    def mean(self, apply=False, indices=Ellipsis):
        """
        Compute mean spatial field and remove from data if specified
        Option:
            apply to remove the mean field from the input data
            indices of spatial object to compute mean
        """
        #-- output spatial object
        temp = spatial(nlon=self.shape[0],nlat=self.shape[1],
            fill_value=self.fill_value)
        #-- copy dimensions
        temp.lon = self.lon.copy()
        temp.lat = self.lat.copy()
        #-- create output mean spatial object
        temp.data = np.mean(self.data[:,:,indices],axis=2)
        temp.mask = np.any(self.mask[:,:,indices],axis=2)
        #-- calculate the mean time
        try:
            val = getattr(self, 'time')
            temp.time = np.mean(val[indices])
        except (AttributeError,TypeError):
            pass
        #-- calculate the spatial anomalies by removing the mean field
        if apply:
            for i,t in enumerate(self.time):
                self.data[:,:,i] -= temp.data[:,:]
        #-- get spacing and dimensions
        temp.update_spacing()
        temp.update_extents()
        temp.update_dimensions()
        #-- update mask
        temp.update_mask()
        return temp

    def reverse(self, axis=0):
        """
        Reverse the order of data and dimensions along an axis
        Option: axis to reorder
        """
        #-- output spatial object
        temp = self.copy()
        temp.expand_dims()
        #-- copy dimensions and reverse order
        if (axis == 0):
            temp.lat = temp.lat[::-1].copy()
            temp.data = temp.data[::-1,:,:].copy()
            temp.mask = temp.mask[::-1,:,:].copy()
        elif (axis == 1):
            temp.lon = temp.lon[::-1].copy()
            temp.data = temp.data[:,::-1,:].copy()
            temp.mask = temp.mask[:,::-1,:].copy()
        #-- squeeze output spatial object
        #-- get spacing and dimensions
        #-- update mask
        temp.squeeze()
        return temp

    def transpose(self, axes=None):
        """
        Reverse or permute the axes of a spatial object
        Option: order of the output axes
        """
        #-- output spatial object
        temp = self.copy()
        #-- copy dimensions and reverse order
        temp.data = np.transpose(temp.data, axes=axes)
        temp.mask = np.transpose(temp.mask, axes=axes)
        #-- get spacing and dimensions
        temp.update_spacing()
        temp.update_extents()
        temp.update_dimensions()
        #-- update mask
        temp.update_mask()
        return temp

    def sum(self, power=1):
        """
        Compute summation of spatial field
        Option: apply a power before calculating summation
        """
        #-- output spatial object
        temp = spatial(nlon=self.shape[0],nlat=self.shape[1],
            fill_value=self.fill_value)
        #-- copy dimensions
        temp.lon = self.lon.copy()
        temp.lat = self.lat.copy()
        #-- create output summation spatial object
        temp.data = np.sum(np.power(self.data,power),axis=2)
        temp.mask = np.any(self.mask,axis=2)
        #-- get spacing and dimensions
        temp.update_spacing()
        temp.update_extents()
        temp.update_dimensions()
        #-- update mask
        temp.update_mask()
        return temp

    def power(self, power):
        """
        Raise a spatial object to a power
        Inputs: power to which the spatial object will be raised
        """
        temp = self.copy()
        temp.data = np.power(self.data,power)
        #-- assign ndim and shape attributes
        temp.update_dimensions()
        return temp

    def max(self):
        """
        Compute maximum value of spatial field
        """
        #-- output spatial object
        temp = spatial(nlon=self.shape[0],nlat=self.shape[1],
            fill_value=self.fill_value)
        #-- copy dimensions
        temp.lon = self.lon.copy()
        temp.lat = self.lat.copy()
        #-- create output maximum spatial object
        temp.data = np.max(self.data,axis=2)
        temp.mask = np.any(self.mask,axis=2)
        #-- get spacing and dimensions
        temp.update_spacing()
        temp.update_extents()
        temp.update_dimensions()
        #-- update mask
        temp.update_mask()
        return temp

    def min(self):
        """
        Compute minimum value of spatial field
        """
        #-- output spatial object
        temp = spatial(nlon=self.shape[0],nlat=self.shape[1],
            fill_value=self.fill_value)
        #-- copy dimensions
        temp.lon = self.lon.copy()
        temp.lat = self.lat.copy()
        #-- create output minimum spatial object
        temp.data = np.min(self.data,axis=2)
        temp.mask = np.any(self.mask,axis=2)
        #-- get spacing and dimensions
        temp.update_spacing()
        temp.update_extents()
        temp.update_dimensions()
        #-- update mask
        temp.update_mask()
        return temp

    def replace_invalid(self, fill_value, mask=None):
        """
        Replace the masked values with a new fill_value
        """
        #-- validate current mask
        self.update_mask()
        #-- update the mask if specified
        if mask is not None:
            if (np.shape(mask) == self.shape):
                self.mask |= mask
            elif (np.ndim(mask) == 2) & (self.ndim == 3):
                #-- broadcast mask over third dimension
                temp = np.repeat(mask[:,:,np.newaxis],self.shape[2],axis=2)
                self.mask |= temp
        #-- update the fill value
        self.fill_value = fill_value
        #-- replace invalid values with new fill value
        self.data[self.mask] = self.fill_value
        return self

    def replace_masked(self):
        """
        Replace the masked values with fill_value
        """
        if self.fill_value is not None:
            self.data[self.mask] = self.fill_value
        return self

def hdf5_read(filename, **kwargs):
    """
    Reads spatial data from HDF5 files

    Arguments
    ---------
    filename: HDF5 file to be opened and read

    Keyword arguments
    -----------------
    date: HDF5 file has date information
    compression: HDF5 file is compressed or streaming as bytes
        gzip
        zip
        bytes
    varname: z variable name in HDF5 file
    lonname: longitude variable name in HDF5 file
    latname: latitude variable name in HDF5 file
    timename: time variable name in HDF5 file

    Returns
    -------
    data: z value of dataset
    lon: longitudinal array
    lat: latitudinal array
    time: time value of dataset
    attributes: HDF5 attributes
    """
    #-- set default keyword arguments
    kwargs.setdefault('date',False)
    kwargs.setdefault('compression',None)
    kwargs.setdefault('varname','z')
    kwargs.setdefault('lonname','lon')
    kwargs.setdefault('latname','lat')
    kwargs.setdefault('timename','time')

    #-- Open the HDF5 file for reading
    if (kwargs['compression'] == 'gzip'):
        #-- read gzip compressed file and extract into in-memory file object
        with gzip.open(os.path.expanduser(filename),'r') as f:
            fid = io.BytesIO(f.read())
        #-- set filename of BytesIO object
        fid.filename = os.path.basename(filename)
        #-- rewind to start of file
        fid.seek(0)
        #-- read as in-memory (diskless) HDF5 dataset from BytesIO object
        fileID = h5py.File(fid, 'r')
    elif (kwargs['compression'] == 'zip'):
        #-- read zipped file and extract file into in-memory file object
        fileBasename,_ = os.path.splitext(os.path.basename(filename))
        with zipfile.ZipFile(os.path.expanduser(filename)) as z:
            #-- first try finding a HDF5 file with same base filename
            #-- if none found simply try searching for a HDF5 file
            try:
                f,=[f for f in z.namelist() if re.match(fileBasename,f,re.I)]
            except:
                f,=[f for f in z.namelist() if re.search(r'\.H(DF)?5$',f,re.I)]
            #-- read bytes from zipfile into in-memory BytesIO object
            fid = io.BytesIO(z.read(f))
        #-- set filename of BytesIO object
        fid.filename = os.path.basename(filename)
        #-- rewind to start of file
        fid.seek(0)
        #-- read as in-memory (diskless) HDF5 dataset from BytesIO object
        fileID = h5py.File(fid, 'r')
    elif (kwargs['compression'] == 'bytes'):
        #-- read as in-memory (diskless) HDF5 dataset
        fileID = h5py.File(filename, 'r')
    else:
        #-- read HDF5 dataset
        fileID = h5py.File(os.path.expanduser(filename), 'r')
    #-- allocate python dictionary for output variables
    dinput = {}
    dinput['attributes'] = {}

    #-- Output HDF5 file information
    logging.info(fileID.filename)
    logging.info(list(fileID.keys()))

    #-- mapping between output keys and HDF5 variable names
    keys = ['lon','lat','data']
    h5keys = [kwargs['lonname'],kwargs['latname'],kwargs['varname']]
    if kwargs['date']:
        keys.append('time')
        h5keys.append(kwargs['timename'])

    #-- list of variable attributes
    attributes_list = ['description','units','long_name','calendar',
        'standard_name','_FillValue','missing_value']
    #-- for each variable
    for key,h5key in zip(keys,h5keys):
        #-- Getting the data from each HDF5 variable
        dinput[key] = fileID[h5key][:].copy()
        #-- Getting attributes of included variables
        dinput['attributes'][key] = {}
        for attr in attributes_list:
            try:
                dinput['attributes'][key][attr] = fileID[h5key].attrs[attr]
            except (KeyError, AttributeError):
                pass

    #-- switching data array to lat/lon if lon/lat
    sz = dinput['data'].shape
    if (dinput['data'].ndim == 2) and (len(dinput['lon']) == sz[0]):
        dinput['data'] = dinput['data'].T

    #-- Global attribute description
    try:
        dinput['attributes']['title'] = fileID.attrs['description']
    except (KeyError, AttributeError):
        pass

    #-- Closing the HDF5 file
    fileID.close()
    return dinput

def hdf5_write(data, lon, lat, tim, **kwargs):
    """
    Writes spatial data to HDF5 files

    Arguments
    ---------
    data: z data
    lon: longitude array
    lat: latitude array
    tim: time array

    Keyword arguments
    -----------------
    filename: HDF5 filename
    varname: z variable name in HDF5 file
    lonname: longitude variable name in HDF5 file
    latname: latitude variable name in HDF5 file
    units: z variable units
    longname: z variable description
    fill_value: missing value for z variable
    time_units: time variable units
    time_longname: time variable description
    title: description attribute of dataset
    reference: reference attribute of dataset
    clobber: will overwrite an existing HDF5 file
    date: data has date information
    """
    kwargs.setdefault('filename',None)
    kwargs.setdefault('date',True)
    kwargs.setdefault('clobber',True)
    kwargs.setdefault('varname','z')
    kwargs.setdefault('lonname','lon')
    kwargs.setdefault('latname','lat')
    kwargs.setdefault('timename','time')
    kwargs.setdefault('units',None)
    kwargs.setdefault('longname',None)
    kwargs.setdefault('fill_value',None)
    kwargs.setdefault('time_units',None)
    kwargs.setdefault('time_longname',None)
    kwargs.setdefault('title',None)
    kwargs.setdefault('reference',None)

    #-- setting HDF5 clobber attribute
    clobber = 'w' if kwargs['clobber'] else 'w-'

    #-- opening HDF5 file for writing
    fileID = h5py.File(kwargs['filename'], clobber)

    #-- Dimensions of time parameters
    n_time = len(np.atleast_1d(tim))
    #-- copy kwargs for variable names
    VARNAME = copy.copy(kwargs['VARNAME'])
    LONNAME = copy.copy(kwargs['LONNAME'])
    LATNAME = copy.copy(kwargs['LATNAME'])
    TIMENAME = copy.copy(kwargs['TIMENAME'])
    #-- Defining the HDF5 dataset variables
    h5 = {}
    h5[LONNAME] = fileID.create_dataset(LONNAME, lon.shape, data=lon,
        dtype=lon.dtype, compression='gzip')
    h5[LATNAME] = fileID.create_dataset(LATNAME, lat.shape, data=lat,
        dtype=lat.dtype, compression='gzip')
    h5[VARNAME] = fileID.create_dataset(VARNAME, data.shape, data=data,
        dtype=data.dtype, fillvalue=kwargs['fillvalue'], compression='gzip')
    if kwargs['date']:
        h5[TIMENAME] = fileID.create_dataset(TIMENAME, (n_time,), data=tim,
            dtype=np.float64, compression='gzip')
    #-- add dimensions
    h5[VARNAME].dims[0].label=LATNAME
    h5[VARNAME].dims[0].attach_scale(h5[LATNAME])
    h5[VARNAME].dims[1].label=LONNAME
    #-- if more than 1 date in file
    if (n_time > 1):
        h5[VARNAME].dims[2].label=TIMENAME
        h5[VARNAME].dims[2].attach_scale(h5[TIMENAME])

    #-- filling HDF5 dataset attributes
    #-- Defining attributes for longitude and latitude
    h5[LONNAME].attrs['long_name'] = 'longitude'
    h5[LONNAME].attrs['units'] = 'degrees_east'
    h5[LATNAME].attrs['long_name'] = 'latitude'
    h5[LATNAME].attrs['units'] = 'degrees_north'
    #-- Defining attributes for dataset
    h5[VARNAME].attrs['long_name'] = kwargs['longname']
    h5[VARNAME].attrs['units'] = kwargs['units']
    #-- Dataset contains missing values
    if (kwargs['fill_value'] is not None):
        h5[VARNAME].attrs['_FillValue'] = kwargs['fill_value']
    #-- Defining attributes for date
    if kwargs['date']:
        h5[TIMENAME].attrs['long_name'] = kwargs['time_longname']
        h5[TIMENAME].attrs['units'] = kwargs['time_units']
    #-- description of file
    if kwargs['title']:
        fileID.attrs['description'] = kwargs['title']
    #-- reference of file
    if kwargs['reference']:
        fileID.attrs['reference'] = kwargs['reference']
    #-- date created
    date_created = time.strftime('%Y-%m-%d',time.localtime())
    fileID.attrs['date_created'] = date_created

    #-- Output HDF5 structure information
    logging.info(kwargs['filename'])
    logging.info(list(fileID.keys()))

    #-- Closing the HDF5 file
    fileID.close()

def ncdf_read(filename, **kwargs):
    """
    Reads spatial data from COARDS-compliant netCDF4 files

    Arguments
    ---------
    filename: netCDF4 file to be opened and read

    Keyword arguments
    -----------------
    date: netCDF4 file has date information
    varname: z variable name in netCDF4 file
    lonname: longitude variable name in netCDF4 file
    latname: latitude variable name in netCDF4 file
    timename: time variable name in netCDF4 file
    compression: netCDF4 file is compressed or streaming as bytes
        gzip
        zip
        bytes

    Returns
    -------
    data: z value of dataset
    lon: longitudinal array
    lat: latitudinal array
    time: time value of dataset
    attributes: netCDF4 attributes
    """
    #-- set default keyword arguments
    kwargs.setdefault('date',False)
    kwargs.setdefault('compression',None)
    kwargs.setdefault('varname','z')
    kwargs.setdefault('lonname','lon')
    kwargs.setdefault('latname','lat')
    kwargs.setdefault('timename','time')

    #-- Open the NetCDF4 file for reading
    if (kwargs['compression'] == 'gzip'):
        #-- read as in-memory (diskless) netCDF4 dataset
        with gzip.open(os.path.expanduser(filename),'r') as f:
            fileID = netCDF4.Dataset(os.path.basename(filename),memory=f.read())
    elif (kwargs['compression'] == 'zip'):
        #-- read zipped file and extract file into in-memory file object
        fileBasename,_ = os.path.splitext(os.path.basename(filename))
        with zipfile.ZipFile(os.path.expanduser(filename)) as z:
            #-- first try finding a netCDF4 file with same base filename
            #-- if none found simply try searching for a netCDF4 file
            try:
                f,=[f for f in z.namelist() if re.match(fileBasename,f,re.I)]
            except:
                f,=[f for f in z.namelist() if re.search(r'\.nc(4)?$',f)]
            #-- read bytes from zipfile as in-memory (diskless) netCDF4 dataset
            fileID = netCDF4.Dataset(uuid.uuid4().hex, memory=z.read(f))
    elif (kwargs['compression'] == 'bytes'):
        #-- read as in-memory (diskless) netCDF4 dataset
        fileID = netCDF4.Dataset(uuid.uuid4().hex, memory=filename.read())
    else:
        #-- read netCDF4 dataset
        fileID = netCDF4.Dataset(os.path.expanduser(filename), 'r')
    #-- create python dictionary for output variables
    dinput = {}
    dinput['attributes'] = {}

    #-- Output NetCDF file information
    logging.info(fileID.filepath())
    logging.info(list(fileID.variables.keys()))

    #-- mapping between output keys and netCDF4 variable names
    keys = ['lon','lat','data']
    nckeys = [kwargs['lonname'],kwargs['latname'],kwargs['varname']]
    if kwargs['date']:
        keys.append('time')
        nckeys.append(kwargs['timename'])
    #-- list of variable attributes
    attributes_list = ['description','units','long_name','calendar',
        'standard_name','_FillValue','missing_value']
    #-- for each variable
    for key,nckey in zip(keys,nckeys):
        #-- Getting the data from each NetCDF variable
        dinput[key] = fileID.variables[nckey][:].data
        #-- Getting attributes of included variables
        dinput['attributes'][key] = {}
        for attr in attributes_list:
            #-- try getting the attribute
            try:
                dinput['attributes'][key][attr] = \
                    fileID.variables[nckey].getncattr(attr)
            except (KeyError,ValueError,AttributeError):
                pass

    #-- switching data array to lat/lon if lon/lat
    sz = dinput['data'].shape
    if (dinput['data'].ndim == 2) and (len(dinput['lon']) == sz[0]):
        dinput['data'] = dinput['data'].T

    #-- Global attribute (title of dataset)
    try:
        title, = [st for st in dir(fileID) if re.match(r'TITLE',st,re.I)]
        dinput['attributes']['title'] = fileID.getncattr(title)
    except (ValueError, KeyError, AttributeError):
        pass
    #-- Closing the NetCDF file
    fileID.close()
    #-- return the output variable
    return dinput

def ncdf_write(data, lon, lat, tim, **kwargs):
    """
    Writes spatial data to COARDS-compliant netCDF4 files

    Arguments
    ---------
    data: z data
    lon: longitude array
    lat: latitude array
    tim: time array

    Keyword arguments
    -----------------
    filename: netCDF4 filename
    varname: z variable name in netCDF4 file
    lonname: longitude variable name in netCDF4 file
    latname: latitude variable name in netCDF4 file
    units: z variable units
    longname: z variable description
    fill_value: missing value for z variable
    time_units: time variable units
    time_longname: time variable description
    title: title attribute of dataset
    reference: reference attribute of dataset
    clobber: will overwrite an existing netCDF4 file
    date: data has date information
    """
    kwargs.setdefault('filename',None)
    kwargs.setdefault('date',True)
    kwargs.setdefault('clobber',True)
    kwargs.setdefault('varname','z')
    kwargs.setdefault('lonname','lon')
    kwargs.setdefault('latname','lat')
    kwargs.setdefault('timename','time')
    kwargs.setdefault('units',None)
    kwargs.setdefault('longname',None)
    kwargs.setdefault('fill_value',None)
    kwargs.setdefault('time_units',None)
    kwargs.setdefault('time_longname',None)
    kwargs.setdefault('title',None)
    kwargs.setdefault('reference',None)

    #-- setting NetCDF clobber attribute
    clobber = 'w' if kwargs['clobber'] else 'a'
    #-- opening NetCDF file for writing
    #-- Create the NetCDF file
    fileID = netCDF4.Dataset(kwargs['filename'],
        clobber, format="NETCDF4")

    # copy kwargs for variable names
    VARNAME = copy.copy(kwargs['VARNAME'])
    LONNAME = copy.copy(kwargs['LONNAME'])
    LATNAME = copy.copy(kwargs['LATNAME'])
    TIMENAME = copy.copy(kwargs['TIMENAME'])
    #-- Defining the NetCDF dimensions
    n_time = len(np.atleast_1d(tim))
    fileID.createDimension(LONNAME, len(lon))
    fileID.createDimension(LATNAME, len(lat))
    fileID.createDimension(TIMENAME, n_time)

    #-- defining the NetCDF variables
    nc = {}
    #-- lat and lon
    nc[LONNAME] = fileID.createVariable(LONNAME, lon.dtype, (LONNAME,))
    nc[LATNAME] = fileID.createVariable(LATNAME, lat.dtype, (LATNAME,))
    #-- spatial data
    if (n_time > 1):
        nc[VARNAME] = fileID.createVariable(VARNAME, data.dtype,
            (LATNAME,LONNAME,TIMENAME,), fill_value=kwargs['fill_value'],
            zlib=True)
    else:
        nc[VARNAME] = fileID.createVariable(VARNAME, data.dtype,
            (LATNAME,LONNAME,), fill_value=kwargs['fill_value'],
            zlib=True)
    #-- time
    if kwargs['date']:
        nc[TIMENAME] = fileID.createVariable(TIMENAME, 'f8', (TIMENAME,))

    #-- filling NetCDF variables
    nc[LONNAME][:] = lon
    nc[LATNAME][:] = lat
    nc[VARNAME][:,:] = data
    if kwargs['date']:
        nc[TIMENAME][:] = tim

    #-- Defining attributes for longitude and latitude
    nc[LONNAME].long_name = 'longitude'
    nc[LONNAME].units = 'degrees_east'
    nc[LATNAME].long_name = 'latitude'
    nc[LATNAME].units = 'degrees_north'
    #-- Defining attributes for dataset
    nc[VARNAME].long_name = kwargs['longname']
    nc[VARNAME].units = kwargs['units']
    #-- Defining attributes for date if applicable
    if kwargs['date']:
        nc[TIMENAME].long_name = kwargs['time_longname']
        nc[TIMENAME].units = kwargs['time_units']
    #-- global variables of NetCDF file
    if kwargs['title']:
        fileID.title = kwargs['title']
    if  kwargs['reference']:
        fileID.reference = kwargs['reference']
    #-- date created
    fileID.date_created = time.strftime('%Y-%m-%d',time.localtime())

    #-- Output NetCDF structure information
    logging.info(kwargs['filename'])
    logging.info(list(fileID.variables.keys()))

    #-- Closing the NetCDF file
    fileID.close()
