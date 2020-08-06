#!/usr/bin/env python
u"""
append_netCDF4.py
Written by Tyler Sutterley (06/2020)
Copy variables from one netCDF4 file to another existing netCDF4 file

CALLING SEQUENCE:
    python append_netCDF4.py <path_to_file_to_be_read> \
        <path_to_file_to_be_appended> <variables>

PYTHON DEPENDENCIES:
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/netCDF4/index.html
UPDATE HISTORY:
    Written 06/2020
"""

import sys
import os
import netCDF4

def append_netCDF4():
    input_file = os.path.expanduser(sys.argv[1])
    append_file = os.path.expanduser(sys.argv[2])
    # open the netCDF files for reading and appending
    with netCDF4.Dataset(input_file, 'r') as src, netCDF4.Dataset(append_file, 'a') as dst:
        # copy dimensions
        for name, dimension in src.dimensions.items():
            if name not in dst.dimensions.keys():
                dst.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
        # for each variable to copy
        for name in sys.argv[3].split(','):
            # check if presently existing in destination dataset
            if name not in dst.variables.keys():
                # copy variable from src
                nc = dst.createVariable(name, src[name].datatype, src[name].dimensions)
                dst[name][:] = src[name][:]
                # copy variable attributes all at once via dictionary
                dst[name].setncatts(src[name].__dict__)
            else:
                dst[name][:] = src[name][:]

# run program
if __name__ == '__main__':
    append_netCDF4()
