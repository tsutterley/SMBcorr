#!/usr/bin/env python
"""
GSFCfdm.py
Written by Tyler Sutterley (04/2026)
Reads GSFC-fdm data products provided by Brooke Medley (NASA GSFC)

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Written 04/2026
"""

from __future__ import print_function

import re
import gzip
import logging
import pathlib
import xarray as xr
import numpy as np

# mapping of netCDF4 variable names to internal variable names
_variable_mapping = {
    "SMB": "SMB",
    "Ra": "rainfall",
    "Ru": "runoff",
    "Sn-Ev": "snowfall",  # technically net as it is minus sublimation
    "Me": "snowmelt",
    "FAC": "zfirn",
    "h_a": "zsurf",
    "Me_a": "zmelt",
    "SMB_a": "zsmb",
    "cum_smb_anomaly": "zsmb",  # old variable name for SMB_a
    "height": "zsurf",  # old variable name for h_a
}

# PROJ4 parameters for GSFC-fdm model projections
# ais: WGS84 / Polar Stereographic South
# gris: WGS84 / NSIDC Sea Ice Polar Stereographic North
proj4_params = dict(ais=3031, gris=3413)


def open_dataset(
    filename: str | pathlib.Path,
    variable: str | list[str],
    chunks: str | None = "auto",
    **kwargs,
):
    """
    Open a netCDF4 file containing GSFC-fdm data

    Parameters
    ----------
    filename: str or pathlib.Path
        Path to netCDF4 file containing GSFC-fdm data
    variable: str or list
        netCDF4 variable name(s) to extract
    chunks: str or None, default 'auto'
        Chunk size for xarray dataset
    compressed: bool, default False
        If True, read gzipped netCDF4 file
    """
    # set default keyword arguments
    kwargs.setdefault("compressed", False)
    # regular expression pattern for extracting parameters
    (region,) = re.findall(r"(ais|gris)", pathlib.Path(filename).stem)
    # read the netCDF4-format file
    logging.debug(filename)
    if kwargs["compressed"]:
        # read gzipped netCDF4 file
        f = gzip.open(filename, "rb")
        tmp = xr.open_dataset(f, mask_and_scale=True, chunks=chunks)
    else:
        tmp = xr.open_dataset(filename, mask_and_scale=True, chunks=chunks)
    # output dataset
    ds = xr.Dataset()
    # decode time variables
    ds["time"] = decode_times(tmp["time"])
    # extract x and y coordinate arrays
    if tmp["x"].ndim == 2:
        ds["x"] = tmp["x"].isel(y=0).squeeze()
        ds["y"] = tmp["y"].isel(x=0).squeeze()
    else:
        ds["x"] = tmp["x"].copy()
        ds["y"] = tmp["y"].copy()
    # check if variable is a string
    if isinstance(variable, str):
        variable = [variable]
    # loop over variables to extract
    for var in variable:
        # read dataset and remove singleton dimensions
        imap = _variable_mapping.get(var, var)
        # convert dimension order from matlab (time, x, y)
        ds[imap] = tmp[var].squeeze().transpose("time", "y", "x")
        # replace points where all values are zero with NaNs
        ds[imap] = ds[imap].where(
            ds[imap].sum(dim="time", skipna=False) != 0, np.nan, drop=False
        )
        # add attributes for variable
        ds[imap].attrs["group"] = var
        ds[imap].attrs["units"] = tmp[var].attrs.get("units", "")
    # drop coordinates that are not in the dimensions
    drop_coords = [c for c in ds.coords if c not in ds.dims]
    ds = ds.drop_vars(drop_coords)
    # add attributes to dataset
    ds.attrs["crs"] = proj4_params[region]
    # return the dataset
    return ds


def decode_times(variable: xr.DataArray) -> xr.DataArray:
    """Decode time variable to be monotonic and in datetime format"""
    try:
        # unit long_name attribute for variable
        long_name = variable.attrs.get("long_name")
        # get time step and unit from long_name attribute
        (step, unit) = re.findall(r"(\d+)(.*?)resolution", long_name).pop()
        # convert time step to numpy timedelta64 format (hours)
        time_step = np.timedelta64(24 * int(step), "h")
    except Exception as exc:
        # if cannot be parsed: return variable without modification
        return variable
    # convert time variables to be monotonically increasing
    start_time = np.datetime64(f"{variable.values[0]:0.0f}", "h")
    variable = start_time + time_step * (0.5 + np.arange(variable.size))
    return variable
