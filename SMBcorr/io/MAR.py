#!/usr/bin/env python
"""
MAR.py
Written by Tyler Sutterley (04/2026)
Reads Modèle Atmosphérique Régional (MAR) data products provided by
Lèige Université (Belgium)

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
from SMBcorr.utilities import import_dependency, dependency_available

# attempt imports
dask = import_dependency("dask")
dask_available = dependency_available("dask")

# mapping of netCDF4 variable names to internal variable names
_variable_mapping = {
    "SMB": "SMB",
    "ZN4": "zfirn",
    "ZN5": "zmelt",
    "ZN6": "zsurf",
}

# PROJ4 parameters for MAR model projections
proj4_params = dict()
# Greenland: Polar Stereographic (Oblique)
# Earth Radius: 6371229 m
# True Latitude: 0
# Center Longitude: -40
# Center Latitude: 70.5
# Coordinate Axis Units: km
proj4_params["Greenland"] = {
    "proj": "sterea",
    "lat_0": 70.5,
    "lat_ts": 0,
    "lon_0": -40,
    "k": 1,
    "x_0": 0,
    "y_0": 0,
    "R": 6371229,
    "units": "km",
    "no_defs": None,
    "type": "crs",
}
# Antarctica: WGS84 / Polar Stereographic
# Modification of EPSG:3031
# Coordinate Axis Units: km
proj4_params["Antarctica"] = {
    "proj": "stere",
    "lat_0": -90,
    "lat_ts": -71,
    "lon_0": 0,
    "datum": "WGS84",
    "units": "km",
    "no_defs": None,
    "type": "crs",
}


def open_mfdataset(
    filenames: list[str] | list[pathlib.Path],
    parallel: bool = False,
    **kwargs,
):
    """
    Open multiple files containing MAR model data

    Parameters
    ----------
    filenames: list of str or pathlib.Path
        Path(s) to file(s) containing MAR data
    parallel: bool, default False
        Open files in parallel using ``dask.delayed``
    kwargs: dict
        Additional keyword arguments for opening MAR files
    """
    # merge multiple granules
    if parallel and dask_available:
        opener = dask.delayed(open_dataset)
    else:
        opener = open_dataset
    # verify that filename is iterable
    if isinstance(filenames, str):
        filenames = [filenames]
    # read each file as xarray dataset and append to list
    d = [opener(f, **kwargs) for f in filenames]
    # read datasets as dask arrays
    if parallel and dask_available:
        (d,) = dask.compute(d)
    # concatenate a single variable over time
    ds = xr.concat(d, dim="time", compat="override", join="override")
    # return xarray dataset
    return ds


def open_dataset(
    filename: str | pathlib.Path,
    variable: str | list[str],
    surface_type: int | list[int] = 4,
    chunks: str | None = "auto",
    **kwargs,
):
    """
    Open a netCDF4 file containing MAR data

    Parameters
    ----------
    filename: str or pathlib.Path
        Path to netCDF4 file containing MAR data
    variable: str or list
        netCDF4 variable name(s) to extract
    surface_type: int or list, default 4
        Surface type(s) to extract (1 = ocean, 4 = land)
    chunks: str or None, default 'auto'
        Chunk size for xarray dataset
    compressed: bool, default False
        If True, read gzipped netCDF4 file
    """
    # set default keyword arguments
    kwargs.setdefault("compressed", False)
    # read the netCDF4-format file
    logging.debug(filename)
    if kwargs["compressed"]:
        # read gzipped netCDF4 file
        f = gzip.open(filename, "rb")
        tmp = xr.open_dataset(f, mask_and_scale=True, chunks=chunks)
    else:
        tmp = xr.open_dataset(filename, mask_and_scale=True, chunks=chunks)
    # get coordinate names
    mapping = {}
    for v in tmp.coords:
        if re.match(r"X", tmp[v].attrs.get("axis", ""), re.I):
            mapping[v] = "x"
        elif re.match(r"Y", tmp[v].attrs.get("axis", ""), re.I):
            mapping[v] = "y"
        elif re.match(r"T", tmp[v].attrs.get("axis", ""), re.I):
            mapping[v] = "time"
        elif re.match(r"sector$", v, re.I):
            mapping[v] = "sector"
    # rename to standardized coordinate names
    tmp = tmp.rename(mapping)
    # attempt to get MAR region from latitude variable
    region = "Greenland" if tmp["LAT"].min() > 0 else "Antarctica"
    # output dataset
    ds = xr.Dataset()
    # extract x, y and time coordinate arrays
    ds["x"] = tmp["x"].copy()
    ds["y"] = tmp["y"].copy()
    ds["time"] = tmp["time"].copy()
    # check if surface_types is a string
    if isinstance(surface_type, int):
        surface_type = [surface_type]
    # extract surface type and ice fraction variables
    valid_surface = tmp["SRF"].isin(surface_type)
    # combine sectors (if applicable) using fractional area
    if "FRA" in tmp and "sector" in tmp["FRA"].dims:
        fa = tmp["FRA"].isel(sector=0) / 100.0
    elif "FRV" in tmp and "sector" in tmp["FRV"].dims:
        fa = tmp["FRV"].isel(sector=0) / 100.0
    # check if variable is a string
    if isinstance(variable, str):
        variable = [variable]
    # loop over variables to extract
    for var in variable:
        # read dataset and remove singleton dimensions
        imap = _variable_mapping.get(var, var)
        v = tmp[var].squeeze()
        # combine sectors (if applicable) using fractional area
        if v.ndim == 4 and "sector" in v.dims:
            ds[imap] = fa * v.isel(sector=0) + (1 - fa) * v.isel(sector=1)
        else:
            ds[imap] = v.copy()
        # reduce to surface type of interest
        ds[imap] = ds[imap].where(valid_surface, np.nan, drop=False)
        # add attributes for variable
        ds[imap].attrs["group"] = var
        ds[imap].attrs["units"] = tmp[var].attrs.get("units", "")
    # drop coordinates that are not in the dimensions
    drop_coords = [c for c in ds.coords if c not in ds.dims]
    ds = ds.drop_vars(drop_coords)
    # verify that chunks are unified (if specified)
    if chunks is not None:
        ds = ds.unify_chunks()
    # add attributes to dataset
    ds.attrs["crs"] = proj4_params[region]
    # return the dataset
    return ds
