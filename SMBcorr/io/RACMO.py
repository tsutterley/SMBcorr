#!/usr/bin/env python
"""
RACMO.py
Written by Tyler Sutterley (04/2026)
Reads Regional Atmospheric and Climate MOdel (RACMO) data products
provided by IMAU (Utrecht University)

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
import timescale.time
from SMBcorr.utilities import import_dependency, dependency_available

# attempt imports
dask = import_dependency("dask")
dask_available = dependency_available("dask")

# mapping of netCDF4 variable names to internal variable names
_variable_mapping = {
    "SMB": "SMB",
    "smb": "SMB",
    "SMB_rec": "SMB",  # downscaled SMB
    "smb_rec": "SMB",  # downscaled SMB
    "precipcorr": "precip",  # downscaled precipitation
    "refreezecorr": "refreeze",  # downscaled meltwater refreeze
    "runoffcorr": "runoff",  # downscaled runoff
    "snowmeltcorr": "snowmelt",  # downscaled snowmelt
    "FirnAir": "zfirn",
    "hgtsrf": "zsurf",
    "zs": "zsurf",
}

# PROJ4 parameters for RACMO model projections
proj4_params = {"rotated_pole": {}, "downscaled": {}}
# RACMO (default) model projections: Rotated Latitude-Longitude
proj4_params["rotated_pole"]["ANT"] = (
    "-m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=-180.0 +lon_0=10.0"
)
proj4_params["rotated_pole"]["ASE"] = (
    "-m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=167.0 +lon_0=53.0"
)
proj4_params["rotated_pole"]["PEN"] = (
    "-m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=-180.0 +lon_0=30.0"
)
proj4_params["rotated_pole"]["GRN"] = (
    "-m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=18.0 +lon_0=-37.5"
)
# RACMO downscaled model projections: Polar Stereographic
# ANT: WGS84 / Polar Stereographic South
# GRN: WGS84 / NSIDC Sea Ice Polar Stereographic North
proj4_params["downscaled"]["ANT"] = 3031
proj4_params["downscaled"]["GRN"] = 3413


def open_mfdataset(
    filenames: list[str] | list[pathlib.Path],
    parallel: bool = False,
    how: str = "merge",
    **kwargs,
):
    """
    Open multiple files containing RACMO model data

    Parameters
    ----------
    filenames: list of str or pathlib.Path
        Path(s) to file(s) containing RACMO data
    parallel: bool, default False
        Open files in parallel using ``dask.delayed``
    how: str, default 'merge'
        How to merge the datasets

        - ``'merge'``: merge variables from multiple files
        - ``'concat'``: concatenate a single variable over time
    kwargs: dict
        Additional keyword arguments for opening RACMO files
    """
    # set default keyword arguments
    kwargs.setdefault("reference", None)
    kwargs.setdefault("range", None)
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
    # merge or concatenate datasets
    if how == "merge":
        # merge variables from multiple files
        ds = xr.merge(d, compat="override", join="override")
    elif how == "concat":
        # concatenate a single variable over time
        ds = xr.concat(d, dim="time", compat="override", join="override")
    else:
        # raise an error for unknown or invalid merge methods
        raise ValueError(f"Invalid merge method: {how}")
    # return xarray dataset
    return ds


def open_dataset(
    filename: str | pathlib.Path,
    format: str = "netcdf",
    **kwargs,
):
    """
    Open a file with RACMO model data

    Parameters
    ----------
    filename: str or pathlib.Path
        Path to file containing RACMO data
    format: str
        Format of RACMO data

        - `'ascii'`: ascii-formatted model output
        - `'downscaled'`: downscaled model output in netCDF4 format
        - `'netcdf'`: daily or monthly model outputs in netCDF4 format
    kwargs: dict
        Additional keyword arguments for opening RACMO files
    """
    if format == "ascii":
        return open_ascii_dataset(filename, **kwargs)
    elif format == "netcdf":
        return open_netcdf_dataset(filename, **kwargs)
    elif format == "downscaled":
        return open_downscaled_dataset(filename, **kwargs)
    else:
        raise ValueError(f"Invalid format: {format}")


def open_ascii_dataset(
    filename: str | pathlib.Path,
    variable: str = "SMB",
    chunks: str | None = "auto",
    **kwargs,
):
    """
    Open an ASCII file with RACMO model data

    Parameters
    ----------
    filename: str or pathlib.Path
        Path to ASCII file containing RACMO data
    variable: str, default 'SMB'
        Variable name in the ASCII file to extract
    chunks: str or None, default 'auto'
        Chunk size for xarray dataset
    """
    # read the tab-delimited text file
    logging.debug(filename)
    tmp = np.loadtxt(filename)
    # latitude and longitude arrays
    lon = tmp[:, 0].copy()
    lat = tmp[:, 1].copy()
    # flattened data matrix
    flattened = tmp[:, 2:]
    # number of time steps in the dataset
    _, nt = flattened.shape
    # grid dimensions for each RACMO region
    if np.max(lat) > 0:
        # XGRN11 grid dimensions
        ny, nx = 312, 306
        # starting epoch and conversion from months to seconds
        epoch, to_secs = [1960, 1, 15], 365.25 * 86400.0 / 12.0
    elif np.min(lat) < 0:
        # XANT27 grid dimensions
        ny, nx = 240, 262
        # starting epoch and conversion from months to seconds
        epoch, to_secs = [1979, 1, 15], 365.25 * 86400.0 / 12.0
    # data dictionary
    var = dict(dims=("y", "x", "time"), coords={}, data_vars={})
    # store the latitude coordinates
    var["data_vars"]["lat"] = {}
    var["data_vars"]["lat"]["dims"] = ("y", "x")
    var["data_vars"]["lat"]["data"] = lat.reshape(ny, nx)
    # store the longitude coordinates
    var["data_vars"]["lon"] = {}
    var["data_vars"]["lon"]["dims"] = ("y", "x")
    var["data_vars"]["lon"]["data"] = lon.reshape(ny, nx)
    # store the data variables
    var["data_vars"][variable] = {}
    var["data_vars"][variable]["dims"] = ("y", "x", "time")
    var["data_vars"][variable]["data"] = flattened.reshape(ny, nx, nt)
    # convert to xarray Dataset from the data dictionary
    ds = xr.Dataset.from_dict(var)
    # replace zeros with NaNs
    ds[variable] = ds[variable].where(
        ds[variable].sum(dim="time", skipna=False) != 0, np.nan, drop=False
    )
    # convert delta times from seconds since epoch to datetime objects
    ts = timescale.from_deltatime(np.arange(nt) * to_secs, epoch=epoch)
    ds["time"] = ts.to_datetime()
    # coerce to specified chunks
    if chunks is not None:
        ds = ds.chunk(chunks)
    # add attributes to dataset
    ds[variable].attrs["units"] = "mm w.e."
    # return the dataset
    return ds


def open_netcdf_dataset(
    filename: str | pathlib.Path,
    variable: str | list[str],
    chunks: str | None = "auto",
    **kwargs,
):
    """
    Open a netCDF4 file with RACMO model data

    Parameters
    ----------
    filename: str or pathlib.Path
        Path to netCDF4 file containing RACMO data
    variable: str or list
        netCDF4 variable name(s) to extract
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
    # regular expression pattern for extracting parameters
    pattern = r"[F|X]?(GRN|ANT|ASE|PEN)[\d+]"
    m = re.search(pattern, pathlib.Path(filename).stem)
    if m:
        region = m.group(1)
    elif "LAT" in tmp.data_vars:
        # attempt to get RACMO region from latitude variable
        latname = "LAT" if "LAT" in tmp.data_vars else "lat"
        region = "GRN" if tmp[latname].min() > 0 else "ANT"
    # get coordinate names
    mapping = {}
    for v in tmp.coords:
        if re.match(r"X", tmp[v].attrs.get("axis", ""), re.I):
            mapping[v] = "x"
        elif re.match(r"Y", tmp[v].attrs.get("axis", ""), re.I):
            mapping[v] = "y"
        elif re.match(r"T", tmp[v].attrs.get("axis", ""), re.I):
            mapping[v] = "time"
    # verify mapping
    if "x" not in mapping and "rlon" in tmp.coords:
        mapping["rlon"] = "x"
    if "y" not in mapping and "rlat" in tmp.coords:
        mapping["rlat"] = "y"
    if "time" not in mapping and "time" in tmp.coords:
        mapping["time"] = "time"
    # rename to standardized coordinate names
    tmp = tmp.rename(mapping)
    # output dataset
    ds = xr.Dataset()
    # extract x, y and time coordinate arrays
    ds["x"] = tmp["x"].copy()
    ds["y"] = tmp["y"].copy()
    ds["time"] = tmp["time"].copy()
    # check if variable is a string
    if isinstance(variable, str):
        variable = [variable]
    # loop over variables to extract
    for var in variable:
        # mapping between netCDF4 variable names and internal variable names
        imap = _variable_mapping.get(var, var)
        # check if variable is in dataset
        if var not in tmp.data_vars:
            logging.info(f"Variable {var} not found in dataset")
            continue
        # read dataset and remove singleton dimensions
        ds[imap] = tmp[var].squeeze()
        # replace zeros with NaNs
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
    if "rotated_pole" in tmp.data_vars:
        # extract rotated pole parameters from dataset attributes
        ds.attrs["crs"] = tmp["rotated_pole"].proj4_params
    else:
        # add default rotated pole parameters for region
        ds.attrs["crs"] = proj4_params["rotated_pole"][region]
    # raise a value error if there are no variables in the dataset
    if len(ds.data_vars) == 0:
        raise ValueError(f"No variables found in dataset: {filename}")
    # return the dataset
    return ds


def open_downscaled_dataset(
    filename: str | pathlib.Path,
    variable: str | list[str],
    chunks: str | None = "auto",
    **kwargs,
):
    """
    Open a netCDF4 file with downscaled RACMO model data

    Parameters
    ----------
    filename: str or pathlib.Path
        Path to netCDF4 file containing RACMO data
    variable: str or list
        netCDF4 variable name(s) to extract
    chunks: str or None, default 'auto'
        Chunk size for xarray dataset
    compressed: bool, default False
        If True, read gzipped netCDF4 file
    """
    # set default keyword arguments
    kwargs.setdefault("compressed", False)
    # read the netCDF4-format file
    if kwargs["compressed"]:
        # read gzipped netCDF4 file
        f = gzip.open(filename, "rb")
        tmp = xr.open_dataset(
            f, mask_and_scale=True, decode_times=False, chunks=chunks
        )
    else:
        tmp = xr.open_dataset(
            filename, mask_and_scale=True, decode_times=False, chunks=chunks
        )
    # regular expression pattern for extracting parameters
    # default to Greenland if region cannot be determined from filename
    m = re.search(r"[F|X]?(GRN|ANT)[\d+]", pathlib.Path(filename).stem)
    region = m.group(1) if m else "GRN"
    # get coordinate names
    mapping = {}
    for v in tmp.coords:
        if re.match(r"X", tmp[v].attrs.get("axis", ""), re.I):
            mapping[v] = "x"
        elif re.match(r"Y", tmp[v].attrs.get("axis", ""), re.I):
            mapping[v] = "y"
        elif re.match(r"T", tmp[v].attrs.get("axis", ""), re.I):
            mapping[v] = "time"
    # rename to standardized coordinate names
    tmp = tmp.rename(mapping)
    # output dataset
    ds = xr.Dataset()
    # cell origins in the downscaled product are on the bottom right
    dx = np.abs(tmp["x"][1] - tmp["x"][0])
    dy = np.abs(tmp["y"][1] - tmp["y"][0])
    # convert x and y arrays to cell centers
    ds["x"] = tmp["x"].copy() - dx / 2.0
    ds["y"] = tmp["y"].copy() - dy / 2.0
    # parse dates from time variable
    epoch, to_secs = timescale.time.parse_date_string(tmp["time"].units)
    # if monthly: convert to seconds using average month lengths
    if re.search(r"month", tmp["time"].units, re.I):
        to_secs = 365.25 * 86400.0 / 12.0
    # convert delta times from seconds since epoch to datetime objects
    ts = timescale.from_deltatime(tmp["time"] * to_secs, epoch=epoch)
    ds["time"] = ts.to_datetime()
    # get the ice mask from the downscaled dataset
    if "icemask" in tmp.data_vars:
        tmp = tmp.where(tmp["icemask"] == 1, drop=False)
    elif "Promicemask" in tmp.data_vars:
        tmp = tmp.where(
            (tmp["Promicemask"] >= 1) & (tmp["Promicemask"] <= 3),
            drop=False,
        )
    # check if variable is a string
    if isinstance(variable, str):
        variable = [variable]
    # loop over variables to extract
    for var in variable:
        # mapping between netCDF4 variable names and internal variable names
        imap = _variable_mapping.get(var, var)
        # check if variable is in dataset
        if var not in tmp.data_vars:
            logging.info(f"Variable {var} not found in dataset")
            continue
        # read dataset and remove singleton dimensions
        ds[imap] = ("time", "y", "x"), tmp[var].squeeze().data
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
    m = re.search(r"EPSG[:]?(\d+)", tmp.attrs.get("grid", ""), re.I)
    if m:
        # extract crs parameters from dataset attributes
        ds.attrs["crs"] = int(m.group(1))
    else:
        # add default downscaled crs for region
        ds.attrs["crs"] = proj4_params["downscaled"][region]
    # raise a value error if there are no variables in the dataset
    if len(ds.data_vars) == 0:
        raise ValueError(f"No variables found in dataset: {filename}")
    # return the dataset
    return ds
