#!/usr/bin/env python
"""
dataset.py
Written by Tyler Sutterley (04/2026)
An xarray.Dataset extension for SMB and firn model data

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    pint: Python package to define, operate and manipulate physical quantities
        https://pypi.org/project/Pint/
        https://pint.readthedocs.io/en/stable
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
        https://pyproj4.github.io/pyproj/
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Written 04/2026
"""

import re
import pint
import pyproj
import warnings
import numpy as np
import xarray as xr

# suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

__all__ = [
    "Dataset",
    "DataArray",
    "register_dataset_subaccessor",
    "register_dataarray_subaccessor",
    "_transform",
    "_coords",
]

# pint unit registry
__ureg__ = pint.UnitRegistry()
# add water and ice equivalents
__ureg__.define("we = 1.0 * g / cm^3")
__ureg__.define("ie = 0.917 * g / cm^3")
__ureg__.define("@alias we = water = water_equivalent")
__ureg__.define("@alias ie = ice = ice_equivalent")
# air equivalent for FAC
__ureg__.define("air = 1.0")
# default units for SMB and firn outputs
_default_units = {
    "mass density": "cm we",
}


@xr.register_dataset_accessor("smb")
class Dataset:
    """Accessor for extending an ``xarray.Dataset`` for SMB and firn data"""

    def __init__(self, ds):
        # initialize Dataset
        self._ds = ds

    def assign_coords(
        self,
        x: np.ndarray,
        y: np.ndarray,
        crs: str | int | dict = 4326,
        **kwargs,
    ):
        """
        Assign new coordinates to the ``Dataset``

        Parameters
        ----------
        x: np.ndarray
            Updated x-coordinates
        y: np.ndarray
            Updated y-coordinates
        crs: str, int, or dict, default 4326 (WGS84 Latitude/Longitude)
            Coordinate reference system of coordinates
        kwargs: keyword arguments
            Keyword arguments for ``xarray.Dataset.assign_coords``

        Returns
        -------
        ds: xarray.Dataset
            ``Dataset`` with updated coordinates
        """
        # assign new coordinates to dataset
        ds = self._ds.assign_coords(dict(x=x, y=y), **kwargs)
        ds.attrs["crs"] = crs
        # return the dataset
        return ds

    def coords_as(
        self,
        x: np.ndarray,
        y: np.ndarray,
        crs: str | int | dict = 4326,
        **kwargs,
    ):
        """
        Transform coordinates into ``DataArrays`` in the ``Dataset``
        coordinate reference system

        Parameters
        ----------
        x: np.ndarray
            Input x-coordinates
        y: np.ndarray
            Input y-coordinates
        crs: str, int, or dict, default 4326 (WGS84 Latitude/Longitude)
            Coordinate reference system of input coordinates

        Returns
        -------
        X: xarray.DataArray
            Transformed x-coordinates
        Y: xarray.DataArray
            Transformed y-coordinates
        """
        # convert coordinate reference system to that of the dataset
        # and format as xarray DataArray with appropriate dimensions
        X, Y = _coords(x, y, source_crs=crs, target_crs=self.crs, **kwargs)
        # return the transformed coordinates
        return X, Y

    def crop(
        self,
        bounds: list | tuple,
        buffer: int | float = 0,
    ):
        """
        Crop ``Dataset`` to input bounding box

        Parameters
        ----------
        bounds: list, tuple
            Bounding box [min_x, max_x, min_y, max_y]
        buffer: int or float, default 0
            Buffer to add to bounds for cropping
        """
        # pad global grids along x-dimension (if necessary)
        lon_wrap = self.crs.to_dict().get("lon_wrap", 0)
        if self.is_global and (lon_wrap == 180) and (np.min(bounds[:2]) < 0):
            # number of points to pad for global grids
            n = int(180 // (self._x[1] - self._x[0]))
            ds = self.pad(n=(n, 0))
        elif self.is_global and (lon_wrap == 0) and (np.max(bounds[:2]) > 180):
            # number of points to pad for global grids
            n = int(180 // (self._x[1] - self._x[0]))
            ds = self.pad(n=(0, n))
        else:
            # copy dataset
            ds = self._ds.copy()
        # check if chunks are present
        if hasattr(ds, "chunks") and ds.chunks is not None:
            ds = ds.chunk(-1).compute()
        # unpack bounds and buffer
        xmin = bounds[0] - buffer
        xmax = bounds[1] + buffer
        ymin = bounds[2] - buffer
        ymax = bounds[3] + buffer
        # crop dataset to bounding box
        ds = ds.where(
            (ds.x >= xmin) & (ds.x <= xmax) & (ds.y >= ymin) & (ds.y <= ymax),
            drop=True,
        )
        # return the cropped dataset
        return ds

    def cumsum(self, **kwargs):
        """
        Calculate cumulative sum of ``Dataset`` along time dimension

        Returns
        -------
        ds: xarray.Dataset
            Cumulative sum of the ``Dataset``
        """
        # calculate cumulative sum along time dimension
        ds = self._ds.cumsum(dim="time", skipna=False, **kwargs)
        # return the cumulative sum dataset
        return ds

    def extrap_like(self, other: xr.Dataset, **kwargs):
        """
        Extrapolate missing values in ``Dataset`` using nearest-neighbors

        Parameters
        ----------
        other: xarray.Dataset
            ``Dataset`` with missing values to be extrapolated
        kwargs: keyword arguments
            Keyword arguments for extrapolation functions

        Returns
        -------
        other: xarray.Dataset
            ``Dataset`` with extrapolated values
        """
        # import extrapolate functions
        from SMBcorr.interpolate import (
            _to_cartesian,
            _build_tree,
            _nearest_neighbors,
        )

        # get extrapolation cutoff distance
        cutoff = kwargs.get("cutoff", np.inf)
        # check if chunks are present
        if hasattr(other, "chunks") and other.chunks is not None:
            other = other.chunk(-1).compute()
        # bounds of other dataset
        bounds = [
            other.x.values.min(),
            other.x.values.max(),
            other.y.values.min(),
            other.y.values.max(),
        ]
        # crop dataset to bounding box of other dataset plus buffer
        if np.isfinite(cutoff) and self.crs.is_geographic:
            # use twice the cutoff distance as a buffer
            cutoff_km = cutoff * __ureg__.parse_units("km")
            a_axis = 6378.137 * __ureg__.parse_units("km")
            buffer = 2.0 * (cutoff_km / a_axis).to(self.axis_units).magnitude
            # crop dataset to bounding box of other dataset plus buffer
            ds = self.crop(bounds=bounds, buffer=buffer)
        elif np.isfinite(cutoff):
            # use twice the cutoff distance as a buffer
            cutoff_km = cutoff * __ureg__.parse_units("km")
            buffer = 2.0 * cutoff_km.to(self.axis_units).magnitude
            # crop dataset to bounding box of other dataset plus buffer
            ds = self.crop(bounds=bounds, buffer=buffer)
        else:
            # copy dataset without cropping
            ds = self._ds.copy()
        # calculate meshgrid of cropped model coordinates
        gridx, gridy = np.meshgrid(ds.x.values, ds.y.values)
        # initialize valid mask for building tree
        valid_mask = np.zeros_like(gridx, dtype=bool)
        tree = None
        # iterate over variables in dataset
        for i, v in enumerate(other.data_vars.keys()):
            # check for missing values
            invalid = other[v].isnull()
            if not invalid.any():
                # no missing values
                continue
            # find valid values
            mask = ds[v].notnull().values
            # build tree if on the first iteration
            # or if the valid mask has changed
            if (tree is None) or (mask != valid_mask).any():
                # get indices of valid points
                valid_indices = np.nonzero(mask)
                # reduce to valid original values
                p_in = _to_cartesian(
                    gridx[valid_indices],
                    gridy[valid_indices],
                    is_geographic=self.crs.is_geographic,
                )
                # build kd-tree for valid points
                tree = _build_tree(p_in)
                # copy valid mask for next iteration
                valid_mask = np.copy(mask)
            # reduce model to valid original values
            flattened = ds[v].values[valid_indices]
            # extrapolate missing values using nearest-neighbors
            if other[v].ndim == 0:
                # single point extrapolation
                p_out = _to_cartesian(
                    other.x.values,
                    other.y.values,
                    is_geographic=self.crs.is_geographic,
                )
                (other[v].values,) = _nearest_neighbors(
                    tree, p_out, flattened, **kwargs
                )
            else:
                # only extrapolate invalid points
                p_out = _to_cartesian(
                    other.x.values[invalid],
                    other.y.values[invalid],
                    is_geographic=self.crs.is_geographic,
                )
                other[v].values[invalid] = _nearest_neighbors(
                    tree, p_out, flattened, **kwargs
                )
        # return xarray dataset
        return other

    def gaussian_filter(
        self,
        sigma: float | list[float] = 1.5,
        **kwargs,
    ):
        """
        Apply Gaussian smoothing to the ``Dataset``

        Parameters
        ----------
        sigma: float or list, default 1.5
            Standard deviation for Gaussian kernel in x and y directions
        **kwargs: dict
            Additional keyword arguments for ``scipy.ndimage.gaussian_filter``

        Returns
        -------
        ds: xarray.Dataset
            Smoothed ``Dataset``
        """
        # import gaussian filter function
        from scipy.ndimage import gaussian_filter

        # set default keyword arguments
        kwargs.setdefault("mode", "constant")
        kwargs.setdefault("cval", 0)
        # create copy of dataset
        ds = self._ds.copy(deep=True)
        # apply Gaussian smoothing to each variable in the dataset
        for v in ds.data_vars.keys():
            # use a gaussian filter to smooth mask
            mask = np.logical_not(ds[v].isnull().any(dim="time")).astype("f")
            kernel = gaussian_filter(mask, sigma=sigma, **kwargs)
            for i, t in enumerate(ds.time):
                # replace fill values with zeros before smoothing data
                tmp = ds[v].isel(time=i).fillna(0.0)
                # smooth spatial field
                smooth = gaussian_filter(tmp, sigma=sigma, **kwargs)
                # scale output smoothed field
                scaled = xr.where(kernel != 0, smooth / kernel, np.nan)
                # replace valid values with original
                ds[v][i, :, :] = xr.where(mask, tmp, scaled)
        # return the smoothed dataset
        return ds

    def grid_interp(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method="linear",
        **kwargs,
    ):
        """
        Interpolate a regular or rectilinear ``Dataset`` to new coordinates

        Parameters
        ----------
        x: np.ndarray
            Interpolation x-coordinates
        y: np.ndarray
            Interpolation y-coordinates
        method: str, default 'linear'
            Interpolation method

        Returns
        -------
        other: xarray.Dataset
            Interpolated ``Dataset``
        """
        # pad global grids along x-dimension (if necessary)
        if self.is_global:
            self._ds = self.pad(n=1)
        # verify longitudinal convention for geographic models
        if self.crs.is_geographic:
            # grid spacing in x-direction
            dx = self._x[1] - self._x[0]
            # adjust input longitudes to be consistent with model
            if (np.min(x) < 0.0) & (self._x.max() > (180.0 + dx)):
                # input points convention (-180:180)
                # model convention (0:360)
                x = xr.where(x < 0.0, x + 360.0, x)
            elif (np.max(x) > 180.0) & (self._x.min() < (0.0 - dx)):
                # input points convention (0:360)
                # model convention (-180:180)
                x = xr.where(x > 180.0, x - 360.0, x)
        # interpolate dataset using built-in xarray methods
        other = self._ds.interp(x=x, y=y, method=method)
        # return xarray dataset
        return other

    def inpaint(self, **kwargs):
        """
        Inpaint over missing data in ``Dataset``

        Parameters
        ----------
        kwargs: keyword arguments
            Keyword arguments for ``SMBcorr.interpolate.inpaint``

        Returns
        -------
        ds: xarray.Dataset
            Interpolated ``Dataset``
        """
        # import inpaint function
        from SMBcorr.interpolate import inpaint

        # create copy of dataset
        ds = self._ds.copy()
        # inpaint each variable in the dataset
        for v in ds.data_vars.keys():
            ds[v].values = inpaint(
                self._x, self._y, self._ds[v].values, **kwargs
            )
        # return the dataset
        return ds

    def interp(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ):
        """
        Interpolate ``Dataset`` to new coordinates

        Parameters
        ----------
        x: np.ndarray
            Interpolation x-coordinates
        y: np.ndarray
            Interpolation y-coordinates
        extrapolate: bool, default False
            Flag to extrapolate values using nearest-neighbors
        cutoff: int or float, default np.inf
            Maximum distance for extrapolation
        **kwargs: dict
            Additional keyword arguments for interpolation functions

        Returns
        -------
        other: xarray.Dataset
            Interpolated ``Dataset``
        """
        # set default keyword arguments
        kwargs.setdefault("method", "linear")
        kwargs.setdefault("extrapolate", False)
        kwargs.setdefault("cutoff", np.inf)
        # use built-in xarray interpolation methods
        other = self.grid_interp(x, y, **kwargs)
        # extrapolate missing values using nearest-neighbors
        if kwargs["extrapolate"]:
            other = self.extrap_like(other, cutoff=kwargs["cutoff"])
        # return xarray dataset
        return other

    def pad(
        self,
        n: int = 1,
        chunks=None,
    ):
        """
        Pad ``Dataset`` by repeating edge values in the x-direction

        Parameters
        ----------
        n: int, default 1
            Number of padding values to add on each side

        Returns
        -------
        ds: xarray.Dataset
            Padded ``Dataset``
        """
        # (possibly) unchunk x-coordinates and pad to wrap at meridian
        x = xr.DataArray(self._x, dims="x").pad(
            x=n, mode="reflect", reflect_type="odd"
        )
        # pad dataset and re-assign x-coordinates
        ds = self._ds.copy()
        ds = ds.pad(x=n, mode="wrap").assign_coords(x=x)
        # rechunk dataset (if specified)
        if chunks is not None:
            ds = ds.chunk(chunks)
        # return the dataset
        return ds

    def to_anomaly(
        self,
        reference: str | None = None,
        climatology: list | None = None,
    ):
        """
        Convert ``Dataset`` to anomalies relative to a reference period

        Parameters
        ----------
        reference: str or None
            Method for referencing anomalies

            - `'first'`: remove first time step
            - `'mean'`: remove mean over a time range
        climatology: list, default None
            Time range for calculating mean reference
        """
        # if referencing anomalies: change from absolute to relative values
        if reference == "first":
            # subtract first time step from all time steps
            z0 = self._ds.isel(time=0)
            ds = self._ds - z0
        elif reference == "mean":
            # get time range for calculating reference period
            if climatology is None:
                # default time range is the full range of the dataset
                tmin = self._ds["time"].values.min()
                tmax = self._ds["time"].values.max()
            elif isinstance(climatology[0], (int, float)):
                # convert years to numpy datetime64 format
                tmin = np.array(climatology[0] - 1970, dtype="datetime64[Y]")
                tmax = np.array(climatology[1] - 1970, dtype="datetime64[Y]")
            else:
                # verify that time range is in datetime64 format
                tmin, tmax = np.array(climatology, dtype="datetime64[D]")
            # subtract mean from all time steps
            zmean = self._ds.where(
                (self._ds["time"] >= tmin) & (self._ds["time"] < tmax + 1),
                drop=True,
            )
            ds = self._ds - zmean.mean(dim="time")
            # add (actual) climatology attributes to variable
            ds.attrs["climatology"] = np.array(
                [zmean.time.values.min(), zmean.time.values.max()]
            ).astype("datetime64[D]")
        else:
            raise ValueError(f"Invalid reference method: {reference}")
        # return the anomaly dataset
        return ds

    def transform_as(
        self,
        x: np.ndarray,
        y: np.ndarray,
        crs: str | int | dict = 4326,
        **kwargs,
    ):
        """
        Transform coordinates to/from the ``Dataset`` coordinate reference system

        Parameters
        ----------
        x: np.ndarray
            Input x-coordinates
        y: np.ndarray
            Input y-coordinates
        crs: str, int, or dict, default 4326 (WGS84 Latitude/Longitude)
            Coordinate reference system of input coordinates
        direction: str, default 'FORWARD'
            Direction of transformation

            - ``'FORWARD'``: from input crs to model crs
            - ``'INVERSE'``: from model crs to input crs

        Returns
        -------
        X: np.ndarray
            Transformed x-coordinates
        Y: np.ndarray
            Transformed y-coordinates
        """
        # convert coordinate reference system to that of the dataset
        X, Y = _transform(x, y, source_crs=crs, target_crs=self.crs, **kwargs)
        # return the transformed coordinates
        return (X, Y)

    def to_units(
        self,
        units: str,
        value: float = 1.0,
    ):
        """Convert ``Dataset`` to specified units

        Parameters
        ----------
        units: str
            Output units
        value: float, default 1.0
            Scaling factor to apply
        """
        # create copy of dataset
        ds = self._ds.copy()
        # convert each variable in the dataset
        for k in ds.data_vars.keys():
            ds[k] = ds[k].smb.to_units(units, value=value)
        # return the dataset
        return ds

    def to_base_units(self):
        """Convert ``Dataset`` to base units"""
        # create copy of dataset
        ds = self._ds.copy()
        # convert each variable in the dataset
        for k in ds.data_vars.keys():
            ds[k] = ds[k].smb.to_base_units()
        # return the dataset
        return ds

    def to_default_units(self):
        """Convert ``Dataset`` to default units"""
        # create copy of dataset
        ds = self._ds.copy()
        # convert each variable in the dataset
        for k in ds.data_vars.keys():
            ds[k] = ds[k].smb.to_default_units()
        # return the dataset
        return ds

    @property
    def crs(self):
        """Coordinate reference system of the ``Dataset``"""
        # return the CRS of the dataset
        # default is EPSG:4326 (WGS84)
        CRS = self._ds.attrs.get("crs", 4326)
        return pyproj.CRS.from_user_input(CRS)

    @property
    def is_global(self) -> bool:
        """Determine if ``Dataset`` covers a global domain"""
        # grid spacing in x-direction
        dx = self._x[1] - self._x[0]
        # check if global grid
        cyclic = np.isclose(self._x[-1] - self._x[0], 360.0 - dx)
        return self.crs.is_geographic and cyclic

    @property
    def area_of_use(self) -> str | None:
        """Area of use from the ``Dataset`` CRS"""
        if self.crs.area_of_use is not None:
            return self.crs.area_of_use.name.replace(".", "").lower()

    @property
    def axis_units(self) -> str:
        """Units of the coordinate axes from the ``Dataset`` CRS"""
        return self.crs.axis_info[0].unit_name

    @property
    def _x(self):
        """x-coordinates of the ``Dataset``"""
        return self._ds.x.values

    @property
    def _y(self):
        """y-coordinates of the ``Dataset``"""
        return self._ds.y.values


@xr.register_dataarray_accessor("smb")
class DataArray:
    """Accessor for extending an ``xarray.DataArray`` for SMB and firn data"""

    def __init__(self, da):
        # initialize DataArray
        self._da = da

    def to_units(
        self,
        units: str,
        value: float = 1.0,
    ):
        """Convert ``DataArray`` to specified units

        Parameters
        ----------
        units: str
            Output units
        value: float, default 1.0
            Scaling factor to apply
        """
        # convert to specified units
        conversion = value * self.quantity.to(units)
        da = self._da * conversion.magnitude
        da.attrs["units"] = str(conversion.units)
        return da

    def to_base_units(self, value=1.0):
        """Convert ``DataArray`` to base units

        Parameters
        ----------
        value: float, default 1.0
            Scaling factor to apply
        """
        # convert to base units
        conversion = value * self.quantity.to_base_units()
        da = self._da * conversion.magnitude
        da.attrs["units"] = str(conversion.units)
        return da

    def to_default_units(self, value=1.0):
        """Convert ``DataArray`` to default units

        Parameters
        ----------
        value: float, default 1.0
            Scaling factor to apply
        """
        # convert to default units
        default_units = _default_units.get(self.group, self.units)
        da = self.to_units(default_units, value=value)
        return da

    @property
    def units(self):
        """Units of the ``DataArray``"""
        try:
            return self._parse_units(self._units)
        except TypeError as exc:
            raise ValueError(f"Unknown units: {self._units}") from exc
        except AttributeError as exc:
            raise AttributeError("DataArray has no attribute 'units'") from exc

    @property
    def quantity(self):
        """``Pint`` Quantity of the ``DataArray``"""
        return 1.0 * self.units

    @property
    def group(self):
        """Variable group of the ``DataArray``"""
        if self.units.is_compatible_with("m"):
            return "elevation"
        elif self.units.is_compatible_with("m/s"):
            return "velocity"
        elif self.units.is_compatible_with("g / cm^2"):
            return "mass density"
        elif self.units.is_compatible_with("g"):
            return "mass"
        elif self.units.is_compatible_with("degrees"):
            return "angle"
        else:
            raise ValueError(f"Unknown unit group: {self._units}")

    @staticmethod
    def _parse_units(units: str):
        """
        Convert units attributes to ``pint`` units
        """
        # fix the exponent notation in units string
        units = re.sub(
            r"(\w)([-]?\d+)",
            lambda m: m.group(1) + r"^" + m.group(2),
            units,
            flags=re.IGNORECASE,
        )
        # remove "of" from units string
        units = re.sub(
            r"of\s(water|ice|air)",
            lambda m: m.group(1),
            units,
            flags=re.IGNORECASE,
        )
        # delete periods between water or ice equivalent units
        units = re.sub(
            r"(w|i)\.e[q]?\.",
            lambda m: m.group(1) + "e",
            units,
            flags=re.IGNORECASE,
        )
        # add a space before water or ice equivalent units
        units = re.sub(
            r"([\w])(we|ie)\b",
            lambda m: m.group(1) + " " + m.group(2),
            units,
            flags=re.IGNORECASE,
        )
        # parse units string using pint
        return __ureg__.parse_units(units.lower())

    @property
    def _units(self):
        """Units attribute of the ``DataArray`` as a string"""
        return self._da.attrs.get("units")

    @property
    def _has_compatible_units(self):
        """Tests that units are compatible with known groups"""
        try:
            unit_group = self.group
        except (TypeError, ValueError, AttributeError) as exc:
            return False
        else:
            return True


def register_dataset_subaccessor(name):
    """Register a custom subaccessor on ``Dataset`` objects

    Parameters
    ----------
    name: str
        Name of the subaccessor
    """
    return xr.core.extensions._register_accessor(name, Dataset)


def register_dataarray_subaccessor(name):
    """Register a custom subaccessor on ``DataArray`` objects

    Parameters
    ----------
    name: str
        Name of the subaccessor
    """
    return xr.core.extensions._register_accessor(name, DataArray)


def _transform(
    i1: np.ndarray,
    i2: np.ndarray,
    source_crs: str | int | dict = 4326,
    target_crs: str | int | dict = None,
    **kwargs,
):
    """
    Transform coordinates to/from the dataset coordinate reference system

    Parameters
    ----------
    i1: np.ndarray
        Input x-coordinates
    i2: np.ndarray
        Input y-coordinates
    source_crs: str, int, or dict, default 4326 (WGS84 Latitude/Longitude)
        Coordinate reference system of input coordinates
    target_crs: str, int, or dict, default None
        Coordinate reference system of output coordinates
    direction: str, default 'FORWARD'
        Direction of transformation

        - ``'FORWARD'``: from input crs to model crs
        - ``'INVERSE'``: from model crs to input crs

    Returns
    -------
    o1: np.ndarray
        Transformed x-coordinates
    o2: np.ndarray
        Transformed y-coordinates
    """
    # set the direction of the transformation
    kwargs.setdefault("direction", "FORWARD")
    assert kwargs["direction"] in ("FORWARD", "INVERSE", "IDENT")
    # get the coordinate reference system and transform
    source_crs = pyproj.CRS.from_user_input(source_crs)
    transformer = pyproj.Transformer.from_crs(
        source_crs, target_crs, always_xy=True
    )
    # convert coordinate reference system
    o1, o2 = transformer.transform(i1, i2, **kwargs)
    # return the transformed coordinates
    return (o1, o2)


def _coords(
    x: np.ndarray,
    y: np.ndarray,
    source_crs: str | int | dict = 4326,
    target_crs: str | int | dict = None,
    **kwargs,
):
    """
    Transform coordinates into DataArrays in a new
    coordinate reference system

    Parameters
    ----------
    x: np.ndarray
        Input x-coordinates
    y: np.ndarray
        Input y-coordinates
    source_crs: str, int, or dict, default 4326 (WGS84 Latitude/Longitude)
        Coordinate reference system of input coordinates
    target_crs: str, int, or dict, default None
        Coordinate reference system of output coordinates
    type: str or None, default None
        Coordinate data type

        If not provided: must specify ``time`` parameter to auto-detect

        - ``None``: determined from input variable dimensions
        - ``'drift'``: drift buoys or satellite/airborne altimetry
        - ``'grid'``: spatial grids or images
        - ``'time series'``: time series at a single point
    time: np.ndarray or None, default None
        Time variable for determining coordinate data type

    Returns
    -------
    X: xarray.DataArray
        Transformed x-coordinates
    Y: xarray.DataArray
        Transformed y-coordinates
    """
    from SMBcorr.spatial import data_type

    # set default keyword arguments
    kwargs.setdefault("type", None)
    kwargs.setdefault("time", None)
    # determine coordinate data type if possible
    if (np.ndim(x) == 0) and (np.ndim(y) == 0):
        coord_type = "time series"
    elif kwargs["type"] is None:
        # must provide time variable to determine data type
        assert kwargs["time"] is not None, (
            "Must provide time parameter when type is not specified"
        )
        coord_type = data_type(x, y, np.ravel(kwargs["time"]))
    else:
        # use provided coordinate data type
        # and verify that it is lowercase
        coord_type = kwargs.get("type").lower()
    # convert coordinates to a new coordinate reference system
    if (coord_type == "grid") and (np.size(x) != np.size(y)):
        gridx, gridy = np.meshgrid(x, y)
        mx, my = _transform(
            gridx,
            gridy,
            source_crs=source_crs,
            target_crs=target_crs,
            direction="FORWARD",
        )
    else:
        mx, my = _transform(
            x,
            y,
            source_crs=source_crs,
            target_crs=target_crs,
            direction="FORWARD",
        )
    # convert to xarray DataArray with appropriate dimensions
    if (np.ndim(x) == 0) and (np.ndim(y) == 0):
        X = xr.DataArray(mx)
        Y = xr.DataArray(my)
    elif coord_type == "grid":
        X = xr.DataArray(mx, dims=("y", "x"))
        Y = xr.DataArray(my, dims=("y", "x"))
    elif coord_type == "drift":
        X = xr.DataArray(mx, dims=("time"))
        Y = xr.DataArray(my, dims=("time"))
    elif coord_type == "time series":
        X = xr.DataArray(mx, dims=("station"))
        Y = xr.DataArray(my, dims=("station"))
    else:
        raise ValueError(f"Unknown coordinate data type: {coord_type}")
    # return the transformed coordinates
    return (X, Y)
