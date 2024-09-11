mar_extrap_daily.py
===================

- Interpolates and extrapolates daily MAR products to times and coordinates using inverse distance weighting

#### Calling Sequence
```python
from SMBcorr.mar_extrap_daily import extrapolate_mar_daily
interp_data = extrapolate_mar_daily(DIRECTORY, EPSG, tdec, X, Y,
    VARIABLE='SMB', SIGMA=1.5, SEARCH='BallTree', FILL_VALUE=np.nan)
```
[Source code](https://github.com/tsutterley/SMBcorr/blob/master/SMBcorr/mar_extrap_daily.py)

#### Arguments
- `DIRECTORY`: full path to the MAR data directory
    * `<path_to_mar>/MARv3.11.2/Greenland/7.5km_ERA5/`
    * `<path_to_mar>/MARv3.11.2/Greenland/10km_ERA5/`
    * `<path_to_mar>/MARv3.11.2/Greenland/15km_ERA5/`
    * `<path_to_mar>/MARv3.11.2/Greenland/20km_ERA5/`
    * `<path_to_mar>/MARv3.11.2/Greenland/20km_NCEP1/`
    * `<path_to_mar>/MARv3.11/Greenland/ERA_1958-2019-15km`
    * `<path_to_mar>/MARv3.11/Greenland/ERA_1958-2019-15km/daily_15km`
    * `<path_to_mar>/MARv3.11/Greenland/NCEP1_1948-2020_20km/daily_20km`
    * `<path_to_mar>/MARv3.10/Greenland/NCEP1_1948-2019_20km/daily_20km`
    * `<path_to_mar>/MARv3.9/Greenland/ERA_1958-2018_10km/daily_10km`
- `EPSG`: projection of input spatial coordinates
- `tdec`: dates to interpolate in year-decimal
- `X`: x-coordinates to interpolate in projection EPSG
- `Y`: y-coordinates to interpolate in projection EPSG

#### Keyword arguments
- `VARIABLE`: MAR product to interpolate
- `SIGMA`: Standard deviation for Gaussian kernel
- `SEARCH`: nearest-neighbor search algorithm (`'BallTree'` or `'KDTree'`)
- `NN`: number of nearest-neighbor points to use
- `POWER`: inverse distance weighting power
- `FILL_VALUE`: output fill_value for invalid points
- `EXTRAPOLATE`: create a regression model to extrapolate out in time

#### Dependencies
- [numpy: Scientific Computing Tools For Python](https://numpy.org)
- [scipy: Scientific Tools for Python](https://docs.scipy.org/doc//)
- [netCDF4: Python interface to the netCDF C library](https://unidata.github.io/netcdf4-python/netCDF4/index.html)
- [pyproj: Python interface to PROJ library](https://pypi.org/project/pyproj/)
- [scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/index.html)
