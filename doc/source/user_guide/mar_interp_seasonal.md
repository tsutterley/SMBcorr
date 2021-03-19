mar_interp_seasonal.py
======================

- Interpolates and extrapolates seasonal MAR products to times and coordinates using surface splines
- Seasonal files are climatology files for each day of the year

#### Calling Sequence
```python
from SMBcorr.mar_interp_seasonal import interpolate_mar_seasonal
interp_data = interpolate_mar_seasonal(DIRECTORY, EPSG, tdec, X, Y,
    VARIABLE='SMB', RANGE=[2000,2019], SIGMA=1.5, FILL_VALUE=np.nan)
```
[Source code](https://github.com/tsutterley/SMBcorr/blob/master/SMBcorr/mar_interp_seasonal.py)

#### Arguments
- `DIRECTORY`: full path to the MAR data directory
    * `<path_to_mar>/MARv3.11.2/Greenland/7.5km_ERA5/`
    * `<path_to_mar>/MARv3.11.2/Greenland/10km_ERA5/`
    * `<path_to_mar>/MARv3.11.2/Greenland/15km_ERA5/`
    * `<path_to_mar>/MARv3.11.2/Greenland/20km_ERA5/`
    * `<path_to_mar>/MARv3.11.2/Greenland/20km_NCEP1/`
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
- `RANGE`: start year and end year of seasonal file
- `SIGMA`: Standard deviation for Gaussian kernel
- `FILL_VALUE`: output fill_value for invalid points
- `EXTRAPOLATE`: create a regression model to extrapolate out in time

#### Dependencies
- [numpy: Scientific Computing Tools For Python](https://numpy.org)
- [scipy: Scientific Tools for Python](https://docs.scipy.org/doc//)
- [netCDF4: Python interface to the netCDF C library](https://unidata.github.io/netcdf4-python/netCDF4/index.html)
- [pyproj: Python interface to PROJ library](https://pypi.org/project/pyproj/)
