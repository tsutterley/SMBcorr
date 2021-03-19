racmo_extrap_mean.py
====================

- Interpolates the mean of downscaled RACMO products to spatial coordinates using inverse distance weighting

#### Calling Sequence
```python
from SMBcorr.racmo_extrap_mean import extrapolate_racmo_mean
interp_data = extrapolate_racmo_mean(base_dir, EPSG, '3.0', tdec, X, Y,
    VARIABLE='SMB', RANGE=[1961,1990], SEARCH='BallTree', FILL_VALUE=np.nan)
```
[Source code](https://github.com/tsutterley/SMBcorr/blob/master/SMBcorr/racmo_extrap_mean.py)

#### Arguments
- `base_dir`: working data directory
- `EPSG`: projection of input spatial coordinates
- `VERSION`: Downscaled RACMO Version
    * `1.0`: RACMO2.3/XGRN11
    * `2.0`: RACMO2.3p2/XGRN11
    * `3.0`: RACMO2.3p2/FGRN055
- `tdec`: dates to interpolate in year-decimal
- `X`: x-coordinates to interpolate in projection EPSG
- `Y`: y-coordinates to interpolate in projection EPSG

#### Keyword arguments
- `VARIABLE`: RACMO product to interpolate
    * `'SMB'`: Surface Mass Balance
    * `'PRECIP'`: Precipitation
    * `'RUNOFF'`: Melt Water Runoff
    * `'SNOWMELT'`: Snowmelt
    * `'REFREEZE'`: Melt Water Refreeze
- `RANGE`: start and end year for mean
- `SEARCH`: nearest-neighbor search algorithm (`'BallTree'` or `'KDTree'`)
- `NN`: number of nearest-neighbor points to use
- `POWER`: inverse distance weighting power
- `FILL_VALUE`: output fill_value for invalid points

#### Dependencies
- [numpy: Scientific Computing Tools For Python](https://numpy.org)
- [scipy: Scientific Tools for Python](https://docs.scipy.org/doc//)
- [netCDF4: Python interface to the netCDF C library](https://unidata.github.io/netcdf4-python/netCDF4/index.html)
- [pyproj: Python interface to PROJ library](https://pypi.org/project/pyproj/)
- [scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/index.html)
