merra_hybrid_extrap.py
======================

- Interpolates and extrapolates MERRA-2 hybrid variables to times and coordinates using inverse distance weighting

#### Calling Sequence
```python
from SMBcorr.merra_hybrid_extrap import extrapolate_merra_hybrid
interp_data = interpolate_merra_hybrid(DIRECTORY, EPSG, 'gris', tdec, X, Y,
    VARIABLE='FAC', SIGMA=1.5, SEARCH='BallTree', FILL_VALUE=np.nan)
```
[Source code](https://github.com/tsutterley/SMBcorr/blob/master/SMBcorr/merra_hybrid_extrap.py)

#### Arguments
- `base_dir`: working data directory
- `EPSG`: projection of input spatial coordinates
- `REGION`: region to interpolate
    * `'gris'`: Greenland
    * `'ais'`: Antarctica
- `tdec`: dates to interpolate in year-decimal
- `X`: x-coordinates to interpolate in projection EPSG
- `Y`: y-coordinates to interpolate in projection EPSG

#### Keyword arguments
- `VARIABLE`: MERRA-2 hybrid product to interpolate
    * `FAC`: firn air content
    * `p_minus_e`: precipitation minus evaporation
    * `melt`: snowmelt
- `SIGMA`: Standard deviation for Gaussian kernel
- `SEARCH`: nearest-neighbor search algorithm (`'BallTree'` or `'KDTree'`)
- `NN`: number of nearest-neighbor points to use
- `POWER`: inverse distance weighting power
- `FILL_VALUE`: output fill_value for invalid points
- `GZIP`: netCDF4 file is locally gzip compressed

#### Dependencies
- [numpy: Scientific Computing Tools For Python](https://numpy.org)
- [scipy: Scientific Tools for Python](https://docs.scipy.org/doc//)
- [netCDF4: Python interface to the netCDF C library](https://unidata.github.io/netcdf4-python/netCDF4/index.html)
- [pyproj: Python interface to PROJ library](https://pypi.org/project/pyproj/)
- [scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/index.html)
