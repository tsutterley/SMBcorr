racmo_interp_mean.py
====================

- Interpolates the mean of downscaled RACMO products to spatial coordinates using surface splines

#### Calling Sequence
```python
from SMBcorr.racmo_interp_mean import interpolate_racmo_mean
interp_data = interpolate_racmo_mean(base_dir, EPSG, '3.0', tdec, X, Y,
    VARIABLE='SMB', RANGE=[1961,1990], FILL_VALUE=np.nan)
```
[Source code](https://github.com/tsutterley/SMBcorr/blob/master/SMBcorr/racmo_interp_mean.py)

#### Inputs
- `base_dir`: working data directory
- `EPSG`: projection of input spatial coordinates  
- `VERSION`: Downscaled RACMO Version
    * `1.0`: RACMO2.3/XGRN11
    * `2.0`: RACMO2.3p2/XGRN11
    * `3.0`: RACMO2.3p2/FGRN055
- `tdec`: dates to interpolate in year-decimal  
- `X`: x-coordinates to interpolate in projection EPSG  
- `Y`: y-coordinates to interpolate in projection EPSG  

#### Options
- `VARIABLE`: RACMO product to interpolate  
    `'SMB'`: Surface Mass Balance
    `'PRECIP'`: Precipitation
    `'RUNOFF'`: Melt Water Runoff
    `'SNOWMELT'`: Snowmelt
    `'REFREEZE'`: Melt Water Refreeze
- `RANGE`: start and end year for mean
- `FILL_VALUE`: output fill_value for invalid points  

#### Dependencies
- [numpy: Scientific Computing Tools For Python](https://numpy.org)  
- [scipy: Scientific Tools for Python](https://docs.scipy.org/doc//)  
- [netCDF4: Python interface to the netCDF C library](https://unidata.github.io/netcdf4-python/netCDF4/index.html)  
- [pyproj: Python interface to PROJ library](https://pypi.org/project/pyproj/)  
