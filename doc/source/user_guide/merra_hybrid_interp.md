merra_hybrid_interp.py
======================

- Interpolates and extrapolates MERRA-2 hybrid variables to times and coordinates using surface splines

#### Calling Sequence
```python
from SMBcorr.merra_hybrid_interp import interpolate_merra_hybrid
interp_data = interpolate_merra_hybrid(DIRECTORY, EPSG, 'gris', tdec, X, Y,
    VARIABLE='FAC', SIGMA=1.5, FILL_VALUE=np.nan)
```
[Source code](https://github.com/tsutterley/SMBcorr/blob/master/SMBcorr/merra_hybrid_interp.py)

#### Inputs
- `base_dir`: working data directory
- `EPSG`: projection of input spatial coordinates  
- `REGION`: region to interpolate
    * `'gris'`: Greenland
    * `'ais'`: Antarctica
- `tdec`: dates to interpolate in year-decimal  
- `X`: x-coordinates to interpolate in projection EPSG  
- `Y`: y-coordinates to interpolate in projection EPSG  

#### Options
- `VARIABLE`: MERRA-2 hybrid product to interpolate  
    * `FAC`: firn air content
    * `p_minus_e`: precipitation minus evaporation
    * `melt`: snowmelt
- `SIGMA`: Standard deviation for Gaussian kernel  
- `FILL_VALUE`: output fill_value for invalid points  

#### Dependencies
- [numpy: Scientific Computing Tools For Python](https://numpy.org)  
- [scipy: Scientific Tools for Python](https://docs.scipy.org/doc//)  
- [netCDF4: Python interface to the netCDF C library](https://unidata.github.io/netcdf4-python/netCDF4/index.html)  
- [pyproj: Python interface to PROJ library](https://pypi.org/project/pyproj/)  
