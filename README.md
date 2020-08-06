SMBcorr
=======

[![Language](https://img.shields.io/badge/python-v3.7-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/tsutterley/SMBcorr/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/smbcorr/badge/?version=latest)](https://smbcorr.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tsutterley/SMBcorr/master)
[![Binder](https://binder.pangeo.io/badge.svg)](https://binder.pangeo.io/v2/gh/tsutterley/SMBcorr/master)

#### Python-based tools for correcting altimetry data for surface mass balance and firn processes  

- [`convert_calendar_decimal.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/convert_calendar_decimal.md) - Converts from calendar date into decimal years taking into account leap years  
- [`convert_julian.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/convert_julian.md) - Return the calendar date and time given Julian date  
- [`mar_extrap_daily.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/mar_extrap_daily.md) - Interpolates and extrapolates daily MAR products to times and coordinates using inverse distance weighting  
- [`mar_extrap_seasonal.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/mar_extrap_seasonal.md) - Interpolates and extrapolates seasonal MAR products to times and coordinates using inverse distance weighting  
- [`mar_interp_daily.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/mar_interp_daily.md) - Interpolates and extrapolates daily MAR products to times and coordinates using surface splines  
- [`mar_interp_seasonal.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/mar_interp_seasonal.md) - Interpolates and extrapolates seasonal MAR products to times and coordinates using surface splines  
- [`mar_smb_cumulative.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/mar_smb_cumulative.md) - Calculates cumulative anomalies of MAR surface mass balance products  
- [`mar_smb_mean.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/mar_smb_mean.md) - Calculates the temporal mean of MAR surface mass balance products  
- [`merra_hybrid_cumulative.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/merra_hybrid_cumulative.md) - Calculates the temporal mean of MAR surface mass balance products  
- [`merra_hybrid_extrap.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/merra_hybrid_extrap.md) - Interpolates and extrapolates MERRA-2 hybrid variables to times and coordinates using inverse distance weighting  
- [`merra_hybrid_interp.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/merra_hybrid_interp.md) - Interpolates and extrapolates MERRA-2 hybrid variables to times and coordinates using surface splines  
- [`merra_smb_cumulative.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/merra_smb_cumulative.md) - Calculates cumulative anomalies of MERRA-2 surface mass balance products  
- [`racmo_downscaled_cumulative.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/racmo_downscaled_cumulative.md) - Calculates cumulative anomalies of downscaled RACMO surface mass balance products  
- [`racmo_downscaled_mean.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/racmo_downscaled_mean.md) - Calculates the temporal mean of downscaled RACMO surface mass balance products  
- [`racmo_extrap_daily.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/racmo_extrap_daily.md) - Interpolates and extrapolates daily RACMO products to times and coordinates using inverse distance weighting  
- [`racmo_extrap_downscaled.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/racmo_extrap_downscaled.md) - Interpolates and extrapolates downscaled RACMO products to times and coordinates using inverse distance weighting  
- [`racmo_extrap_firn_height.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/racmo_extrap_firn_height.md) - Interpolates and extrapolates firn heights to times and coordinates using inverse distance weighting  
- [`racmo_extrap_mean.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/racmo_extrap_mean.md) - Interpolates the mean of downscaled RACMO products to spatial coordinates using inverse distance weighting  
- [`racmo_integrate_firn_height.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/racmo_integrate_firn_height.md) - Integrate RACMO firn heights for each Promice ice class  
- [`racmo_interp_daily.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/racmo_interp_daily.md) - Interpolates and extrapolates daily RACMO products to times and coordinates using surface splines  
- [`racmo_interp_downscaled.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/racmo_interp_downscaled.md) - Interpolates and extrapolates downscaled RACMO products to times and coordinates using surface splines  
- [`racmo_interp_firn_height.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/racmo_interp_firn_height.md) - Interpolates and extrapolates firn heights to times and coordinates using surface splines  
- [`racmo_interp_mean.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/racmo_interp_mean.md) - Interpolates the mean of downscaled RACMO products to spatial coordinates using surface splines  
- [`regress_model.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/regress_model.md) - Estimates a modeled time series for extrapolation by least-squares regression  
- [`time.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/time.rst) - Utilities for calculating time operations  
- [`utilities.py`](https://github.com/tsutterley/SMBcorr/blob/master/doc/source/user_guide/utilities.rst) - Download and management utilities for syncing time and auxiliary files  

#### Dependencies
 - [numpy: Scientific Computing Tools For Python](https://www.numpy.org)  
 - [scipy: Scientific Tools for Python](https://www.scipy.org/)  
 - [pyproj: Python interface to PROJ library](https://pypi.org/project/pyproj/)  
 - [netCDF4: Python interface to the netCDF C library](https://unidata.github.io/netcdf4-python/)  
 - [scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/index.html)  

#### Download
The program homepage is:  
https://github.com/tsutterley/SMBcorr  
A zip archive of the latest version is available directly at:  
https://github.com/tsutterley/SMBcorr/archive/master.zip  

#### Disclaimer
This program is not sponsored or maintained by the Universities Space Research Association (USRA) or NASA.  It is provided here for your convenience but _with no guarantees whatsoever_.

#### License
The content of this project is licensed under the [Creative Commons Attribution 4.0 Attribution license](https://creativecommons.org/licenses/by/4.0/) and the source code is licensed under the [MIT license](LICENSE).
