======================
mar_interp_seasonal.py
======================

- Interpolates and extrapolates seasonal MAR products to times and coordinates using surface splines
- Seasonal files are climatology files for each day of the year

`Source code`__

.. __: https://github.com/tsutterley/SMBcorr/blob/main/SMBcorr/mar_interp_seasonal.py

General Methods
===============

.. autofunction:: SMBcorr.interpolate_mar_seasonal

.. autofunction:: SMBcorr.mar_interp_seasonal.find_valid_triangulation

Directories
###########

- ``<path_to_mar>/MARv3.11.2/Greenland/7.5km_ERA5/``
- ``<path_to_mar>/MARv3.11.2/Greenland/10km_ERA5/``
- ``<path_to_mar>/MARv3.11.2/Greenland/15km_ERA5/``
- ``<path_to_mar>/MARv3.11.2/Greenland/20km_ERA5/``
- ``<path_to_mar>/MARv3.11.2/Greenland/20km_NCEP1/``
- ``<path_to_mar>/MARv3.11/Greenland/ERA_1958-2019-15km/daily_15km/``
- ``<path_to_mar>/MARv3.11/Greenland/NCEP1_1948-2020_20km/daily_20km/``
- ``<path_to_mar>/MARv3.10/Greenland/NCEP1_1948-2019_20km/daily_20km/``
- ``<path_to_mar>/MARv3.9/Greenland/ERA_1958-2018_10km/daily_10km/``
