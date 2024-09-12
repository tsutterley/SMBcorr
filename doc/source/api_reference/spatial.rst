=======
spatial
=======

Utilities for reading, writing and operating on spatial data

 - Can read ascii, netCDF4, HDF5 or geotiff files
 - Can output to ascii, netCDF4, HDF5 or geotiff files

Calling Sequence
----------------

Reading a netCDF4 file

.. code-block:: python

    import SMBcorr.spatial
    dinput = SMBcorr.spatial.from_netCDF4(path_to_netCDF4_file)

Reading a HDF5 file

.. code-block:: python

    import SMBcorr.spatial
    dinput = SMBcorr.spatial.from_HDF5(path_to_HDF5_file)

`Source code`__

.. __: https://github.com/tsutterley/SMBcorr/blob/main/SMBcorr/spatial.py

General Methods
===============


.. autofunction:: SMBcorr.spatial.case_insensitive_filename

.. autofunction:: SMBcorr.spatial.data_type

.. autofunction:: SMBcorr.spatial.from_file

.. autofunction:: SMBcorr.spatial.from_ascii

.. autofunction:: SMBcorr.spatial.from_netCDF4

.. autofunction:: SMBcorr.spatial.from_HDF5

.. autofunction:: SMBcorr.spatial.from_geotiff

.. autofunction:: SMBcorr.spatial.to_ascii

.. autofunction:: SMBcorr.spatial.to_netCDF4

.. autofunction:: SMBcorr.spatial._drift_netCDF4

.. autofunction:: SMBcorr.spatial._grid_netCDF4

.. autofunction:: SMBcorr.spatial._time_series_netCDF4

.. autofunction:: SMBcorr.spatial.to_HDF5

.. autofunction:: SMBcorr.spatial.to_geotiff

.. autofunction:: SMBcorr.spatial.expand_dims

.. autofunction:: SMBcorr.spatial.default_field_mapping

.. autofunction:: SMBcorr.spatial.inverse_mapping

.. autofunction:: SMBcorr.spatial.convert_ellipsoid

.. autofunction:: SMBcorr.spatial.compute_delta_h

.. autofunction:: SMBcorr.spatial.wrap_longitudes

.. autofunction:: SMBcorr.spatial.to_cartesian

.. autofunction:: SMBcorr.spatial.to_sphere

.. autofunction:: SMBcorr.spatial.to_geodetic

.. autofunction:: SMBcorr.spatial._moritz_iterative

.. autofunction:: SMBcorr.spatial._bowring_iterative

.. autofunction:: SMBcorr.spatial._zhu_closed_form

.. autofunction:: SMBcorr.spatial.scale_areas

.. autofunction:: SMBcorr.spatial.build_tree

.. autofunction:: SMBcorr.spatial.find_valid_triangulation
