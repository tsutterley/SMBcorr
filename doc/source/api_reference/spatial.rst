==========
spatial.py
==========

Spatial data class for reading, writing and processing spatial data

Calling Sequence
================

Reading a netCDF4 file

.. code-block:: python

    from SMBcorr.spatial import spatial
    grid = spatial().from_netCDF4(path_to_netCDF4_file)

Reading a HDF5 file

.. code-block:: python

    from SMBcorr.spatial import spatial
    grid = spatial().from_HDF5(path_to_HDF5_file)

`Source code`__

.. __: https://github.com/tsutterley/SMBcorr/blob/main/SMBcorr/spatial.py

General Attributes and Methods
==============================

.. autoclass:: SMBcorr.spatial
   :members:
