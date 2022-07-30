============
utilities.py
============

Download and management utilities for syncing time and auxiliary files

 - Can list a directory on a ftp host
 - Can download a file from a ftp or http host
 - Can download a file from CDDIS via https when NASA Earthdata credentials are supplied
 - Checks ``MD5`` or ``sha1`` hashes between local and remote files

`Source code`__

.. __: https://github.com/tsutterley/SMBcorr/blob/main/SMBcorr/utilities.py

General Methods
===============

.. autofunction:: SMBcorr.utilities.get_data_path

.. autofunction:: SMBcorr.utilities.get_hash

.. autofunction:: SMBcorr.utilities.url_split

.. autofunction:: SMBcorr.utilities.get_unix_time

.. autofunction:: SMBcorr.utilities.isoformat

.. autofunction:: SMBcorr.utilities.even

.. autofunction:: SMBcorr.utilities.ceil

.. autofunction:: SMBcorr.utilities.copy

.. autofunction:: SMBcorr.utilities.check_ftp_connection

.. autofunction:: SMBcorr.utilities.ftp_list

.. autofunction:: SMBcorr.utilities.from_ftp

.. autofunction:: SMBcorr.utilities.http_list

.. autofunction:: SMBcorr.utilities.from_http

.. autofunction:: SMBcorr.utilities.build_opener

.. autofunction:: SMBcorr.utilities.gesdisc_list

.. autofunction:: SMBcorr.utilities.cmr_filter_json

.. autofunction:: SMBcorr.utilities.cmr

.. autofunction:: SMBcorr.utilities.build_request
