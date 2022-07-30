===================
append_SMB_ATL11.py
===================

- Interpolates daily model firn estimates to the coordinates of an ATL11 file

`Source code`__

.. __: https://github.com/tsutterley/SMBcorr/blob/main/scripts/append_SMB_ATL11.py


Calling Sequence
################

.. argparse::
    :filename: ../../scripts/append_SMB_ATL11.py
    :func: arguments
    :prog: append_SMB_ATL11.py
    :nodescription:
    :nodefault:

    --model -m : @replace
        Regional firn model to run `(see list of models) <#models>`_

Models
######

* Greenland

    - ``MARv3.11.5-ERA-10km``
    - ``MARv3.11.5-ERA-15km``
    - ``MARv3.11.5-ERA-20km``
    - ``RACMO2.3p2-FGRN055``
    - ``GSFC-fdm-v1.2``
* Antarctica

    - ``GSFC-fdm-v1``
    - ``GSFC-fdm-v1.1``
