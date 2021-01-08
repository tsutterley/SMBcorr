===========================
interp_SMB_ICESat2_ATL15.py
===========================

- Interpolates daily firn model estimates to the times and locations of ICESat-2 ATL15 gridded land ice height change data

    * Relative Snow Height Change (``delta_h_surf``)
    * Relative Snow Height Change due to Compaction (``delta_h_firn``)
    * Relative Snow Height Change due to Surface Melt (``delta_h_melt``)
    * Relative Snow Height Change due to Surface Mass Balance (``delta_h_smb``)
    * Relative Snow Height Change due to Surface Accumulation (``delta_h_accum``)

`Source code`__

.. __: https://github.com/tsutterley/SMBcorr/blob/main/scripts/interp_SMB_ICESat2_ATL15.py


Calling Sequence
################

.. argparse::
    :filename: interp_SMB_ICESat2_ATL15.py
    :func: arguments
    :prog: interp_SMB_ICESat2_ATL15.py
    :nodescription:
    :nodefault:

    --model -m : @replace
        Regional firn model to run `(see list of models) <#models>`_

Models
######

* Greenland

    - ``MARv3.9-ERA``
    - ``MARv3.10-ERA``
    - ``MARv3.11-NCEP``
    - ``MARv3.11-ERA``
    - ``MARv3.11.2-ERA-6km``
    - ``MARv3.11.2-ERA-7.5km``
    - ``MARv3.11.2-ERA-10km``
    - ``MARv3.11.2-ERA-15km``
    - ``MARv3.11.2-ERA-20km``
    - ``MARv3.11.2-NCEP-20km``
    - ``MARv3.11.5-ERA-6km``
    - ``MARv3.11.5-ERA-10km``
    - ``MARv3.11.5-ERA-15km``
    - ``MARv3.11.5-ERA-20km``
    - ``RACMO2.3-XGRN11``
    - ``RACMO2.3p2-XGRN11``
    - ``RACMO2.3p2-FGRN055``
    - ``GSFC-fdm-v0``
    - ``GSFC-fdm-v1``
    - ``GSFC-fdm-v1.0``
    - ``GSFC-fdm-v1.1``
    - ``GSFC-fdm-v1.2``
    - ``GSFC-fdm-v1.2.1``
* Antarctica

    - ``GSFC-fdm-v0``
    - ``GSFC-fdm-v1``
    - ``GSFC-fdm-v1.1``
    - ``GSFC-fdm-v1.2.1``
