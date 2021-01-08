========================
interp_SMB_correction.py
========================

- Interpolates SMB and firn model estimates for correcting surface elevations

    * Snow Height Change (``zsurf``)
    * Snow Height Change due to Compaction (``zfirn``)
    * Snow Height Change due to Surface Melt (``zmelt``)
    * Snow Height Change due to Surface Mass Balance (``zsmb``)
    * Snow Height Change due to Surface Accumulation (``zaccum``)
    * Cumulative Surface Mass Balance (``SMB``)

`Source code`__

.. __: https://github.com/tsutterley/SMBcorr/blob/main/scripts/interp_SMB_correction.py


Calling Sequence
################

.. argparse::
    :filename: interp_SMB_correction.py
    :func: arguments
    :prog: interp_SMB_correction.py
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
