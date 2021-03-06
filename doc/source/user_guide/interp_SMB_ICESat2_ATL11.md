interp_SMB_ICESat2_ATL11.py
===========================

- Interpolates daily firn model estimates to the times and locations of ICESat-2 ATL11 annual land ice height data
    * Snow Height Change (zsurf)
    * Snow Height Change due to Compaction (zfirn)
    * Snow Height Change due to Surface Melt (zmelt)
    * Snow Height Change due to Surface Mass Balance (zsmb)
    * Snow Height Change due to Surface Accumulation (zaccum)
    * Cumulative Surface Mass Balance (SMB)
- Interpolates firn model estimates for both along-track and across-track locations

#### Calling Sequence
```bash
python interp_SMB_ICESat2_ATL11.py --directory <path_to_directory> input_file
```
[Source code](https://github.com/tsutterley/SMBcorr/blob/master/scripts/interp_SMB_ICESat2_ATL11.py)

#### Inputs
1. `input_file`: input ICESat-2 ATL11 file

#### Command Line Options
- `-D X`, `--directory X`: Working data directory
- `-m X`, `--model X`: Regional firn model to run
    * Greenland
        - `'MARv3.9-ERA'`
        - `'MARv3.10-ERA'`
        - `'MARv3.11-NCEP'`
        - `'MARv3.11-ERA'`
        - `'MARv3.11.2-ERA-6km'`
        - `'MARv3.11.2-ERA-7.5km'`
        - `'MARv3.11.2-ERA-10km'`
        - `'MARv3.11.2-ERA-15km'`
        - `'MARv3.11.2-ERA-20km'`
        - `'MARv3.11.2-NCEP-20km'`
        - `'MARv3.11.5-ERA-6km'`
        - `'MARv3.11.5-ERA-10km'`
        - `'MARv3.11.5-ERA-15km'`
        - `'MARv3.11.5-ERA-20km'`
        - `'RACMO2.3-XGRN11'`
        - `'RACMO2.3p2-XGRN11'`
        - `'RACMO2.3p2-FGRN055'`
        - `'GSFC-fdm-v0'`
        - `'GSFC-fdm-v1'`
        - `'GSFC-fdm-v1.0'`
        - `'GSFC-fdm-v1.1'`
    * Antarctica
        - `'GSFC-fdm-v0'`
        - `'GSFC-fdm-v1'`
- `-C`, `--crossovers`: Run ATL11 Crossovers
- `-V`, `--verbose`: Output information about each created file
- `-M X`, `--mode X`: Permission mode of output file
