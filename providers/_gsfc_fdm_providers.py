"""
_gsfc_fdm_providers.py (04/2026)
Create GSFC-fdm providers for SMBcorr database
"""

import re
import json
import inspect
import pathlib
import posixpath
import argparse

# current file path
filename = inspect.getframeinfo(inspect.currentframe()).filename
filepath = pathlib.Path(filename).absolute().parent


# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Create GSFC-fdm providers for SMBcorr database"
            """,
        fromfile_prefix_chars="@",
    )
    # command line parameters
    parser.add_argument(
        "--pretty", "-p", action="store_true", help="Pretty print the json file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    return parser


def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args, _ = parser.parse_known_args()

    # available models
    models = dict(ais=[], gris=[])
    # GSFC-fdm
    models["gris"] = []
    models["gris"].append("GSFC-fdm-v0")
    models["gris"].append("GSFC-fdm-v1")
    models["gris"].append("GSFC-fdm-v1.0")
    models["gris"].append("GSFC-fdm-v1.1")
    models["gris"].append("GSFC-fdm-v1.2.1")
    models["ais"] = []
    models["ais"].append("GSFC-fdm-v0")
    models["ais"].append("GSFC-fdm-v1")
    models["ais"].append("GSFC-fdm-v1.1")
    models["ais"].append("GSFC-fdm-v1.2.1")

    # regular expression pattern for extracting version
    rx = re.compile(r"GSFC-fdm-((v\d+)(\.\d+)?(\.\d+)?)$")
    # create output dictionary
    output = {}
    for region in models.keys():
        for model_version in models[region]:
            # get GSFC-fdm version and major version
            fdm_version = rx.match(model_version).group(1)
            # keyword arguments for MERRA-2 interpolation programs
            if fdm_version in ("v0",):
                version = rx.match(model_version).group(2)
                # netCDF4 variable names
                variables = ["FAC"]
                # netCDF4 file
                hybrid_file = f"gsfc_FAC_{region}.nc"
            elif fdm_version in ("v0", "v1", "v1.0"):
                version = rx.match(model_version).group(2)
                # netCDF4 variable names
                variables = ["FAC", "cum_smb_anomaly", "height"]
                # netCDF4 file
                hybrid_file = f"gsfc_fdm_{version}_{region}.nc"
            else:
                version = fdm_version.replace(".", "_")
                # netCDF4 variable names
                variables = ["FAC", "SMB_a", "h_a"]
                # netCDF4 file
                hybrid_file = f"gsfc_fdm_{version}_{region}.nc"
            # relative path to model file
            filename = posixpath.join("GSFC-fdm", fdm_version, hybrid_file)
            # build output dictionary for model version and region
            if model_version in output:
                output[model_version][region] = {}
            else:
                output[model_version] = {region: {}}
            # append to output dictionary
            output[model_version][region]["model_file"] = filename
            output[model_version][region]["variables"] = variables
            output[model_version][region]["format"] = "GSFC-fdm"

    # writing model parameters to JSON database file
    json_file = filepath.joinpath("GSFC.json")
    print(f"Writing to {json_file}") if args.verbose else None
    with open(json_file, "w") as fid:
        indent = 4 if args.pretty else None
        json.dump(output, fid, indent=indent, sort_keys=True)


if __name__ == "__main__":
    main()
