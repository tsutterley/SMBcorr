"""
_mar_providers.py (04/2026)
Create MAR providers for SMBcorr database
"""

import re
import copy
import json
import inspect
import pathlib
import posixpath
import argparse
import xarray as xr
import SMBcorr.utilities

# current file path
filename = inspect.getframeinfo(inspect.currentframe()).filename
filepath = pathlib.Path(filename).absolute().parent


# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Create MAR providers for SMBcorr database"
            """,
        fromfile_prefix_chars="@",
    )
    # command line parameters
    parser.add_argument("url", type=str, help="MAR ftp url")
    parser.add_argument(
        "--directory",
        "-D",
        type=pathlib.Path,
        help="MAR directory for checking files",
    )
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

    # parse FTP url
    ftp = SMBcorr.utilities.urlparse.urlparse(args.url)

    # short names for regions
    regions = dict(Antarctica="ais", Greenland="gris")
    # available MAR models
    models = dict(Antarctica=[], Greenland=[])
    # MAR
    models["Antarctica"].append("MARv3.12")
    models["Antarctica"].append("MARv3.14")
    models["Greenland"].append("MARv3.9-ERA")
    models["Greenland"].append("MARv3.10-ERA")
    models["Greenland"].append("MARv3.11-NCEP")
    models["Greenland"].append("MARv3.11-ERA")
    models["Greenland"].append("MARv3.11.2-ERA-6km")
    models["Greenland"].append("MARv3.11.2-ERA-7.5km")
    models["Greenland"].append("MARv3.11.2-ERA-10km")
    models["Greenland"].append("MARv3.11.2-ERA-15km")
    models["Greenland"].append("MARv3.11.2-ERA-20km")
    models["Greenland"].append("MARv3.11.2-NCEP-20km")
    models["Greenland"].append("MARv3.11.5-ERA-6km")
    models["Greenland"].append("MARv3.11.5-ERA-10km")
    models["Greenland"].append("MARv3.11.5-ERA-15km")
    models["Greenland"].append("MARv3.11.5-ERA-20km")
    models["Greenland"].append("MARv3.14-ERA-1km")
    models["Greenland"].append("MARv3.14-ERA-10km")
    # MAR model subdirectories
    subdirectories = dict(Antarctica={}, Greenland={})
    subdirectories["Antarctica"]["MARv3.12"] = []
    subdirectories["Antarctica"]["MARv3.14"] = []
    subdirectories["Greenland"]["MARv3.9-ERA"] = [
        "ERA_1958-2018_10km",
        "daily_10km",
    ]
    subdirectories["Greenland"]["MARv3.10-ERA"] = [
        "ERA_1958-2019-15km",
        "daily_15km",
    ]
    subdirectories["Greenland"]["MARv3.11-NCEP"] = [
        "NCEP1_1948-2020_20km",
        "daily_20km",
    ]
    subdirectories["Greenland"]["MARv3.11-ERA"] = [
        "ERA_1958-2019-15km",
        "daily_15km",
    ]
    subdirectories["Greenland"]["MARv3.11.2-ERA-6km"] = ["6km_ERA5"]
    subdirectories["Greenland"]["MARv3.11.2-ERA-7.5km"] = ["7.5km_ERA5"]
    subdirectories["Greenland"]["MARv3.11.2-ERA-10km"] = ["10km_ERA5"]
    subdirectories["Greenland"]["MARv3.11.2-ERA-15km"] = ["15km_ERA5"]
    subdirectories["Greenland"]["MARv3.11.2-ERA-20km"] = ["20km_ERA5"]
    subdirectories["Greenland"]["MARv3.11.2-NCEP-20km"] = ["20km_NCEP1"]
    subdirectories["Greenland"]["MARv3.11.5-ERA-6km"] = ["ERA5_1950-2021-6km"]
    subdirectories["Greenland"]["MARv3.11.5-ERA-10km"] = [
        "ERA5_1950-2020-10km",
        "daily_10km",
    ]
    subdirectories["Greenland"]["MARv3.11.5-ERA-15km"] = ["15km_ERA5"]
    subdirectories["Greenland"]["MARv3.11.5-ERA-20km"] = ["20km_ERA5"]
    subdirectories["Greenland"]["MARv3.14-ERA-1km"] = ["ERA5-1km-monthly"]
    subdirectories["Greenland"]["MARv3.14-ERA-10km"] = ["ERA5-10km-daily"]

    # MAR data variables of interest
    default_variables = ["SMB", "ZN4", "ZN5", "ZN6"]

    # create output dictionary
    output = {}
    for model_region in models.keys():
        region = regions[model_region]
        for model_version in models[model_region]:
            if args.verbose:
                print(f"Processing {model_version} in {model_region}")
            subdirectory = subdirectories[model_region][model_version]
            # get ftp subdirectory for model version
            match_object = re.match(r"((MARv\d+\.\d+)(.\d+)?)", model_version)
            # check if model version is in ftp path
            local_version = match_object.group(0)
            short_version = match_object.group(2)
            # full path for local files
            model_directory = posixpath.join(
                "MAR", local_version, model_region, *subdirectory
            )
            # try to get list of files from ftp
            url = [
                ftp.netloc,
                ftp.path,
                short_version,
                model_region,
                *subdirectory,
            ]
            if args.verbose:
                print(f"\tQuerying {posixpath.join(*url)}")
            filenames, mtimes = SMBcorr.utilities.ftp_list(
                url, basename=True, pattern=r"\d+", sort=True
            )
            how = "ftp"
            if not filenames:
                how = "local"
                # if not available from ftp: check local directory
                pattern = rf"{short_version}(.*?)-(\d+)(_subset)?.nc$"
                directory = args.directory.joinpath(model_directory)
                print(f"\tSearching: {directory}") if args.verbose else None
                if directory.exists():
                    filenames = [
                        i.name
                        for i in directory.iterdir()
                        if re.match(pattern, i.name)
                    ]
            # skip model version if no files found (on ftp or locally)
            if how == "local" and filenames:
                ds = xr.open_dataset(directory.joinpath(filenames[0]))
                variables = [v for v in default_variables if v in ds.variables]
            elif how == "ftp" and filenames:
                variables = copy.copy(default_variables)
            elif not filenames:
                print("\tNo files found") if args.verbose else None
                continue
            # relative path to model files
            model_files = [
                posixpath.join(model_directory, f) for f in filenames
            ]
            if args.verbose:
                print(f"\t{len(model_files)} files found")
            # build output dictionary for model version and region
            if model_version in output:
                output[model_version][region] = {}
            else:
                output[model_version] = {region: {}}
            # append to output dictionary
            output[model_version][region]["model_file"] = sorted(model_files)
            output[model_version][region]["variables"] = variables
            output[model_version][region]["format"] = "MAR"

    # writing model parameters to JSON database file
    json_file = filepath.joinpath("MAR.json")
    print(f"Writing to {json_file}") if args.verbose else None
    with open(json_file, "w") as fid:
        indent = 4 if args.pretty else None
        json.dump(output, fid, indent=indent, sort_keys=True)


if __name__ == "__main__":
    main()
