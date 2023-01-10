#!/usr/bin/env python
u"""
version.py (04/2021)
Gets version number of a package
"""
from pkg_resources import get_distribution

# get version
version = get_distribution("SMBcorr").version
# append "v" before the version
full_version = "v{0}".format(version)
# get project name
project_name = get_distribution("SMBcorr").project_name
