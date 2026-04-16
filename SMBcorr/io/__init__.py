"""
Input/output functions for reading and writing SMB and firn data
"""

import os
from . import GSFCfdm
from . import MAR
from . import RACMO
from . import dataset
from .model import model, load_database

# set environmental variable for anonymous s3 access
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
