#!/usr/bin/env python
u"""
scale_areas.py
Written by Tyler Sutterley (07/2022)
Calculates area scaling factors for a polar stereographic projection
    including special case of at the exact pole

Scaling factor is defined as:
    scale = 1/k^2, where:

    ecc2 is the Earth's eccentricity (calculated from ellipsoidal flattening)
    ecc = sqrt(ecc2)

    m = cos(lat)/sqrt(1 - ecc2*sin(lat)^2)
    t = tan(pi/4 - lat/2)/((1 - ecc*sin(lat))/(1 + ecc*sin(lat)))^(ecc/2)
    mref is m at the reference latitude
    tref is t at the reference latitude

    k = (mref/m)*(t/tref)

For the special case of the pole:
    kp = 1/2*mref*sqrt((1+ecc)^(1+ecc)*(1-ecc)^(1-ecc))/tref

INPUTS:
    lat: input latitude array (degrees North)

OPTIONS:
    flat: ellipsoidal flattening (default WGS84)
    ref: Standard parallel (latitude with no distortion, e.g. +70/-71)

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python (https://numpy.org)

REFERENCES:
    Snyder, J P (1982) Map Projections used by the U.S. Geological Survey
        Forward formulas for the ellipsoid.  Geological Survey Bulletin 1532,
        U.S. Government Printing Office.
    JPL Technical Memorandum 3349-85-101

UPDATE HISTORY
    Updated 07/2022: updated docstrings to numpy documentation format
    Updated 10/2021: add pole case in stereographic area scale calculation
    Updated 12/2020: added function docstrings, updated comments for release
    Updated 06/2014: updated comments
    Written 06/2013
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def scale_areas(lat, flat=1.0/298.257223563, ref=70.0):
    """
    Calculates area scaling factors for a polar stereographic projection
    including special case of at the exact pole

    Parameters
    ----------
    lat: float,
        latitude (degrees north)
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening
    ref: float, default 70.0
        reference latitude (true scale latitude)

    Returns
    -------
    scale: float
        area scaling factors at input latitudes

    References
    ----------
    .. [1] Snyder, J P (1982) Map Projections used by the U.S. Geological Survey
        Forward formulas for the ellipsoid.  Geological Survey Bulletin
        1532, U.S. Government Printing Office.
    .. [2] JPL Technical Memorandum 3349-85-101
    """
    #-- convert latitude from degrees to positive radians
    theta = np.abs(lat)*np.pi/180.0
    #-- convert reference latitude from degrees to positive radians
    theta_ref = np.abs(ref)*np.pi/180.0
    #-- square of the eccentricity of the ellipsoid
    #-- ecc2 = (1-b**2/a**2) = 2.0*flat - flat^2
    ecc2 = 2.0*flat - flat**2
    #-- eccentricity of the ellipsoid
    ecc = np.sqrt(ecc2)
    #-- calculate ratio at input latitudes
    m = np.cos(theta)/np.sqrt(1.0 - ecc2*np.sin(theta)**2)
    t = np.tan(np.pi/4.0 - theta/2.0)/((1.0 - ecc*np.sin(theta)) / \
        (1.0 + ecc*np.sin(theta)))**(ecc/2.0)
    #-- calculate ratio at reference latitude
    mref = np.cos(theta_ref)/np.sqrt(1.0 - ecc2*np.sin(theta_ref)**2)
    tref = np.tan(np.pi/4.0 - theta_ref/2.0)/((1.0 - ecc*np.sin(theta_ref)) / \
        (1.0 + ecc*np.sin(theta_ref)))**(ecc/2.0)
    #-- distance scaling
    k = (mref/m)*(t/tref)
    kp = 0.5*mref*np.sqrt(((1.0+ecc)**(1.0+ecc))*((1.0-ecc)**(1.0-ecc)))/tref
    #-- area scaling
    scale = np.where(np.isclose(theta,np.pi/2.0),1.0/(kp**2),1.0/(k**2))
    return scale
