#!/usr/bin/env python
u"""
scale_areas.py
Written by Tyler Sutterley (12/2020)
Calculates area scaling factors for a polar stereographic projection

Scaling factor is defined as:
    scale = 1/k^2, where:

    ecc2 is the Earth's eccentricity (calculated from ellipsoidal flattening)
    ecc = sqrt(ecc2)

    m = cos(lat)/sqrt(1 - ecc2*sin(lat)^2)
    t = tan(pi/4 - lat/2)/((1 - ecc*sin(lat))/(1 + ecc*sin(lat)))^(ecc/2)
    mref is m at the reference latitude
    tref is t at the reference latitude

    k = (mref/m)*(t/tref)

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
    Updated 12/2020: added function docstrings, updated comments for release
    Updated 06/2014: updated comments
    Written 06/2013
"""
import numpy as np

def scale_areas(lat, flat=1.0/298.257223563, ref=70.0):
    """
    Calculates area scaling factors for a polar stereographic projection

    Arguments
    ---------
    lat: latitude

    Keyword arguments
    -----------------
    flat: ellipsoidal flattening
    ref: reference latitude

    Returns
    -------
    scale: area scaling factors at input latitudes
    """
    #-- convert latitude from degrees to positive radians
    theta = np.abs(lat)*np.pi/180.0
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
    #-- area scaling
    k = (mref/m)*(t/tref)
    scale = 1.0/(k**2)
    return scale
