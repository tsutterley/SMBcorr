#!/usr/bin/env python
u"""
regress_model.py
Written by Tyler Sutterley (07/2019)
Estimates a modeled time series for extrapolation by least-squares regression

CALLING SEQUENCE:
    d_out = regress_model(t_in, d_in, t_out, ORDER=2,
        CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=t_in[0])

INPUTS:
    t_in: input time array (year-decimal)
    d_in: input data array
    t_out: output time array for calculating modeled values (year-decimal)

OUTPUTS:
    model: reconstructed time series

OPTIONS:
    ORDER: polynomial order in design matrix (default quadratic)
    CYCLES: harmonic cycles in design matrix (year-decimal)
    RELATIVE: set polynomial fits relative to some value

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python (http://www.numpy.org)

UPDATE HISTORY:
    Written 07/2019
"""
import numpy as np

#-- PURPOSE: calculate a regression model for extrapolating values
def regress_model(t_in, d_in, t_out, ORDER=2, CYCLES=None, RELATIVE=None):

    #-- remove singleton dimensions
    t_in = np.squeeze(t_in)
    d_in = np.squeeze(d_in)
    t_out = np.squeeze(t_out)
    #-- check dimensions of output
    if (np.ndim(t_out) == 0):
        t_out = np.array([t_out])

    #-- CREATING DESIGN MATRIX FOR REGRESSION
    DMAT = []
    MMAT = []
    #-- add polynomial orders (0=constant, 1=linear, 2=quadratic)
    for o in range(ORDER+1):
        DMAT.append((t_in-RELATIVE)**o)
        MMAT.append((t_out-RELATIVE)**o)
    #-- add cyclical terms (0.5=semi-annual, 1=annual)
    for c in CYCLES:
        DMAT.append(np.sin(2.0*np.pi*t_in/np.float(c)))
        DMAT.append(np.cos(2.0*np.pi*t_in/np.float(c)))
        MMAT.append(np.sin(2.0*np.pi*t_out/np.float(c)))
        MMAT.append(np.cos(2.0*np.pi*t_out/np.float(c)))

    #-- Calculating Least-Squares Coefficients
    #-- Standard Least-Squares fitting (the [0] denotes coefficients output)
    beta_mat = np.linalg.lstsq(np.transpose(DMAT), d_in, rcond=-1)[0]

    #-- return modeled time-series
    return np.dot(np.transpose(MMAT),beta_mat)
