#!/usr/bin/env python
u"""
regress_model.py
Written by Tyler Sutterley (07/2022)
Estimates a modeled time series for extrapolation by least-squares regression

CALLING SEQUENCE:
    d_out = regress_model(t_in, d_in, t_out, ORDER=2,
        CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=t_in[0])

INPUTS:
    t_in: input time array
    d_in: input data array
    t_out: output time array for calculating modeled values

OUTPUTS:
    d_out: reconstructed time series

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python (https://numpy.org)

UPDATE HISTORY:
    Updated 07/2022: updated docstrings to numpy documentation format
    Updated 05/2021: define int/float precision to prevent deprecation warning
    Updated 07/2020: added function docstrings
    Written 07/2019
"""
import numpy as np

def regress_model(t_in, d_in, t_out,
    ORDER=2,
    CYCLES=[0.25, 0.5, 1.0, 2.0, 4.0, 5.0],
    RELATIVE=Ellipsis):
    """
    Fits a synthetic signal to data over a time period by
        ordinary or weighted least-squares

    Parameters
    ----------
    t_in: float
        input time array
    d_in: float
        input data array
    ORDER: int, default 2
        maximum polynomial order in fit

            * ``0``: constant
            * ``1``: linear
            * ``2``: quadratic
    CYCLES: list, default [0.25,0.5,1.0,2.0,4.0,5.0]
        list of cyclical terms
    RELATIVE: float or List, default Ellipsis
        Epoch for calculating relative dates

            - float: use exact value as epoch
            - list: use mean from indices of available times
            - ``Ellipsis``: use mean of all available times

    Returns
    -------
    d_out: float
        reconstructed time series
    """

    # remove singleton dimensions
    t_in = np.squeeze(t_in)
    d_in = np.squeeze(d_in)
    t_out = np.squeeze(t_out)
    # check dimensions of output
    t_out = np.atleast_1d(t_out)
    # calculate epoch for calculating relative times
    if isinstance(RELATIVE, (list, np.ndarray)):
        t_rel = t_in[RELATIVE].mean()
    elif isinstance(RELATIVE, (float, int, np.float64, np.int_)):
        t_rel = np.copy(RELATIVE)
    elif (RELATIVE == Ellipsis):
        t_rel = t_in[RELATIVE].mean()

    # create design matrix based on polynomial order and harmonics
    DMAT = []
    MMAT = []
    # add polynomial orders (0=constant, 1=linear, 2=quadratic)
    for o in range(ORDER+1):
        DMAT.append((t_in-t_rel)**o)
        MMAT.append((t_out-t_rel)**o)
    # add cyclical terms (0.5=semi-annual, 1=annual)
    for c in CYCLES:
        DMAT.append(np.sin(2.0*np.pi*t_in/np.float64(c)))
        DMAT.append(np.cos(2.0*np.pi*t_in/np.float64(c)))
        MMAT.append(np.sin(2.0*np.pi*t_out/np.float64(c)))
        MMAT.append(np.cos(2.0*np.pi*t_out/np.float64(c)))

    # Calculating Least-Squares Coefficients
    # Standard Least-Squares fitting (the [0] denotes coefficients output)
    beta_mat = np.linalg.lstsq(np.transpose(DMAT), d_in, rcond=-1)[0]

    # return modeled time-series
    return np.dot(np.transpose(MMAT), beta_mat)
