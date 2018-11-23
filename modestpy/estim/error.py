# -*- coding: utf-8 -*-

"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.
This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import pandas as pd
import numpy as np


def calc_err(result, ideal, forgetting=False, ftype='RMSE'):
    """
    Returns a dictionary with Normalised Root Mean Square Errors
    for each variable in ideal. The dictionary contains also a key
    ``tot`` with a sum of all errors

    If ``forgetting`` = ``True``, the error function is multiplied
    by a linear function with y=0 for the first time step and y=1
    for the last time step. In other words, the newer the error,
    the most important it is. It is used in the UKF tuning,
    since UKF needs some time to converge and the path it takes
    to converge should not be taken into account.

    :param result: DataFrame
    :param ideal: DataFrame
    :param forgetting: bool, if True, the older the error the lower weight
    :param string ftype: Cost function type, currently 'RMSE' or 'NRMSE'
    :return: dictionary
    """
    logger = logging.getLogger("error")

    for v in ideal.columns:
        assert v in result.columns, \
            'Columns in ideal and model solution not matching: {} vs. {}' \
            .format(ideal.columns, result.columns)

    # Get original variable names
    variables = list(ideal.columns)

    # Rename columns
    ideal = ideal.rename(columns=lambda x: x + '_ideal')
    result = result.rename(columns=lambda x: x + '_model')

    # Concatenate and interpolate
    comp = pd.concat([ideal, result], sort=False)
    comp = comp.sort_index().interpolate().bfill()

    if forgetting:
        forget_weights = np.linspace(0., 1., len(comp.index))
    else:
        forget_weights = None

    # Calculate error
    error = dict()
    for v in variables:
        comp[v + '_se'] = np.square(comp[v + '_ideal'] - comp[v + '_model'])

        if forgetting:
            # Cost function multiplied by a linear function
            # (0 for the oldest timestep, 1 for the newest)
            comp[v + '_se'] = comp[v + '_se'] * forget_weights

        mse = comp[v + '_se'].mean()  # Mean square error
        rmse = mse ** 0.5  # Root mean square error

        ideal_mean = comp[v + '_ideal'].abs().mean()

        if ideal_mean != 0.:
            nrmse = rmse / ideal_mean  # Normalized root mean square error
        else:
            msg = "Ideal solution for variable '{}' is null, " \
                  "so the error cannot be normalized.".format(v)
            logger.error(msg)
            raise ZeroDivisionError(msg)

        # Choose error function type
        if ftype == 'NRMSE':
            error[v] = nrmse
        elif ftype == 'RMSE':
            error[v] = rmse
        else:
            raise ValueError('Cost function type unknown: {}'.format(ftype))

        logger.debug('Calculated partial error ({}) = {}'
                     .format(ftype, error[v]))

    # Calculate total error (sum of partial errors)
    assert 'tot' not in error, "'tot' is not an allowed name " \
                               "for output variables..."
    error['tot'] = 0
    for v in variables:
        error['tot'] += error[v]

    logger.debug('Calculated total error ({}) = {}'.format(ftype,
                                                           error['tot']))

    return error


