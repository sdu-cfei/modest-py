"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

from modestpy.log_init import LogInit
LOG_INIT = LogInit(__name__)
LOGGER = LOG_INIT.get_logger()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_err(result, ideal, forgetting=False):
    """
    Returns a dictionary with Normalised Root Mean Square Errors for each variable in ideal.
    The dictionary contains also a key ``tot`` with a sum of all errors

    If ``forgetting`` = ``True``, the error function is multiplied by a linear function with y=0 for the first time step
    and y=1 for the last time step. In other words, the newer the error, the most important it is. It is used
    in the UKF tuning, since UKF needs some time to converge and the path it takes to converge should not be taken
    into account.

    :param result: DataFrame
    :param ideal: DataFrame
    :param forgetting: bool, if True, the older the error the lower weight (used in Kalman filter tuning)
    :return: dictionary
    """

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
    comp = pd.concat([ideal, result])
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
            # Cost function multiplied by a linear function (0 for the oldest timestep, 1 for the newest)
            comp[v + '_se'] = comp[v + '_se'] * forget_weights

        mse = comp[v + '_se'].mean()  # Mean square error
        rmse = mse ** 0.5  # Root mean square error

        ideal_mean = comp[v + '_ideal'].abs().mean()

        if ideal_mean != 0.:
            nrmse = rmse / ideal_mean # Normalized root mean square error
        else:
            # Division by zero attempt
            DivisionByZero.warning(v, (ideal.index[0], ideal.index[-1]))
            nrmse = rmse

        error[v] = nrmse  # Choose error from above

    # Calculate total error (sum of partial errors)
    assert 'tot' not in error, "'tot' is not an allowed name for output variables..."
    error['tot'] = 0
    for v in variables:
        error['tot'] += error[v]

    return error


class DivisionByZero:

    def __init__(self):
        pass

    ACCEPT_ZERO_DIVISION = False

    @staticmethod
    def warning(variable, period):
        if DivisionByZero.ACCEPT_ZERO_DIVISION is False:
            LOGGER.warning("[WARNING] Ideal solution for variable '{}' in the period ({}, {}) is null, " \
                  "so the error cannot be normalized. " \
                  "You can cancel or assume NRMSE=RMSE.".format(variable, period[0], period[1]))
            LOGGER.warning("If you accpet NRMSE=RMSE, it will be assumed also in future cases during this simulation.")

            decision = raw_input("Enter 'c' to cancel or any other key to continue with NRMSE=RMSE: ")

            if decision == 'c':
                raise ZeroDivisionError
            else:
                DivisionByZero.ACCEPT_ZERO_DIVISION = True

if __name__ == '__main__':
    zero_warning('TEST', (0, 999))

