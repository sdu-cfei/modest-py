"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Plot files
DPI = 150
FIG_SIZE = (10, 6)


def plot_comparison(sim_res, ideal_res, f=None):
    """ Plots comparison: simulation vs. ideal results.

    :param sim_res: DataFrame
    :param ideal_res: DataFrame
    :param f: string
    :return: axes
    """
    simulated = sim_res.copy()
    measured = ideal_res.copy()
    measured.columns = [x + '_meas' for x in measured.columns]

    variables = list(simulated.columns)
    assert len(variables) > 0, 'No output variables to be compared'

    fig, axes = plt.subplots(nrows=len(variables), ncols=1)
    i = 0
    for var in variables:
        if len(variables) > 1:
            ax = axes[i]
        else:
            ax = axes
        var_meas = var + '_meas'
        ax.plot(measured.index / 3600., measured[var_meas], label='$' + var + '_{meas}$')
        ax.plot(simulated.index / 3600., simulated[var], label='$' + var + '$')
        ax.legend()
        ax.set_xlim(measured.index[0] / 3600)
        i += 1
    if len(variables) > 1:
        ax_last = axes[-1]
    else:
        ax_last = axes
    ax_last.set_xlabel('time [h]')

    if f:
        fig.set_size_inches(FIG_SIZE)
        plt.savefig(f, dpi=DPI)
    return axes


def plot_error_evo(errors, f=None):
    """ Plots evolution of errors.

    :param errors: DataFrame
    :param f: string
    :return: axes
    """
    fig, ax = plt.subplots()
    ax.plot(errors)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error (NRMSE)')
    if f:
        fig = ax.get_figure()
        fig.set_size_inches(FIG_SIZE)
        fig.savefig(f, dpi=DPI, figsize=FIG_SIZE)
    return ax


def plot_parameter_evo(parameters, file=None):
    """ Plots parameter evolution.

    :param parameters: DataFrame
    :param file: string
    :return: axes
    """
    par_evo = parameters.copy()
    # Get axes
    axes = par_evo.plot(subplots=True)
    fig = axes[0].get_figure()
    # Extend y lim
    axes = _extend_ylim(axes, par_evo)
    # x label
    axes[-1].set_xlabel('Iteration')

    if file:
        fig.set_size_inches(FIG_SIZE)
        fig.savefig(file, dpi=DPI)
    return axes


def plot_inputs(inputs, file=None):
    """ Plots inputs.

    :param inputs: DataFrame
    :param file: string
    :return: axes
    """
    axes = inputs.plot(subplots=True)
    fig = axes[0].get_figure()
    # x label
    axes[-1].set_xlabel('Time [s]')

    if file:
        fig.set_size_inches(FIG_SIZE)
        fig.savefig(file, dpi=DPI)
    return axes


def _extend_ylim(axes, df):
    # Extend y lim a bit and assign 3 y ticks in each subplot
    i = 0
    for p in list(df.columns):
        minimum = float(df[p].min())
        maximum = float(df[p].max())
        if maximum - minimum > 0.0001:
            # Varying values
            y_range = maximum - minimum
            ext = y_range * 0.2
            y_ext_range = y_range + 2 * ext
            axes[i].set_ylim([minimum - ext, maximum + ext])
            axes[i].set_yticks(np.arange(minimum - ext, maximum + ext * 1.1, y_ext_range / 2))
        else:
            # Probably constant value
            avg = (maximum + minimum) / 2  # ...but just in case take the average
            if avg != 0.:  # set_ylim and set_yticks wouldn't work if average == 0
                ext = avg * 0.2
                axes[i].set_ylim([minimum - ext, maximum + ext])
                axes[i].set_yticks(np.arange(avg - ext, avg + ext * 1.1, ext))
        i += 1
    return axes
