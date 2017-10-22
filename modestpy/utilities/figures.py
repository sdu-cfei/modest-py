# -*- coding: utf-8 -*-

"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.
This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

# Plot files
DPI = 100
FIG_SIZE = (15, 10)

def get_figure(ax):
    """
    Retrieves figure from axes. Axes can be either an instance
    of Matplotlib.Axes or a 1D/2D array of Matplotlib.Axes.

    :param ax: Axes or vector/array of Axes
    :return: Matplotlib.Figure
    """
    fig = None
    try:
        # Single plot
        fig = ax.get_figure()
    except AttributeError:
        # Subplots
        try:
            # 1D grid
            fig = ax[0].get_figure()
        except AttributeError:
            # 2D grid
            fig = ax[0][0].get_figure()
    # Adjust size
    fig.set_size_inches(FIG_SIZE)
    return fig