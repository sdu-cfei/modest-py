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

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modestpy import Estimation
from modestpy.utilities.sysarch import get_sys_arch
from modestpy.fmi.compiler import mo_2_fmu
from modestpy.fmi.model import Model


if __name__ == "__main__":
    """
    This file is supposed to be run from the root directory.
    Otherwise the paths have to be corrected.
    """

    # Compile FMU =====================================================
    platform = get_sys_arch()
    mo_path = os.path.join('examples', 'lin', 'resources', 'lin_model.mo')
    fmu_path = os.path.join('examples', 'lin', 'resources', 'lin_model_{}.fmu'.format(platform))
    model_name = "lin_model"

    print('Compiling FMU for this platform')
    mo_2_fmu(model_name, mo_path, fmu_path)

    # Simulate FMU ====================================================
    model = Model(fmu_path)

    # Inputs
    inp = pd.DataFrame()
    time = np.arange(0, 86401, 100)
    inp['time'] = time
    inp['u1'] = np.full(time.shape, 2.)
    inp['u2'] = np.sin(time / 3000.)
    inp = inp.set_index('time')
    #inp.to_csv(os.path.join('examples', 'lin', 'resources', 'input.csv'))

    # True parameters
    a = 4.0
    b = 2.0
    true_par = pd.DataFrame(index=[0])
    true_par['a'] = a
    true_par['b'] = b
    #true_par.to_csv(os.path.join('examples', 'lin', 'resources', 'true_parameters.csv'), index=False)

    model.inputs_from_df(inp)
    model.parameters_from_df(true_par)
    model.specify_outputs(['y'])
    ideal = model.simulate(com_points=inp.index.size - 1)
    #ideal.to_csv(os.path.join('examples', 'lin', 'resources', 'ideal.csv'))

    # Grid search ==============================================

    # Working directory
    workdir = os.path.join('examples', 'lin', 'workdir')
    if not os.path.exists(workdir):
        os.mkdir(workdir)
        assert os.path.exists(workdir), "Work directory does not exist"

    a_bounds = (0., 8.1)
    b_bounds = (-4., 8.1)
    step = 0.5

    a_grid = np.arange(a_bounds[0], a_bounds[1], step)
    b_grid = np.arange(b_bounds[0], b_bounds[1], step)
    rmse = pd.DataFrame(index=pd.Index(a_grid, name='a'), columns=pd.Series(b_grid, name='b'))

    # Cost function shape
    for ai in a_grid:
        for bi in b_grid:
            par = pd.DataFrame(index=[0])
            par['a'] = ai
            par['b'] = bi

            model.parameters_from_df(par)
            yi = model.simulate(com_points=inp.index.size - 1)
            yi['ideal'] = ideal['y']
            rmse.loc[ai, bi] = ((yi['ideal'] - yi['y']) ** 2).mean()

    rmse.to_csv(os.path.join(workdir, 'rmse.csv'))
    rmse = rmse.astype(float)
    ax = sns.heatmap(rmse.iloc[::-1]) # Use reversed index
    ax.set_title('RMSE')
    fig = ax.get_figure()
    fig.set_size_inches(10, 7)
    fig.savefig(os.path.join(workdir, 'RMSE_grid.png'), dpi=100)

    # Search path
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ab = pd.read_csv(os.path.join(workdir, 'summary_1.csv')).set_index('_iter_')
    ax.set_xlabel('b')
    ax.set_ylabel('a')

    x = b_grid
    y = a_grid
    
    ax.plot(ab['b'].values, ab['a'].values, color='r', marker='x')
    ax.set_xlim(b_bounds[0], b_bounds[1])
    ax.set_ylim(a_bounds[0], a_bounds[1])

    z = rmse.values
    cs = ax.contour(x, y, rmse.values, levels=[0.5, 3, 6, 10, 15, 25])
    ax.clabel(cs)

    fig.savefig(os.path.join(workdir, 'search_path.png'), dpi=100)

    plt.show()

    # Delete FMU ==============================================
    os.remove(fmu_path)
