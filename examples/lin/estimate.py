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
    par = pd.DataFrame(index=[0])
    par['a'] = a
    par['b'] = b
    #par.to_csv(os.path.join('examples', 'lin', 'resources', 'true_parameters.csv'), index=False)

    model.inputs_from_df(inp)
    model.parameters_from_df(par)
    model.specify_outputs(['y'])
    ideal = model.simulate(com_points=inp.index.size - 1)
    #ideal.to_csv(os.path.join('examples', 'lin', 'resources', 'ideal.csv'))

    # Estimation ==============================================

    # Working directory
    workdir = os.path.join('examples', 'lin', 'workdir')
    if not os.path.exists(workdir):
        os.mkdir(workdir)
        assert os.path.exists(workdir), "Work directory does not exist"

    # Estimated and known parameters
    known = {}
    est = {'a': (5., 0., 10.), 'b': (5., 0., 10.)}

    # Session
    session = Estimation(workdir, fmu_path, inp, known, est, ideal,
                         lp_n=1, lp_len=86400/2, lp_frame=(0, 86400/2), vp = (86400/2, 86400),
                         methods=('SQP',),
                         ga_opts={'maxiter': 200, 'tol': 1e-6, 'lhs': True},
                         ps_opts={'maxiter': 500, 'tol': 1e-8},
                         sqp_opts={},
                         ftype='RMSE', seed=1)

    estimates = session.estimate()
    err, res = session.validate()

    # Check estimates =========================================
    epsilon = 1e-2
    a_err = abs(estimates['a'].iloc[0] - a)
    b_err = abs(estimates['b'].iloc[0] - b)
    if a_err < epsilon and b_err < epsilon:
        print("ESTIMATED PARAMETERS ARE CORRECT (ERRORS BELOW {})".format(epsilon))
    else:
        print("ESTIMATED PARAMETERS INCORRECT: a_err={}, b_err={} > {}".format(a_err, b_err, epsilon))

    # Delete FMU ==============================================
    os.remove(fmu_path)