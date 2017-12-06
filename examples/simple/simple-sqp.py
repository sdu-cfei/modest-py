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
from modestpy import Estimation
from modestpy.utilities.sysarch import get_sys_arch
from modestpy.estim.sqp.sqp import SQP
import matplotlib.pyplot as plt


if __name__ == "__main__":
    """
    This file is supposed to be run from the root directory.
    Otherwise the paths have to be corrected.
    """

    # DATA PREPARATION ==============================================
    # Resources
    platform = get_sys_arch()
    assert platform, 'Unsupported platform type!'
    fmu_file = 'Simple2R1C_ic_' + platform + '.fmu'

    fmu_path = os.path.join('examples', 'simple', 'resources', fmu_file)
    inp_path = os.path.join('examples', 'simple', 'resources', 'inputs.csv')
    ideal_path = os.path.join('examples', 'simple', 'resources', 'result.csv')
    est_path = os.path.join('examples', 'simple', 'resources', 'est_validate_hj.json')
    known_path = os.path.join('examples', 'simple', 'resources', 'known_validate_hj.json')

    # Working directory
    workdir = os.path.join('examples', 'simple', 'workdir')
    if not os.path.exists(workdir):
        os.mkdir(workdir)
        assert os.path.exists(workdir), "Work directory does not exist"

    # Load inputs
    inp = pd.read_csv(inp_path).set_index('time')

    # Load measurements (ideal results)
    ideal = pd.read_csv(ideal_path).set_index('time')

    # Load definition of estimated parameters (name, initial value, bounds)
    with open(est_path) as f:
        est = json.load(f)

    # Load definition of known parameters (name, value)
    with open(known_path) as f:
        known = json.load(f)

    # MODEL IDENTIFICATION ==========================================
    # session = Estimation(workdir, fmu_path, inp, known, est, ideal,
    #                      lp_n=3, lp_len=25000, lp_frame=(0, 25000),
    #                      vp = (150000, 215940), ic_param={'Tstart': 'T'},
    #                      ga_pop=20, ga_iter=20, ps_iter=30, ga_tol=0.001, ps_tol=0.0001,
    #                      seed=1, ftype='RMSE', lhs=True)  # seed is used to make the results repetitive in this example

    # estimates = session.estimate()
    # err, res = session.validate()
    sqp = SQP(fmu_path, inp, known, est, ideal, ftype='RMSE')

    par = sqp.estimate()
    print(par)

    res = sqp.res
    res['ideal'] = ideal['T']
    res.plot()
    plt.show()

    print('ERROR={}'.format(sqp.best_err))
