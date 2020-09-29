"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.
This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""
import time
import os
import pandas as pd
import numpy as np
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
    fmu_path = os.path.join('examples', 'lin', 'resources',
                            'lin_model_{}.fmu'.format(platform))
    model_name = "lin_model"

    print('Compiling FMU for this platform')
    mo_2_fmu(model_name, mo_path, fmu_path)

    # Simulate FMU ====================================================
    model = Model(fmu_path)

    # Inputs
    inp = pd.DataFrame()
    t = np.arange(0, 86401, 100)
    inp['time'] = t
    inp['u1'] = np.full(t.shape, 2.)
    inp['u2'] = np.sin(t / 3000.)
    inp = inp.set_index('time')
    # inp.to_csv(os.path.join('examples', 'lin', 'resources', 'input.csv'))

    # True parameters
    a = 4.0
    b = 2.0
    par = pd.DataFrame(index=[0])
    par['a'] = a
    par['b'] = b
    # par.to_csv(os.path.join('examples', 'lin', 'resources',
    #            'true_parameters.csv'), index=False)

    model.inputs_from_df(inp)
    model.parameters_from_df(par)
    model.specify_outputs(['y'])
    ideal = model.simulate()
    # ideal.to_csv(os.path.join('examples', 'lin', 'resources', 'ideal.csv'))

    # Estimation ==============================================

    # Working directory
    workdir = os.path.join('examples', 'lin', 'workdir')
    if not os.path.exists(workdir):
        os.mkdir(workdir)
        assert os.path.exists(workdir), "Work directory does not exist"

    # Estimated and known parameters
    known = {}
    est = {'a': (5., 0., 8.), 'b': (5., -4., 8.)}

    # Session
    session = Estimation(workdir, fmu_path, inp, known, est, ideal,
                         lp_n=1, lp_len=86400/2, lp_frame=(0, 86400/2),
                         vp=(86400/2, 86400),
                         methods=('GA_LEGACY', 'SCIPY'),
                         ga_opts={'maxiter': 10, 'tol': 1e-8, 'lhs': True},
                         ps_opts={'maxiter': 1000, 'tol': 1e-12},
                         scipy_opts={'solver': 'L-BFGS-B',
                                     'options': {'eps': 1e-12}},
                         ftype='RMSE')

    t0 = time.time()
    estimates = session.estimate()
    t1 = time.time()
    err, res = session.validate()

    print("ELAPSED TIME: {}".format(t1 - t0))

    # Check estimates =========================================
    epsilon = 1e-3
    a_err = abs(estimates['a'] - a)
    b_err = abs(estimates['b'] - b)
    if a_err < epsilon and b_err < epsilon:
        print("ESTIMATED PARAMETERS ARE CORRECT: a_err={}, b_err={} < {}"
              .format(a_err, b_err, epsilon))
    else:
        print("ESTIMATED PARAMETERS INCORRECT: a_err={}, b_err={} > {}"
              .format(a_err, b_err, epsilon))

    # Delete FMU ==============================================
    os.remove(fmu_path)
