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

import unittest
import tempfile
import shutil
import json
import os
import pandas as pd
from modestpy import Estimation
from modestpy.utilities.sysarch import get_sys_arch


class TestEstimation(unittest.TestCase):

    def setUp(self):
        
        # Platform (win32, win64, linux32, linux 64)
        platform = get_sys_arch()
        assert platform, "Unsupported platform type!"

        # Temp directory
        self.tmpdir = tempfile.mkdtemp()

        # Parent directory
        parent = os.path.dirname(__file__)

        # Resources
        self.fmu_path = os.path.join(parent, 'resources', 'simple2R1C_ic', 'Simple2R1C_ic_{}.fmu'.format(platform))
        inp_path = os.path.join(parent, 'resources', 'simple2R1C_ic', 'inputs.csv')
        ideal_path = os.path.join(parent, 'resources', 'simple2R1C_ic', 'result.csv')
        est_path = os.path.join(parent, 'resources', 'simple2R1C_ic', 'est.json')
        known_path = os.path.join(parent, 'resources', 'simple2R1C_ic', 'known.json')

        self.inp = pd.read_csv(inp_path).set_index('time')
        self.ideal = pd.read_csv(ideal_path).set_index('time')

        with open(est_path) as f:
            self.est = json.load(f)
        with open(known_path) as f:
            self.known = json.load(f)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_estimation_basic(self):
        ga_opts = {'maxiter': 3}
        ps_opts = {'maxiter': 3}
        session = Estimation(self.tmpdir, self.fmu_path, self.inp,
                             self.known, self.est, self.ideal,
                             ga_opts=ga_opts, ps_opts=ps_opts)
        estimates = session.estimate()
        err, res = session.validate()

        self.assertIsNotNone(estimates)
        self.assertGreater(len(estimates), 0)
        self.assertIsNotNone(err)
        self.assertIsNotNone(res)
        self.assertGreater(len(res.index), 1)
        self.assertGreater(len(res.columns), 0)

    def test_estimation_all_args(self):
        ga_opts = {'maxiter': 3, 'pop_size': 10, 'trm_size': 5, 'lhs': True}
        ps_opts = {'maxiter': 3}
        session = Estimation(self.tmpdir, self.fmu_path, self.inp,
                             self.known, self.est, self.ideal,
                             lp_n=2, lp_len=3600, lp_frame=(0, 3600),
                             vp=(20000, 40000), ic_param={'Tstart': 'T'}, methods=('GA', 'PS'),
                             ga_opts=ga_opts, ps_opts=ps_opts, seed=1, ftype='NRMSE')

        estimates = session.estimate()
        err, res = session.validate()

        self.assertIsNotNone(estimates)
        self.assertGreater(len(estimates), 0)
        self.assertIsNotNone(err)
        self.assertIsNotNone(res)
        self.assertGreater(len(res.index), 1)
        self.assertGreater(len(res.columns), 0)
        self.assertEqual(session.lp[0][0], 0)
        self.assertEqual(session.lp[0][1], 3600)
        # raw_input('Continue...') # <-- enabling this line triggers the Matplotlib error (issue #20)
        self.assertLess(err['tot'], 1.65)  # NRMSE

    def test_estimation_rmse(self):
        ga_opts = {'maxiter': 3, 'pop_size': 8, 'trm_size': 3}
        ps_opts = {'maxiter': 3}

        session = Estimation(self.tmpdir, self.fmu_path, self.inp,
                            self.known, self.est, self.ideal,
                            lp_n=1, lp_len=3600, lp_frame=(0, 3600),
                            vp = (20000, 40000), ic_param={'Tstart': 'T'}, methods=('GA', 'PS'),
                            ga_opts=ga_opts, ps_opts=ps_opts, seed=1, ftype='RMSE')

        estimates = session.estimate()
        err, res = session.validate()

        self.assertIsNotNone(estimates)
        self.assertGreater(len(estimates), 0)
        self.assertIsNotNone(err)
        self.assertIsNotNone(res)
        self.assertGreater(len(res.index), 1)
        self.assertGreater(len(res.columns), 0)
        self.assertLess(err['tot'], 1.48)
        
    def test_ga_only(self):
        ga_opts = {'maxiter': 1}
        ps_opts = {'maxiter': 0}
        session = Estimation(self.tmpdir, self.fmu_path, self.inp,
                    self.known, self.est, self.ideal,
                    lp_n=1, lp_len=3600, lp_frame=(0, 3600),
                    vp = (20000, 40000), ic_param={'Tstart': 'T'}, methods=('GA', ),
                    ga_opts=ga_opts, ps_opts=ps_opts, seed=1, ftype='RMSE')
        estimates = session.estimate()

    def test_ps_only(self):
        ga_opts = {'maxiter': 0}
        ps_opts = {'maxiter': 1}
        session = Estimation(self.tmpdir, self.fmu_path, self.inp,
                    self.known, self.est, self.ideal,
                    lp_n=1, lp_len=3600, lp_frame=(0, 3600),
                    vp = (20000, 40000), ic_param={'Tstart': 'T'}, methods=('PS', ),
                    ga_opts=ga_opts, ps_opts=ps_opts, seed=1, ftype='RMSE')
        estimates = session.estimate()

    # def test_sqp_only(self):
    #     session = Estimation(self.tmpdir, self.fmu_path, self.inp,
    #                 self.known, self.est, self.ideal,
    #                 lp_n=1, lp_len=3600, lp_frame=(0, 3600),
    #                 vp = (20000, 40000), ic_param={'Tstart': 'T'}, methods=('SQP', ), 
    #                 ga_opts=ga_opts, ps_opts=ps_opts, seed=1, ftype='RMSE')
    #     estimates = session.estimate()

    def test_opts(self):
        ga_opts = {'maxiter': 10, 'pop_size': 10, 'look_back': 10,
                   'tol': 0.001, 'mut': 0.02, 'mut_inc': 0.3, 'trm_size': 3}
        ps_opts = {'maxiter': 10, 'rel_step': 0.1, 'tol': 0.001, 'try_lim': 10}
        session = Estimation(self.tmpdir, self.fmu_path, self.inp,
                    self.known, self.est, self.ideal,
                    ga_opts=ga_opts, ps_opts=ps_opts)
        ga_return = session.GA_OPTS
        ps_return = session.PS_OPTS
        self.assertDictContainsSubset(ga_opts, ga_return)
        self.assertDictContainsSubset(ps_opts, ps_return)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestEstimation('test_estimation_basic'))
    suite.addTest(TestEstimation('test_estimation_all_args'))
    suite.addTest(TestEstimation('test_estimation_rmse'))
    suite.addTest(TestEstimation('test_ga_only'))
    suite.addTest(TestEstimation('test_ps_only'))
    suite.addTest(TestEstimation('test_opts'))

    return suite


if __name__ == "__main__":
    unittest.main()