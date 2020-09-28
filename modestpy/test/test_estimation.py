# -*- coding: utf-8 -*-
"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.
This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""
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
        self.fmu_path = os.path.join(parent, 'resources', 'simple2R1C_ic',
                                     'Simple2R1C_ic_{}.fmu'.format(platform))
        inp_path = os.path.join(parent, 'resources', 'simple2R1C_ic',
                                'inputs.csv')
        ideal_path = os.path.join(parent, 'resources', 'simple2R1C_ic',
                                  'result.csv')
        est_path = os.path.join(parent, 'resources', 'simple2R1C_ic',
                                'est.json')
        known_path = os.path.join(parent, 'resources', 'simple2R1C_ic',
                                  'known.json')

        # Assert there is an FMU for this platform
        assert os.path.exists(self.fmu_path), \
            "FMU for this platform ({}) doesn't exist.\n".format(platform) + \
            "No such file: {}".format(self.fmu_path)

        self.inp = pd.read_csv(inp_path).set_index('time')
        self.ideal = pd.read_csv(ideal_path).set_index('time')

        with open(est_path) as f:
            self.est = json.load(f)
        with open(known_path) as f:
            self.known = json.load(f)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_estimation_basic(self):
        """Will use default methods ('MODESTGA', 'PS')"""
        modestga_opts = {'generations': 2, 'workers': 1, 'pop_size': 8, 'trm_size': 3}
        ps_opts = {'maxiter': 2}
        session = Estimation(self.tmpdir, self.fmu_path, self.inp,
                             self.known, self.est, self.ideal,
                             modestga_opts=modestga_opts, ps_opts=ps_opts,
                             default_log=False)
        estimates = session.estimate()
        err, res = session.validate()

        self.assertIsNotNone(estimates)
        self.assertGreater(len(estimates), 0)
        self.assertIsNotNone(err)
        self.assertIsNotNone(res)
        self.assertGreater(len(res.index), 1)
        self.assertGreater(len(res.columns), 0)

    def test_estimation_basic_parallel(self):
        """Will use default methods ('MODESTGA', 'PS')"""
        modestga_opts = {'generations': 2, 'workers': 2, 'pop_size': 16, 'trm_size': 3}
        ps_opts = {'maxiter': 2}
        session = Estimation(self.tmpdir, self.fmu_path, self.inp,
                             self.known, self.est, self.ideal,
                             modestga_opts=modestga_opts, ps_opts=ps_opts,
                             default_log=False)
        estimates = session.estimate()
        err, res = session.validate()

        self.assertIsNotNone(estimates)
        self.assertGreater(len(estimates), 0)
        self.assertIsNotNone(err)
        self.assertIsNotNone(res)
        self.assertGreater(len(res.index), 1)
        self.assertGreater(len(res.columns), 0)

    def test_estimation_all_args(self):
        modestga_opts = {'generations': 2, 'workers': 2, 'pop_size': 16, 'trm_size': 3}
        ps_opts = {'maxiter': 3}
        session = Estimation(self.tmpdir, self.fmu_path, self.inp,
                             self.known, self.est, self.ideal,
                             lp_n=2, lp_len=3600, lp_frame=(0, 3600),
                             vp=(20000, 40000), ic_param={'Tstart': 'T'},
                             methods=('MODESTGA', 'PS'),
                             modestga_opts=modestga_opts, ps_opts=ps_opts,
                             seed=1, ftype='NRMSE',
                             default_log=False)

        estimates = session.estimate()
        err, res = session.validate()  # Standard validation period
        err2, res2 = session.validate(vp=(25000, 28600))

        self.assertIsNotNone(estimates)
        self.assertGreater(len(estimates), 0)
        self.assertIsNotNone(err)
        self.assertIsNotNone(res)
        self.assertIsNotNone(err2)
        self.assertIsNotNone(res2)
        self.assertGreater(len(res.index), 1)
        self.assertGreater(len(res.columns), 0)
        self.assertGreater(len(res2.index), 1)
        self.assertGreater(len(res2.columns), 0)
        self.assertEqual(session.lp[0][0], 0)
        self.assertEqual(session.lp[0][1], 3600)
        self.assertLess(err['tot'], 1.7)  # NRMSE

    def test_estimation_rmse(self):
        modestga_opts = {'generations': 8}
        ps_opts = {'maxiter': 16}

        session = Estimation(self.tmpdir, self.fmu_path, self.inp,
                             self.known, self.est, self.ideal,
                             lp_n=1, lp_len=3600, lp_frame=(0, 3600),
                             vp=(20000, 40000), ic_param={'Tstart': 'T'},
                             methods=('MODESTGA', 'PS'),
                             modestga_opts=modestga_opts,
                             ps_opts=ps_opts,
                             seed=1, ftype='RMSE',
                             default_log=False)

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
        modestga_opts = {'generations': 1}
        ps_opts = {'maxiter': 0}
        session = Estimation(self.tmpdir, self.fmu_path, self.inp,
                             self.known, self.est, self.ideal,
                             lp_n=1, lp_len=3600, lp_frame=(0, 3600),
                             vp=(20000, 40000), ic_param={'Tstart': 'T'},
                             methods=('MODESTGA', ),
                             modestga_opts=modestga_opts, ps_opts=ps_opts,
                             seed=1, ftype='RMSE',
                             default_log=False)
        session.estimate()

    def test_seed(self):
        modestga_opts = {'generations': 10}
        ps_opts = {'maxiter': 5}
        # Run 1
        session1 = Estimation(self.tmpdir, self.fmu_path, self.inp,
                              self.known, self.est, self.ideal,
                              lp_n=1, lp_len=3600, lp_frame=(0, 3600),
                              vp=(20000, 40000), ic_param={'Tstart': 'T'},
                              methods=('MODESTGA', ),
                              modestga_opts=modestga_opts, ps_opts=ps_opts,
                              seed=1, ftype='RMSE',
                              default_log=False)
        estimates1 = session1.estimate()
        # Run 2
        session2 = Estimation(self.tmpdir, self.fmu_path, self.inp,
                              self.known, self.est, self.ideal,
                              lp_n=1, lp_len=3600, lp_frame=(0, 3600),
                              vp=(20000, 40000), ic_param={'Tstart': 'T'},
                              methods=('MODESTGA', ),
                              modestga_opts=modestga_opts, ps_opts=ps_opts,
                              seed=1, ftype='RMSE',
                              default_log=False)
        estimates2 = session2.estimate()
        # Check if estimates are the same
        same = True
        for key in estimates1:
            if estimates1[key] != estimates2[key]:
                same = False
        self.assertTrue(
            same,
            "Different estimates obtained despite the same seed"
            )

    def test_ps_only(self):
        modestga_opts = {'generations': 0}
        ps_opts = {'maxiter': 1}
        session = Estimation(self.tmpdir, self.fmu_path, self.inp,
                             self.known, self.est, self.ideal,
                             lp_n=1, lp_len=3600, lp_frame=(0, 3600),
                             vp=(20000, 40000), ic_param={'Tstart': 'T'},
                             methods=('PS', ),
                             modestga_opts=modestga_opts, ps_opts=ps_opts, seed=1,
                             ftype='RMSE', default_log=False)
        session.estimate()

    def test_opts(self):
        modestga_opts = {
            'workers': 2,              # CPU cores to use
            'generations': 10,         # Max. number of generations
            'pop_size': 40,            # Population size
            'mut_rate': 0.05,          # Mutation rate
            'trm_size': 10,            # Tournament size
            'tol': 1e-4,               # Solution tolerance
            'inertia': 20              # Max. number of non-improving generations
        }
        ps_opts = {'maxiter': 10, 'rel_step': 0.1, 'tol': 0.001, 'try_lim': 10}
        session = Estimation(self.tmpdir, self.fmu_path, self.inp,
                             self.known, self.est, self.ideal,
                             methods=('MODESTGA', 'PS'),
                             modestga_opts=modestga_opts, ps_opts=ps_opts,
                             default_log=False)
        modestga_return = session.MODESTGA_OPTS
        ps_return = session.PS_OPTS

        def extractDictAFromB(A, B):
            return dict([(k, B[k]) for k in A.keys() if k in B.keys()])

        self.assertEqual(modestga_opts, extractDictAFromB(modestga_opts, modestga_return))
        self.assertEqual(ps_opts, extractDictAFromB(ps_opts, ps_return))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestEstimation('test_estimation_basic'))
    suite.addTest(TestEstimation('test_estimation_basic_parallel'))
    suite.addTest(TestEstimation('test_estimation_all_args'))
    suite.addTest(TestEstimation('test_estimation_rmse'))
    suite.addTest(TestEstimation('test_ga_only'))
    suite.addTest(TestEstimation('test_ps_only'))
    suite.addTest(TestEstimation('test_opts'))
    suite.addTest(TestEstimation('test_seed'))

    return suite


if __name__ == "__main__":
    unittest.main()
