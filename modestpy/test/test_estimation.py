"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.

Author: Krzysztof Arendt
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
        session = Estimation(self.tmpdir, self.fmu_path, self.inp,
                             self.known, self.est, self.ideal,
                             ga_iter=3, ps_iter=3)
        estimates = session.estimate()
        err, res = session.validate()

        self.assertIsNotNone(estimates)
        self.assertGreater(len(estimates), 0)
        self.assertIsNotNone(err)
        self.assertIsNotNone(res)
        self.assertGreater(len(res.index), 1)
        self.assertGreater(len(res.columns), 0)
    
    def test_estimation_multiple_lp(self):
        session = Estimation(self.tmpdir, self.fmu_path, self.inp,
                             self.known, self.est, self.ideal,
                             lp_n=2, lp_len=3600,
                             ga_iter=3, ps_iter=3)

        estimates = session.estimate()
        err, res = session.validate()

        self.assertIsNotNone(estimates)
        self.assertGreater(len(estimates), 0)
        self.assertIsNotNone(err)
        self.assertIsNotNone(res)
        self.assertGreater(len(res.index), 1)
        self.assertGreater(len(res.columns), 0)

    def test_estimation_all_args(self):
        session = Estimation(self.tmpdir, self.fmu_path, self.inp,
                             self.known, self.est, self.ideal,
                             lp_n=2, lp_len=3600, lp_frame=(0, 20000),
                             vp = (20000, 40000), ic_param={'Tstart': 'T'},
                             ga_iter=3, ps_iter=3)

        estimates = session.estimate()
        err, res = session.validate()

        self.assertIsNotNone(estimates)
        self.assertGreater(len(estimates), 0)
        self.assertIsNotNone(err)
        self.assertIsNotNone(res)
        self.assertGreater(len(res.index), 1)
        self.assertGreater(len(res.columns), 0)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestEstimation('test_estimation_basic'))
    suite.addTest(TestEstimation('test_estimation_multiple_lp'))
    suite.addTest(TestEstimation('test_estimation_all_args'))
    
    return suite


if __name__ == "__main__":
    unittest.main()