"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

import unittest
import tempfile
import shutil
import os
import json
import pandas as pd
from modestpy.estim.learnman import LearnMan
from modestpy.utilities.sysarch import get_sys_arch


class TestLM(unittest.TestCase):

    def setUp(self):

        # Platform (win32, win64, linux32, linix64)
        platform = get_sys_arch()
        assert platform, 'Unsupported platform type!'

        # Temp directory
        self.tmpdir = tempfile.mkdtemp()

        # Parent directory
        parent = os.path.dirname(__file__)

        # Resources
        self.fmu_path = os.path.join(parent, 'resources', 'simple2R1C', 'Simple2R1C_{}.fmu'.format(platform))
        inp_path = os.path.join(parent, 'resources', 'simple2R1C', 'inputs.csv')
        ideal_path = os.path.join(parent, 'resources', 'simple2R1C', 'result.csv')
        est_path = os.path.join(parent, 'resources', 'simple2R1C', 'est.json')
        known_path = os.path.join(parent, 'resources', 'simple2R1C', 'known.json')
        self.workdir = self.tmpdir

        self.inp = pd.read_csv(inp_path).set_index('time')
        self.ideal = pd.read_csv(ideal_path).set_index('time')

        with open(est_path) as f:
            self.est = json.load(f)
        with open(known_path) as f:
            self.known = json.load(f)

        # LM settings
        LearnMan.GA_GENERATIONS = 5
        LearnMan.GA_POP_SIZE = 8
        LearnMan.PS_MAX_ITER = 5
        self.lp_n = 3
        self.lp_bounds = (0., 215940.)
        self.lp_length = 215940. / 3.
        self.vp = (0., 215940.)
        # self.sensors = {'TInitial': 'T',
        #                 'CO2PpmInitial': 'CO2'}

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_lm(self):
        self.lm = LearnMan(self.workdir, self.fmu_path, self.inp, self.known,
                           self.est, self.ideal)

        # Estimation
        # self.lm.ic_from_sensors(self.sensors)
        self.lm.select_learning_periods(self.lp_n, self.lp_length, self.lp_bounds)
        self.estimates = self.lm.estimate()

        # Validation
        self.lm.select_validation_period(location=(self.vp[0], self.vp[1]))
        self.lm.validate(self.lm.get_wavg_est())

        # Results
        wavg = self.lm.get_wavg_est()
        best = self.lm.get_best_est()

        self.assertIsNotNone(wavg, 'get_wavg_est() returned None')
        self.assertIsNotNone(best, 'get_best_est() returned None')

        self.lm.save_best_est('best.csv', incl_known=True)
        self.lm.save_wavg_est('wavg.csv', incl_known=True)

        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, 'best.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, 'wavg.csv')))

        self.lm.save_plots()
        # TODO: Check if all plots are saved


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestLM('test_lm'))

    return suite


if __name__ == '__main__':
    unittest.main()
