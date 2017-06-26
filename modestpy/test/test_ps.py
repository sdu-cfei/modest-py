"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

import unittest
import shutil
import tempfile
import json
import os
import pandas as pd
from modestpy.estim.ps.ps import PS
from modestpy.utilities.sysarch import get_sys_arch


class TestPS(unittest.TestCase):

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

        self.inp = pd.read_csv(inp_path).set_index('time')
        self.ideal = pd.read_csv(ideal_path).set_index('time')

        with open(est_path) as f:
            self.est = json.load(f)
        with open(known_path) as f:
            self.known = json.load(f)

        # PS settings
        self.max_iter = 10
        self.try_lim = 2

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_ps(self):
        self.ps = PS(self.fmu_path, self.inp, self.known,
                     self.est, self.ideal, max_iter=self.max_iter,
                     try_lim=self.try_lim)
        self.estimates = self.ps.estimate()

        # Generate plots
        self.ps.plot_comparison(os.path.join(self.tmpdir, 'ps_comparison.png'))
        self.ps.plot_error_evo(os.path.join(self.tmpdir, 'ps_error_evo.png'))
        self.ps.plot_parameter_evo(os.path.join(self.tmpdir, 'ps_param_evo.png'))

        # Make sure plots are created
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, 'ps_comparison.png')))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, 'ps_error_evo.png')))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, 'ps_param_evo.png')))

        # Make sure errors do not increase
        errors = self.ps.get_errors()
        for i in range(1, len(errors)):
            prev_err = errors[i-1]
            next_err = errors[i]
            self.assertGreaterEqual(prev_err, next_err)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestPS('test_ps'))

    return suite


if __name__ == '__main__':
    unittest.main()
