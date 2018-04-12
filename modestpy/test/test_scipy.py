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
import shutil
import tempfile
import json
import os
import pandas as pd
from modestpy.estim.scipy.scipy import SCIPY
from modestpy.utilities.sysarch import get_sys_arch


class TestSCIPY(unittest.TestCase):

    def setUp(self):

        # Platform (win32, win64, linux32, linix64)
        platform = get_sys_arch()
        assert platform, 'Unsupported platform type!'

        # Temp directory
        self.tmpdir = tempfile.mkdtemp()

        # Parent directory
        parent = os.path.dirname(__file__)

        # Resources
        self.fmu_path = os.path.join(parent, 'resources', 'simple2R1C',
                                     'Simple2R1C_{}.fmu'.format(platform))
        inp_path = os.path.join(parent, 'resources', 'simple2R1C',
                                'inputs.csv')
        ideal_path = os.path.join(parent, 'resources', 'simple2R1C',
                                  'result.csv')
        est_path = os.path.join(parent, 'resources', 'simple2R1C', 'est.json')
        known_path = os.path.join(parent, 'resources', 'simple2R1C',
                                  'known.json')

        self.inp = pd.read_csv(inp_path).set_index('time')
        self.ideal = pd.read_csv(ideal_path).set_index('time')

        with open(est_path) as f:
            self.est = json.load(f)
        with open(known_path) as f:
            self.known = json.load(f)

        # PS settings
        self.max_iter = 3
        self.try_lim = 2

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_scipy(self):
        self.scipy = SCIPY(self.fmu_path, self.inp, self.known,
                           self.est, self.ideal,
                           solver='L-BFGS-B')
        self.estimates = self.scipy.estimate()

        # Generate plots
        self.scipy.plot_comparison(os.path.join(self.tmpdir,
                                             'scipy_comparison.png'))
        self.scipy.plot_error_evo(os.path.join(self.tmpdir,
                                            'scipy_error_evo.png'))
        self.scipy.plot_parameter_evo(os.path.join(self.tmpdir,
                                                'scipy_param_evo.png'))

        # Make sure plots are created
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir,
                                                    'scipy_comparison.png')))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir,
                                                    'scipy_error_evo.png')))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir,
                                                    'scipy_param_evo.png')))

        # Make last error is lower than initial
        errors = self.scipy.get_errors()
        self.assertGreaterEqual(errors[0], errors[-1])


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestSCIPY('test_scipy'))

    return suite


if __name__ == '__main__':
    unittest.main()
