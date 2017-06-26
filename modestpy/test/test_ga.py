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
from modestpy.estim.ga.ga import GA
import pandas as pd
import json
import os
from modestpy.utilities.sysarch import get_sys_arch


class TestGA(unittest.TestCase):

    def setUp(self):

        # Platform (win32, win64, linux32, linux64)
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

        # GA settings
        self.gen = 6
        self.pop = 9
        self.trm = 4

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_ga(self):
        self.ga = GA(self.fmu_path, self.inp, self.known,
                     self.est, self.ideal, generations=self.gen,
                     pop_size=self.pop, trm_size=self.trm)
        self.estimates = self.ga.estimate()

        # Generate plot
        plot_path = file=os.path.join(self.tmpdir, 'popevo.png')
        self.ga.plot_pop_evo(plot_path)

        # Make sure plot is created
        self.assertTrue(os.path.exists(plot_path))

        # Make sure errors do not increase
        errors = self.ga.get_errors()
        for i in range(1, len(errors)):
            prev_err = errors[i-1]
            next_err = errors[i]
            self.assertGreaterEqual(prev_err, next_err)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestGA('test_ga'))

    return suite


if __name__ == '__main__':
    unittest.main()
