"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

import unittest
import json
import os
import pandas as pd
from estim.ps.ps import PS

class TestPS(unittest.TestCase):

    def setUp(self):

        # Resources
        self.fmu_path = os.path.join('tests', 'resources', 'simple2R1C', 'Simple2R1C.fmu')
        inp_path = os.path.join('tests', 'resources', 'simple2R1C', 'inputs.csv')
        ideal_path = os.path.join('tests', 'resources', 'simple2R1C', 'result.csv')
        est_path = os.path.join('tests', 'resources', 'simple2R1C', 'est.json')
        known_path = os.path.join('tests', 'resources', 'simple2R1C', 'known.json')

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
        pass

    def test_ps(self):
        self.ps = PS(self.fmu_path, self.inp, self.known,
                     self.est, self.ideal, max_iter=self.max_iter,
                     try_lim=self.try_lim)
        self.estimates = self.ps.estimate()

        # Make sure errors do not increase
        errors = self.ps.get_errors()
        for i in range(1, len(errors)):
            prev_err = errors[i-1]
            next_err = errors[i]
            self.assertGreaterEqual(prev_err, next_err)

if __name__ == '__main__':
    unittest.main()
