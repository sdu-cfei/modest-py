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
import pandas as pd
import json
import os
from modestpy.estim.ga.ga import GA
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

    def test_init_pop(self):
        init_pop = pd.DataFrame({'R1': [0.1, 0.2, 0.3], 'R2': [0.15, 0.25, 0.35], 'C': [1000., 1100., 1200.]})
        pop_size = 3
        self.ga = GA(self.fmu_path, self.inp, self.known,
                     self.est, self.ideal, generations=self.gen,
                     pop_size=pop_size, trm_size=self.trm, init_pop=init_pop)
        i1 = self.ga.pop.individuals[0]
        i2 = self.ga.pop.individuals[1]
        i3 = self.ga.pop.individuals[2]
        R1_lo = self.est['R1'][1]
        R1_hi = self.est['R1'][2]
        R2_lo = self.est['R2'][1]
        R2_hi = self.est['R2'][2]
        C_lo = self.est['C'][1]
        C_hi = self.est['C'][2]
        assert i1.genes == {'C': (1000. - C_lo) / (C_hi - C_lo),
                            'R1': (0.1 - R1_lo) / (R1_hi - R1_lo),
                            'R2': (0.15 - R2_lo) / (R2_hi - R2_lo)}
        assert i2.genes == {'C': (1100. - C_lo) / (C_hi - C_lo),
                            'R1': (0.2 - R1_lo) / (R1_hi - R1_lo),
                            'R2': (0.25 - R2_lo) / (R2_hi - R2_lo)}
        assert i3.genes == {'C': (1200. - C_lo) / (C_hi - C_lo),
                            'R1': (0.3 - R1_lo) / (R1_hi - R1_lo),
                            'R2': (0.35 - R2_lo) / (R2_hi - R2_lo)}

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestGA('test_ga'))
    suite.addTest(TestGA('test_init_pop'))

    return suite


if __name__ == '__main__':
    unittest.main()
