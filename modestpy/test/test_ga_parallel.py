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
import random
import pandas as pd
import numpy as np
from modestpy.estim.ga_parallel.ga_parallel import MODESTGA
from modestpy.utilities.sysarch import get_sys_arch
from modestpy.loginit import config_logger


class TestMODESTGA(unittest.TestCase):

    def setUp(self):

        # Platform (win32, win64, linux32, linux64)
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

        # MODESTGA settings
        self.gen = 2
        self.pop = None      # Set individually
        self.trm = None      # Set individually
        self.workers = None  # Set individually

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_modestga_default(self):
        ga = MODESTGA(self.fmu_path, self.inp, self.known, self.est, self.ideal,
                      generations=self.gen)
        par_df = ga.estimate()
        assert type(par_df) is pd.DataFrame

    def test_modestga_simulation_fail(self):
        est_fail = {
            "R1": (-99, -100., 0.),
            "R2": (-1e6, -1e7, -1e5),
            "C": (0., -1e-10, 1e-10)
        }
        gen = 3
        ga = MODESTGA(self.fmu_path, self.inp, self.known, est_fail, self.ideal,
                      generations=gen)
        par_df = ga.estimate()
        assert type(par_df) is pd.DataFrame

    def test_modestga_1_worker(self):
        ga = MODESTGA(self.fmu_path, self.inp, self.known, self.est, self.ideal,
                      generations=self.gen,
                      pop_size=6,
                      trm_size=self.trm,
                      tol=1e-3,
                      inertia=5,
                      workers=1)
        par_df = ga.estimate()
        assert type(par_df) is pd.DataFrame

    def test_modestga_2_workers_small_pop(self):
        ga = MODESTGA(self.fmu_path, self.inp, self.known, self.est, self.ideal,
                      generations=self.gen,
                      pop_size=2,
                      trm_size=1,
                      tol=1e-3,
                      inertia=5,
                      workers=2)
        par_df = ga.estimate()
        assert type(par_df) is pd.DataFrame

    def test_modestga_2_workers_large_pop(self):
        ga = MODESTGA(self.fmu_path, self.inp, self.known, self.est, self.ideal,
                      generations=self.gen,
                      pop_size=32,
                      trm_size=3,
                      tol=1e-3,
                      inertia=5,
                      workers=2)
        par_df = ga.estimate()
        assert type(par_df) is pd.DataFrame


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestMODESTGA('test_modestga_default'))
    suite.addTest(TestMODESTGA('test_modestga_simulation_fail'))
    suite.addTest(TestMODESTGA('test_modestga_1_worker'))
    suite.addTest(TestMODESTGA('test_modestga_2_workers_small_pop'))
    suite.addTest(TestMODESTGA('test_modestga_2_workers_large_pop'))
    return suite


if __name__ == '__main__':
    config_logger(filename='unit_tests.log', level='DEBUG')
    unittest.main()
