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
from modestpy.fmi.model import Model
from modestpy.utilities.sysarch import get_sys_arch
from modestpy.loginit import config_logger


class TestFMPy(unittest.TestCase):

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

        # Assert there is an FMU for this platform
        assert os.path.exists(self.fmu_path), \
            "FMU for this platform ({}) doesn't exist.\n".format(platform) + \
            "No such file: {}".format(self.fmu_path)

        self.inp = pd.read_csv(inp_path).set_index('time')
        self.ideal = pd.read_csv(ideal_path).set_index('time')

        with open(est_path) as f:
            self.est = json.load(f)
        with open(known_path) as f:
            known_dict = json.load(f)
            known_records = dict()
            for k, v in known_dict.items():
                known_records[k] = [v]
            self.known_df = pd.DataFrame.from_dict(known_records)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_simulation(self):
        output_names = self.ideal.columns.tolist()
        model = Model(self.fmu_path)
        model.inputs_from_df(self.inp)
        model.specify_outputs(output_names)
        model.parameters_from_df(self.known_df)

        res1 = model.simulate(reset=True)
        res2 = model.simulate(reset=False)

        self.assertTrue(res1.equals(res2), 'Dataframes not equal')

        input_size = self.inp.index.size
        result_size = res1.index.size
        self.assertTrue(input_size == result_size, 'Result size different than input')


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestFMPy('test_simulation'))

    return suite


if __name__ == '__main__':
    config_logger(filename='unit_tests.log', level='DEBUG')
    unittest.main()

