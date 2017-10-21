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
from modestpy.test import test_ga
from modestpy.test import test_lm
from modestpy.test import test_ps
from modestpy.test import test_estimation
from modestpy.test import test_utilities


def all_suites():

    suites = [
        test_ga.suite(),
        test_ps.suite(),
        test_estimation.suite(),
        test_utilities.suite()
    ]

    all_suites = unittest.TestSuite(suites)
    return all_suites

def tests():
    runner = unittest.TextTestRunner()
    test_suite = all_suites()
    runner.run (test_suite)

if __name__ == '__main__':
    tests()
