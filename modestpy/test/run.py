"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.

Author: Krzysztof Arendt
"""

import unittest
import test_ga
import test_lm
import test_ps
import test_estimation
import test_utilities


def all_suites():

    suites = [
        test_ga.suite(),
        test_lm.suite(),
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
