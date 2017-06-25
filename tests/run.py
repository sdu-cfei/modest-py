import unittest
import test_ga
import test_lm
import test_ps
import test_utilities


def all_suites():

    suites = [
        test_ga.suite(),
        test_lm.suite(),
        test_ps.suite(),
        test_utilities.suite()
    ]

    all_suites = unittest.TestSuite(suites)
    return all_suites

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    test_suite = all_suites()
    runner.run (test_suite)
