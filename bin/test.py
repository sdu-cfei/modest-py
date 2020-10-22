#!/usr/bin/env python
from modestpy.test import run
from modestpy.loginit import config_logger

if __name__ == "__main__":
    config_logger(filename='unit_tests.log', level='DEBUG')
    run.tests()
