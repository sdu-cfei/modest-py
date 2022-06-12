#!/usr/bin/env python
from modestpy.loginit import config_logger
from modestpy.test import run

if __name__ == "__main__":
    config_logger(filename="unit_tests.log", level="DEBUG")
    run.tests()
