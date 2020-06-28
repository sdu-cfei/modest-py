#!/usr/bin/env python
from modestpy.test import run
from modestpy.loginit import config_logger

config_logger(filename='unit_tests.log', level='DEBUG')
run.tests()
