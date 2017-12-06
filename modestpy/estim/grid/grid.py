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

from modestpy.log_init import LogInit
LOG_INIT = LogInit(__name__)
LOGGER = LOG_INIT.get_logger()

from modestpy.estim.model import Model
from modestpy.estim.error import calc_err
import modestpy.estim.plots as plots
import pandas as pd
import copy
import os
from random import random

class Grid:

    def __init__(self, fmu_path, inp, known, est, ideal, opts=None, ftype='RMSE'):
        """
        :param fmu_path: string, absolute path to the FMU
        :param inp: DataFrame, columns with input timeseries, index in seconds
        :param known: Dictionary, key=parameter_name, value=value
        :param est: Dictionary, key=parameter_name, value=tuple (guess value, lo limit, hi limit), guess can be None
        :param ideal: DataFrame, ideal solution to be compared with model outputs (variable names must match)
        :param dict opts: Additional FMI options to be passed to the simulator (consult FMI specification)
        :param string ftype: Cost function type. Currently 'NRMSE' (advised for multi-objective estimation) or 'RMSE'.
        """
        assert inp.index.equals(ideal.index), 'inp and ideal indexes are not matching'
        pass # TODO: To be completed

    def search(self, N):
        """
        Performs grid search, slicing each dimension into `N` points.
        Returns DataFrame with all parameters and errors.

        :param int N: Number of points in each dimension
        :return: DataFrame
        """
        pass # TODO: To be completed
