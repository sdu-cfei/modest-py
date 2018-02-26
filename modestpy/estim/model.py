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

import logging
from modestpy.fmi.model import Model as FmiModel

# PyFmi log level (controls the amount of information saved to the log file)
# (watch out for the file writing overhead)
FMI_NOTHING = 0
FMI_FATAL = 1
FMI_ERROR = 2
FMI_WARNING = 3
FMI_INFO = 4
FMI_VERBOSE = 5
FMI_DEBUG = 6
FMI_ALL = 7

# Printing on the screen
VERBOSE = True


class Model(object):
    """ Model for static parameter estimation """
    def __init__(self, fmu_path, opts=None):
        self.logger = logging.getLogger(type(self).__name__)

        self.model = FmiModel(fmu_path, opts=opts)

        # Log level
        try:
            self.model.model.set_log_level(FMI_WARNING)
        except AttributeError as e:
            self.logger.error(e.message)
            self.logger.error('Proceeding with standard log level...')

        # Simulation count
        self.sim_count = 0

    def set_input(self, df, exclude=list()):
        """ Sets inputs.

        :param df: Dataframe, time given in seconds
        :param exclude: list of strings, names of columns to be excluded
        :return: None
        """
        self.model.inputs_from_df(df, exclude)

    def set_param(self, df):
        """ Sets parameters. It is possible to set only a subset of model parameters.

        :param df: Dataframe with header and a single row of data
        :return: None
        """
        self.model.parameters_from_df(df)

    def set_outputs(self, outputs):
        """ Sets output variables.

        :param outputs: list of strings
        :return: None
        """
        self.model.specify_outputs(outputs)

    def simulate(self, com_points=None):
        # TODO: com_points should be adjusted to the number of samples
        self.sim_count += 1
        self.info('Simulation count = ' + str(self.sim_count))
        return self.model.simulate(com_points=com_points)

    def info(self, txt):
        class_name = self.__class__.__name__
        if VERBOSE:
            if isinstance(txt, str):
                print('[' + class_name + '] ' + txt)
            else:
                print('[' + class_name + '] ' + repr(txt))
