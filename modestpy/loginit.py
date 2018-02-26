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


def config_logger(filename='modestpy.log', level='DEBUG'):
    """
    Configure logger using logging.basicConfig. Use only if you don't
    have your own logger in your application.
    :param str filename: Log file name
    :param str level: Logging level ('DEBUG', 'WARNING', 'ERROR', 'INFO')
    """
    import logging

    logging.basicConfig(filename=filename, filemode='w', level=level,
                        format='[%(asctime)s][%(name)s][%(levelname)s] '
                               '%(message)s')
