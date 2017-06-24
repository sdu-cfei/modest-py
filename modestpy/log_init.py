"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

import logging
import os

class LogInit:
    """
    Used for logger initialization. Ensures, that only the top one logger gets new handlers.
    The top logger is found at runtime. 
    
    Add the following lines to all modules:
    
    from log_init import LogInit
    LOG_INIT = LogInit(__name__)
    LOGGER = LOG_INIT.get_logger()
    """

    ROOT_ADDED = False
    FIRST_MOD = None  # Name of the module calling this Class first

    # Set to true if multiprocessing is in use (affects logging)
    MULTIPROCESSING = False

    def __init__(self, name):
        """
        Initializes new logging.Logger 

        Parameters
        ----------
        name: str
            Module name
        """
        if LogInit.ROOT_ADDED is False:
            logger = logging.getLogger('modest')
            logger.setLevel(logging.INFO)

            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)

            filename = 'modest.log'
            if LogInit.MULTIPROCESSING is True:
                filename = 'modest_{}.log'.format(os.getpid())

            fh = logging.FileHandler(filename, mode='w')
            fh.setLevel(logging.DEBUG)

            formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
            sh.setFormatter(formatter)
            fh.setFormatter(formatter)

            logger.addHandler(sh)
            logger.addHandler(fh)

            LogInit.ROOT_ADDED = True
            LogInit.FIRST_MOD = name

            logger.setLevel(logging.INFO)
            logger.info('Log file created on demand of [{}]'.format(name))
            logger.info('Process PID: {}'.format(os.getpid()))

        self.logger = logging.getLogger('modest.' + name)
        self.logger.setLevel(logging.DEBUG)


    def get_logger(self):
        """
        Returns the assigned logger
        
        :return: logging.Logger
        """
        return self.logger
