"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

import multiprocessing
import pandas as pd

class GAPS:

    def __init__(self, workdir, fmu_path, inp, known, est, ideal):
        """
        :param workdir: string, working directory (for data and plot saving)
        :param fmu_path: string
        :param inp: DataFrame, time in seconds (float) as index
        :param known: dict, key=parameter_name, value=value
        :param est: dict, key=parameter_name, value=tuple (guess value, lo limit, hi limit), guess can be None
        :param ideal: DataFrame, time in seconds as index  # TODO: accept datetime
        """
        # Sanity checks
        assert inp.index.equals(ideal.index), 'inp and ideal indexes are not matching'
        for v in est:
            assert (est[v][LearnMan.INIT] >= est[v][LearnMan.LO]) and (est[v][LearnMan.INIT] <= est[v][LearnMan.HI]), \
                'Initial value out of limits ({})'.format(v)

        # Input data
        self.workdir = workdir
        self.fmu_path = fmu_path
        self.inp = inp
        self.known = known
        self.est = est
        self.ideal = ideal

        # Estimation algorithm parameters
        self.GA_POP_SIZE = 3 * len(est.keys())
        self.GA_GENERATIONS = 50
        self.PS_MAX_ITER = 100

        # Multiprocessing
        self.cpu_num = multiprocessing.cpu_count() - 1  # Leave one spare core for other tasks
        if self.cpu_num == 0:  # Minimum one
            self.cpu_num = 1

    def select_lp():
        """
        Select learning periods. If not selected, entire inp data is used.
        """
        pass

    def select_vp():
        """
        Select validation period. If not selected, entire inp data set is used.
        """
        pass

    def set_ic_par():
        """
        Sets parameters that have to read IC from sensors.
        """
        pass
    
    def estimate():
        """
        Selects learning periods and performs the esimation using multiple cores
        """
        pass
    
    def validate():
        """
        Validation simulation
        """
        pass