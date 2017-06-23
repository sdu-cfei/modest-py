"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

from modestpy.log_init import LogInit
LOG_INIT = LogInit(__name__)
LOGGER = LOG_INIT.get_logger()

import random
import copy
import matplotlib.pyplot as plt
import pandas as pd
from modestpy.estim.ga.ga import GA
from modestpy.estim.ps.ps import PS
from modestpy.estim.model import Model
from modestpy.estim.error import calc_err
from modestpy.estim.plots import plot_comparison
from modestpy.estim.plots import plot_parameter_evo


class Estimation:

    # Number of attempts to find nonzero learning data set
    NONZERO_ATTEMPTS = 20

    def __init__(self, workdir, fmu_path, inp, known, est, ideal, \
                 lp_n=None, lp_len=None, lp_frame=None, vp=None, \
                 ic_pars=None):
        """
        Returns an ``Estimation`` instance.

        Index in DataFrames ``inp`` and ``ideal`` must be named 'time'
        and given in seconds. The index name assertion check is
        implemented to avoid situations in which a user reads DataFrame
        from a csv and forgets to use ``DataFrame.set_index(column_name)``
        (it happens quite often...).

        Parameters:
        -----------
        workdir: str
            Output directory
        fmu_path: str
            Path to the FMU
        inp: pandas.DataFrame
            Input data, index given in seconds and named ``time``
        known: dict
            Dictionary with known parameters (``parameter_name: value``)
        est: dict
            Dictionary with estimated parameters
            (``par_name: tuple(guess value, lo limit, hi limit)``)
        ideal: pandas.DataFrame
            Ideal solution (usually measurements), 
            index in seconds and named ``time``
        lp_n: int, None
            Number of learning periods, one if ``None``
        lp_len: float, None
            Length of a single learning period, entire ``lp_frame`` if ``None``
        lp_frame: tupe of floats or None
            Learning period time frame, entire data set if ``None``
        vp: tuple of floats
            Validation period, entire data set if ``None``
        ic_pars: dict
            Parameters defining initial condition, initialized from ``ideal``
            (``par_name``: ``ideal_column_name``)
        """
        # est tuple indices
        est_init = 0  # Initial value
        est_lo = 1    # Lower bound
        est_hi = 2    # Upper bound

        # Sanity checks
        assert inp.index.equals(ideal.index), 'inp and ideal indexes are not matching'
        for v in est:
            assert  (est[v][est_init] >= est[v][est_lo])  \
                and (est[v][est_init] <= est[v][est_hi]), \
                'Initial value out of limits ({})'.format(v)

        # Input data
        self.workdir = workdir
        self.fmu_path = fmu_path
        self.inp = inp
        self.known = known
        self.est = est
        self.ideal = ideal

        # Estimation parameters
        self.GA_POP_SIZE = max((3 * len(est.keys()), 10))
        self.GA_GENERATIONS = 50
        self.PS_MAX_ITER = 100

        # Learning periods
        self.lp = self._select_lp(lp_n, lp_len, lp_frame)

        # Validation period
        if vp is not None:
            self.vp = vp
        else:
            self.vp = (ideal.index[0], ideal.index[-1])

        # Initial condition parameters
        self.ic_pars = ic_pars     # dict (par_name: ideal_column_name)

        # Variables storing results
        self.plots = dict()        # plot_name: matplotlib.Axes
        self.all = pd.DataFrame()  # Estimates and errors from all runs
        self.eee = pd.DataFrame()  # Error evolution envelope
                                   # Columns:
                                   # iter, method1, method2, ..., err1, err2, ...

    # PUBLIC METHODS =====================================================

    def estimate(self):
        """
        Estimates parameters using the previously defined settings.

        Returns
        -------
        dict
        """
        n = 0
        for period in self.lp:
            # Slice data
            start, stop = period[0], period[1]
            inp_slice = self.inp.loc[start:stop]
            ideal_slice = self.ideal.loc[start:stop]

            # Get data for IC parameters and add to known parameters
            if self.ic_pars:
                for par in self.ic_pars:
                    ic = ideal_slice[self.ic_pars[par]].iloc[0] 
                    self.known[par] = ic

            # Initialize GA
            ga = GA(self.fmu_path, inp_slice, self.known, self.est, \
                    ideal_slice, generations=self.GA_GENERATIONS, \
                    epsilon=0.001, look_back=10, pop_size=self.GA_POP_SIZE, \
                    uniformity=0.5, mut=0.1, mut_inc=0.3, trm_size=6)

            # Run GA
            ga_estimates = ga.estimate()
            ga_est_dict = ga_estimates.to_dict('records')[0]

            # Update est dictionary
            for p in ga_est_dict:
                try:
                    new_value = ga_est_dict[p]
                    lo_limit = self.est[p][1]
                    hi_limit = self.est[p][2]
                    self.est[p] = (new_value, lo_limit, hi_limit)

                except KeyError as e:
                    LOGGER.error('Key not found: {}\n'.format(p))
                    raise e


    # PRIVATE METHODS ====================================================

    def _select_lp(self, lp_n=None, lp_len=None, lp_frame=None):
        """
        Selects random learning periods within ``lp_frame``.

        Each learning period has the length of ``lp_len``. Periods may overlap.
        Ensures that a period with null data for any ``ideal`` variable is not selected.
        If ``None`` is given for any of

        Parameters
        ----------
        lp_n: int, optional
            Number of periods, default: 1
        lp_len: int, optional
            Period length in seconds, default: all data
        lp_frame: tuple of floats or ints, optional
            Learning periods are selected within this time frame (start, end), default: all data

        Returns
        -------
        list of tuples of floats
        """
        # Defaults
        if lp_n is None:
            lp_n = 1
        if lp_len is None:
            lp_len = self.ideal.index[-1] - self.ideal.index[0]
        if lp_frame is None:
            lp_frame = (self.ideal.index[0], self.ideal.index[-1])

        # Assign time frame
        t0 = lp_frame[0]
        tend = lp_frame[1]
        assert lp_len <= tend - t0, 'Learning period length cannot be longer than data length!'
        # lp_len = int(lp_len)  # TODO: figure out if this line is needed

        # Return variable
        lp = []

        for i in range(lp_n):
            chosen_lp = False
            tries_left = Estimation.NONZERO_ATTEMPTS

            while not chosen_lp and tries_left > 0:
                new_t0 = random.randint(t0, tend - lp_len)
                new_tend = new_t0 + lp_len

                ideal_nonzero = self._all_columns_nonzero(self.ideal.loc[new_t0:new_tend])

                if ideal_nonzero:
                    lp.append((new_t0, new_tend))
                    chosen_lp = True
                else:
                    tries_left -= 1
                    LOGGER.warning('Zero ideal solution not allowed, selecting another one...')
                    LOGGER.warning('Number of tries left: {}'.format(tries_left))

            if tries_left == 0:
                LOGGER.error('Nonzero ideal solution not found ({} attempts)'
                      .format(Estimation.NONZERO_ATTEMPTS))
                raise Exception

        return lp

    def _set_ic_par(self):
        """
        Sets parameters that have to read IC from sensors.
        """
        pass

    def _validate(self):
        """
        Validation simulation
        """
        pass

    def _all_columns_nonzero(self, df):
        """
        Checks whether all columns in DataFrame are nonzero.

        Parameters
        ----------
        df: DataFrame
        
        Returns
        -------
        boolean
        """
        assert df.empty is False, 'This DataFrame should not be empty. Something is wrong.'
        is_zero = (df == 0).all()
        for col in is_zero:
            if col == True:  # Never use ``is`` with numpy.bool objects
                return False
        return True


if __name__ == "__main__":

    # Example
    import json

    workdir = "home/krzysztof/Desktop/temp"
    fmu_path = "./tests/resources/simple2R1C/Simple2R1C.fmu"
    inp = pd.read_csv("./tests/resources/simple2R1C/inputs.csv").set_index('time')
    known = json.load(open("./tests/resources/simple2R1C/known.json"))
    est = json.load(open("./tests/resources/simple2R1C/est.json"))
    ideal = pd.read_csv("./tests/resources/simple2R1C/result.csv").set_index('time')

    session = Estimation(workdir, fmu_path, inp, known, est, ideal,
                         lp_n=5, lp_len=100, lp_frame=None)
    estimates = session.estimate()