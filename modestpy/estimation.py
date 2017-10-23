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

import random
import copy
import os
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import numpy as np
from pyfmi.fmi import FMUException
try:
    from pandas.plotting import scatter_matrix
except ImportError:
    from pandas.tools.plotting import scatter_matrix
from modestpy.estim.ga.ga import GA
from modestpy.estim.ps.ps import PS
from modestpy.estim.sqp.sqp import SQP
from modestpy.estim.model import Model
import modestpy.estim.error
from modestpy.estim.plots import plot_comparison
import modestpy.utilities.figures as figures

class Estimation:
    """
    Public API of ``modestpy``.

    This class allows to use multiple estimation methods in a single
    estimation pipeline. The user needs only to instantiate this class,
    define the desired sequence of estimation methods and call
    ``estimate()``. All results are saved in the working directory 
    ``workdir``.

    .. note:: ``inp`` and ``ideal`` DataFrames **must have** index
              named ``time``. This is to avoid a common user mistake 
              of loading DataFrame from a csv and forgetting to set 
              the right index. The index should be in seconds.
              TODO: Is it still true?

    Methods
    -------
    estimate(get='best')
        Estimates parameters, saves results to ``workdir`` and
        returns chosen type of estimates ('avg' or 'best').
    validate()
        Performs a validation of the model on the chosen validation period

    Examples
    --------
    >>> from modestpy import Estimation
    >>> session = Estimation(workdir, fmu_path, inp, known, est, ideal,
                             lp_n=2, lp_len=25000, lp_frame=(0, 25000),
                             vp = (150000, 215940), ic_param={'Tstart': 'T'},
                             methods=('GA', 'PS'),
                             ga_opts={'maxiter': 5, 'tol': 0.001},
                             ps_opts={'maxiter': 20, 'tol': 0.0001},
                             ftype='RMSE') 

    >>> estimates = session.estimate()
    >>> err, res = session.validate()
    """

    # Number of attempts to find nonzero learning data set
    NONZERO_ATTEMPTS = 20

    # Ploting settings
    FIG_DPI = 150
    FIG_SIZE = (10, 6)

    def __init__(self, workdir, fmu_path, inp, known, est, ideal,
                 lp_n=None, lp_len=None, lp_frame=None, vp=None,
                 ic_param=None, methods=('GA', 'PS'), ga_opts={}, ps_opts={}, sqp_opts={}, 
                 fmi_opts={}, ftype='RMSE', seed=None):
        """
        Index in DataFrames ``inp`` and ``ideal`` must be named 'time'
        and given in seconds. The index name assertion check is
        implemented to avoid situations in which a user reads DataFrame
        from a csv and forgets to use ``DataFrame.set_index(column_name)``
        (it happens quite often...). TODO: Check index name assertion.

        Currently available estimation methods:
            - GA - genetic algorithm
            - PS - pattern search (Hooke-Jeeves)

        .. note:: Guess value of estimated parameters is not taken into account in GA.

        Parameters:
        -----------
        workdir: str
            Output directory, must exist
        fmu_path: str
            Absolute path to the FMU
        inp: pandas.DataFrame
            Input data, index given in seconds and named ``time``
        known: dict(str: float)
            Dictionary with known parameters (``parameter_name: value``)
        est: dict(str: tuple(float, float, float))
            Dictionary defining estimated parameters,
            (``par_name: (guess value, lo limit, hi limit)``)
        ideal: pandas.DataFrame
            Ideal solution (usually measurements), 
            index in seconds and named ``time``
        lp_n: int or None
            Number of learning periods, one if ``None``
        lp_len: float or None
            Length of a single learning period, entire ``lp_frame`` if ``None``
        lp_frame: tupe of floats or None
            Learning period time frame, entire data set if ``None``
        vp: tuple(float, float) or None
            Validation period, entire data set if ``None``
        ic_param: dict(str, str) or None
            Mapping between model parameters used for IC and variables from ``ideal``
        methods: tuple(str, str)
            List of methods to be used in the pipeline
        ga_opts: dict
            Genetic algorithm options
        ps_opts: dict
            Pattern search options
        sqp_opts: dict
            SQP solver options
        fmi_opts: dict
            Additional options to be passed to the FMI model (e.g. solver tolerance)
        ftype: string
            Cost function type. Currently 'NRMSE' (advised for multi-objective estimation) or 'RMSE'.
        seed: None or int
            Random number seed. If None, current time or OS specific randomness is used.
        """
        # Sanity checks
        assert inp.index.equals(ideal.index), 'inp and ideal indexes are not matching'

        init, lo, hi = 0, 1 ,2  # Initial value, lower bound, upper bound indices
        for v in est:
            assert  (est[v][init] >= est[v][lo])  \
                and (est[v][init] <= est[v][hi]), \
                'Initial value out of limits ({})'.format(v)

        # Random seed
        if seed is not None:
            LOGGER.info('Setting random seed: {}'.format(seed))
            random.seed(seed)
            np.random.seed(seed)  # Important for other libraries, like pyDOE

        # Input data
        self.workdir = workdir
        self.fmu_path = fmu_path
        self.inp = inp
        self.known = known
        self.est = est
        self.ideal = ideal
        self.methods = methods
        self.ftype = ftype

        # Results placeholders
        self.best_per_run = pd.DataFrame()
        self.final = pd.DataFrame()

        # Estimation options
        # GA options
        self.GA_OPTS = {
            'maxiter':      50,
            'pop_size':     max((4 * len(est.keys()), 20)),
            'tol':          1e-6,
            'mut':          0.05,
            'mut_inc':      0.3,
            'uniformity':   0.5,
            'look_back':    50,
            'lhs':          False,
            'ftype':        ftype,
            'fmi_opts':     fmi_opts
        }  # Default
        self.GA_OPTS['trm_size'] = max(self.GA_OPTS['pop_size']//5, 1)  # Default
        self.GA_OPTS = self._update_opts(self.GA_OPTS, ga_opts, 'GA')  # User options

        # PS options
        self.PS_OPTS = {
            'maxiter':  500,
            'rel_step': 0.02,
            'tol':      1e-11,
            'try_lim':  1000,
            'ftype':    ftype,
            'fmi_opts': fmi_opts
        }  # Default
        self.PS_OPTS = self._update_opts(self.PS_OPTS, ps_opts, 'PS')  # User options

        # SQP options
        self.SQP_OPTS = {
            'scipy_opts': {'disp': True,
                           'iprint': 2,
                           'maxiter': 150,
                           'full_output': True},
            'ftype': ftype,
            'fmi_opts': fmi_opts
        } # Default
        self.SQP_OPTS = self._update_opts(self.SQP_OPTS, sqp_opts, 'SQP')  # User options

        # Method dictionary
        self.method_dict = {
            'GA': (GA, self.GA_OPTS),
            'PS': (PS, self.PS_OPTS),
            'SQP': (SQP, self.SQP_OPTS)
        }  # Key -> method name, value -> (method class, method options)

        # List of learning periods (tuples with start, stop)
        self.lp = self._select_lp(lp_n, lp_len, lp_frame)

        # Validation period (a tuple with start, stop)
        if vp is not None:
            self.vp = vp
        else:
            self.vp = (ideal.index[0], ideal.index[-1])

        # Initial condition parameters
        # Take first value from time series from 'ideal' column
        self.ic_param = ic_param  # dict (par_name: ideal_column_name)

    # PUBLIC METHODS =====================================================

    def estimate(self, get='best'):
        """
        Estimates parameters.

        Returns average or best estimates depending on ``get``.
        Average parameters are calculated as arithmetic average
        from all learning periods. Best parameters are those which
        resulted in the lowest error during respective learning period.
        It is advised to use 'best' parameters.

        The chosen estimates ('avg' or 'best') are saved
        in a csv file ``final.csv`` in the working directory.
        In addition estimates and errors from all learning periods 
        are saved in ``best_per_run.csv``.

        Parameters
        ----------
        get: str, default 'best'
            Type of returned estimates: 'avg' or 'best'

        Returns
        -------
        dict(str: float)
        """
        # (0) Sanity checks
        allowed_types = ['best', 'avg']
        assert get in allowed_types, 'get={} is not allowed'.format(get)

        # (1) Initialize local variables
        methods = self.methods      # Tuple with method names, e.g. ('GA', 'PS'), ('GA', 'SQP') or ('GA', )
        plots = list()              # List of plots to saved

        cols = ['_method_', '_error_'] + [par_name for par_name in self.est]
        summary = pd.DataFrame(columns=cols)  # Estimates and errors from all iterations from all methods
        summary.index.name = '_iter_'

        summary_list = list()  # List of DataFrames with summaries from all runs

        # (2) Double step estimation
        n = 1  # Learning period counter

        for period in self.lp:
            # (2.1) Copy initial parameters
            est = copy.copy(self.est)

            # (2.2) Slice data
            start, stop = period[0], period[1]
            inp_slice = self.inp.loc[start:stop]
            ideal_slice = self.ideal.loc[start:stop]

            # (2.3) Get data for IC parameters and add to known parameters
            if self.ic_param:
                for par in self.ic_param:
                    ic = ideal_slice[self.ic_param[par]].iloc[0] 
                    self.known[par] = ic

            # (2.4) Iterate over estimation methods (append results from all)
            m = 0  # Method counter
            for m_name in methods:
                # (2.4.1) Instantiate method class
                m_class = self.method_dict[m_name][0]
                m_opts = self.method_dict[m_name][1]

                m_inst = m_class(self.fmu_path, inp_slice, self.known, est, ideal_slice,
                                 **m_opts)

                # (2.4.2) Estimate
                m_estimates = m_inst.estimate()

                # (2.4.3) Update current estimates (stored in self.est dictionary)
                for key in est:
                    new_value = m_estimates[key].iloc[0]
                    est[key] = (new_value, est[key][1], est[key][2])

                # (2.4.4) Append summary
                full_traj = m_inst.get_full_solution_trajectory()
                if m > 0:
                    full_traj.index += summary.index[-1]  # Add iterations from previous methods
                summary = summary.append(full_traj, verify_integrity=True)
                summary.index.rename('_iter_', inplace=True)

                # (2.4.5) Save method's plots
                plots = m_inst.get_plots()
                for p in plots:
                    fig = figures.get_figure(p['axes'])
                    fig_file = os.path.join(self.workdir, "{}_{}.png".format(p['name'], n))
                    fig.set_size_inches(Estimation.FIG_SIZE)
                    fig.savefig(fig_file, dpi=Estimation.FIG_DPI)
                plt.close('all')

                # (2.4.6) Increase method counter
                m += 1

            # (2.5) Add summary from this run to the list of all summaries
            summary_list.append(summary)
            summary = pd.DataFrame(columns=cols)  # Reset

            # (2.6) Increase learning period counter
            n += 1

        # (3) Get and save best estimates per run and final estimates
        best_per_run = self._get_finals(summary_list)
        best_per_run.to_csv(os.path.join(self.workdir, 'best_per_run.csv'))
        
        if get == 'best':
            cond = best_per_run['_error_'] == best_per_run['_error_'].min()
            final = best_per_run.loc[cond].iloc[0:1]  # Take only one if more than one present
            final = final.drop('_error_', axis=1)
        elif get == 'avg':
            final = best_per_run.drop('_error_', axis=1).mean().to_frame().T
        else:
            # This shouldn't happen, because the type is checked at (0)
            raise RuntimeError('Unknown type of estimates: {}'.format(get))

        final_file = os.path.join(self.workdir, 'final.csv')
        final.to_csv(final_file, index=False)

        # (4) Save summaries from all learning periods
        for s, n in zip(summary_list, range(1, len(summary_list) + 1)):
            sfile = os.path.join(self.workdir, 'summary_{}.csv'.format(n))
            s.to_csv(sfile)

        # (5) Save error plot including all learning periods
        ax = self._plot_error_per_run(summary_list, err_type=self.ftype)
        fig = figures.get_figure(ax)
        fig.savefig(os.path.join(self.workdir, 'errors.png'))

        # (6) Assign results to instance attributes
        self.best_per_run = best_per_run
        self.final = final

        # (7) Return final estimates
        return final

    def validate(self):
        """
        Performs a simulation with estimated parameters (average or best) 
        for the previously selected validation period.

        Returns
        -------
        dict
            Validation error, keys: 'tot', '<var1>', '<var2>', ...
        pandas.DataFrame
            Simulation result
        """
        # Get estimates
        est = self.final
        est.index = [0] # Reset index (needed by model.set_param())

        LOGGER.info('Validation of parameters: {}'.format(str(est.iloc[0].to_dict())))

        # Slice data
        start, stop = self.vp[0], self.vp[1]
        inp_slice = self.inp.loc[start:stop]
        ideal_slice = self.ideal.loc[start:stop]

        # Initialize IC parameters and add to known
        if self.ic_param:
            for par in self.ic_param:
                ic = ideal_slice[self.ic_param[par]].iloc[0] 
                self.known[par] = ic

        # Initialize model
        model = Model(self.fmu_path)
        model.set_input(inp_slice)
        model.set_param(est)
        model.set_param(self.known)
        model.set_outputs(list(self.ideal.columns))

        # Simulate and get error
        com_points = len(ideal_slice) - 1
        try:
            result = model.simulate(com_points=com_points)
        except FMUException as e:
            msg = 'Problem found inside FMU. Did you set all parameters? Log:\n'
            msg += str(model.model.model.print_log())
            LOGGER.error(msg)
            raise FMUException(e)

        err = modestpy.estim.error.calc_err(result, ideal_slice)

        # Create validation plot
        plots = dict()
        ax = plot_comparison(result, ideal_slice, f=None)
        fig = figures.get_figure(ax)
        fig.set_size_inches(Estimation.FIG_SIZE)
        fig.savefig(os.path.join(self.workdir, 'validation.png'), dpi=Estimation.FIG_DPI)

        # Return
        return err, result

    # PRIVATE METHODS ====================================================

    def _get_finals(self, summary_list):
        """
        Returns final estimates and errors from all learning periods

        :param list(DataFrame) summary_list: List of all summaries from all runs
        :param bool avg: If true, return average estimates, else return best estimates
        :return: DataFrame with final estimates
        """
        finals = pd.DataFrame()
        for s in summary_list:
            finals = finals.append(s.drop('_method_', axis=1).iloc[-1:], ignore_index=True)
        finals.index += 1  # Start from 1
        finals.index.name = '_run_'

        return finals

    def _update_opts(self, opts, new_opts, method):
        """
        Updates the dictionary with method options.

        :param dict opts: Options to be updated
        :param dict new_opts: New options (can contain a subset of opts keys)
        :param str method: Method name, 'GA', 'PS' etc. (used only for logging)
        :return: Updated dict
        """
        if len(new_opts) > 0:
            for key in new_opts:
                if key not in opts.keys():
                    msg = 'Unknown key: {}'.format(key)
                    LOGGER.error(msg)
                    raise KeyError(msg)
                LOGGER.info('User defined option ({}): {} = {}'.format(method, key, new_opts[key]))
                opts[key] = new_opts[key]
        return opts

    def _plot_error_per_run(self, summary_list, err_type):
        """
        :param list(DataFrame) summary_list: Summary list
        :param str err_type: Error type
        :return: Axes
        """
        # Error evolution per estimation run
        err = pd.DataFrame()
        for s, n in zip(summary_list, range(1, len(summary_list) +1)):
            next_err = pd.Series(data=s['_error_'], name='error #{}'.format(n))
            err = pd.concat([err, next_err], axis=1)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=Estimation.FIG_SIZE, dpi=Estimation.FIG_DPI)
        err.plot(ax=ax)

        # Get line colors
        lines = ax.get_lines()
        colors = [l.get_color() for l in lines]

        # Method switch marks
        xloc, yloc = self._get_method_switch_xy(summary_list)

        mltp = len(xloc) // len(colors)  # In cases there is more switches than lines
        if mltp > 1:
            colors_copy = copy.copy(colors)
            colors = list()
            for c in colors_copy:
                for i in range(mltp):
                    colors.append(c)

        for x, y, c in zip(xloc, yloc, colors * mltp):
            ax.scatter(x, y, marker='o', c='white', edgecolors=c, lw=1.5, zorder=10)

        # Formatting
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Error ({})".format(err_type))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        return ax

    def _get_method_switch_xy(self, summary_list):
        """
        Returns a tuple with two lists describing x, y coordinates
        marking when there was a switch to a next method in the estimation.

        The first list contains x coordinates, the second list contains y coordinates.
        x coordinates represent iterations. y coordinates represent simulation error.

        :param list(DataFrame) summary_list: List of DataFrames with summary
        :return: tuple(list(int), list(int))
        """

        # Construct an array with indices marking method switches.
        # E.g. if there were 3 methods used in 2 estimation runs,
        # the returned list looks as follows:
        #
        # [[11, 34, 20],  -> estimation run #1
        #  [23, 49, 15]]  -> estimation run #2
        #    |   |   |
        #    m1  m2  m3
        #
        # where m1, m2, m3 are iteration numbers when method 1, 2, 3 started.
        switch_array = list()
        for s in summary_list:
            methods = s['_method_'].values
            switch = list()
            last = None
            for n, i in zip(methods, range(len(methods))):
                if last is None:
                    last = n
                if n != last:
                    switch.append(i)
                last = n
            switch_array.append(switch)

        # Generate x, y lists
        xloc = list()
        yloc = list()
        i = 0
        for run in switch_array:
            for index in run:
                xloc.append(index)
                yloc.append(summary_list[i]['_error_'].iloc[index-1])
            i += 1

        return xloc, yloc

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
