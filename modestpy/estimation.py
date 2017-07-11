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
import os
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
from pyfmi.fmi import FMUException
try:
    from pandas.plotting import scatter_matrix
except ImportError:
    from pandas.tools.plotting import scatter_matrix
from modestpy.estim.ga.ga import GA
from modestpy.estim.ps.ps import PS
from modestpy.estim.model import Model
import modestpy.estim.error
from modestpy.estim.plots import plot_comparison

class Estimation:
    """
    Public API of ``modestpy``.

    This class wraps genetic algorithm (GA) and pattern search (PS)
    methods into a single combined estimation algorithm. The user
    needs only to instantiate this class and then call the method
    ``estimate()``. All results are saved in the working directory 
    ``workdir``.

    .. note:: To switch off GA or PS, simply set the number of
              iterations to 0, e.g. ``ga_iter=0`` or ``ps_iter=0``.

    .. note:: Guess values of estimated parameters are taken into
              account only if GA is switched off, i.e.
              ``ga_iter = 0``. Otherwise GA selects random initial
              guesses itself.

    .. note:: ``inp`` and ``ideal`` DataFrames **must have** index
              named ``time``. This is to avoid a common user mistake 
              of loading DataFrame from a csv and forgetting to set 
              the right index. The index should be in seconds.

    Methods
    -------
    estimate(get_type='avg')
        Estimates parameters, saves results to ``workdir`` and
        returns chosen type of estimates ('avg' or 'best')
    validate(use_type='avg')
        Performs a validation of the model using chosen
        type of estimates ('avg' or 'best')

    Examples
    --------
    >>> from modestpy import Estimation
    >>> session = Estimation(workdir, fmu_path, inp, known, est, ideal,
                             lp_n=3, lp_len=3600, lp_frame=None, vp=None,
                             ic_param={'Tstart': 'T'}, ga_iter=30, ps_iter=30)
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
                 ic_param=None, ga_iter=None, ga_tol=None,
                 ps_iter=None, ps_tol=None):
        """
        Constructor.

        Index in DataFrames ``inp`` and ``ideal`` must be named 'time'
        and given in seconds. The index name assertion check is
        implemented to avoid situations in which a user reads DataFrame
        from a csv and forgets to use ``DataFrame.set_index(column_name)``
        (it happens quite often...).

        Guess value of estimated parameters is taken into account only
        if GA is switched of, i.e. ``ga_iter = 0``. Otherwise, GA
        selects random guesses itself.

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
        ga_iter: int or None
            Maximum number of GA iterations (generations). If 0, GA is switched off. Default: 50.
        ga_tol: float or None
            GA tolerance (accepted error)
        ps_iter: int or None
            Maximum number of PS iterations. If 0, PS is switched off. Default: 50.
        ps_tol: float or None
            PS tolerance (accepted error)
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
        self.GA_POP_SIZE = max((4 * len(est.keys()), 20))   # Default
        self.GA_GENERATIONS = 50                            # Default
        self.GA_LOOK_BACK = 10                              # Default
        self.GA_TOL = 1e-6                                  # Default
        self.PS_MAX_ITER = 50                               # Dafault
        self.PS_TOL = 1e-7                                  # Default
        if ga_iter is not None:
            self.GA_GENERATIONS = ga_iter
        if ps_iter is not None:
            self.PS_MAX_ITER = ps_iter
        if ga_tol is not None:
            self.GA_TOL = ga_tol
        if ps_tol is not None:
            self.PS_TOL = ps_tol

        # Learning periods
        self.lp = self._select_lp(lp_n, lp_len, lp_frame)

        # Validation period
        if vp is not None:
            self.vp = vp
        else:
            self.vp = (ideal.index[0], ideal.index[-1])

        # Initial condition parameters
        self.ic_param = ic_param  # dict (par_name: ideal_column_name)

    # PUBLIC METHODS =====================================================

    def estimate(self, get_type='avg'):
        """
        Estimates parameters using the previously defined settings.

        Returns average or best estimates depending on ``get_type``.
        Average parameters are calculated as arithmetic average
        from all learning periods. Best parameters are those which
        resulted in the lowest error during respective learning period.
        Estimates obtained with 'avg' may be suboptimal, but there is
        higher chance to avoid overfitting. Estimates obtained with 'best'
        sometimes perform better (especially when the error function is 
        convex), and sometimes can be overfitted.

        The chosen type of estimates ('avg' or 'best') is saved
        in a csv file ``final_estimates.csv`` in the working directory.
        In addition estimates and errors from all learning periods 
        are saved in ``all_estimates.csv``.

        Parameters
        ----------
        get_type: str, default 'avg'
            Type of returned estimates: 'avg' or 'best'

        Returns
        -------
        dict(str: float)
        """
        if (self.GA_GENERATIONS <= 0) and (self.PS_MAX_ITER <= 0):
            msg = 'Both GA and PS are switched off, cannot estimate!'
            LOGGER.error(msg)
            raise RuntimeError(msg)

        plots = dict()

        ga_estimates = None
        ps_estimates = None
        final_estimates = None
        all_estimates = pd.DataFrame()
        err_evo = pd.DataFrame(columns=['iter'])

        n = 0
        for period in self.lp:
            # Slice data
            start, stop = period[0], period[1]
            inp_slice = self.inp.loc[start:stop]
            ideal_slice = self.ideal.loc[start:stop]

            # Get data for IC parameters and add to known parameters
            if self.ic_param:
                for par in self.ic_param:
                    ic = ideal_slice[self.ic_param[par]].iloc[0] 
                    self.known[par] = ic

            # Genetic algorithm
            if self.GA_GENERATIONS > 0:
                # Initialize GA
                ga = GA(self.fmu_path, inp_slice, self.known, self.est,
                        ideal_slice, generations=self.GA_GENERATIONS,
                        tolerance=self.GA_TOL,
                        look_back=10,
                        pop_size=self.GA_POP_SIZE,
                        uniformity=0.5,
                        mut=0.15,
                        mut_inc=0.5,
                        trm_size=self.GA_POP_SIZE/5)
                # Run GA
                ga_estimates = ga.estimate()
                # Update self.est dictionary
                ga_est_dict = ga_estimates.to_dict('records')[0]
                for p in ga_est_dict:
                    try:
                        new_value = ga_est_dict[p]
                        lo_limit = self.est[p][1]
                        hi_limit = self.est[p][2]
                        self.est[p] = (new_value, lo_limit, hi_limit)
                    except KeyError as e:
                        LOGGER.error('Key not found: {}\n'.format(p))
                        raise e
                # GA errors
                ga_errors = ga.get_errors()
                # Evolution visualization
                plots['ga_{}'.format(n)] = ga.plot_pop_evo()
            else:
                # GA not used, assign empty list
                ga_errors = list()

            # Pattern search
            if self.PS_MAX_ITER > 0:
                # Initialize PS
                ps = PS(self.fmu_path, inp_slice, self.known, self.est, \
                        ideal_slice, max_iter=self.PS_MAX_ITER, tolerance=self.PS_TOL)
                # Run PS
                ps_estimates = ps.estimate()
                # PS errors
                ps_errors = ps.get_errors()
                # PS parameter evolution
                plots['ps_{}'.format(n)] = ps.plot_parameter_evo()
            else:
                # PS not used, assign empty list
                ps_errors = list()

            # Generate error evolution (err_evo) for this learning period
            err_evo_n = self._get_err_evo_n(ga_errors, ps_errors, n)

            # Merge err_evo_n (this run) with err_evo (all runs)
            err_evo = err_evo.merge(err_evo_n, on='iter', how='outer')

            # Increase learning period counter
            n += 1

            # Current estimates
            current_estimates = ps_estimates if ps_estimates is not None else ga_estimates
            current_estimates['error'] = ps_errors[-1] if ps_errors is not None else ga_errors  # BUG if ps_iter = 0

            # Append all estimates
            all_estimates = all_estimates.append(current_estimates, ignore_index=True)

        # Final estimates
        final_estimates = self._get_avg_estimates(all_estimates) if get_type == 'avg' \
                          else self._get_best_estimates(all_estimates) 

        # Generate plots
        plots['err_evo'] = self._plot_err_evo(err_evo)

        if n > 1:
            try:
                plots['all_estimates'] = self._plot_all_estimates(all_estimates)
            except Exception as e:
                # TODO: Why it fails sometimes?
                LOGGER.error('Unable to plot scatter matrix')
                LOGGER.error(e.message)

        # Save plots
        self._save_plots(plots)

        # Save csv files
        err_evo.set_index('iter').to_csv(os.path.join(self.workdir, 'err_evo.csv'))
        all_estimates.to_csv(os.path.join(self.workdir, 'all_estimates.csv'), index=False)
        final_estimates.to_csv(os.path.join(self.workdir, 'final_estimates.csv'), index=False)

        # Return
        return final_estimates.to_dict('records')[0]

    def validate(self, use_type='avg'):
        """
        Performs a simulation with estimated parameters (average or best) 
        for the previously selected validation period.

        Parameters
        ----------
        use_type: string, default 'avg'
            Type of estimates to use ('avg' or 'best')

        Returns
        -------
        float
            Validation error
        pandas.DataFrame
            Simulation result
        """
        # Get estimates
        all_est = pd.read_csv(os.path.join(self.workdir, 'all_estimates.csv'))
        est = self._get_avg_estimates(all_est) if use_type == 'avg' \
              else self._get_best_estimates(all_est)

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
            msg += model.model.model.print_log()
            LOGGER.error(msg)
            raise FMUException(e)

        err = modestpy.estim.error.calc_err(result, ideal_slice)

        # Create validation plot
        plots = dict()
        plots['validation_'+use_type] = plot_comparison(result, ideal_slice, f=None)

        # Save plot
        self._save_plots(plots)

        # Return
        return err, result

    # PRIVATE METHODS ====================================================

    def _get_best_estimates(self, all_estimates):
        """
        Returns best estimates from ``all_estimates``.

        Parameters
        ----------
        all_estimates: pandas.DataFrame
            Estimates and errors from all learning periods
        
        Returns
        -------
        pandas.DataFrame
        """
        best = all_estimates.loc[all_estimates['error'] == all_estimates['error'].min()]
        best = best.drop('error', axis=1)
        best.index = [0]
        return best

    def _get_avg_estimates(self, all_estimates):
        """
        Returns average estimates from ``all_estimates``.

        Parameters
        ----------
        all_estimates: pandas.DataFrame
            Estimates and errors from all learning periods
        
        Returns
        -------
        pandas.DataFrame
        """
        avg = all_estimates.mean().to_frame().T
        avg = avg.drop('error', axis=1)
        return avg

    def _save_plots(self, plots):
        """
        Saves all plots from ``plots`` in the working directory.

        Parameters
        ----------
        plots: list(matplotlib.Axes)

        Returns
        -------
        None
        """
        LOGGER.info('Saving plots...')
        for name in plots:
            LOGGER.info('Saving {}'.format(name))
            ax = plots[name]
            # Get figure
            try:
                # Single plot
                fig = ax.get_figure()
            except AttributeError:
                # Subplots
                try:
                    # 1D grid
                    fig = ax[0].get_figure()
                except AttributeError:
                    # 2D grid
                    fig = ax[0][0].get_figure()
            # Adjust size
            fig.set_size_inches(Estimation.FIG_SIZE)
            # Save file
            filepath = os.path.join(self.workdir, name + '.png')
            fig.savefig(filepath, dpi=Estimation.FIG_DPI)
        # Close all plots (to release instances for garbage collecting)
        plt.close('all')

    def _plot_all_estimates(self, all_estimates):
        """
        Generates a scatter matrix plot for all estimates and errors.

        Parameters
        ----------
        all_estimates: pandas.DataFrame
            All estimates and errors

        Returns
        -------
        matplotlib.Axes
        """
        ax = scatter_matrix(all_estimates, marker='o', alpha=0.5)
        return ax

    def _plot_err_evo(self, err_evo):
        """
        Generates a plot for error evolution.

        Parameters
        ----------
        err_evo: pandas.DataFrame

        Returns
        -------
        matplotlib.Axes
        """
        # Plot lines
        ax = err_evo[[x for x in err_evo.columns if 'err' in x]].plot()
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Total RMSE')
        # Plot circles marking transition from GA to PS
        num_of_lp = len([x for x in err_evo.columns if 'err' in x])
        x_list = list()
        y_list = list()
        for n in range(num_of_lp):
            method = 'method{}'.format(n)
            err = 'err{}'.format(n)
            x_circ = len(err_evo.loc[err_evo[method] == 'GA'])
            if x_circ == len(err_evo[method].dropna()):
                # There are no PS records, move x_circ one back (to the last index)
                x_circ -= 1
            y_circ = err_evo[err].iloc[x_circ]
            x_list.append(x_circ)
            y_list.append(y_circ)
        ax.scatter(x_list, y_list, marker='o', c='grey', edgecolors='k')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        return ax

    def _get_err_evo_n(self, ga_errors, ps_errors, lp_count):
        """
        Generates DataFrame from lists ``ga_errors`` and ``ps_errors``,
        representing error evolution. The DataFrame has the
        following collumns: ``iter, err0, method0, ..., errN, methodN``.
        Methods are described with a string ``GA`` or ``PS``.

        Parameters
        ----------
        ga_errors: list
        ps_errors: list
        lp_count: int

        Returns
        -------
        pandas.DataFrame
        """
        err_evo_ga = pd.DataFrame({
            'iter': [x for x in range(len(ga_errors))],
            'err' + str(lp_count): ga_errors,
            'method' + str(lp_count): ['GA' for x in range(len(ga_errors))]
        })

        err_evo_ps = pd.DataFrame({
            'iter': [x for x in range(len(ga_errors), len(ga_errors) + len(ps_errors))],
            'err' + str(lp_count): ps_errors,
            'method' + str(lp_count): ['PS' for x in range(len(ps_errors))]
        })

        err_evo = err_evo_ga.append(err_evo_ps)

        return err_evo

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


if __name__ == "__main__":

    # Example
    import json

    workdir = "/home/krza/Desktop/temp"
    fmu_path = "./examples/simple/resources/Simple2R1C_ic_linux64.fmu"
    inp = pd.read_csv("./examples/simple/resources/inputs.csv").set_index('time')
    known = json.load(open("./examples/simple/resources/known.json"))
    est = json.load(open("./examples/simple/resources/est.json"))
    ideal = pd.read_csv("./examples/simple/resources/result.csv").set_index('time')

    session = Estimation(workdir, fmu_path, inp, known, est, ideal,
                         lp_n=3, lp_len=3600, lp_frame=None, vp=(3600, 20000),
                         ic_param={'Tstart': 'T'}, ga_iter=3, ps_iter=3)
    estimates = session.estimate()
    err, res = session.validate('avg')
    err, res = session.validate('best')
