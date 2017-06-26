"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.

Author: Krzysztof Arendt
"""

from modestpy.log_init import LogInit
LOG_INIT = LogInit(__name__)
LOGGER = LOG_INIT.get_logger()

import pandas as pd
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from modestpy.estim.ga.ga import GA
from modestpy.estim.ps.ps import PS
from modestpy.estim.model import Model
from modestpy.estim.error import calc_err
from modestpy.estim.plots import plot_comparison
from modestpy.estim.plots import plot_parameter_evo
from pyfmi.fmi import FMUException
from os.path import join
try:
    from pandas.plotting import scatter_matrix
except ImportError:
    from pandas.tools.plotting import scatter_matrix

class LearnMan:

    # GA and PS settings
    GA_GENERATIONS = 30
    PS_MAX_ITER = 200

    # Est object
    INIT = 0
    LO = 1
    HI = 2

    # Number of attempts to find nonzero learning period
    NONZERO_ATTEMPTS = 50

    # Plots size
    DPI = 100
    FIG_SIZE = (15, 10)

    def __init__(self, workdir, fmu_path, inp, known, est, ideal):
        """
        :param workdir: string, working directory (for data and plot saving)
        :param fmu_path: string
        :param inp: DataFrame, time in seconds (float) as index
        :param known: dict, key=parameter_name, value=value
        :param est: dict, key=parameter_name, value=tuple (guess value, lo limit, hi limit), guess can be None
        :param ideal: DataFrame, time in seconds as index
        """
        assert inp.index.equals(ideal.index), 'inp and ideal indexes are not matching'
        assert inp.index.name == 'time', "Index must be named 'time'"
        assert ideal.index.name == 'time', "Index must be named 'time'"
        for v in est:
            assert (est[v][LearnMan.INIT] >= est[v][LearnMan.LO]) and (est[v][LearnMan.INIT] <= est[v][LearnMan.HI]), \
                'Initial value out of limits ({})'.format(v)

        # Input data
        self.workdir = workdir
        self.fmu_path = fmu_path
        self.inp = inp.copy()
        self.known = copy.deepcopy(known)
        self.est = copy.deepcopy(est)
        self.ideal = ideal.copy()

        # Estimation algorithm parameters
        self.GA_POP_SIZE = 3 * len(est.keys())
        self.GA_GENERATIONS = LearnMan.GA_GENERATIONS
        self.PS_MAX_ITER = LearnMan.PS_MAX_ITER

        # Learning periods
        self.lp = []     # List of tuples (start, stop) (int, int)
        self.n = None    # Number of learning periods (int)
        self.all = None  # Final estimates and errors from all runs (DataFrame)
        self.eee = None  # Error evolution envelope (DataFrame). Columns:
                         # iter, method1, method2, method3, ..., err1, err2, err3, ...

        # Validation period
        self.vp = None   # Tuple (start, stop)

        # IC from sensors
        self.sensor_map = dict()
        self.is_sensor_ic = False

        # Plots
        # Axes are added to the dictionary, dedicated method is used to show/save them in workdir
        # key: value -> plot name: axes
        self.plots = {}

    def select_validation_period(self, location):
        """
        Selects the validation period.

        :param location: tuple, (t_start [s], t_stop [s]), if present the validation period is assigned exactly
        :return: tuple (start, stop)
        """
        start = location[0]
        end = location[1]
        self.vp = (start, end)
        return self.vp

    def select_learning_periods(self, n, length, bounds=None):
        """
        Splits input and ideal data into random ``n_periods``, each one with the length of ``len_period``.
        Periods may overlap. Ensures that a period with null data for any ``ideal`` variable is not selected.
        Indexes of ``self.ideal`` and ``self.inp`` must match.

        :param n: int, number of periods
        :param length: int, period length in seconds
        :param bounds: tuple, (t_start [s], t_stop [s]), if present learning periods are assigned within this frame
        :return: list of tuples (start, stop)
        """
        self.n = n

        if bounds:
            t0 = bounds[0]
            tend = bounds[1]
        else:
            t0 = self.ideal.index[0]
            tend = self.ideal.index[-1]
        assert length <= tend - t0, 'Learning period lenght cannot be longer than data length!'

        length = int(length)
        lp = []

        for i in range(n):
            chosen_lp = False
            tries = LearnMan.NONZERO_ATTEMPTS

            while not chosen_lp and tries > 0:
                new_t0 = random.randint(t0, tend - length)
                new_tend = new_t0 + length

                ideal_nonzero = LearnMan._all_columns_nonzero(self.ideal.loc[new_t0:new_tend])

                if ideal_nonzero:
                    lp.append((new_t0, new_tend))
                    chosen_lp = True
                else:
                    tries -= 1
                    LOGGER.warning('[LearnMan] Zero ideal solution not allowed, selecting another one...')
                    LOGGER.warning('[LearnMan] Number of tries left: {}'.format(tries))

            if tries == 0:
                LOGGER.error('[LearnMan] Nonzero ideal solution not found ({} attempts)'.format(LearnMan.NONZERO_ATTEMPTS))
                raise Exception

        self.lp = lp
        return lp

    def ic_from_sensors(self, par_from_ideal):
        """
        Initial condition from sensors.
        Updates or extends ``known`` dictionary with sensor readings.
        Sensor readings are the first values from the chosen variables in ``ideal`` data.

        :param par_from_ideal: dictionary, key=parameter_name, value=ideal_var_name
        :return: None
        """
        self.sensor_map = par_from_ideal
        self.is_sensor_ic = True

    def estimate(self):
        """
        Runs through the estimation loop. Returns estimates and errors for all learning periods.

        :return: DataFrame with estimates and errors
        """
        all_est = pd.DataFrame()
        all_err = np.zeros(self.n)

        eee_left = None

        n = 0
        for lp in self.lp:
            # Slice data
            start, stop = lp[0], lp[1]
            inp_slice = self.inp.loc[start:stop]
            ideal_slice = self.ideal.loc[start:stop]

            # Update known with sensor data
            known = copy.deepcopy(self.known)
            if self.is_sensor_ic:
                for par in self.sensor_map:
                    var_name = self.sensor_map[par]
                    known[par] = ideal_slice[var_name].iloc[0]

            # Initialize GA
            ga = GA(self.fmu_path, inp_slice, known, self.est, ideal_slice,
                    generations=self.GA_GENERATIONS,
                    pop_size=self.GA_POP_SIZE)

            # Run GA
            est_df = ga.estimate()
            est = LearnMan.get_new_est(est_df, self.est)

            # Add GA errors to error evolution envelope (eee)
            ga_errs = ga.get_errors()
            eee_right = pd.DataFrame(
            {
                'iter': [x for x in range(len(ga_errs))],
                'err'+str(n): ga_errs,
                'method'+str(n): ['GA' for x in range(len(ga_errs))]
            })

            self.plots['pop_evo_{}'.format(n)] = ga.plot_pop_evo()

            # Initialize PS, taking estimates from GA
            ps = PS(self.fmu_path, inp_slice, known, est, ideal_slice,
                    max_iter=self.PS_MAX_ITER)

            # Run PS
            est_df = ps.estimate()

            # Add PS errors to error evolution envelope (eee)
            ps_errs = ps.get_errors()
            eee_right = eee_right.append(pd.DataFrame({
                'iter': [x for x in range(len(ga_errs), len(ga_errs) + len(ps_errs))],
                'err'+str(n): ps_errs,
                'method' + str(n): ['PS' for x in range(len(ps_errs))]
            }))

            # Merge complete eee (left) with current (right)
            if eee_left is None:
                # First learning period
                eee_left = eee_right
            else:
                # Next learning period
                eee_left = eee_left.merge(eee_right, on='iter')

            # Save final error
            all_err[n] = ps.get_error()

            # Save estimates
            all_est = all_est.append(est_df, ignore_index=True)

            # Increase learning period counter
            n += 1

        assert n >= 0, 'No learning period found...'

        # Merge estimates and errors into single DataFrame
        all = all_est
        all['error'] = pd.Series(data=all_err)

        # Assign local to global
        self.all = all
        self.eee = eee_left.set_index('iter')

        # Create eee plot
        self.plots['err_env'] = self.plot_eee()

        # Create scatter matrix plot
        if n > 1:
            try:
                self.plots['scatter'] = self.plot_scat_matrix(all)
            except Exception as e:
                # TODO: Why does it fail sometimes? Because it's not normalized?
                LOGGER.warning('[ERROR] Unable to plot scatter matrix')
                LOGGER.warning(e.message)
        return all

    def plot_eee(self):
        ax = self.eee.plot(color='grey')
        emax = self.eee[['err'+str(x) for x in range(len(self.lp))]].max(axis=1)
        emin = self.eee[['err'+str(x) for x in range(len(self.lp))]].min(axis=1)
        ax.fill_between(self.eee.index, emin, emax, facecolor='grey', alpha=0.5)
        ax.set_ylabel('Total NRMSE')
        return ax

    def plot_scat_matrix(self, all_est):
        ax = scatter_matrix(all_est, alpha=0.5)
        return ax

    def get_avg_est(self):
        """
        Returns average estimates

        :return: DataFrame
        """
        est = self.all.drop('error', axis=1)
        est = est.mean().to_frame().T
        est.index = [0]
        return est

    def get_wavg_est(self):
        """
        Returns weighted average estimates, with the accuracy as the weight.
        Accuracy is defined as 1/error.

        :return: DataFrame
        """
        allest = self.all.copy()
        allest['accuracy'] = 1 / allest['error']
        allest = allest.drop('error', axis=1)
        allest = allest.apply(LearnMan.wavg, axis=0, args=(allest['accuracy'],)).to_frame().T.drop('accuracy', axis=1)
        allest.index = [0]
        return allest

    def get_best_est(self):
        """
        Returns best estimates from all runs.

        :return: DataFrame
        """
        best = self.all.loc[self.all['error'] == self.all['error'].min()]
        best = best.drop('error', axis=1)
        best.index = [0]
        return best

    def get_worst_est(self):
        """
        Returns worst estimates from all runs

        :return: DataFrame
        """
        worst = self.all.loc[self.all['error'] == self.all['error'].max()]
        worst = worst.drop('error', axis=1)
        worst.index = [0]
        return worst

    def save_best_est(self, filename, incl_known=True):
        """
        Saves best estimates in work directory

        :param filename: string
        :param incl_known: bool, if True, known parameters are included
        :return: None
        """
        best = self.get_best_est()
        if incl_known:
            best = self.add_from_dict(best, self.known)
        best.to_csv(join(self.workdir, filename), index=False)

    def save_wavg_est(self, filename, incl_known=True):
        """
        Saves WAVG estimates in work directory

        :param filename: string
        :param incl_known: bool, if True, known parameters are included
        :return: None
        """
        wavg = self.get_wavg_est()
        if incl_known:
            wavg = self.add_from_dict(wavg, self.known)
        wavg.to_csv(join(self.workdir, filename), index=False)

    def save_avg_est(self, filename, incl_known=True):
        """
        Saves AVG estimates in work directory

        :param filename: string
        :param incl_known: bool, if True, known parameters are included
        :return: None
        """
        avg = self.get_avg_est()
        if incl_known:
            avg = self.add_from_dict(avg, self.known)
        avg.to_csv(join(self.workdir, filename), index=False)

    def add_from_dict(self, df, d):
        for key in d:
            df[key] = [d[key]]
        return df

    @staticmethod
    def wavg(series, weight):
        """
        Returns weighted average.

        :param series: pandas.Series
        :param weight: pandas.Series
        :return: float
        """
        var = series
        w = weight
        try:
            return (var * w).sum() / w.sum()
        except ZeroDivisionError:
            return var.mean()

    def validate(self, estimates, return_res=False, plot_suffix=None):
        """
        Performs a simulation with estimated parameters for the previously selected validation period.

        :param estimates: DataFrame with estimated parameters
        :param return_res: bool, if True, returns a tuple (err, res), otherwise returns err
        :param plot_suffix: string or None
        :return: dictionary with Normalised Root Mean Square Errors for all variables, and the total error (key=tot)
        """
        # Slice data
        start, stop = self.vp[0], self.vp[1]
        inp_slice = self.inp.loc[start:stop]
        ideal_slice = self.ideal.loc[start:stop]

        # Update self.est
        self.est = LearnMan.get_new_est(estimates, self.est)

        # Update known with sensor data
        known = copy.deepcopy(self.known)
        if self.is_sensor_ic:
            for par in self.sensor_map:
                var_name = self.sensor_map[par]
                known[par] = ideal_slice[var_name].iloc[0]

        # Initialize model
        model = LearnMan._get_model_instance(self.fmu_path,
                                             inp_slice,
                                             LearnMan._known_2_df(known),
                                             estimates,
                                             ideal_slice.columns.tolist())

        # Simulate and get error
        com_points = len(ideal_slice) - 1

        try:
            result = model.simulate(com_points=com_points)

        except FMUException as e:
            LOGGER.error('[ERROR] Problem found inside FMU. Log:')
            LOGGER.error(model.model.model.print_log())
            for key in self.est:
                if self.est[key][0] is None:
                    LOGGER.error("Parameter '{}' does not have any value!".format(key))
            for key in self.known:
                if self.known[key] is None:
                    LOGGER.error("Parameter '{}' does not have any value!".format(key))

            raise FMUException(e)

        err = calc_err(result, ideal_slice)

        # Create validation plot
        if plot_suffix:
            plot_suffix = '_' + plot_suffix
        else:
            plot_suffix = ''
        self.plots['validation' + plot_suffix] = plot_comparison(result, ideal_slice, f=None)

        # Return
        LOGGER.info('Validation error: {}'.format(err))

        if return_res:
            return err, result
        else:
            return err

    def save_plots(self):
        """
        Saves all plots into workdir. Closes all figures afterwards.

        :return: None
        """
        LOGGER.info('[LearnMan] Saving plots...')
        for name in self.plots:
            LOGGER.info("Plotting: {}".format(name))
            # Get figure
            ax = self.plots[name]
            try:
                # Single plot
                fig = ax.get_figure()
            except AttributeError:
                # Subplots
                try:
                    fig = ax[0].get_figure()
                except AttributeError:
                    fig = ax[0][0].get_figure()

            # Adjust size
            fig.set_size_inches(LearnMan.FIG_SIZE)

            # Save file
            filepath = join(self.workdir, name + '.png')
            fig.savefig(filepath, dpi=LearnMan.DPI)

        # Close all plots (release instances for garbage collecting)
        plt.close('all')

    @staticmethod
    def get_new_est(df, est):
        """
        Returns a copy of ``est`` dictionary with new values from ``df``.

        :param df: DataFrame, result of GA or PS estimation
        :param est: dictionary, keys = variable names, values = (value, lo limit, hi limit)
        :return: dictionary, keys = variable names, values = (value, lo limit, hi limit)
        """
        new_est = copy.deepcopy(est)
        for p in est:
            try:
                new_est[p] = (df[p][0], est[p][1], est[p][2])
            except KeyError as e:
                print '[ERROR] Key not found: {}'.format(p)
                print 'Compare keys to find problem:'
                print 'df: ', df
                print 'est: ', est
                print e
                raise e

        return new_est

    @staticmethod
    def _get_model_instance(fmu_path, inputs, known, est, output):
        """
        :param fmu_path: string
        :param inputs: DataFrame
        :param known: DataFrame
        :param est: DataFrame
        :param output: list of strings
        :return:
        """
        model = Model(fmu_path)
        model.set_input(inputs)
        model.set_param(known)
        model.set_param(est)
        model.set_outputs(output)
        return model

    @staticmethod
    def _all_columns_nonzero(df):
        """
        Checks whether all columns in DataFrame are nonzero.

        :param df: DataFrame
        :return: boolean
        """
        assert df.empty is False, 'This DataFrame should not be empty. Something is wrong.'
        is_zero = (df == 0).all()
        for col in is_zero:
            if col == True:  # Never use ``is`` with numpy.bool objects
                return False
        return True

    @staticmethod
    def _known_2_df(known):
        known_df = pd.DataFrame()
        for key in known:
            known_df[key] = [known[key]]
        return known_df

    @staticmethod
    def _est_2_df(est):
        est_df = pd.DataFrame()
        for key in est:
            est_df[key] = [est[key][0]]
        return est_df


if __name__ == '__main__':

    # Some tests
    eee = pd.DataFrame(columns=['err' + str(x) for x in range(3)]
                              +['method' + str(x) for x in range(3)])

    eee.index = eee.index.rename('iter')

    print eee.index
