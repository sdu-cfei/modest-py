"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

from modestpy.log_init import LogInit
LOG_INIT = LogInit(__name__)
LOGGER = LOG_INIT.get_logger()

from modestpy.estim.model import Model
from modestpy.estim.estpar import estpars_2_df
from modestpy.estim.estpar import EstPar
from modestpy.estim.error import calc_err
import modestpy.estim.plots as plots
import pandas as pd
import copy
import os
from random import random


class PS:
    """
    Pattern _search (Hooke-Jeeves) algorithm for FMU parameter estimation.
    """

    COM_POINTS = 500  # Default number of communication points, should be adjusted to the number of samples
    STEP_CEILING = 0.2  # Maximum allowed relative step
    STEP_INC = 1.2  # Step is multiplied by this factor if solution improves
    STEP_DEC = 2.  # Step is divided by this factor if solution does not improve

    def __init__(self, fmu_path, inp, known, est, ideal, rel_step=0.05, tolerance=0.001, try_lim=30, max_iter=300):
        """
        :param fmu_path: string, absolute path to the FMU
        :param inp: DataFrame, columns with input timeseries, index in seconds
        :param known: Dictionary, key=parameter_name, value=value
        :param est: Dictionary, key=parameter_name, value=tuple (guess value, lo limit, hi limit), guess can be None
        :param ideal: DataFrame, ideal solution to be compared with model outputs (variable names must match)
        :param rel_step: float, initial relative step when modifying parameters
        :param tolerance: float, stopping criterion, when rel_step becomes smaller than tolerance algorithm stops
        :param try_lim: integer, maximum number of tries to decrease rel_step
        :param max_iter: integer, maximum number of iterations
        """
        assert inp.index.equals(ideal.index), 'inp and ideal indexes are not matching'
        assert rel_step > tolerance, 'Relative step must not be smaller than the stop criterion'

        # Ideal solution
        self.ideal = ideal

        # Adjust COM_POINTS
        PS.COM_POINTS = len(self.ideal) - 1  # CVODE solver complained without "-1"

        # Inputs
        self.inputs = inp

        # Known parameters to DataFrame
        known_df = pd.DataFrame()
        for key in known:
            assert known[key] is not None, 'None is not allowed in known parameters (parameter {})'.format(key)
            known_df[key] = [known[key]]

        # est: dictionary to a list with EstPar instances
        self.est = list()
        for key in est:
            lo = est[key][1]
            hi = est[key][2]
            if est[key][0] is None:
                v = lo + random() * (hi - lo)
            else:
                v = est[key][0]
            self.est.append(EstPar(name=key, value=v, lo=lo, hi=hi))
        est = self.est

        # Model
        output_names = [var for var in ideal]
        self.model = PS._get_model_instance(fmu_path, inp, known_df, est, output_names)

        # Initial value for relative parameter step (0-1)
        self.rel_step = rel_step

        # Min. allowed relative parameter change (0-1) - PS stops when self.max_change < tolerance
        self.tolerance = tolerance

        # Max. number of iterations without moving to a new point
        self.try_lim = try_lim

        # Max. number of iterations in total
        self.max_iter = max_iter

        # Outputs
        self.summary = pd.DataFrame()
        self.res = pd.DataFrame()

        LOGGER.info('Pattern Search initialized... =========================')

    def estimate(self):
        """
        Proxy method. Each algorithm from ``estim`` package should have this method

        :return: DataFrame
        """
        return self._search()

    def get_error(self):
        """
        :return: float, last error
        """
        return float(self.summary['error'].iloc[-1])

    def get_errors(self):
        """
        :return: list, all errors from all iterations
        """
        return self.summary['error'].tolist()

    def save_plots(self, workdir):
        self.plot_comparison(os.path.join(workdir, 'ps_comparison.png'))
        self.plot_error_evo(os.path.join(workdir, 'ps_error_evo.png'))
        self.plot_parameter_evo(os.path.join(workdir, 'ps_param_evo.png'))

    def plot_comparison(self, file=None):
        return plots.plot_comparison(self.res, self.ideal, file)

    def plot_error_evo(self, file=None):
        err_df = pd.DataFrame(self.summary['error'])
        return plots.plot_error_evo(err_df, file)

    def plot_parameter_evo(self, file=None):
        par_df = self.summary.drop('error', axis=1)
        return plots.plot_parameter_evo(par_df, file)

    def plot_inputs(self, file=None):
        return plots.plot_inputs(self.inputs, file)

    def _search(self):
        """
        Pattern _search loop.

        :return: DataFrame with estimates
        """
        initial_estimates = copy.deepcopy(self.est)
        best_estimates = copy.deepcopy(initial_estimates)
        current_estimates = copy.deepcopy(initial_estimates)

        initial_result = self.model.simulate(com_points=PS.COM_POINTS)
        self.res = initial_result
        initial_error = calc_err(initial_result, self.ideal)['tot']
        best_err = initial_error

        # First line of the summary
        summary = estpars_2_df(current_estimates)
        summary['error'] = [initial_error]
        summary['rel_step'] = [self.rel_step]
        summary.index.names = ['iter']

        # Counters
        n_try = 0
        iteration = 0

        # Search loop
        while (n_try < self.try_lim) and (iteration < self.max_iter) and (self.rel_step > self.tolerance):
            iteration += 1
            LOGGER.info('Iteration no. {} ========================='.format(iteration))
            improved = False

            # Iterate over all parameters
            for par in current_estimates:
                for sign in ['+', '-']:
                    # Calculate new parameter
                    new_par = self._get_new_estpar(par, self.rel_step, sign)

                    # Simulate and calculate error
                    self.model.set_param(estpars_2_df([new_par]))
                    result = self.model.simulate(com_points=PS.COM_POINTS)
                    err = calc_err(result, self.ideal)['tot']

                    # Save point if solution improved
                    if err < best_err:
                        self.res = result
                        best_err = err
                        # best_estimates = PS._replace_par(best_estimates, new_par)  # Shortest path search
                        best_estimates = PS._replace_par(current_estimates, new_par)  # Orthogonal search
                        improved = True

                    # Reset model parameters
                    self.model.set_param(estpars_2_df(current_estimates))

            # Go to the new point
            current_estimates = copy.deepcopy(best_estimates)

            # Update summary
            current_estimates_df = estpars_2_df(current_estimates)
            current_estimates_df.index = [iteration]
            summary = pd.concat([summary, current_estimates_df])
            summary['error'][iteration] = best_err
            summary['rel_step'][iteration] = self.rel_step

            if not improved:
                n_try += 1
                self.rel_step /= PS.STEP_DEC
                LOGGER.info('Solution did not improve...')
                LOGGER.debug('Step reduced to {}'.format(self.rel_step))
                LOGGER.debug('Tries left: {}'.format(self.try_lim - n_try))
            else:
                # Solution improved, reset n_try counter
                n_try = 0
                self.rel_step *= PS.STEP_INC
                if self.rel_step > PS.STEP_CEILING:
                    self.rel_step = PS.STEP_CEILING
                LOGGER.info('Solution improved')
                LOGGER.debug('Current step is {}'.format(self.rel_step))
                LOGGER.info('New error: {}'.format(best_err))
                LOGGER.debug('New estimates:\n{}'.format(estpars_2_df(current_estimates)))

        # Reorder columns in summary
        s_cols = summary.columns.tolist()
        s_cols.remove('rel_step')
        s_cols.append('rel_step')
        s_cols.remove('error')
        s_cols.append('error')
        summary = summary[s_cols]

        # Print summary
        reason = 'Unknown'
        if n_try >= self.try_lim:
            reason = 'Maximum number of tries to decrease the step reached'
        elif iteration >= self.max_iter:
            reason = 'Maximum number of iterations reached'
        elif self.rel_step <= self.tolerance:
            reason = 'Relative step smaller than the stoping criterion'

        LOGGER.info('Pattern search finished. Reason: {}'.format(reason))
        LOGGER.info('Summary:\n{}'.format(summary))

        # Final assignments
        self.summary = summary
        final_estimates = estpars_2_df(best_estimates)

        return final_estimates

    def _get_new_estpar(self, estpar, rel_step, sign):
        """
        Returns new ``EstPar`` object with modified value, according to ``sign`` and ``max_change``.

        :param estpar: EstPar
        :param rel_step: float, (0-1)
        :param sign: string, '+' or '-'
        :return: EstPar
        """
        sign_mltp = None
        if sign == '+':
            sign_mltp = 1.
        elif sign == '-':
            sign_mltp = -1.
        else:
            print 'Unrecognized sign ({})'.format(sign)

        new_value = estpar.value * (1 + rel_step * sign_mltp)

        if new_value > estpar.hi:
            new_value = estpar.hi
        if new_value < estpar.lo:
            new_value = estpar.lo

        return EstPar(estpar.name, estpar.lo, estpar.hi, new_value)

    @staticmethod
    def _get_model_instance(fmu_path, inputs, known_pars, est, output_names):
        model = Model(fmu_path)
        model.set_input(inputs)
        model.set_param(known_pars)
        model.set_param(estpars_2_df(est))
        model.set_outputs(output_names)
        return model

    @staticmethod
    def _replace_par(estpar_list, estpar):
        """
        Puts ``estpar`` in ``estpar_list``, replacing object with the same ``name``.

        :param estpar_list: list of EstPar objects
        :param estpar: EstPar
        :return: list of EstPar objects
        """
        new_list = copy.deepcopy(estpar_list)
        for i in range(len(new_list)):
            if new_list[i].name == estpar.name:
                new_list[i] = copy.deepcopy(estpar)
        return new_list

