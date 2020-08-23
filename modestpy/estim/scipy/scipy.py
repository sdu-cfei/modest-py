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
import os
import pandas as pd
import numpy as np
from random import random
from scipy.optimize import minimize
from modestpy.fmi.model import Model
from modestpy.estim.estpar import EstPar
from modestpy.estim.estpar import estpars_2_df
from modestpy.estim.error import calc_err
import modestpy.estim.plots as plots
import modestpy.utilities.figures as figures


class SCIPY(object):
    """
    Interface to `scipy.optimize.minimize()`.
    """
    # Summary placeholder
    TMP_SUMMARY = pd.DataFrame()

    # Ploting settings
    FIG_DPI = 150
    FIG_SIZE = (10, 6)

    NAME = 'SCIPY'
    METHOD = '_method_'
    ITER = '_iter_'
    ERR = '_error_'

    def __init__(self, fmu_path, inp, known, est, ideal,
                 solver, options={}, fmi_opts=None, ftype='RMSE'):
        """
        :param fmu_path: string, absolute path to the FMU
        :param inp: DataFrame, columns with input timeseries, index in seconds
        :param known: Dictionary, key=parameter_name, value=value
        :param est: Dictionary, key=parameter_name, value=tuple
                    (guess value, lo limit, hi limit), guess can be None
        :param ideal: DataFrame, ideal solution to be compared with model
                      outputs (variable names must match)
        :param solver: str, solver type (e.g. 'TNC', 'L-BFGS-B', 'SLSQP')
        :param options: dict, additional options passed to the SciPy's solver
        :param fmi_opts: dict, Additional FMI options to be passed to
                         the simulator (consult FMI specification)
        :param ftype: str, cost function type. Currently 'NRMSE' (advised
                      for multi-objective estimation) or 'RMSE'.
        """
        self.logger = logging.getLogger(type(self).__name__)

        assert inp.index.equals(ideal.index), \
            'inp and ideal indexes are not matching'

        # Solver type
        self.solver = solver

        # Default solver options
        self.options = {'disp': True, 'iprint': 2, 'maxiter': 500}

        if len(options) > 0:
            for key in options:
                self.options[key] = options[key]

        # Cost function type
        self.ftype = ftype

        # Ideal solution
        self.ideal = ideal

        # Adjust COM_POINTS
        # CVODE solver complains without "-1"
        SCIPY.COM_POINTS = len(self.ideal) - 1

        # Inputs
        self.inputs = inp

        # Known parameters to DataFrame
        known_df = pd.DataFrame()
        for key in known:
            assert known[key] is not None, \
                'None is not allowed in known parameters (parameter {})' \
                .format(key)
            known_df[key] = [known[key]]

        # est: dictionary to a list with EstPar instances
        self.est = list()
        for key in est:
            lo = est[key][1]
            hi = est[key][2]
            if est[key][0] is None:  # If guess is None, assume random guess
                v = lo + random() * (hi - lo)
            else:  # Else, take the guess passed in est
                v = est[key][0]
            self.est.append(EstPar(name=key, value=v, lo=lo, hi=hi))
        est = self.est

        # Model
        output_names = [var for var in ideal]
        self.model = SCIPY._get_model_instance(fmu_path, inp, known_df, est,
                                               output_names, fmi_opts)

        # Outputs
        self.summary = pd.DataFrame()
        self.res = pd.DataFrame()
        self.best_err = 1e7

        # Temporary placeholder for summary
        # It needs to be stored as class variable, because it has to be updated
        # from a static method used as callback
        self.summary_cols = \
            [x.name for x in self.est] + [SCIPY.ERR, SCIPY.METHOD]
        SCIPY.TMP_SUMMARY = pd.DataFrame(columns=self.summary_cols)

        # Log
        self.logger.info('SCIPY initialized... =========================')

    def estimate(self):

        # Initial error
        initial_result = self.model.simulate()
        self.res = initial_result
        initial_error = calc_err(initial_result, self.ideal,
                                 ftype=self.ftype)['tot']
        self.best_err = initial_error

        def objective(x):
            """Returns model error"""
            # Updated parameters are stored in x. Need to update the model.
            self.logger.debug('objective(x={})'.format(x))

            parameters = pd.DataFrame(index=[0])
            try:
                for v, ep in zip(x, self.est):
                    parameters[ep.name] = SCIPY.rescale(v, ep.lo, ep.hi)
            except TypeError as e:
                print(x)
                raise e
            self.model.set_param(parameters)
            result = self.model.simulate()
            err = calc_err(result, self.ideal, ftype=self.ftype)['tot']
            # Update best error and result
            if err < self.best_err:
                self.best_err = err
                self.res = result

            return err

        # Initial guess
        x0 = [SCIPY.scale(x.value, x.lo, x.hi) for x in self.est]
        self.logger.debug('SciPy x0 = {}'.format(x0))

        # Save initial guess in summary
        row = pd.DataFrame(index=[0])
        for x, c in zip(x0, SCIPY.TMP_SUMMARY.columns):
            row[c] = x
        row[SCIPY.ERR] = np.nan
        row[SCIPY.METHOD] = SCIPY.NAME
        SCIPY.TMP_SUMMARY = SCIPY.TMP_SUMMARY.append(row, ignore_index=True)

        # Parameter bounds
        b = [(0., 1.) for x in self.est]

        out = minimize(objective, x0, bounds=b, constraints=[],
                       method=self.solver, callback=SCIPY._callback,
                       options=self.options)

        outx = [SCIPY.rescale(x, ep.lo, ep.hi) for x, ep in
                zip(out.x.tolist(), self.est)]

        self.logger.debug('SciPy x = {}'.format(outx))

        # Update summary
        self.summary = SCIPY.TMP_SUMMARY.copy()
        self.summary.index += 1  # Adjust iteration counter
        self.summary.index.name = SCIPY.ITER  # Rename index

        # Update error
        self.summary[SCIPY.ERR] = \
            list(map(objective,
                     self.summary[[x.name for x in self.est]].values))

        for ep in self.est:
            name = ep.name
            # list(map(...)) - for Python 2/3 compatibility
            self.summary[name] = \
                list(map(lambda x: SCIPY.rescale(x, ep.lo, ep.hi),
                         self.summary[name]))  # Rescale

        # Add solver name to column `method`
        self.summary[SCIPY.METHOD] += '[' + self.solver + ']'

        # Reset temp placeholder
        SCIPY.TMP_SUMMARY = pd.DataFrame(columns=self.summary_cols)

        # Return DataFrame with estimates
        par_vec = outx
        par_df = pd.DataFrame(columns=[x.name for x in self.est], index=[0])
        for col, x in zip(par_df.columns, par_vec):
            par_df[col] = x

        return par_df

    @staticmethod
    def scale(v, lo, hi):
        # scaled = (rescaled - lo) / (hi - lo)
        return (v - lo) / (hi - lo)

    @staticmethod
    def rescale(v, lo, hi):
        # rescaled = lo + scaled * (hi - lo)
        return lo + v * (hi - lo)

    def get_plots(self):
        """
        Returns a list with important plots produced by this estimation method.
        Each list element is a dictionary with keys 'name' and 'axes'. The name
        should be given as a string, while axes as matplotlib.Axes instance.

        :return: list(dict)
        """
        plots = list()
        plots.append({'name': 'SCIPY-{}'.format(self.solver),
                      'axes': self.plot_parameter_evo()})
        return plots

    def save_plots(self, workdir):
        self.plot_comparison(os.path.join(workdir, 'ps_comparison.png'))
        self.plot_error_evo(os.path.join(workdir, 'ps_error_evo.png'))
        self.plot_parameter_evo(os.path.join(workdir, 'ps_param_evo.png'))

    def plot_comparison(self, file=None):
        return plots.plot_comparison(self.res, self.ideal, file)

    def plot_error_evo(self, file=None):
        err_df = pd.DataFrame(self.summary[SCIPY.ERR])
        return plots.plot_error_evo(err_df, file)

    def plot_parameter_evo(self, file=None):
        par_df = self.summary.drop([SCIPY.METHOD], axis=1)
        par_df = par_df.rename(columns={
            x: 'error' if x == SCIPY.ERR else x for x in par_df.columns
            })

        # Get axes
        axes = par_df.plot(subplots=True)
        fig = figures.get_figure(axes)
        # x label
        axes[-1].set_xlabel('Iteration')
        # ylim for error
        axes[-1].set_ylim(0, None)

        if file:
            fig.set_size_inches(SCIPY.FIG_SIZE)
            fig.savefig(file, dpi=SCIPY.FIG_DPI)
        return axes

    def get_full_solution_trajectory(self):
        """
        Returns all parameters and errors from all iterations.
        The returned DataFrame contains columns with parameter names,
        additional column '_error_' for the error and the index
        named '_iter_'.

        :return: DataFrame
        """
        return self.summary

    def get_error(self):
        """
        :return: float, last error
        """
        return float(self.summary[SCIPY.ERR].iloc[-1])

    def get_errors(self):
        """
        :return: list, all errors from all iterations
        """
        return self.summary[SCIPY.ERR].tolist()

    # PRIVATE METHODS

    @staticmethod
    def _callback(xk):
        # New row
        row = pd.DataFrame(index=[0])
        for x, c in zip(xk, SCIPY.TMP_SUMMARY.columns):
            row[c] = x

        row[SCIPY.ERR] = np.nan
        row[SCIPY.METHOD] = SCIPY.NAME

        # Append
        SCIPY.TMP_SUMMARY = SCIPY.TMP_SUMMARY.append(row, ignore_index=True)

    @staticmethod
    def _get_model_instance(fmu_path, inputs, known_pars, est, output_names,
                            fmi_opts=None):
        model = Model(fmu_path, fmi_opts)
        model.set_input(inputs)
        model.set_param(known_pars)
        model.set_param(estpars_2_df(est))
        model.set_outputs(output_names)
        return model
