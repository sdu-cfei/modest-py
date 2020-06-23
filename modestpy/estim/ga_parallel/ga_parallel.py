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
from multiprocessing import Manager
from multiprocessing.managers import BaseManager
from modestga import minimize
from modestpy.fmi.model import Model
from modestpy.estim.estpar import EstPar
from modestpy.estim.estpar import estpars_2_df
from modestpy.estim.error import calc_err
import modestpy.estim.plots as plots
import modestpy.utilities.figures as figures


class ObjectiveFun:
    def __init__(self, fmu_path, inp, known, est, ideal,
                 fmi_opts=None, ftype='RMSE', com_points=500):
        self.logger = logging.getLogger(type(self).__name__)
        self.model = None
        self.fmu_path = fmu_path
        self.inp = inp

        # Known parameters to DataFrame
        known_df = pd.DataFrame()
        for key in known:
            assert known[key] is not None, \
                'None is not allowed in known parameters (parameter {})' \
                .format(key)
            known_df[key] = [known[key]]
        self.known = known_df

        self.est = est
        self.ideal = ideal
        self.output_names = [var for var in ideal]
        self.fmi_opts = fmi_opts
        self.ftype = ftype
        self.com_points = com_points
        self.best_err = 1e7
        self.res = pd.DataFrame()

        self.logger.debug(f"fmu_path = {fmu_path}")
        self.logger.debug(f"inp = {inp}")
        self.logger.debug(f"known = {known}")
        self.logger.debug(f"est = {est}")
        self.logger.debug(f"ideal = {ideal}")
        self.logger.debug(f"output_names = {self.output_names}")

    def rescale(self, v, lo, hi):
        return lo + v * (hi - lo)

    def __call__(self, x, *args):
        # Instantiate the model
        self.logger.debug(f"x = {x}")
        if self.model is None:
            self.model = self._get_model_instance(
                self.fmu_path, self.inp, self.known, self.est, 
                self.output_names, self.fmi_opts
            )
            logging.debug(f"Model instance returned: {self.model}")

        # Updated parameters are stored in x. Need to update the model.
        parameters = pd.DataFrame(index=[0])
        try:
            for v, ep in zip(x, self.est):
                parameters[ep.name] = self.rescale(v, ep.lo, ep.hi)
        except TypeError as e:
            raise e
        self.logger.debug(f"parameters = {parameters}")
        self.model.parameters_from_df(parameters)
        self.logger.debug(f"est: {self.est}")
        self.logger.debug(f"parameters: {parameters}")
        self.logger.debug(f"model: {self.model}")
        self.logger.debug("Calling simulation...")
        result = self.model.simulate(com_points=self.com_points)
        self.logger.debug(f"result: {result}")
        err = calc_err(result, self.ideal, ftype=self.ftype)['tot']
        # Update best error and result
        if err < self.best_err:
            self.best_err = err
            self.res = result

        return err

    def _get_model_instance(self, fmu_path, inputs, known_pars, est,
                            output_names, fmi_opts=None):
        self.logger.debug("Getting model instance...")
        self.logger.debug(f"inputs = {inputs}")
        self.logger.debug(f"known_pars = {known_pars}")
        self.logger.debug(f"est = {est}")
        self.logger.debug(f"estpars_2_df(est) = {estpars_2_df(est)}")
        self.logger.debug(f"output_names = {output_names}")
        model = Model(fmu_path, fmi_opts)
        model.inputs_from_df(inputs)
        model.parameters_from_df(known_pars)
        model.parameters_from_df(estpars_2_df(est))
        model.specify_outputs(output_names)
        self.logger.debug(f"Model instance initialized: {model}")
        self.logger.debug(f"Model instance initialized: {model.model}")
        res = model.simulate(500)
        self.logger.debug(f"test result: {res}")
        return model


class MODESTGA(object):
    """
    Using GA with interface similar to `scipy.optimize.minimize()`.
    """
    # Default number of communication points, should be adjusted
    # to the number of samples
    COM_POINTS = 500

    # Summary placeholder
    TMP_SUMMARY = pd.DataFrame()

    # Ploting settings
    FIG_DPI = 150
    FIG_SIZE = (10, 6)

    NAME = 'MODESTGA'
    METHOD = '_method_'
    ITER = '_iter_'
    ERR = '_error_'

    def __init__(self, fmu_path, inp, known, est, ideal,
                 options={}, fmi_opts=None, ftype='RMSE',
                 generations=None, pop_size=None, mut_rate=None,
                 trm_size=None, tol=None, inertia=None, workers=None):
        """
        :param fmu_path: string, absolute path to the FMU
        :param inp: DataFrame, columns with input timeseries, index in seconds
        :param known: Dictionary, key=parameter_name, value=value
        :param est: Dictionary, key=parameter_name, value=tuple
                    (guess value, lo limit, hi limit), guess can be None
        :param ideal: DataFrame, ideal solution to be compared with model
                      outputs (variable names must match)
        :param options: dict, additional options passed to the solver (not used here)
        :param fmi_opts: dict, Additional FMI options to be passed to
                         the simulator (consult FMI specification)
        :param ftype: str, cost function type. Currently 'NRMSE' (advised
                      for multi-objective estimation) or 'RMSE'. 
        """
        self.logger = logging.getLogger(type(self).__name__)

        assert inp.index.equals(ideal.index), \
            'inp and ideal indexes are not matching'

        self.fmu_path = fmu_path
        self.inp = inp
        self.known = known
        self.ideal = ideal
        self.fmi_opts = fmi_opts
        self.ftype = ftype
        self.com_points = len(self.ideal) - 1  # CVODE solver complains without "-1"

        # Default solver options
        self.workers = 3                # CPU cores to use
        self.options = {
            'generations': 50,          # Max. number of generations
            'pop_size': 50,             # Population size
            'mut_rate': 0.01,           # Mutation rate
            'trm_size': 20,             # Tournament size
            'tol': 1e-3,                # Solution tolerance
            'inertia': 100,             # Max. number of non-improving generations
            'xover_ratio': 0.5          # Crossover ratio
        }

        # User options
        if workers is not None:
            self.workers = workers
        if generations is not None:
            self.options['generations'] = generations
        if pop_size is not None:
            self.options['pop_size'] = pop_size
        if mut_rate is not None:
            self.options['mut_rate'] = mut_rate
        if trm_size is not None:
            self.options['trm_size'] = trm_size
        if tol is not None:
            self.options['tol'] = tol
        if inertia is not None:
            self.options['inertia'] = inertia

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
        # self.model = MODESTGA._get_model_instance(fmu_path, inp, known_df, est,
        #                                           output_names, fmi_opts)

        # Outputs
        self.summary = pd.DataFrame()
        self.res = pd.DataFrame()
        self.best_err = 1e7

        # Temporary placeholder for summary
        # It needs to be stored as class variable, because it has to be updated
        # from a static method used as callback
        self.summary_cols = \
            [x.name for x in self.est] + [MODESTGA.ERR, MODESTGA.METHOD]
        MODESTGA.TMP_SUMMARY = pd.DataFrame(columns=self.summary_cols)

        # Log
        self.logger.info('MODESTGA initialized... =========================')

    def estimate(self):
        # Objective function
        self.logger.debug('Instantiating ObjectiveFun')
        objective_fun = ObjectiveFun(
            self.fmu_path, self.inp, self.known, self.est, self.ideal,
            self.fmi_opts, self.ftype, self.com_points
        )
        self.logger.debug(f'ObjectiveFun: {objective_fun}')

        # Initial guess
        x0 = [MODESTGA.scale(x.value, x.lo, x.hi) for x in self.est]
        self.logger.debug('modestga x0 = {}'.format(x0))

        # Save initial guess in summary
        row = pd.DataFrame(index=[0])
        for x, c in zip(x0, MODESTGA.TMP_SUMMARY.columns):
            row[c] = x
        row[MODESTGA.ERR] = np.nan
        row[MODESTGA.METHOD] = MODESTGA.NAME
        MODESTGA.TMP_SUMMARY = MODESTGA.TMP_SUMMARY.append(row, ignore_index=True)

        # Parameter bounds
        b = [(0., 1.) for x in self.est]
        self.logger.debug(f'bounds = {b}')

        out = minimize(
            objective_fun,
            bounds=b,
            x0=x0,
            args=(),
            callback=MODESTGA._callback,
            options=self.options,
            workers=self.workers)

        self.logger.debug(f'out = {out}')
        outx = [MODESTGA.rescale(x, ep.lo, ep.hi) for x, ep in
                zip(out.x.tolist(), self.est)]

        self.logger.debug('modestga x = {}'.format(outx))

        # Update summary
        self.summary = MODESTGA.TMP_SUMMARY.copy()
        self.summary.index += 1  # Adjust iteration counter
        self.summary.index.name = MODESTGA.ITER  # Rename index

        # Update error
        self.summary[MODESTGA.ERR] = \
            list(map(objective_fun,
                     self.summary[[x.name for x in self.est]].values))

        for ep in self.est:
            name = ep.name
            # list(map(...)) - for Python 2/3 compatibility
            self.summary[name] = \
                list(map(lambda x: MODESTGA.rescale(x, ep.lo, ep.hi),
                         self.summary[name]))  # Rescale

        # Reset temp placeholder
        MODESTGA.TMP_SUMMARY = pd.DataFrame(columns=self.summary_cols)

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
        plots.append({'name': 'MODESTGA',
                      'axes': self.plot_parameter_evo()})
        return plots

    def save_plots(self, workdir):
        self.plot_comparison(os.path.join(workdir, 'ps_comparison.png'))
        self.plot_error_evo(os.path.join(workdir, 'ps_error_evo.png'))
        self.plot_parameter_evo(os.path.join(workdir, 'ps_param_evo.png'))

    def plot_comparison(self, file=None):
        return plots.plot_comparison(self.res, self.ideal, file)

    def plot_error_evo(self, file=None):
        err_df = pd.DataFrame(self.summary[MODESTGA.ERR])
        return plots.plot_error_evo(err_df, file)

    def plot_parameter_evo(self, file=None):
        par_df = self.summary.drop([MODESTGA.METHOD], axis=1)
        par_df = par_df.rename(columns={
            x: 'error' if x == MODESTGA.ERR else x for x in par_df.columns
            })

        # Get axes
        axes = par_df.plot(subplots=True)
        fig = figures.get_figure(axes)
        # x label
        axes[-1].set_xlabel('Iteration')
        # ylim for error
        axes[-1].set_ylim(0, None)

        if file:
            fig.set_size_inches(MODESTGA.FIG_SIZE)
            fig.savefig(file, dpi=MODESTGA.FIG_DPI)
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
        return float(self.summary[MODESTGA.ERR].iloc[-1])

    def get_errors(self):
        """
        :return: list, all errors from all iterations
        """
        return self.summary[MODESTGA.ERR].tolist()

    # PRIVATE METHODS

    @staticmethod
    def _callback(xk, fx, ng, *args):  # TODO: it must be pickable for multiprocessing
        # New row
        row = pd.DataFrame(index=[0])
        for x, c in zip(xk, MODESTGA.TMP_SUMMARY.columns):
            row[c] = x

        row[MODESTGA.ERR] = np.nan
        row[MODESTGA.METHOD] = MODESTGA.NAME

        # Append
        MODESTGA.TMP_SUMMARY = MODESTGA.TMP_SUMMARY.append(row, ignore_index=True)

    @staticmethod
    def _get_model_instance(fmu_path, inputs, known_pars, est, output_names,
                            fmi_opts=None):
        model = Model(fmu_path, fmi_opts)
        model.set_input(inputs)
        model.set_param(known_pars)
        model.set_param(estpars_2_df(est))
        model.set_outputs(output_names)
        return model
