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
import pandas as pd
import numpy as np
import copy
import os
from random import random
from scipy.optimize import minimize
from scipy.optimize import fmin_slsqp
from modestpy.estim.model import Model
from modestpy.estim.estpar import EstPar
from modestpy.estim.estpar import estpars_2_df
from modestpy.estim.error import calc_err
import modestpy.estim.plots as plots


class SQP(object):
    """
    SQP (Sequential Quadratic Programmic) algorithm for FMU parameter estimation. Based on SLSQP solver from SciPy.
    """
    COM_POINTS = 500  # Default number of communication points, should be adjusted to the number of samples
    TMP_SUMMARY = pd.DataFrame()  # Summary placeholder

    NAME = 'SQP'
    METHOD = '_method_'
    ITER = '_iter_'
    ERR = '_error_'

    def __init__(self, fmu_path, inp, known, est, ideal, scipy_opts={}, fmi_opts=None, ftype='RMSE'):
        """
        :param fmu_path: string, absolute path to the FMU
        :param inp: DataFrame, columns with input timeseries, index in seconds
        :param known: Dictionary, key=parameter_name, value=value
        :param est: Dictionary, key=parameter_name, value=tuple (guess value, lo limit, hi limit), guess can be None
        :param ideal: DataFrame, ideal solution to be compared with model outputs (variable names must match)
        :param dict scipy_opts: Additional options passed to the SLSQP solver in SciPy
        :param dict fmi_opts: Additional FMI options to be passed to the simulator (consult FMI specification)
        :param string ftype: Cost function type. Currently 'NRMSE' (advised for multi-objective estimation) or 'RMSE'.
        """
        self.logger = logging.getLogger(type(self).__name__)

        assert inp.index.equals(ideal.index), 'inp and ideal indexes are not matching'

        # Warning regarding limited functionality of SQP
        warning_msg = "SQP solver chosen. SQP is not well tested yet and has a limited functionality. " + \
                      "While the final solution should be OK, the intermediate results obtained from SciPy seem to be incorrect... "
        self.logger.warning(warning_msg)

        # SLSQP soler options
        self.scipy_opts = {'disp': True, 'iprint': 2, 'maxiter': 150, 'full_output': True}
        if len(scipy_opts) > 0:
            for key in scipy_opts:
                self.scipy_opts[key] = scipy_opts[key]

        # Cost function type
        self.ftype = ftype

        # Ideal solution
        self.ideal = ideal

        # Adjust COM_POINTS
        SQP.COM_POINTS = len(self.ideal) - 1  # CVODE solver complained without "-1"

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
            if est[key][0] is None:  # If guess is None, assume random guess
                v = lo + random() * (hi - lo)
            else:  # Else, take the guess passed in est
                v = est[key][0]
            self.est.append(EstPar(name=key, value=v, lo=lo, hi=hi))
        est = self.est

        # Model
        output_names = [var for var in ideal]
        self.model = SQP._get_model_instance(fmu_path, inp, known_df, est, output_names, fmi_opts)

        # Outputs
        self.summary = pd.DataFrame()
        self.res = pd.DataFrame()
        self.best_err = 1e7

        # Temporary placeholder for summary
        # It needs to be stored as class variable, because it has to be updated
        # from a static method used as callback
        self.summary_cols = [x.name for x in self.est] + [SQP.ERR, SQP.METHOD]
        SQP.TMP_SUMMARY = pd.DataFrame(columns=self.summary_cols)

        # Log
        self.logger.info('SQP initialized... =========================')

    def estimate(self):

        # Initial error
        initial_result = self.model.simulate(com_points=SQP.COM_POINTS)
        self.res = initial_result
        initial_error = calc_err(initial_result, self.ideal, ftype=self.ftype)['tot']
        self.best_err = initial_error

        def objective(x):
            """Returns model error"""
            # Updated parameters are stored in x. Need to update the model.
            parameters = pd.DataFrame(index=[0])
            try:
                for v, ep in zip(x, self.est):
                    parameters[ep.name] = SQP.rescale(v, ep.lo, ep.hi)
            except TypeError as e:
                print(x)
                raise e
            self.model.set_param(parameters)
            result = self.model.simulate(com_points=SQP.COM_POINTS)
            err = calc_err(result, self.ideal, ftype=self.ftype)['tot']
            # Update best error and result
            if err < self.best_err:
                self.best_err = err
                self.res = result

            return err

        x0 = [SQP.scale(x.value, x.lo, x.hi) for x in self.est]  # Initial guess
        b = [(0, 1) for x in self.est]  # Parameter bounds

        out = minimize(objective, x0, bounds=b, constraints=[],
                       method='SLSQP', callback=SQP._callback,
                       options=self.scipy_opts)

        outx = [SQP.rescale(x, ep.lo, ep.hi) for x, ep in zip(out.x.tolist(), self.est)]

        # Update summary
        self.summary = SQP.TMP_SUMMARY.copy()
        self.summary.index += 1  # Adjust iteration counter
        self.summary.index.name = SQP.ITER  # Rename index
        self.summary[SQP.ERR] = map(objective, self.summary[[x.name for x in self.est]].values)  # Update error
        for ep in self.est:
            name = ep.name
            self.summary[name] = map(lambda x: SQP.rescale(x, ep.lo, ep.hi), self.summary[name])  # Rescale
        SQP.TMP_SUMMARY = pd.DataFrame(columns=self.summary_cols) # Reset temp placeholder

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
        return list()

    def get_full_solution_trajectory(self):
        """
        Returns all parameters and errors from all iterations.
        The returned DataFrame contains columns with parameter names,
        additional column '_error_' for the error and the index
        named '_iter_'.

        :return: DataFrame
        """
        return self.summary

    # PRIVATE METHODS

    @staticmethod
    def _callback(xk):
        # New row
        row = pd.DataFrame(index=[0])
        for x, c in zip(xk, SQP.TMP_SUMMARY.columns):
            row[c] = x

        row[SQP.ERR] = np.nan
        row[SQP.METHOD] = SQP.NAME

        # Append
        SQP.TMP_SUMMARY = SQP.TMP_SUMMARY.append(row, ignore_index=True)

    @staticmethod
    def _get_model_instance(fmu_path, inputs, known_pars, est, output_names, fmi_opts=None):
        model = Model(fmu_path, fmi_opts)
        model.set_input(inputs)
        model.set_param(known_pars)
        model.set_param(estpars_2_df(est))
        model.set_outputs(output_names)
        return model
