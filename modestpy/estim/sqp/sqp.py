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

import pandas as pd
import copy
import os
from random import random
from scipy.optimize import minimize
from modestpy.estim.model import Model
from modestpy.estim.estpar import EstPar
from modestpy.estim.estpar import estpars_2_df
from modestpy.estim.error import calc_err
import modestpy.estim.plots as plots


class SQP:
    """
    SQP (Sequential Quadratic Programmic) algorithm for FMU parameter estimation. Based on SLSQP solver from SciPy.
    """

    COM_POINTS = 500  # Default number of communication points, should be adjusted to the number of samples
    TMP_SUMMARY = pd.DataFrame()  # Summary placeholder

    def __init__(self, fmu_path, inp, known, est, ideal, sqp_opts={}, opts=None, ftype='NRMSE'):
        """
        :param fmu_path: string, absolute path to the FMU
        :param inp: DataFrame, columns with input timeseries, index in seconds
        :param known: Dictionary, key=parameter_name, value=value
        :param est: Dictionary, key=parameter_name, value=tuple (guess value, lo limit, hi limit), guess can be None
        :param ideal: DataFrame, ideal solution to be compared with model outputs (variable names must match)
        :param dict sqp_opts: Additional options passed to the SLSQP solver in SciPy
        :param dict opts: Additional FMI options to be passed to the simulator (consult FMI specification)
        :param string ftype: Cost function type. Currently 'NRMSE' (advised for multi-objective estimation) or 'RMSE'.
        """
        assert inp.index.equals(ideal.index), 'inp and ideal indexes are not matching'

        # SLSQP soler options
        self.sqp_opts = {'disp': True, 'iprint': 2, 'maxiter': 150, 'ftol': 0.00000001}
        if len(sqp_opts) > 0:
            for key in sqp_opts:
                self.sqp_opts[key] = sqp_opts[key]

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
        self.model = SQP._get_model_instance(fmu_path, inp, known_df, est, output_names, opts)

        # Outputs
        self.summary = pd.DataFrame()
        self.res = pd.DataFrame()
        self.best_err = 9999.

        # Temporary placeholder for summary
        # It needs to be stored as class variable, because it has to be updated
        # from a static method used as callback
        SQP.TMP_SUMMARY = pd.DataFrame(columns=[x.name for x in self.est])

        # Log
        LOGGER.info('SQP initialized... =========================')

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
            for v, ep in zip(x, self.est):
                parameters[ep.name] = v
            self.model.set_param(parameters)
            result = self.model.simulate(com_points=SQP.COM_POINTS)
            err = calc_err(result, self.ideal, ftype=self.ftype)['tot']
            # Update best error and result
            if err < self.best_err:
                self.best_err = err
                self.res = result

            return err

        x0 = [x.value for x in self.est]      # Initial guess
        b = [(x.lo, x.hi) for x in self.est]  # Parameter bounds

        xres = minimize(objective, x0, bounds=b, constraints=[],
                        method='SLSQP', callback=SQP._callback,
                        options={'disp': True, 'iprint': 2, 'maxiter': 50})
        par = xres.x

        # Save summary
        self.summary = SQP.TMP_SUMMARY.copy()
        SQP.TMP_SUMMARY = pd.DataFrame(columns=[x.name for x in self.est]) # Reset temp placeholder
        self.summary.to_csv('sqp_test.csv')
        print(self.summary)

        return par

    @staticmethod
    def _callback(xk):
        SQP.TMP_SUMMARY = SQP.TMP_SUMMARY.append({n: v for n, v in zip(SQP.TMP_SUMMARY.columns, xk)},
                                                 ignore_index=True)

    @staticmethod
    def _get_model_instance(fmu_path, inputs, known_pars, est, output_names, opts=None):
        model = Model(fmu_path, opts)
        model.set_input(inputs)
        model.set_param(known_pars)
        model.set_param(estpars_2_df(est))
        model.set_outputs(output_names)
        return model
